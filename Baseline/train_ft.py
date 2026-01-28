
"""
Fine-Tuning (FT) - Full model fine-tuning
All CLIP parameters are trainable

Usage:
    # 基本用法（只需指定数据集名称）
    python train_ft.py --dataset stanford_dogs  --batch_size 64
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import yaml
from models import CLIPClassifier
from baseline_utils.dataset import create_dataloaders
from baseline_utils.training import (
    train_epoch,
    evaluate,
    get_optimizer,
    CosineScheduleWithWarmup,
)


class ModelEMA:
    """Exponential Moving Average (EMA) for model parameters.

    - Keeps a shadow copy of model parameters updated as:
        ema = decay * ema + (1 - decay) * param
    - Provide apply_shadow()/restore() to evaluate/save with EMA weights.
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = None
        # Only track trainable parameters for EMA
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                # In case new params appear
                self.shadow[name] = p.detach().clone()
                continue
            ema_p = self.shadow[name]
            ema_p.mul_(d).add_(p.detach(), alpha=(1.0 - d))

    @torch.no_grad()
    def apply_shadow(self, model: torch.nn.Module):
        # Backup current params then load EMA to model
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[name] = p.detach().clone()
            if name in self.shadow:
                p.data.copy_(self.shadow[name].data)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module):
        if self.backup is None:
            return
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name in self.backup:
                p.data.copy_(self.backup[name].data)
        self.backup = None


def set_seed(seed: int):
    """Set random seeds for reproducibility across Python/NumPy/PyTorch.

    Note: Deterministic CuDNN is enabled which may slightly reduce performance.
    """
    if seed is None or seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic behavior for cuDNN (may impact speed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def main(args):
    """Fine-Tuning 训练主函数

    Args:
        args: 命令行参数对象（已与 config 合并）
    """
    # --- 设备设置 ---
    # 优先使用命令行指定的 GPU，否则自动选择第一个可用的 GPU
    if args.gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using specified GPU: cuda:{args.gpu_id}")
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using default GPU: cuda:0")
    else:
        device = torch.device('cpu')
        print(f"Using device: CPU")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设定随机种子（可选）
    if getattr(args, 'seed', None) is not None and args.seed >= 0:
        print(f"Seeding everything with seed={args.seed}")
        set_seed(args.seed)

    # Load data（根据配置选择 STN 或 Base (transfer-style) 预处理）
    dataloaders, num_classes = create_dataloaders(
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        preprocess=getattr(args, 'preprocess', 'stn'),
        seed=getattr(args, 'seed', None)
    )
    
    # Create model
    print(f"\n{'='*50}")
    print(f"Creating CLIP model: {args.model_name}")
    print(f"{'='*50}")
    model = CLIPClassifier(
        model_name=args.model_name,
        num_classes=num_classes,
        device=device,
        classifier_type=getattr(args, 'classifier_type', 'linear'),
        head_hidden_dim=getattr(args, 'head_hidden_dim', None),
        head_dropout=getattr(args, 'head_dropout', 0.0)
    )
    # Set to fine-tuning mode (all parameters trainable)
    model.unfreeze_backbone()
    
    total_params, trainable_params = model.count_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    # Loss and optimizer
    # 从配置读取标签平滑（默认0.1；配置A建议 0.0）
    criterion = nn.CrossEntropyLoss(label_smoothing=getattr(args, 'label_smoothing', 0.1))
    optimizer = get_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer,
        betas=(getattr(args, 'beta1', 0.9), getattr(args, 'beta2', 0.999)),
        eps=getattr(args, 'eps', 1e-8),
        momentum=getattr(args, 'momentum', 0.9),
        nesterov=getattr(args, 'nesterov', False),
        dampening=getattr(args, 'dampening', 0.0),
        llrd_decay=getattr(args, 'llrd_decay', 0.65),
        head_lr_mult=getattr(args, 'head_lr_mult', 5.0),
    )
    # 简要打印参数组学习率，便于确认 LLRD 是否生效
    try:
        group_lrs = [pg.get('lr', args.lr) for pg in optimizer.param_groups]
        print(f"Optimizer param groups: {len(group_lrs)} groups")
        print("  LRs (first 8 shown):", [f"{lr:.6e}" for lr in group_lrs[:8]], ("..." if len(group_lrs) > 8 else ""))
    except Exception:
        pass
    
    # --- 调度器设置（统一 step-based） ---
    steps_per_epoch = len(dataloaders['train'])
    num_training_steps = int(args.epochs * steps_per_epoch)
    if getattr(args, 'warmup_epochs', 0) and args.warmup_epochs > 0:
        num_warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    else:
        num_warmup_steps = int(getattr(args, 'warmup_factor', 0.0) * num_training_steps)
    min_factor = 0.0
    if getattr(args, 'eta_min', None) is not None:
        min_factor = float(args.eta_min) / float(max(args.lr, 1e-12))
    num_cycles = getattr(args, 'num_cycles', 0.5)
    scheduler = CosineScheduleWithWarmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_factor=min_factor,
        num_cycles=num_cycles,
    )
    print(f"  - Scheduler: Cosine (step-based) with warmup")
    print(f"    - Total steps: {num_training_steps}, Warmup steps: {num_warmup_steps}, min_factor: {min_factor:.6f}, num_cycles: {getattr(scheduler, 'num_cycles', 0.5)}")

    # --- EMA 设置 ---
    ema = None
    if getattr(args, 'ema_enabled', True):
        ema_decay = float(getattr(args, 'ema_decay', 0.999))
        ema = ModelEMA(model, decay=ema_decay)
        print(f"  - EMA: enabled (decay={ema_decay})")
    else:
        print("  - EMA: disabled")
    
    # Training loop
    print(f"\n{'='*50}")
    print("Starting Fine-Tuning...")
    print(f"{'='*50}\n")
    
    best_val_acc = 0.0
    best_epoch_no_improve = 0  # 早停计数器
    results = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'best_val_acc': 0.0,
        'best_epoch': 0
    }

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model,
            dataloaders['train'],
            criterion,
            optimizer,
            device,
            epoch,
            scheduler=scheduler,  # step-based: 传入 per-step scheduler
            max_grad_norm=getattr(args, 'max_grad_norm', 1.0),
            ema=ema,
        )
        
        # Validate
        if ema is not None:
            ema.apply_shadow(model)
        val_loss, val_acc, _, _ = evaluate(
            model, dataloaders['val'], criterion, device, split='Val'
        )
        if ema is not None:
            ema.restore(model)
        

        
        # Save results
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            results['best_val_acc'] = best_val_acc
            results['best_epoch'] = epoch
            best_epoch_no_improve = 0
            
            save_path = os.path.join(args.output_dir, 'best_model_ft.pth')
            # 用 EMA 权重保存最佳模型（若启用）
            if ema is not None:
                ema.apply_shadow(model)
                model.save_checkpoint(save_path, epoch, optimizer, best_val_acc)
                ema.restore(model)
            else:
                model.save_checkpoint(save_path, epoch, optimizer, best_val_acc)
            print(f"  ✓ New best model! Val Acc: {val_acc:.4f}")
        
        print(f"  Best Val Acc so far: {best_val_acc:.4f} (Epoch {results['best_epoch']})")
        print(f"{'-'*50}\n")

        # 早停逻辑：连续若干epoch未提升则停止（从配置中读取patience，默认10）
        if val_acc <= best_val_acc + 1e-8:  # 未提升
            best_epoch_no_improve += 1
        if best_epoch_no_improve >= getattr(args, 'patience', 10):
            print(f"⏹️  早停: 连续{best_epoch_no_improve}个epoch验证准确率未提升，停止训练。")
            break
    
    # Test evaluation
    if 'test' in dataloaders:
        print(f"\n{'='*50}")
        print("Evaluating on test set...")
        print(f"{'='*50}\n")
        
        # 加载最佳模型
        checkpoint_path = os.path.join(args.output_dir, 'best_model_ft.pth')
        model.load_checkpoint(checkpoint_path)
        
        # 评估（若启用 EMA 则使用 EMA 权重）
        if ema is not None:
            ema.apply_shadow(model)
        test_loss, test_acc, _, _ = evaluate(
            model, dataloaders['test'], criterion, device, split='Test'
        )
        if ema is not None:
            ema.restore(model)
        
        results['test_loss'] = test_loss
        results['test_acc'] = test_acc
        
        print(f"\nTest Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Acc: {test_acc:.4f}")
    
    # Save final results
    results_path = os.path.join(args.output_dir, 'results_ft.json')
    # Inline save results as JSON (save_results helper removed)
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"✓ Results saved to {results_path}")
    
    print(f"\n{'='*50}")
    print("Fine-Tuning completed!")
    print(f"Best Val Acc: {best_val_acc:.4f} (Epoch {results['best_epoch']})")
    if 'test_acc' in results:
        print(f"Test Acc: {results['test_acc']:.4f}")
    print(f"{'='*50}\n")


# Note: `load_config` was removed; configuration is read inline in __main__ using yaml.safe_load


def merge_config_with_args(config, args):
    """
    合并配置文件和命令行参数
    
    功能：将YAML配置文件的参数加载到args对象中
    
    策略：
    - 训练超参数（lr, epochs, optimizer等）：直接使用配置文件，不可覆盖
    - 硬件参数（batch_size, num_workers）：命令行优先，配置文件作为默认值
    - 输出路径：命令行优先，配置文件作为默认值
    
    Args:
        config: 从YAML文件加载的配置字典
        args: 命令行参数对象
    
    Returns:
        args: 更新后的参数对象
    """
    # 从配置文件读取所有必需参数
    args.dataset_name = config['dataset']['name']
    args.data_root = config['dataset']['data_root']
    args.model_name = config['model']['name']
    # 数据预处理方案（图片尺寸由预处理方案内部硬编码：Base=224, STN=448）
    args.preprocess = config['dataset'].get('preprocess', 'stn')
    
    # FT训练配置（从配置文件读取）
    ft_config = config['ft']
    
    # --- 1. 加载通用训练参数 ---
    args.epochs = ft_config['epochs']
    args.patience = ft_config.get('patience', 10)
    args.scheduler = ft_config.get('scheduler', 'cosine')
    args.warmup_epochs = ft_config.get('warmup_epochs', 0)
    # 兼容打印与备用计算路径所需的 warmup_factor 字段
    args.warmup_factor = ft_config.get('warmup_factor', 0.1)
    args.eta_min = ft_config.get('eta_min', 0.0)
    args.max_grad_norm = ft_config.get('max_grad_norm', 1.0)
    args.label_smoothing = ft_config.get('label_smoothing', 0.0)
    args.head_lr_mult = ft_config.get('head_lr_mult', 5.0)

    # --- 2. 根据 optimizer 选择加载特定参数 ---
    optimizer_type = ft_config.get('optimizer', 'adamw').lower()
    args.optimizer = optimizer_type
    
    opt_params_key = f"{optimizer_type}_params"
    if opt_params_key in ft_config:
        # 新式/分组配置（推荐）
        opt_params = ft_config[opt_params_key]
        args.lr = float(opt_params['lr'])
        args.weight_decay = float(opt_params['weight_decay'])
        args.llrd_decay = float(opt_params.get('llrd_decay', 1.0))  # 对 SGD 默认为 1.0（无 LLRD）

        if optimizer_type == 'adamw':
            args.beta1 = opt_params.get('beta1', 0.9)
            args.beta2 = opt_params.get('beta2', 0.999)
            args.eps = opt_params.get('eps', 1e-8)
        elif optimizer_type == 'sgd':
            args.momentum = opt_params.get('momentum', 0.9)
            args.nesterov = opt_params.get('nesterov', False)
            args.dampening = opt_params.get('dampening', 0.0)
    else:
        # 兼容旧式/扁平配置（当前 CUB 配置使用该形式）
        args.lr = float(ft_config.get('lr', 2.5e-5))
        args.weight_decay = float(ft_config.get('weight_decay', 0.01))
        args.llrd_decay = float(ft_config.get('llrd_decay', 0.65))
        if optimizer_type == 'adamw':
            args.beta1 = ft_config.get('beta1', 0.9)
            args.beta2 = ft_config.get('beta2', 0.999)
            args.eps = ft_config.get('eps', 1e-8)
        elif optimizer_type == 'sgd':
            args.momentum = ft_config.get('momentum', 0.9)
            args.nesterov = ft_config.get('nesterov', False)
            args.dampening = ft_config.get('dampening', 0.0)

    # 分类头配置（可选，向后兼容）
    clf_cfg = ft_config.get('classifier', {})
    args.classifier_type = str(clf_cfg.get('type', 'linear')).lower()
    # 校验 classifier_type
    if args.classifier_type not in ('linear', 'mlp'):
        raise ValueError("ft.classifier.type must be 'linear' or 'mlp'")
    args.head_hidden_dim = clf_cfg.get('hidden_dim', None)
    args.head_dropout = float(clf_cfg.get('dropout', 0.0))

    # 说明：当 type 为 'linear' 时，hidden_dim 与 dropout 将在模型构建时被忽略，无需在此处修改其值。

    # EMA 设置（可选）
    args.ema_enabled = ft_config.get('ema_enabled', True)
    args.ema_decay = ft_config.get('ema_decay', 0.999)

    # 可选参数：命令行可覆盖配置文件
    if args.batch_size is None:
        # 从 FT 配置读取默认 batch_size（配置文件中按实验细分）
        args.batch_size = config['ft'].get('batch_size', 32)
    if args.num_workers is None:
        args.num_workers = config['dataset']['num_workers']
    
    # Use output directory defined under FT section; enforce presence
    try:
        args.output_dir = config['ft']['output_dir']
    except KeyError as e:
        raise KeyError("配置缺少 ft.output_dir，请在对应数据集的 YAML 中添加 ft.output_dir") from e
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args

#加载配置文件参数或获取命令行参数
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-Tuning (FT) for CLIP')
    
    # ============================================================================
    # 数据集参数 (必需)
    # ============================================================================
    parser.add_argument('--dataset', type=str, required=True,
                        help='数据集名称\n'
                             '例: stanford_dogs, cub, food101, oxford_pets, dtd, etc.\n'
                             '自动加载 configs/<dataset>.yaml 配置文件\n'
                             '使用方式: python train_ft.py --dataset stanford_dogs')
    
    # ============================================================================
    # 可选参数 (用于覆盖配置文件中的值)
    # ============================================================================
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小 (可选，覆盖配置文件)\n'
                             '根据GPU显存调整，FT默认32')
    
    parser.add_argument('--num_workers', type=int, default=None,
                        help='数据加载进程数 (可选，覆盖配置文件)\n'
                             '根据CPU核心数调整，默认4-8')
    
    parser.add_argument('--gpu_id', type=int, default=None,
                        help='指定单个GPU的编号 (可选, 例如: 0 或 1)\n'
                             '默认使用第一个可用的GPU')

    parser.add_argument('--seed', type=int, default=42,
                        help='随机数种子（默认42）。设置为 -1 以不固定随机性')
    
    args = parser.parse_args()
    
    # 根据数据集名称自动加载配置文件
    config_file = f"configs/{args.dataset}.yaml"
    print(f"数据集: {args.dataset}")
    print(f"配置文件: {config_file}")
    
    # 检查配置文件是否存在
    if not os.path.exists(config_file):
        print(f"\n❌ 错误: 配置文件不存在: {config_file}")
        print(f"\n可用的数据集:")
        if os.path.exists('configs'):
            import glob
            available_configs = glob.glob('configs/*.yaml')
            for cfg in sorted(available_configs):
                dataset_name = os.path.basename(cfg).replace('.yaml', '')
                print(f"  - {dataset_name}")
        parser.error(f"配置文件不存在: {config_file}\n请检查数据集名称是否正确")
    
    # 加载配置文件（inline: 直接读取 YAML，而不使用单独的 load_config 函数）
    print(f"正在加载配置...")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    args = merge_config_with_args(config, args)
    print(f"✓ 配置加载成功")
    print(f"  数据路径: {args.data_root}")
    print(f"  模型: {args.model_name}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  训练轮数: {args.epochs}, 预热: {args.warmup_epochs} (factor={args.warmup_factor}), eta_min: {args.eta_min}")
    print(f"  学习率: {args.lr}, 权重衰减: {args.weight_decay}, LLRD: {getattr(args,'llrd_decay', 0.65)}")
    print(f"  EMA: {'on' if getattr(args,'ema_enabled', True) else 'off'} (decay={getattr(args,'ema_decay', 0.999)})")
    print()
    
    main(args)


