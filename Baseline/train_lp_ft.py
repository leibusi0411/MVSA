"""
LP-FT (Linear Probe then Fine-Tune)
Two-stage training:
1. Stage 1: Linear Probing (freeze backbone, train classifier)
2. Stage 2: Fine-Tuning (unfreeze backbone, train all)

Usage:
    # 基本用法（只需指定数据集名称）
    python train_lp_ft.py --dataset stanford_dogs
    
    # 分阶段批次大小（推荐明确指定或在 YAML 中配置）
    python train_lp_ft.py --dataset cub --lp_batch_size 96 --ft_batch_size 32
    
    # 使用shell脚本运行
    bash scripts/run_single.sh stanford_dogs lp_ft
"""
import os
import argparse
import torch
import torch.nn as nn
import yaml
from models import CLIPClassifier
from baseline_utils.dataset import create_dataloaders
from baseline_utils.training import train_epoch, evaluate, get_optimizer, get_scheduler, save_results, CosineScheduleWithWarmup


def linear_probing_stage(model, dataloaders, device, args):
    """
    第一阶段：线性探测（Linear Probing）
    
    功能：
    1. 冻结CLIP backbone，只训练分类头
    2. 快速学习数据集特定的分类边界
    3. 为第二阶段提供良好的初始化
    
    训练策略：
    - 不使用学习率调度器（简单训练）
    - 使用较大的学习率快速收敛
    - 保存最佳验证准确率的模型
    
    Args:
        model: CLIP分类器模型
        dataloaders: 包含train/val的数据加载器字典
        device: 计算设备
        args: 命令行参数（包含lp_epochs, lp_lr等）
    
    Returns:
        results: 训练结果字典（包含loss和acc历史）
    """
    print(f"\n{'='*50}")
    print("Stage 1: Linear Probing")
    print(f"{'='*50}\n")
    
    # Freeze backbone
    model.freeze_backbone()
    
    total_params, trainable_params = model.count_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    # Loss and optimizer（LP 使用自己的优化器与权重衰减）
    criterion = nn.CrossEntropyLoss()
    # 优化器附加参数（与 FT 对齐）
    lp_opt_kwargs = {}
    if getattr(args, 'lp_optimizer', 'adamw').lower() == 'adamw':
        beta1 = getattr(args, 'lp_beta1', None)
        beta2 = getattr(args, 'lp_beta2', None)
        if beta1 is not None and beta2 is not None:
            lp_opt_kwargs['betas'] = (float(beta1), float(beta2))
        eps = getattr(args, 'lp_eps', None)
        if eps is not None:
            lp_opt_kwargs['eps'] = float(eps)
    else:
        # SGD 参数
        momentum = getattr(args, 'lp_momentum', None)
        if momentum is not None:
            lp_opt_kwargs['momentum'] = float(momentum)
        dampening = getattr(args, 'lp_dampening', None)
        if dampening is not None:
            lp_opt_kwargs['dampening'] = float(dampening)
        nesterov = getattr(args, 'lp_nesterov', None)
        if nesterov is not None:
            lp_opt_kwargs['nesterov'] = bool(nesterov)

    # 可选：分类头倍率（与 LLRD 配合）
    head_lr_mult = float(getattr(args, 'lp_head_lr_mult', 5.0))

    optimizer = get_optimizer(
        model,
        lr=args.lp_lr,
        weight_decay=getattr(args, 'lp_weight_decay', 0.0),
        optimizer_type=getattr(args, 'lp_optimizer', 'adamw'),
        head_lr_mult=head_lr_mult,
        **lp_opt_kwargs,
    )

    # 调度器：epoch-based 或 step-based（带线性 warmup）
    lp_sched_type = str(getattr(args, 'lp_scheduler', 'none')).lower()
    epoch_scheduler = None
    step_scheduler = None
    if lp_sched_type in ('cosine', 'cos', 'cosineannealing', 'cosineannealinglr'):
        epoch_scheduler = get_scheduler(
            optimizer,
            scheduler_type='cosine',
            epochs=int(getattr(args, 'lp_epochs', 50)),
            eta_min=float(getattr(args, 'lp_eta_min', 0.0)),
        )
    elif lp_sched_type in ('cosine_step', 'step_cosine', 'cosinewarmup', 'cosine_warmup'):
        steps_per_epoch = max(1, len(dataloaders['train']))
        total_steps = steps_per_epoch * int(getattr(args, 'lp_epochs', 50))
        warmup_epochs = int(getattr(args, 'lp_warmup_epochs', 5))
        warmup_steps = warmup_epochs * steps_per_epoch
        min_factor = float(getattr(args, 'lp_min_factor', 0.0))
        step_scheduler = CosineScheduleWithWarmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5,
            min_factor=min_factor,
        )
    
    best_val_acc = 0.0
    best_epoch = 0
    results = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
    }
    
    no_improve = 0
    lp_patience = int(getattr(args, 'lp_patience', 0))

    for epoch in range(1, args.lp_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, dataloaders['train'], criterion, optimizer, device, epoch,
            scheduler=step_scheduler,
            max_grad_norm=float(getattr(args, 'lp_max_grad_norm', 1.0)),
        )
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(
            model, dataloaders['val'], criterion, device, split='Val'
        )
        
        # Save results
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.lp_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Track best model / 早停
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            # Save best LP model
            save_path = os.path.join(args.output_dir, 'best_model_lp_stage.pth')
            model.save_checkpoint(save_path, epoch, optimizer, best_val_acc)
            print(f"  ✓ New best model! Val Acc: {val_acc:.4f}")
            no_improve = 0
        else:
            if lp_patience > 0:
                no_improve += 1
                if no_improve >= lp_patience:
                    print(f"⏹️  LP 早停: 连续 {no_improve} 个 epoch 未提升。")
                    print(f"  Best Val Acc so far: {best_val_acc:.4f} (Epoch {best_epoch})")
                    break

        # epoch-based 调度器按 epoch 更新
        if epoch_scheduler is not None:
            epoch_scheduler.step()
        
        print(f"  Best Val Acc so far: {best_val_acc:.4f} (Epoch {best_epoch})")
        print(f"{'-'*50}\n")
    
    results['best_val_acc'] = best_val_acc
    results['best_epoch'] = best_epoch
    
    # Load best LP model
    checkpoint_path = os.path.join(args.output_dir, 'best_model_lp_stage.pth')
    model.load_checkpoint(checkpoint_path)
    
    print(f"\n✓ Stage 1 completed! Best Val Acc: {best_val_acc:.4f} (Epoch {best_epoch})\n")
    
    return results


def fine_tuning_stage(model, dataloaders, device, args):
    """
    第二阶段：精调（Fine-Tuning）
    
    功能：
    1. 解冻CLIP backbone，训练整个模型
    2. 基于Stage 1的分类头初始化继续优化
    3. 使用较小的学习率和学习率调度器
    
    训练策略：
    - 使用cosine annealing学习率调度器
    - 学习率通常比Stage 1小（如1e-6 vs 1e-3）
    - 训练更多epochs以精细调整特征
    
    优势：
    - 避免从随机初始化训练的不稳定性
    - 比直接FT收敛更快更稳定
    - 通常能达到最佳性能
    
    Args:
        model: CLIP分类器模型（已完成LP阶段）
        dataloaders: 包含train/val的数据加载器字典
        device: 计算设备
        args: 命令行参数（包含ft_epochs, ft_lr等）
    
    Returns:
        results: 训练结果字典（包含loss和acc历史）
    """
    print(f"\n{'='*50}")
    print("Stage 2: Fine-Tuning")
    print(f"{'='*50}\n")
    
    # Unfreeze backbone
    model.unfreeze_backbone()
    
    total_params, trainable_params = model.count_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    # Loss and optimizer (smaller learning rate for fine-tuning)
    criterion = nn.CrossEntropyLoss()
    # 传递 AdamW 的 betas/eps（若配置提供），以及可选的 LLRD/头部倍率
    ft_kwargs = {}
    if getattr(args, 'ft_optimizer', 'adamw').lower() == 'adamw':
        beta1 = getattr(args, 'ft_beta1', None)
        beta2 = getattr(args, 'ft_beta2', None)
        if beta1 is not None and beta2 is not None:
            ft_kwargs['betas'] = (float(beta1), float(beta2))
        eps = getattr(args, 'ft_eps', None)
        if eps is not None:
            ft_kwargs['eps'] = float(eps)
    # 可选：从配置注入 llrd/head 倍率（若未来加入，对缺省保持训练库默认值）
    if hasattr(args, 'llrd_decay'):
        ft_kwargs['llrd_decay'] = float(args.llrd_decay)
    if hasattr(args, 'head_lr_mult'):
        ft_kwargs['head_lr_mult'] = float(args.head_lr_mult)

    optimizer = get_optimizer(
        model,
        lr=args.ft_lr,
        weight_decay=getattr(args, 'ft_weight_decay', 0.0),
        optimizer_type=getattr(args, 'ft_optimizer', 'adamw'),
        **ft_kwargs,
    )
    scheduler = get_scheduler(
        optimizer,
        scheduler_type=getattr(args, 'ft_scheduler', 'cosine'),
        epochs=args.ft_epochs,
        eta_min=float(getattr(args, 'ft_eta_min', 0.0)),
    )
    
    best_val_acc = 0.0
    best_epoch = 0
    results = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
    }
    
    no_improve = 0
    ft_patience = int(getattr(args, 'ft_patience', 0))

    for epoch in range(1, args.ft_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, dataloaders['train'], criterion, optimizer, device, epoch,
            max_grad_norm=float(getattr(args, 'ft_max_grad_norm', 1.0)),
        )
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(
            model, dataloaders['val'], criterion, device, split='Val'
        )
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Save results
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.ft_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Track best model / 早停
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            # Save best FT model
            save_path = os.path.join(args.output_dir, 'best_model_lp_ft.pth')
            model.save_checkpoint(save_path, epoch, optimizer, best_val_acc)
            print(f"  ✓ New best model! Val Acc: {val_acc:.4f}")
            no_improve = 0
        else:
            if ft_patience > 0:
                no_improve += 1
                if no_improve >= ft_patience:
                    print(f"⏹️  FT 早停: 连续 {no_improve} 个 epoch 未提升。")
                    print(f"  Best Val Acc so far: {best_val_acc:.4f} (Epoch {best_epoch})")
                    break
        
        print(f"  Best Val Acc so far: {best_val_acc:.4f} (Epoch {best_epoch})")
        print(f"{'-'*50}\n")
    
    results['best_val_acc'] = best_val_acc
    results['best_epoch'] = best_epoch
    
    # Load best FT model
    checkpoint_path = os.path.join(args.output_dir, 'best_model_lp_ft.pth')
    model.load_checkpoint(checkpoint_path)
    
    print(f"\n✓ Stage 2 completed! Best Val Acc: {best_val_acc:.4f} (Epoch {best_epoch})\n")
    
    return results


def main(args):
    """
    LP-FT两阶段训练主函数
    
    流程：
    1. 初始化设备和输出目录
    2. 加载数据集（使用数据增强）
    3. 创建CLIP模型
    4. Stage 1: Linear Probing（冻结backbone）
    5. Stage 2: Fine-Tuning（解冻backbone）
    6. 在测试集上评估最终模型
    7. 保存完整训练结果
    
    与单阶段方法的比较：
    - LP: 只执行Stage 1，快速但性能受限
    - FT: 直接训练全模型，可能不稳定
    - LP-FT: 两阶段训练，兼顾稳定性和性能（推荐）
    
    Args:
        args: 命令行参数，包含数据集名称、两阶段训练配置等
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data with configured preprocessing（分阶段可使用不同的 batch_size）
    # 注意：
    # - preprocess='stn' 时：训练随机裁剪+翻转，验证/测试中心裁剪；默认尺寸 448
    # - preprocess='base' 时：严格对齐 transfer-learning，Resize→CenterCrop→ToTensor；默认尺寸 224
    # LP 阶段 dataloaders
    dataloaders_lp, num_classes = create_dataloaders(
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        batch_size=int(getattr(args, 'lp_batch_size', getattr(args, 'batch_size', 64))),
        num_workers=args.num_workers,
        preprocess=getattr(args, 'preprocess', 'stn')
    )
    
    # Create model
    print(f"\n{'='*50}")
    print(f"Creating CLIP model: {args.model_name}")
    print(f"{'='*50}")
    model = CLIPClassifier(
        model_name=args.model_name,
        num_classes=num_classes,
        device=device,
    )
    
    # Stage 1: Linear Probing
    lp_results = linear_probing_stage(model, dataloaders_lp, device, args)
    
    # Stage 2 之前，若 FT 批次大小不同，则重建 dataloaders
    dataloaders_ft, _ = create_dataloaders(
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        batch_size=int(getattr(args, 'ft_batch_size', getattr(args, 'lp_batch_size', getattr(args, 'batch_size', 64)))),
        num_workers=args.num_workers,
        preprocess=getattr(args, 'preprocess', 'stn')
    )

    # Stage 2: Fine-Tuning
    ft_results = fine_tuning_stage(model, dataloaders_ft, device, args)
    
    # Combine results
    all_results = {
        'lp_stage': lp_results,
        'ft_stage': ft_results,
    }
    
    # Test evaluation
    if 'test' in dataloaders_ft:
        print(f"\n{'='*50}")
        print("Evaluating on test set...")
        print(f"{'='*50}\n")
        
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, _, _ = evaluate(
            model, dataloaders_ft['test'], criterion, device, split='Test'
        )
        
        all_results['test_loss'] = test_loss
        all_results['test_acc'] = test_acc
        
        print(f"\nTest Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Acc: {test_acc:.4f}")
    
    # Save final results
    results_path = os.path.join(args.output_dir, 'results_lp_ft.json')
    save_results(all_results, results_path)
    
    print(f"\n{'='*50}")
    print("LP-FT completed!")
    print(f"LP Stage - Best Val Acc: {lp_results['best_val_acc']:.4f}")
    print(f"FT Stage - Best Val Acc: {ft_results['best_val_acc']:.4f}")
    if 'test_acc' in all_results:
        print(f"Final Test Acc: {all_results['test_acc']:.4f}")
    print(f"{'='*50}\n")


def load_config(config_path):
    """
    从YAML文件加载配置
    
    功能：读取配置文件，返回包含所有训练参数的字典
    
    Args:
        config_path: YAML配置文件路径
    
    Returns:
        config: 配置字典，包含dataset, model, lp_ft, output等配置
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(config, args):
    """
    合并配置文件和命令行参数
    
    功能：将YAML配置文件的参数加载到args对象中
    
    策略：
    - 训练超参数（lp_lr, ft_lr, epochs等）：直接使用配置文件，不可覆盖
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
    
    # 预处理方案：与数据集配置对齐（base / stn / transfer）
    args.preprocess = config['dataset'].get('preprocess', 'stn')

    # LP-FT训练配置（从配置文件读取）
    lp_ft_config = config['lp_ft']
    # Stage 1 (LP)
    args.lp_epochs = lp_ft_config['lp_epochs']
    args.lp_lr = lp_ft_config['lp_lr']
    args.lp_weight_decay = lp_ft_config.get('lp_weight_decay', 0.0)
    args.lp_optimizer = lp_ft_config.get('lp_optimizer', 'adamw')
    # LP 调度与稳定性
    args.lp_scheduler = lp_ft_config.get('lp_scheduler', 'cosine_step')  # 'none' | 'cosine' | 'cosine_step'
    args.lp_eta_min = lp_ft_config.get('lp_eta_min', 0.0)  # epoch-based CosineAnnealingLR
    args.lp_warmup_epochs = lp_ft_config.get('lp_warmup_epochs', 5)  # step-based warmup
    args.lp_min_factor = lp_ft_config.get('lp_min_factor', 0.05)  # step-based 余弦下限比例
    args.lp_patience = lp_ft_config.get('lp_patience', 10)
    args.lp_max_grad_norm = lp_ft_config.get('lp_max_grad_norm', 1.0)
    args.lp_head_lr_mult = lp_ft_config.get('lp_head_lr_mult', 5.0)
    # LP 优化器附加参数
    args.lp_beta1 = lp_ft_config.get('lp_beta1', 0.9)
    args.lp_beta2 = lp_ft_config.get('lp_beta2', 0.999)
    args.lp_eps = lp_ft_config.get('lp_eps', 1e-8)
    args.lp_momentum = lp_ft_config.get('lp_momentum', 0.9)
    args.lp_nesterov = lp_ft_config.get('lp_nesterov', False)
    args.lp_dampening = lp_ft_config.get('lp_dampening', 0.0)
    # Stage 2 (FT)
    args.ft_epochs = lp_ft_config['ft_epochs']
    args.ft_lr = lp_ft_config['ft_lr']
    args.ft_weight_decay = lp_ft_config.get('ft_weight_decay', 1e-4)
    args.ft_optimizer = lp_ft_config.get('ft_optimizer', 'adamw')
    args.ft_scheduler = lp_ft_config.get('ft_scheduler', 'cosine')
    args.ft_eta_min = lp_ft_config.get('ft_eta_min', 0.0)
    args.ft_warmup_epochs = lp_ft_config.get('ft_warmup_epochs', 0)
    args.ft_warmup_factor = lp_ft_config.get('ft_warmup_factor', 0.0)
    args.ft_max_grad_norm = lp_ft_config.get('ft_max_grad_norm', 1.0)
    args.ft_patience = lp_ft_config.get('ft_patience', 0)
    
    # 分阶段 batch_size（不再支持单一 batch_size 兼容模式）
    # 提醒：若仍提供 --batch_size，将被忽略
    if getattr(args, 'batch_size', None) is not None:
        try:
            print("[警告] --batch_size 已废弃，将被忽略。请使用 --lp_batch_size 与 --ft_batch_size。")
        except Exception:
            pass

    # 从命令行读取，若未提供则从配置读取；缺失即报错
    if getattr(args, 'lp_batch_size', None) is None:
        if 'lp_batch_size' not in lp_ft_config:
            raise KeyError("配置缺少 lp_ft.lp_batch_size，请在对应数据集的 YAML 中添加该字段")
        args.lp_batch_size = int(lp_ft_config['lp_batch_size'])
    if getattr(args, 'ft_batch_size', None) is None:
        if 'ft_batch_size' not in lp_ft_config:
            raise KeyError("配置缺少 lp_ft.ft_batch_size，请在对应数据集的 YAML 中添加该字段")
        args.ft_batch_size = int(lp_ft_config['ft_batch_size'])

    if args.num_workers is None:
        args.num_workers = config['dataset']['num_workers']
    if args.output_dir is None:
        # 优先从 lp_ft 段读取输出目录；若缺失则回退到通用默认路径
        args.output_dir = lp_ft_config.get('output_dir', f"./outputs/LP_FT/{args.dataset_name}")
    
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LP-FT (Linear Probe then Fine-Tune) for CLIP')
    
    # ============================================================================
    # 数据集参数 (必需)
    # ============================================================================
    parser.add_argument('--dataset', type=str, required=True,
                        help='数据集名称\n'
                             '例: stanford_dogs, cub, food101, oxford_pets, dtd, etc.\n'
                             '自动加载 configs/<dataset>.yaml 配置文件\n'
                             '使用方式: python train_lp_ft.py --dataset stanford_dogs')
    
    # ============================================================================
    # 可选参数 (用于覆盖配置文件中的值)
    # ============================================================================
    parser.add_argument('--batch_size', type=int, default=None,
                    help='批次大小 (可选，覆盖配置文件；若未提供分阶段参数，则两阶段共用)\n'
                        '根据GPU显存调整，默认按配置文件 lp_ft.batch_size 或 64')
    parser.add_argument('--lp_batch_size', type=int, default=None,
                    help='LP 阶段批次大小 (可选，覆盖配置文件 lp_ft.lp_batch_size)')
    parser.add_argument('--ft_batch_size', type=int, default=None,
                    help='FT 阶段批次大小 (可选，覆盖配置文件 lp_ft.ft_batch_size)')
    
    parser.add_argument('--num_workers', type=int, default=None,
                        help='数据加载进程数 (可选，覆盖配置文件)\n'
                             '根据CPU核心数调整，默认4-8')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录 (可选，覆盖配置文件)\n'
                             '自定义结果保存路径')
    
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
    
    # 加载配置文件
    print(f"正在加载配置...")
    config = load_config(config_file)
    args = merge_config_with_args(config, args)
    print(f"✓ 配置加载成功")
    print(f"  数据路径: {args.data_root}")
    print(f"  模型: {args.model_name}")
    print(f"  预处理: {args.preprocess}")
    print(f"  Batch size (LP/FT): {args.lp_batch_size}/{args.ft_batch_size}")
    print()
    
    main(args)


