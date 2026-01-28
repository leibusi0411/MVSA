"""
Linear Probing (LP) - Only train the classifier head
CLIP backbone is frozen

Usage:
    # 基本用法（只需指定数据集名称）
    python train_lp.py --dataset stanford_dogs
    
    # 调整batch size（根据GPU显存）
    python train_lp.py --dataset cub --batch_size 256
    
    # 使用shell脚本运行
    bash scripts/run_single.sh stanford_dogs lp
"""
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from models import CLIPClassifier
from baseline_utils.dataset import create_dataloaders
from baseline_utils.training import train_epoch, evaluate, get_optimizer, get_scheduler, CosineScheduleWithWarmup
from tqdm import tqdm

def extract_features(model, dataloader, device):
    """
    从冻结的CLIP backbone提取特征
    
    功能：遍历数据集，使用冻结的CLIP模型提取所有图像的特征向量
    用途：为sklearn的LogisticRegression提供输入特征
    
    Args:
        model: CLIP分类器模型（backbone已冻结）
        dataloader: 数据加载器
        device: 计算设备（cuda/cpu）
    
    Returns:
        features: numpy数组，shape=(N, feature_dim)，所有样本的特征
        labels: numpy数组，shape=(N,)，所有样本的标签
    """
    model.eval()
    
    all_features = []
    all_labels = []
    
    print("Extracting features...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            features = model.get_features(images)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    print(f"✓ Extracted features: {features.shape}")
    return features, labels

def train_with_sklearn(args, model, dataloaders, device):
    """
    使用sklearn的LogisticRegression训练线性探测（推荐方法）
    
    功能：
    1. 提取训练集和验证集的CLIP特征
    2. 使用网格搜索找到最优的正则化参数C
    3. 将最佳模型的权重加载到PyTorch模型中
    
    优点：
    - 速度快，无需迭代训练
    - 自动超参数搜索（100个C值）
    - 数值稳定性好
    
    Args:
        args: 命令行参数
        model: CLIP分类器模型
        dataloaders: 包含train/val的数据加载器字典
        device: 计算设备
    
    Returns:
        best_val_acc: 最佳验证集准确率
        results: 训练结果字典（包含所有C值和对应准确率）
    """
    print(f"\n{'='*50}")
    print("Training with sklearn LogisticRegression")
    print(f"{'='*50}\n")
    
    # Extract features
    train_features, train_labels = extract_features(model, dataloaders['train'], device)
    val_features, val_labels = extract_features(model, dataloaders['val'], device)
    
    # Train logistic regression with different C values (config-driven)
    num_cs = int(getattr(args, 'lp_num_cs', 100))
    start_c = float(getattr(args, 'lp_start_c', -7))
    end_c = float(getattr(args, 'lp_end_c', 2))
    max_iter = int(getattr(args, 'lp_max_iter', 200))
    random_state = int(getattr(args, 'lp_random_state', 0))
    warm_start = bool(getattr(args, 'lp_warm_start', True))

    C_values = np.logspace(start_c, end_c, num_cs)
    best_val_acc = 0.0
    best_C = None
    best_clf = None
    
    print(f"\nSearching for best C value (testing {len(C_values)} values)...")
    
    results = {'C_values': [], 'val_accs': []}
    
    if warm_start:
        # 复用同一个分类器以启用 warm_start 带来的加速
        clf = LogisticRegression(
            C=float(C_values[0]),
            max_iter=max_iter,
            random_state=random_state,
            warm_start=True,
            n_jobs=-1,
        )
        for C in tqdm(C_values):
            clf.set_params(C=float(C))
            clf.fit(train_features, train_labels)
            
            val_pred = clf.predict(val_features)
            val_acc = (val_pred == val_labels).mean()
            
            results['C_values'].append(float(C))
            results['val_accs'].append(float(val_acc))
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_C = C
                best_clf = clf
    else:
        for C in tqdm(C_values):
            clf = LogisticRegression(
                C=float(C),
                max_iter=max_iter,
                random_state=random_state,
                n_jobs=-1
            )
            clf.fit(train_features, train_labels)
        
            val_pred = clf.predict(val_features)
            val_acc = (val_pred == val_labels).mean()
            
            results['C_values'].append(float(C))
            results['val_accs'].append(float(val_acc))
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_C = C
                best_clf = clf
    
    print(f"\n✓ Best C: {best_C:.2e}, Best Val Acc: {best_val_acc:.4f}")
    
    # Set classifier weights to the model
    coef = torch.from_numpy(best_clf.coef_).float()
    intercept = torch.from_numpy(best_clf.intercept_).float()
    
    model.classifier.weight.data = coef.to(device)
    model.classifier.bias.data = intercept.to(device)
    
    results['best_C'] = float(best_C)
    results['best_val_acc'] = float(best_val_acc)
    
    return best_val_acc, results


def train_with_pytorch(args, model, dataloaders, device):
    """
    使用PyTorch优化器训练线性探测（备选方法）
    
    功能：使用标准的梯度下降方法训练分类头
    
    适用场景：
    - 需要更灵活的训练策略
    - 需要使用特定的优化器或学习率调度
    
    注意：通常比sklearn方法慢，且需要手动调整超参数
    
    Args:
        args: 命令行参数（包含epochs, lr等）
        model: CLIP分类器模型
        dataloaders: 包含train/val的数据加载器字典
        device: 计算设备
    
    Returns:
        best_val_acc: 最佳验证集准确率
        results: 训练结果字典（包含每个epoch的loss和acc）
    """
    print(f"\n{'='*50}")
    print("Training with PyTorch optimizer")
    print(f"{'='*50}\n")
    
    # Freeze backbone
    model.freeze_backbone()
    
    total_params, trainable_params = model.count_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # 解析优化器额外参数
    opt_kwargs = {}
    if str(args.optimizer).lower() == 'adamw':
        opt_kwargs.update(
            betas=(float(getattr(args, 'beta1', 0.9)), float(getattr(args, 'beta2', 0.999))),
            eps=float(getattr(args, 'eps', 1e-8)),
        )
    elif str(args.optimizer).lower() == 'sgd':
        opt_kwargs.update(
            momentum=float(getattr(args, 'momentum', 0.9)),
            dampening=float(getattr(args, 'dampening', 0.0)),
            nesterov=bool(getattr(args, 'nesterov', False)),
        )

    optimizer = get_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer,
        head_lr_mult=float(getattr(args, 'lp_head_lr_mult', 5.0)),
        **opt_kwargs,
    )

    # 调度器：支持 epoch-based 与 step-based（带 warmup）
    lp_sched_type = str(getattr(args, 'lp_scheduler', 'none')).lower()
    epoch_scheduler = None
    step_scheduler = None
    if lp_sched_type in ('cosine', 'cos', 'cosineannealing', 'cosineannealinglr'):
        epoch_scheduler = get_scheduler(
            optimizer,
            scheduler_type='cosine',
            epochs=int(getattr(args, 'epochs', 100)),
            eta_min=float(getattr(args, 'lp_eta_min', 0.0)),
        )
    elif lp_sched_type in ('cosine_step', 'step_cosine', 'cosinewarmup', 'cosine_warmup'):
        steps_per_epoch = max(1, len(dataloaders['train']))
        total_steps = steps_per_epoch * int(getattr(args, 'epochs', 100))
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
    results = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'best_val_acc': 0.0,
        'best_epoch': 0
    }
    
    no_improve = 0
    patience = int(getattr(args, 'lp_patience', 0))  # 0 表示不早停

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, dataloaders['train'], criterion, optimizer, device, epoch,
            scheduler=step_scheduler,  # 若使用 step-based 调度器，这里生效；否则为 None
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
        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model / Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            results['best_val_acc'] = best_val_acc
            results['best_epoch'] = epoch
            no_improve = 0
            
            save_path = os.path.join(args.output_dir, 'best_model_lp.pth')
            model.save_checkpoint(save_path, epoch, optimizer, best_val_acc)
            print(f"  ✓ New best model! Val Acc: {val_acc:.4f}")
        else:
            if patience > 0:
                no_improve += 1
                if no_improve >= patience:
                    print(f"⏹️  提前停止: 验证集连续 {no_improve} 个 epoch 未提升。")
                    print(f"  Best Val Acc so far: {best_val_acc:.4f} (Epoch {results['best_epoch']})")
                    break

        # Epoch-based 调度器在每个 epoch 结束后 step
        if epoch_scheduler is not None:
            epoch_scheduler.step()

        print(f"  Best Val Acc so far: {best_val_acc:.4f} (Epoch {results['best_epoch']})")
        print(f"{'-'*50}\n")
    
    return best_val_acc, results


def main(args):
    """
    Linear Probing训练主函数
    
    流程：
    1. 初始化设备和输出目录
    2. 加载数据集（使用CLIP标准预处理）
    3. 创建CLIP模型并冻结backbone
    4. 训练线性分类头（sklearn或PyTorch）
    5. 在测试集上评估
    6. 保存结果和模型权重
    
    Args:
        args: 命令行参数，包含数据集名称、训练配置等
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data（按预处理方案使用固定尺寸：Base=224, STN=448）
    dataloaders, num_classes = create_dataloaders(
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        batch_size=args.batch_size,
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
        device=device
    )
    
    # Train linear probe
    if args.use_sklearn:
        best_val_acc, results = train_with_sklearn(args, model, dataloaders, device)
    else:
        best_val_acc, results = train_with_pytorch(args, model, dataloaders, device)
    
    # Test evaluation
    if 'test' in dataloaders:
        print(f"\n{'='*50}")
        print("Evaluating on test set...")
        print(f"{'='*50}\n")
        
        if not args.use_sklearn:
            # Load best model for PyTorch training
            checkpoint_path = os.path.join(args.output_dir, 'best_model_lp.pth')
            model.load_checkpoint(checkpoint_path)
        
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, _, _ = evaluate(
            model, dataloaders['test'], criterion, device, split='Test'
        )
        
        results['test_loss'] = test_loss
        results['test_acc'] = test_acc
        
        print(f"\nTest Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Acc: {test_acc:.4f}")
    
    # Save brief experiment report (no utility function, inline logic)
    results_path = os.path.join(args.output_dir, 'results_lp.json')
    try:
        import json
        brief = {
            "dataset": args.dataset_name,
            "model": args.model_name,
            "use_sklearn": bool(args.use_sklearn),
            "best_val_acc": float(best_val_acc),
        }
        if 'test_acc' in results:
            try:
                brief['test_acc'] = float(results['test_acc'])
            except Exception:
                pass
        os.makedirs(os.path.dirname(results_path) or '.', exist_ok=True)
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(brief, f, ensure_ascii=False, indent=4)
        print(f"✓ Results saved to {results_path}")
    except Exception as e:
        print(f"⚠️ Failed to save brief results to {results_path}: {e}")
    
    # Save classifier for LP-FT
    classifier_path = os.path.join(args.output_dir, 'classifier_lp.pth')
    torch.save(model.classifier.state_dict(), classifier_path)
    print(f"✓ Saved classifier to {classifier_path}")
    
    print(f"\n{'='*50}")
    print("Linear Probing completed!")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    if 'test_acc' in results:
        print(f"Test Acc: {results['test_acc']:.4f}")
    print(f"{'='*50}\n")


def load_config(config_path):
    """
    从YAML文件加载配置
    
    功能：读取配置文件，返回包含所有训练参数的字典
    
    Args:
        config_path: YAML配置文件路径
    
    Returns:
        config: 配置字典，包含dataset, model, lp, output等配置
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(config, args):
    """
    合并配置文件和命令行参数
    
    功能：将YAML配置文件的参数加载到args对象中
    
    策略：
    - 训练超参数（lr, epochs等）：直接使用配置文件，不可覆盖
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
    # 与 FT/LP-FT 对齐：读取预处理方案（Base=224 | STN=448）
    args.preprocess = config['dataset'].get('preprocess', 'stn')
    
    # LP训练配置（从配置文件读取）
    lp_config = config['lp']
    args.use_sklearn = lp_config.get('use_sklearn', True)
    args.epochs = lp_config.get('epochs', 100)
    args.lr = lp_config.get('lr', 1e-3)
    args.weight_decay = lp_config.get('weight_decay', 0)
    args.optimizer = lp_config.get('optimizer', 'adamw')

    # sklearn 搜索超参数（移动到配置文件控制）
    args.lp_num_cs = lp_config.get('num_cs', 100)
    args.lp_start_c = lp_config.get('start_c', -7)
    args.lp_end_c = lp_config.get('end_c', 2)
    args.lp_max_iter = lp_config.get('max_iter', 200)
    args.lp_random_state = lp_config.get('random_state', 0)
    args.lp_warm_start = lp_config.get('warm_start', True)
    # LP (PyTorch) 标准训练配置（可选）
    args.lp_scheduler = lp_config.get('scheduler', 'none')  # 'none' | 'cosine' | 'cosine_step'
    args.lp_eta_min = lp_config.get('eta_min', 0.0)
    args.lp_warmup_epochs = lp_config.get('warmup_epochs', 5)
    args.lp_min_factor = lp_config.get('min_factor', 0.0)
    args.lp_patience = lp_config.get('patience', 0)  # 0 表示不早停
    args.lp_max_grad_norm = lp_config.get('max_grad_norm', 1.0)
    args.lp_head_lr_mult = lp_config.get('head_lr_mult', 5.0)
    # 优化器附加参数（与 FT 对齐）
    args.beta1 = lp_config.get('beta1', 0.9)
    args.beta2 = lp_config.get('beta2', 0.999)
    args.eps = lp_config.get('eps', 1e-8)
    args.momentum = lp_config.get('momentum', 0.9)
    args.nesterov = lp_config.get('nesterov', False)
    args.dampening = lp_config.get('dampening', 0.0)
    
    # 可选参数：命令行可覆盖配置文件
    if args.batch_size is None:
        args.batch_size = lp_config.get('batch_size', 256)
    if args.num_workers is None:
        args.num_workers = config['dataset']['num_workers']
    # 输出目录兼容：优先 lp.output_dir；否则回退到 output.base_dir/LP 或 ./outputs/LP/<dataset>
    lp_out = lp_config.get('output_dir', None)
    if lp_out is not None:
        args.output_dir = lp_out
    else:
        out_base = None
        if 'output' in config and isinstance(config['output'], dict):
            out_base = config['output'].get('base_dir', None)
        if out_base:
            args.output_dir = os.path.join(out_base, 'LP')
        else:
            args.output_dir = os.path.join('./outputs/LP', args.dataset_name)
    
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Probing (LP) for CLIP')
    
    # ============================================================================
    # 数据集参数 (必需)
    # ============================================================================
    parser.add_argument('--dataset', type=str, required=True,
                        help='数据集名称\n'
                             '例: stanford_dogs, cub, food101, oxford_pets, dtd, etc.\n'
                             '自动加载 configs/<dataset>.yaml 配置文件\n'
                             '使用方式: python train_lp.py --dataset stanford_dogs')
    
    # ============================================================================
    # 可选参数 (用于覆盖配置文件中的值)
    # ============================================================================
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小 (可选，覆盖配置文件)\n'
                             '根据GPU显存调整，LP默认256')
    
    parser.add_argument('--num_workers', type=int, default=None,
                        help='数据加载进程数 (可选，覆盖配置文件)\n'
                             '根据CPU核心数调整，默认4-8')
    
    # Note: output_dir is taken from the dataset config under lp.output_dir
    
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
    print(f"  Batch size: {args.batch_size}")
    print()
    
    main(args)


