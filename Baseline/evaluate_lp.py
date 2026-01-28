import os
import argparse
import torch
import torch.nn as nn
import yaml
import sys
from tqdm import tqdm

# 添加当前目录到 path
sys.path.append(os.getcwd())

from models import CLIPClassifier
from baseline_utils.dataset import create_dataloaders
from baseline_utils.training import evaluate

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(args):
    # 1. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. 加载配置
    config_file = f"configs/{args.dataset}.yaml"
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    print(f"Loading config from: {config_file}")
    config = load_config(config_file)
    
    # 提取必要的配置参数
    data_root = config['dataset']['data_root']
    model_name = config['model']['name']
    preprocess = config['dataset'].get('preprocess', 'stn')
    
    # 确定 Checkpoint 路径
    if args.ckpt:
        checkpoint_path = args.ckpt
    else:
        # 尝试从配置文件推断
        lp_out = config.get('lp', {}).get('output_dir', None)
        if not lp_out:
            lp_out = os.path.join('./outputs/LP', args.dataset)
        checkpoint_path = os.path.join(lp_out, 'best_model_lp.pth')
    
    print(f"Model Checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # 3. 准备数据加载器
    # 注意：这里我们只关心测试集，但 create_dataloaders 通常会一起创建。
    # 为了避免 OOM，我们强制使用较小的 batch_size 进行评估
    eval_batch_size = args.batch_size
    print(f"Creating dataloaders (Eval Batch Size: {eval_batch_size})...")
    
    dataloaders, num_classes = create_dataloaders(
        dataset_name=args.dataset,
        data_root=data_root,
        batch_size=eval_batch_size,
        num_workers=args.num_workers,
        preprocess=preprocess
    )
    
    if 'test' in dataloaders:
        test_loader = dataloaders['test']
        split_name = 'Test'
    elif 'val' in dataloaders:
        print("⚠️ Warning: No 'test' set found, falling back to 'val' set.")
        test_loader = dataloaders['val']
        split_name = 'Validation'
    else:
        raise ValueError("No validation or test set found in dataloaders.")

    # 4. 初始化模型
    print(f"Creating CLIP model: {model_name} (Classes: {num_classes})")
    model = CLIPClassifier(
        model_name=model_name,
        num_classes=num_classes,
        device=device
    )
    
    # 5. 加载权重
    print("Loading checkpoint weights...")
    # 假设 checkpoint 保存的是 model.state_dict() 或包含它的字典
    # 根据 train_lp.py 的 save_checkpoint，它可能保存了完整信息
    try:
        model.load_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"Standard load failed ({e}), trying strict=False load...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    model.eval()
    
    # 6. 开始评估
    criterion = nn.CrossEntropyLoss()
    print(f"\n{'='*50}")
    print(f"Starting Evaluation on {split_name} Set")
    print(f"{'='*50}\n")
    
    loss, acc, _, _ = evaluate(
        model, test_loader, criterion, device, split=split_name
    )
    
    print(f"\n{'-'*50}")
    print(f"FINAL RESULT ({split_name})")
    print(f"Dataset: {args.dataset}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Loss:     {loss:.4f}")
    print(f"{'-'*50}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Trained LP Model')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., imagenet)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation (default: 64)')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--ckpt', type=str, default=None, help='Specific path to checkpoint (optional)')
    
    args = parser.parse_args()
    main(args)
