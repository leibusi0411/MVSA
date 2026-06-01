"""
无监督多视角STN模型测试脚本

功能：
1. 自动匹配 UN-STN-Config 配置和 checkpoints/unsupervised 下的模型
2. 支持 --ckpt_path 直接指定检查点文件
3. 支持 --list 列出可用检查点
4. 保留 STN 多视角可视化能力

使用示例：
  # 自动匹配最佳模型
  python test_unsupervised_stn.py --dataset_name oxford_pets

  # 指定检查点文件
  python test_unsupervised_stn.py --dataset_name oxford_pets --ckpt_path checkpoints/unsupervised/oxford_pets/xxx_best.pth

  # 列出可用检查点
  python test_unsupervised_stn.py --dataset_name oxford_pets --list

  # 控制可视化输出
  python test_unsupervised_stn.py --dataset_name oxford_pets --visual_batches 3 --max_vis_samples 8
"""

import argparse
import os
import random
import traceback
from glob import glob
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
from torchvision import transforms

from clip import clip
from utils import imagenet_a_lt, imagenet_r_lt
from data_preprocess import load_multi_view_dataset
from stn.multi_view_stn import MultiViewSTNModel
from train_unsupervised_ddp import build_unsupervised_checkpoint_paths


def accuracy(output, target, n, dataset_name):
    """计算Top-1准确率"""
    if dataset_name in ['imagenet-a', 'imageneta']:
        _, pred = output[:, imagenet_a_lt].max(1)
    elif dataset_name in ['imagenet-r', 'imagenetr']:
        _, pred = output[:, imagenet_r_lt].max(1)
    else:
        _, pred = output.max(1)
    correct = pred.eq(target)
    return float(correct.float().sum().cpu().numpy()) / n * 100


def visualize_preprocessed_vs_stn(preprocessed_images, transformed_images, dataset_name,
                                   config=None,
                                   save_dir="visualizations/test_unsupervised_stn_transforms",
                                   batch_idx=0, max_samples=None,
                                   theta_matrices=None, position_params=None):
    """可视化输入图与STN多视角输出"""
    if config is not None:
        stn_config = config.get('stn_config', {})
        model_size = config.get('model_size', 'ViT-B/32')
        num_views = stn_config.get('num_views', 4)
        fusion_mode = stn_config.get('fusion_mode', 'concat')
        hidden_dim = stn_config.get('hidden_dim', 256)

        config_parts = [
            f"model_{model_size.replace('/', '_')}",
            f"views{num_views}",
            f"{fusion_mode}",
            f"dim{hidden_dim}",
        ]
        dataset_save_dir = os.path.join(save_dir, dataset_name, "_".join(config_parts))
    else:
        dataset_save_dir = os.path.join(save_dir, dataset_name, "default_config")

    os.makedirs(dataset_save_dir, exist_ok=True)

    batch_size = preprocessed_images.size(0) if max_samples is None else min(preprocessed_images.size(0), max_samples)
    num_views = transformed_images.size(1)

    def denormalize(tensor):
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(tensor.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(tensor.device)
        return tensor * std + mean

    def tensor_to_pil(tensor):
        tensor = torch.clamp(tensor, 0, 1).cpu()
        return transforms.ToPILImage()(tensor)

    for i in range(batch_size):
        fig_height = 6 if (theta_matrices is not None or position_params is not None) else 4
        fig, axes = plt.subplots(1, 1 + num_views, figsize=(4 * (1 + num_views), fig_height))

        preprocessed_denorm = denormalize(preprocessed_images[i])
        axes[0].imshow(tensor_to_pil(preprocessed_denorm))
        axes[0].set_title('Input 448x448', fontsize=10)
        axes[0].axis('off')

        for view_idx in range(num_views):
            transformed_denorm = denormalize(transformed_images[i, view_idx])
            axes[1 + view_idx].imshow(tensor_to_pil(transformed_denorm))

            title = f'View {view_idx + 1} 224x224'
            if position_params is not None:
                param_x = position_params[i, view_idx * 2].item()
                param_y = position_params[i, view_idx * 2 + 1].item()
                title += f'\nPos: ({param_x:.3f}, {param_y:.3f})'

            axes[1 + view_idx].set_title(title, fontsize=9)
            axes[1 + view_idx].axis('off')

        plt.tight_layout()
        save_path = os.path.join(dataset_save_dir, f'batch{batch_idx}_sample{i:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()


def stn_precise_testing(stn_model, dataloader, device, dataset_name,
                       precomputed_text_features, config=None, model_tag=None,
                       visual_batches=10, max_samples=None):
    """无监督模型测试"""
    stn_model.eval()
    all_image_features = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (images_448, labels) in enumerate(tqdm(dataloader, desc="Testing")):
            images_448 = images_448.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()

            result = stn_model(images_448, mode='test')
            if not (isinstance(result, tuple) and len(result) == 2):
                raise RuntimeError("stn_model(mode='test') 需要返回 (batch_features, vis_data)")

            batch_features, vis_data = result

            if batch_idx < visual_batches:
                try:
                    save_base = "visualizations/test_unsupervised_stn_transforms"
                    if model_tag:
                        save_base = os.path.join(save_base, model_tag)

                    visualize_preprocessed_vs_stn(
                        preprocessed_images=images_448,
                        transformed_images=vis_data['view_images'],
                        dataset_name=dataset_name,
                        config=config,
                        save_dir=save_base,
                        batch_idx=batch_idx,
                        max_samples=max_samples,
                        theta_matrices=vis_data.get('theta_matrices', None),
                        position_params=vis_data.get('position_params', None)
                    )
                except Exception as vis_e:
                    print(f"  Visualization failed: {vis_e}")

            all_image_features.append(batch_features)
            all_targets.append(labels)

    all_image_features = torch.cat(all_image_features, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    total_samples = all_image_features.size(0)

    logits = all_image_features @ precomputed_text_features
    stn_acc = accuracy(logits, all_targets, total_samples, dataset_name)
    print(f"  Tested {total_samples} samples, Top-1 Acc: {stn_acc:.2f}%")
    return stn_acc


def load_state_dict_flexible(stn_model, ckpt_path, device):
    """兼容 best(pure state_dict) 与 latest(含 model_state_dict 的 checkpoint)"""
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict) and 'model_state_dict' in obj:
        state_dict = obj['model_state_dict']
    else:
        state_dict = obj
    stn_model.load_state_dict(state_dict, strict=True)


def resolve_config_path(dataset_name, config_dir='UN-STN-Config'):
    """自动解析配置文件路径"""
    exact_path = os.path.join(config_dir, f"{dataset_name}.yaml")
    if os.path.exists(exact_path):
        return exact_path

    yaml_files = sorted(glob(os.path.join(config_dir, '*.yaml')))
    for ypath in yaml_files:
        try:
            with open(ypath, 'r', encoding='utf-8') as f:
                ycfg = yaml.safe_load(f)
            if (ycfg or {}).get('dataset') == dataset_name:
                return ypath
        except Exception:
            continue

    raise FileNotFoundError(f"未找到数据集 '{dataset_name}' 的配置文件")


def find_checkpoints(ckpt_dir):
    """返回目录下所有 best.pth 文件，按修改时间降序"""
    files = sorted(
        glob(os.path.join(ckpt_dir, '*_best.pth')),
        key=os.path.getmtime,
        reverse=True
    )
    return files


def extract_ckpt_info(ckpt_path, config):
    """从检查点文件名和配置中提取关键参数信息"""
    name = os.path.basename(ckpt_path)
    info = {}
    # 提取 warmup_epochs
    import re
    m = re.search(r'twostage_w(\d+)_', name)
    if m:
        info['warmup_epochs'] = int(m.group(1))
    m = re.search(r'_kl([\d.]+)_', name)
    if m:
        info['kl_weight'] = float(m.group(1))
    m = re.search(r'_td([\d.]+)_', name)
    if m:
        info['dec_target_temp'] = float(m.group(1))
    return info


def main():
    parser = argparse.ArgumentParser(description='无监督多视角STN模型测试')
    parser.add_argument('--dataset_name', type=str, default='oxford_pets',
                        help='数据集名称')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='直接指定检查点文件路径（优先级最高）')
    parser.add_argument('--list', action='store_true',
                        help='列出可用检查点，不运行测试')
    parser.add_argument('--batch_size', type=int, default=32, help='测试批次大小')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--device', type=str, default=None, help='计算设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--visual_batches', type=int, default=3,
                        help='前N个batch保存可视化')
    parser.add_argument('--max_vis_samples', type=int, default=8,
                        help='每个batch最多保存样本数，0=全部')
    parser.add_argument('--all', action='store_true',
                        help='测试目录下所有best检查点')
    args = parser.parse_args()

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("=" * 50)
    print("  无监督多视角STN模型测试")
    print("=" * 50)
    print(f"数据集:   {args.dataset_name}")
    print(f"设备:     {args.device}")

    try:
        # 加载配置
        config_path = resolve_config_path(args.dataset_name)
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        model_size = config['model_size']
        stn_config = config['stn_config']
        data_path = config['data_path']
        config_dataset = config.get('dataset', args.dataset_name)

        print(f"配置:     {config_path}")
        print(f"CLIP:     {model_size}")
        print(f"STN:      views={stn_config.get('num_views', 4)}, "
              f"fusion={stn_config.get('fusion_mode', 'simple')}, "
              f"dim={stn_config.get('hidden_dim', 512)}")

        # 检查点目录
        ckpt_dir = os.path.join('checkpoints', 'unsupervised', config_dataset)
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"检查点目录不存在: {ckpt_dir}")

        # --list 模式
        if args.list:
            print(f"\n检查点目录: {ckpt_dir}/")
            all_ckpts = find_checkpoints(ckpt_dir)
            if not all_ckpts:
                print("  (无检查点文件)")
            else:
                for i, p in enumerate(all_ckpts):
                    info = extract_ckpt_info(p, config)
                    mtime = os.path.getmtime(p)
                    from datetime import datetime
                    dt = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                    size_mb = os.path.getsize(p) / 1024 / 1024
                    extra = ', '.join(f'{k}={v}' for k, v in info.items())
                    print(f"  [{i+1}] {os.path.basename(p)}")
                    print(f"       {dt}, {size_mb:.0f}MB, {extra}")
            return

        # 确定要测试的检查点
        if args.ckpt_path:
            ckpt_list = [args.ckpt_path]
            if not os.path.exists(args.ckpt_path):
                raise FileNotFoundError(f"检查点不存在: {args.ckpt_path}")
        elif args.all:
            ckpt_list = find_checkpoints(ckpt_dir)
            if not ckpt_list:
                raise FileNotFoundError(f"目录下无检查点: {ckpt_dir}")
            print(f"\n扫描到 {len(ckpt_list)} 个检查点")
        else:
            ckpt_paths = build_unsupervised_checkpoint_paths(config_dataset, config)
            config_ckpt = ckpt_paths['best']
            if not os.path.exists(config_ckpt):
                raise FileNotFoundError(
                    f"配置文件构造的检查点不存在:\n"
                    f"  {config_ckpt}\n"
                    f"用 --ckpt_path 直接指定路径，或 --all 扫描目录下所有检查点"
                )
            ckpt_list = [config_ckpt]
            print(f"\n配置匹配: {os.path.basename(config_ckpt)}")

        # 加载模型
        print(f"\n加载 CLIP: {model_size}")
        clip_model, _ = clip.load(model_size, device=args.device)
        clip_model = clip_model.float()

        num_views = stn_config.get('num_views', 4)
        stn_model = MultiViewSTNModel(clip_model, stn_config, num_views=num_views).to(args.device).float()

        # 加载数据
        test_dataloader = load_multi_view_dataset(
            dataset_name=config_dataset,
            data_path=data_path,
            split='test',
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            target_size=448,
            scale_short_edge=512,
            flip_prob=0.0,
            center_crop=True,
            persistent_workers=False,
        )

        # 加载文本特征
        model_name = model_size.replace('/', '_')
        text_features_path = f"text_features/{config_dataset}_{model_name}.pt"
        if not os.path.exists(text_features_path):
            raise FileNotFoundError(f"文本特征不存在: {text_features_path}")
        text_features = torch.load(text_features_path, map_location=args.device).float().to(args.device)

        max_samples = None if args.max_vis_samples == 0 else args.max_vis_samples
        results = []

        # 遍历测试所有检查点
        for ckpt_path in ckpt_list:
            ckpt_name = os.path.basename(ckpt_path)
            info = extract_ckpt_info(ckpt_path, config)
            tag = '_'.join(f'{k}{v}' for k, v in info.items()) if info else 'best'

            print(f"\n{'─' * 40}")
            print(f"  测试: {ckpt_name}")
            if info:
                print(f"  参数: {info}")
            print(f"{'─' * 40}")

            load_state_dict_flexible(stn_model, ckpt_path, args.device)
            print(f"  Model loaded successfully")

            acc = stn_precise_testing(
                stn_model=stn_model,
                dataloader=test_dataloader,
                device=args.device,
                dataset_name=config_dataset,
                precomputed_text_features=text_features,
                config=config,
                model_tag=tag,
                visual_batches=args.visual_batches,
                max_samples=max_samples,
            )
            results.append((ckpt_name, acc))

        # 汇总
        print(f"\n{'=' * 50}")
        print(f"  测试汇总")
        print(f"{'=' * 50}")
        for name, acc in results:
            print(f"  {acc:.2f}%  |  {name}")
        if len(results) > 1:
            best_name, best_acc = max(results, key=lambda x: x[1])
            print(f"\n  最佳: {best_acc:.2f}% ({best_name})")

    except Exception as e:
        print(f"\nTest failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
