"""
无监督多视角STN模型测试脚本

基于 test_multi_stn.py 改造：
1. 使用 UN-STN-Config/{dataset}.yaml 自动匹配无监督配置
2. 加载 checkpoints/unsupervised/{dataset}/ 下的无监督模型
3. 保留 STN 多视角可视化能力
4. 只测试 best 模型（自动按配置构建路径）

使用命令示例：

# 1) 测试默认best模型（推荐）
python test_unsupervised_stn.py --dataset_name oxford_pets

# 2) 控制可视化输出：前3个batch可视化，每个batch最多保存8张
python test_unsupervised_stn.py --dataset_name oxford_pets --visual_batches 3 --max_vis_samples 8

"""

import argparse
import os
import random
import traceback
from glob import glob

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
    """计算Top-1准确率（兼容ImageNet-A/R子集映射）"""
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
        model_size = config.get('model_size', 'ViT-B_32')

        num_views = stn_config.get('num_views', 4)
        fusion_mode = stn_config.get('fusion_mode', 'concat')
        hidden_dim = stn_config.get('hidden_dim', 256)
        kl_weight = stn_config.get('kl_consistency_weight', 0.0)
        fair_weight = stn_config.get('fairness_weight', 0.0)
        dec_weight = stn_config.get('decorrelation_weight', 0.0)

        config_parts = [
            f"model_{model_size.replace('/', '_')}",
            f"views{num_views}",
            f"{fusion_mode}",
            f"dim{hidden_dim}",
            f"kl{kl_weight}",
            f"fair{fair_weight}",
            f"dec{dec_weight}",
        ]
        dataset_save_dir = os.path.join(save_dir, dataset_name, "_".join(config_parts))
    else:
        dataset_save_dir = os.path.join(save_dir, dataset_name, "default_config")

    os.makedirs(dataset_save_dir, exist_ok=True)

    batch_size = preprocessed_images.size(0) if max_samples is None else min(preprocessed_images.size(0), max_samples)
    num_views = transformed_images.size(1)

    def denormalize(tensor, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]):
        mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
        return tensor * std + mean

    def tensor_to_pil(tensor):
        tensor = torch.clamp(tensor, 0, 1).cpu()
        return transforms.ToPILImage()(tensor)

    for i in range(batch_size):
        fig_height = 6 if (theta_matrices is not None or position_params is not None) else 4
        fig, axes = plt.subplots(1, 1 + num_views, figsize=(4 * (1 + num_views), fig_height))

        preprocessed_denorm = denormalize(preprocessed_images[i])
        axes[0].imshow(tensor_to_pil(preprocessed_denorm))
        axes[0].set_title('Input_image 448x448', fontsize=10)
        axes[0].axis('off')

        for view_idx in range(num_views):
            transformed_denorm = denormalize(transformed_images[i, view_idx])
            axes[1 + view_idx].imshow(tensor_to_pil(transformed_denorm))

            title = f'STN View {view_idx + 1} 224x224'
            if position_params is not None:
                param_x = position_params[i, view_idx * 2].item()
                param_y = position_params[i, view_idx * 2 + 1].item()
                title += f'\nPos: ({param_x:.3f}, {param_y:.3f})'

            axes[1 + view_idx].set_title(title, fontsize=9)
            axes[1 + view_idx].axis('off')

        plt.tight_layout()

        if theta_matrices is not None:
            fig.text(0.5, 0.02,
                     f'Sample {i}: Transformation matrices show scale and translation parameters',
                     ha='center', fontsize=8, style='italic')

        save_path = os.path.join(dataset_save_dir, f'batch{batch_idx}_sample{i:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()


def stn_precise_testing(stn_model, dataloader, device, dataset_name,
                       precomputed_text_features, config=None, model_tag=None,
                       visual_batches=10, max_samples=None):
    """无监督模型测试（始终使用可视化模式提取特征）。"""
    stn_model.eval()
    all_image_features = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (images_448, labels) in enumerate(tqdm(dataloader, desc="STN特征提取")):
            images_448 = images_448.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()

            # 始终使用test模式，统一走可视化分支（同时返回特征与可视化数据）
            result = stn_model(images_448, mode='test')
            if not (isinstance(result, tuple) and len(result) == 2):
                raise RuntimeError("stn_model(mode='test') 需要返回 (batch_features, vis_data)")

            batch_features, vis_data = result

            # 仅控制保存可视化图片的数量，不改变特征提取流程
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
                    print(f"⚠️ 可视化失败，继续测试: {vis_e}")

            all_image_features.append(batch_features)
            all_targets.append(labels)

    all_image_features = torch.cat(all_image_features, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    total_samples = all_image_features.size(0)

    logits = all_image_features @ precomputed_text_features
    stn_acc = accuracy(logits, all_targets, total_samples, dataset_name)
    print(f"✅ STN测试完成，共测试 {total_samples} 张图片，Top-1 Acc: {stn_acc:.2f}%")
    return stn_acc


def load_state_dict_flexible(stn_model, ckpt_path, device):
    """兼容 best(纯state_dict) 与 latest(包含model_state_dict的checkpoint)"""
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict) and 'model_state_dict' in obj:
        state_dict = obj['model_state_dict']
    else:
        state_dict = obj
    stn_model.load_state_dict(state_dict, strict=True)


def resolve_unstn_config_path(dataset_name, config_dir='UN-STN-Config'):
    """
    自动解析UN-STN-Config配置路径：
    1) 优先使用 UN-STN-Config/{dataset_name}.yaml
    2) 若不存在，则扫描目录并按yaml内 dataset 字段匹配
    """
    exact_path = os.path.join(config_dir, f"{dataset_name}.yaml")
    if os.path.exists(exact_path):
        return exact_path

    yaml_files = sorted(glob(os.path.join(config_dir, '*.yaml')))
    matched = []
    for ypath in yaml_files:
        try:
            with open(ypath, 'r', encoding='utf-8') as f:
                ycfg = yaml.load(f, Loader=yaml.FullLoader)
            y_dataset = (ycfg or {}).get('dataset', None)
            if y_dataset == dataset_name:
                matched.append(ypath)
        except Exception:
            continue

    if len(matched) == 1:
        return matched[0]
    if len(matched) > 1:
        raise RuntimeError(f"检测到多个匹配数据集'{dataset_name}'的配置文件，请保留唯一配置: {matched}")

    raise FileNotFoundError(
        f"未找到数据集 '{dataset_name}' 对应的UN-STN配置。"
        f"请确认存在 {config_dir}/{dataset_name}.yaml。"
    )


def main():
    parser = argparse.ArgumentParser(description='无监督多视角STN模型测试系统')
    parser.add_argument('--dataset_name', type=str, default='oxford_pets',
                        choices=['cub', 'imagenet', 'food101', 'oxford_pets', 'dtd',
                                 'fgvc-aircraft', 'stanford_cars', 'stanford_dogs', 'flowers102',
                                 'eurosat', 'places365', 'sun397',
                                 'imagenetv2', 'imagenet-v2', 'imagenet-r', 'imagenetr',
                                 'imagenet-s', 'imagenets', 'imagenet-sketch',
                                 'imagenet-a', 'imageneta'],
                        help='数据集名称')
    parser.add_argument('--batch_size', type=int, default=32, help='测试批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--device', type=str, default=None, help='计算设备(cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--visual_batches', type=int, default=10,
                        help='前N个batch保存可视化图片（仅支持>=1）')
    parser.add_argument('--max_vis_samples', type=int, default=0,
                        help='每个可视化batch最多保存样本数；0表示保存整个batch')
    args = parser.parse_args()

    if args.visual_batches <= 0:
        raise ValueError("--visual_batches 必须 >= 1；该脚本仅支持可视化模式")

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

    print("=== 无监督多视角STN模型测试系统 ===")
    print(f"数据集: {args.dataset_name}")
    print(f"设备: {args.device}")
    print(f"批次大小: {args.batch_size}")

    try:
        config_path = resolve_unstn_config_path(
            dataset_name=args.dataset_name,
            config_dir='UN-STN-Config'
        )

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        model_size = config['model_size']
        stn_config = config['stn_config']
        data_path = config['data_path']
        config_dataset = config.get('dataset', args.dataset_name)

        print(f"✅ 已加载配置: {config_path}")
        print(f"🗂️ 配置数据集: {config_dataset}")
        print(f"🤖 CLIP模型: {model_size}")
        print(f"🧩 STN参数: views={stn_config.get('num_views', 4)}, fusion={stn_config.get('fusion_mode', 'simple')}, dim={stn_config.get('hidden_dim', 512)}")

        clip_model, _ = clip.load(model_size, device=args.device)
        clip_model = clip_model.float()

        num_views = stn_config.get('num_views', 4)
        stn_model = MultiViewSTNModel(clip_model, stn_config, num_views=num_views).to(args.device).float()

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
        )

        model_name = model_size.replace('/', '_')
        text_features_path = f"text_features/{config_dataset}_{model_name}.pt"
        if not os.path.exists(text_features_path):
            raise FileNotFoundError(f"未找到文本特征文件: {text_features_path}")

        precomputed_text_features = torch.load(text_features_path, map_location=args.device).float().to(args.device)

        # 只测试best模型（根据数据集对应配置自动构建路径）
        ckpt_tag = 'best'
        ckpt_path = build_unsupervised_checkpoint_paths(config_dataset, config)['best']

        max_samples = None if args.max_vis_samples == 0 else args.max_vis_samples

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"模型文件不存在: {ckpt_path}")

        print(f"\n📥 加载模型({ckpt_tag}): {ckpt_path}")
        load_state_dict_flexible(stn_model, ckpt_path, args.device)

        acc = stn_precise_testing(
            stn_model=stn_model,
            dataloader=test_dataloader,
            device=args.device,
            dataset_name=config_dataset,
            precomputed_text_features=precomputed_text_features,
            config=config,
            model_tag=ckpt_tag,
            visual_batches=args.visual_batches,
            max_samples=max_samples,
        )

        print("\n=== 测试汇总 ===")
        print(f"{ckpt_tag}: {acc:.2f}%")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
