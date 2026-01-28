"""
Dataset utilities for baseline experiments

本模块支持两套可切换的图像预处理方案：
1) STN 方案（默认向后兼容）：
    - 图像尺寸：448×448
    - 预处理：短边缩放512 → 裁剪448×448 → 翻转 → CLIP归一化
    - 训练集：随机裁剪 + 随机翻转；验证/测试：中心裁剪 + 无翻转

2) Base 方案（transfer-style，参考 transfer_learning 项目中的 CLIP fine-tuning）：
    - 建议图像尺寸：224×224
    - 预处理：Resize(224, bicubic) → CenterCrop(224) → ToTensor（不做 Normalize；归一化在模型内完成）
    - 训练/验证/测试均使用确定性 CenterCrop 流程（strict 对齐 transfer_learning，无随机翻转）

可通过 create_dataloaders(..., preprocess="Base") 开启第二种方案。
"""
import os
import sys
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder

# 动态添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入数据集类
from my_datasets.cub import CUBDataset
from my_datasets.food101 import Food101
from my_datasets.oxford_pets import OxfordIIITPet
from my_datasets.dtd import DTD
from my_datasets.fgvc_aircraft import FGVCAircraft
from my_datasets.stanford_cars import StanfordCars
from my_datasets.stanford_dogs import Dogs
from my_datasets.flowers102 import Flowers102

# 导入数据预处理器
from data_preprocess import MultiViewDataPreprocessor, CLIP_MEAN, CLIP_STD

# 数据集固定类别数（硬编码，适用于中小规模数据集）
NUM_CLASSES_MAP = {
    'imagenet': 1000,
    'cub': 200,
    'food101': 101,
    'oxford_pets': 37,
    'dtd': 47,
    'fgvc-aircraft': 100,
    'fgvc_aircraft': 100,
    'stanford_cars': 196,
    'stanford_dogs': 120,
    'flowers102': 102,
}

#构造单个 split 的 DataLoader（创建 transform、实例化 dataset、创建 DataLoader）
def create_dataloader(dataset_name, data_root, split='train',
                      batch_size=32, num_workers=4,
                      preprocess: str = "stn", seed: int | None = None):
    """
    创建单个dataloader（简化版本，参考主实验的load_multi_view_dataset设计）
    
    Args:
        dataset_name (str): 数据集名称 (e.g., 'stanford_dogs', 'cub')
        data_root (str): 数据集根目录
        split (str): 数据集分割 ('train', 'val', 'test')
        batch_size (int): 批次大小
        num_workers (int): 数据加载进程数
        image_size (int): 目标图像尺寸 (default: 448)
    
    Returns:
        DataLoader: 数据加载器
    """
    # 1. 创建预处理器（根据方案与 split 自动选择策略）
    is_train = (split == 'train')
    preprocess_key = str(preprocess).lower()

    if preprocess_key == "base":
        # Base 方案固定 224
        image_size = 224
        # Base：训练采用最基础的空间增强（RRC + Flip）；验证/测试保持确定性流程
        if is_train:
            transform = T.Compose([
                # 轻量随机缩放+裁剪（不改变进入ViT的224尺寸）
                T.RandomResizedCrop(
                    image_size,
                    scale=(0.90, 1.0),
                    ratio=(0.95, 1.05),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ])
        else:
            transform = T.Compose([
                T.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ])
    elif preprocess_key == "stn":
        # STN 方案固定 448
        image_size = 448
        # STN 方案（向后兼容）
        transform = MultiViewDataPreprocessor(
            target_size=image_size,
            scale_short_edge=512,
            flip_prob=0.5 if is_train else 0.0,    # 训练时翻转，验证/测试时不翻转
            center_crop=not is_train                # 训练时随机裁剪，验证/测试时中心裁剪
        )
    else:
        raise ValueError(
            f"不支持的预处理方案: '{preprocess}'\n"
            f"可用方案: 'base', 'stn'"
        )
    
    # 2. 数据集映射
    dataset_mapping = {
        'imagenet': ImageFolder,
        'cub': CUBDataset,
        'food101': Food101,
        'oxford_pets': OxfordIIITPet,
        'dtd': DTD,
        'fgvc-aircraft': FGVCAircraft,
        'fgvc_aircraft': FGVCAircraft,
        'stanford_cars': StanfordCars,
        'stanford_dogs': Dogs,
        'flowers102': Flowers102,
    }
    
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower not in dataset_mapping:
        raise ValueError(
            f"不支持的数据集: {dataset_name}\n"
            f"可用数据集: {list(dataset_mapping.keys())}"
        )
    
    # 3. 创建数据集实例
    dataset_class = dataset_mapping[dataset_name_lower]
    if dataset_name_lower == 'imagenet':
        # ImageNet 特殊处理 (ImageFolder 不接受 split，需要手动拼接路径)
        # 假设 data_root/train 和 data_root/val 存在
        split_dir = 'val' if split == 'test' else split # Test set usually use val
        data_path = os.path.join(data_root, split_dir)
        if not os.path.exists(data_path) and split == 'test':
             data_path = os.path.join(data_root, 'val')

        if not os.path.exists(data_path):
             raise ValueError(f"ImageNet path not found: {data_path}")

        dataset = dataset_class(
            root=data_path,
            transform=transform
        )
    elif dataset_name_lower in ['fgvc-aircraft', 'fgvc_aircraft']:
        dataset = dataset_class(
            root=data_root,
            split=split,
            annotation_level='variant',  # 100个类别
            transform=transform
        )
    else:
        dataset = dataset_class(
            root=data_root,
            split=split,
            transform=transform
        )
    
    # 4. 创建dataloader
    # 可复现性：为 DataLoader 设置生成器与每个 worker 的随机种子
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    def _seed_worker(worker_id: int):
        if seed is None:
            return
        worker_seed = int(seed) + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=is_train,
        worker_init_fn=_seed_worker if seed is not None else None,
        generator=generator
    )
    
    return dataloader

#为 train/val/test 三个 split 逐个调用 create_dataloader，。   调用create_dataloader
def create_dataloaders(dataset_name, data_root, batch_size=32, num_workers=4,
                       preprocess: str = "stn", seed: int | None = None):
    """
    创建train/val/test的所有dataloaders（简化版本）
    
    Args:
        dataset_name (str): 数据集名称
        data_root (str): 数据集根目录
        batch_size (int): 批次大小
        num_workers (int): 数据加载进程数
        image_size (int): 目标图像尺寸 (default: 448)
    
    Returns:
        dict: {'train': loader, 'val': loader, 'test': loader}
        int: 类别数量
    """
    print(f"\n{'='*60}")
    print(f"Loading dataset: {dataset_name}")
    preprocess_key = str(preprocess).lower()
    # 按方案选择默认尺寸用于日志展示（Base=224, STN=448）
    chosen_size = 224 if preprocess_key == "base" else 448
    if preprocess_key == "base":
        print(f"✨ Preprocess: Base | Size: {chosen_size}x{chosen_size} | Train: light aug (RRC+Flip) + CLIP Normalize | Val/Test: Resize→CenterCrop + CLIP Normalize")
    else:
        print(f"✨ Preprocess: STN | Size: {chosen_size}x{chosen_size} | Train: 随机裁剪+翻转 | Val/Test: 中心裁剪+无翻转")
    if seed is not None:
        print(f"🔒 Seeded dataloaders with seed={seed}")
    print(f"{'='*60}\n")
    
    loaders = {}
    
    # 加载所有splits
    for split in ['train', 'val', 'test']:
        try:
            loader = create_dataloader(
                dataset_name=dataset_name,
                data_root=data_root,
                split=split,
                batch_size=batch_size,
                num_workers=num_workers,
                preprocess=preprocess,
                seed=seed
            )
            loaders[split] = loader
            print(f"✓ {split.capitalize():5s}: {len(loader.dataset):5d} samples, "
                  f"{len(loader):4d} batches")
        except Exception as e:
            print(f"⚠️  {split.capitalize():5s}: Not available")
            if split == 'train':
                raise ValueError(f"训练集必须存在！错误: {e}")
    
    # 获取类别数量（优先使用硬编码表，回退到 dataset 属性或简单推断）
    if 'train' not in loaders:
        raise ValueError("训练集不存在！")

    dataset_key = dataset_name.lower()
    # 仅使用硬编码表来确定类别数量；若找不到则直接报错，避免隐式或不可靠的推断
    if dataset_key in NUM_CLASSES_MAP:
        num_classes = NUM_CLASSES_MAP[dataset_key]
    elif hasattr(loaders['train'].dataset, 'classes'):
         num_classes = len(loaders['train'].dataset.classes)
    else:
        raise ValueError(
            f"无法确定数据集 '{dataset_name}' 的类别数。"
            " 请在 `NUM_CLASSES_MAP` 中添加映射，或使用支持 `num_classes` 属性的数据集类。"
        )
    
    print(f"\n✓ Dataset loaded: {num_classes} classes")
    print(f"{'='*60}\n")
    
    return loaders, num_classes  #loaders：字典，键为 split 名（函数内按顺序尝试 'train', 'val', 'test'），成功加载的 split 会出现在字典中（典型包含 'train' 和 'val'，若 test 存在也会包含）。每个值是对应的 torch.utils.data.DataLoader 实例
