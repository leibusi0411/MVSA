"""
多视角STN数据预处理模块


核心设计理念：
1. 双分辨率处理：原始448x448用于STN变换，224x224用于ViT编码
2. 渐进式数据增强：缩放→裁剪→翻转→张量化
3. 兼容CLIP预处理：保持 CLIP 归一化参数
4. 批量高效处理：支持多进程数据加载

预处理流程：
输入图像 → 等比例缩放(短边512) → 随机裁剪(448x448) → 随机翻转(50%) → 
张量化 → [原始448x448, CLIP处理224x224] → 双重输出

Multi-view STN preprocessing with dual-resolution support.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile
import numpy as np
import os
import random
from typing import Tuple, Dict, Any, Optional, List

# 数据集导入
from my_datasets.cub import CUBDataset
from my_datasets.food101 import Food101
from my_datasets.oxford_pets import OxfordIIITPet
from my_datasets.dtd import DTD
from my_datasets.fgvc_aircraft import FGVCAircraft
from my_datasets.stanford_cars import StanfordCars
from my_datasets.stanford_dogs import Dogs
from my_datasets.flowers102 import Flowers102
from my_datasets.places365 import Places365

# 启用PIL的truncated图像支持
ImageFile.LOAD_TRUNCATED_IMAGES = True

def robust_image_loader(path):
    """
    处理损坏图像文件的安全加载器，仅用于大数据集
    """
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            img.load()  # 强制加载以检测截断
            return img.convert('RGB')
    except (OSError, IOError, Image.UnidentifiedImageError) as e:
        # 图像损坏，抛出通用异常用于跳过
        raise Exception(f"损坏的图像文件: {path} - {type(e).__name__}: {str(e)}")


def safe_collate_fn(batch):
    """
    安全的collate函数，跳过损坏的图像样本
    """
    # 过滤掉None值（损坏图像返回的）
    batch = [item for item in batch if item is not None]
    
    # 使用默认的collate函数处理有效样本
    from torch.utils.data import default_collate
    return default_collate(batch)


class RobustDataset:
    """
    包装数据集，处理损坏图像的跳过
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.corrupted_count = 0
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            return self.dataset[idx]
        except Exception as e:
            # 记录损坏文件并返回None
            self.corrupted_count += 1
            if self.corrupted_count % 10 == 1:  # 每10个损坏文件打印一次
                print(f"⚠️ 跳过损坏图像 (第{self.corrupted_count}个): {str(e)}")
            return None

# CLIP标准预处理参数 (ImageNet统计值)
# 与CLIP官方clip._transform()函数中的Normalize参数完全一致
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]




# 统一的多视角STN数据集加载器 
def load_multi_view_dataset(dataset_name: str,
                           data_path: str,
                           split: str = 'train',
                           batch_size: int = 32,
                           num_workers: int = 4,
                           target_size: int = 448,
                           scale_short_edge: int = 512,
                           flip_prob: float = 0.5,
                           center_crop: bool = False) -> DataLoader:
    """
    统一的数据加载器 - 新架构
    直接使用基础数据集类 + transform，无需MultiViewDataset包装
    
    Args:
        dataset_name (str): 数据集名称
        data_path (str): 数据集路径
        split (str): 数据集分割 ('train', 'val', 'test')
        batch_size (int): 批次大小
        num_workers (int): 数据加载进程数
        target_size (int): 目标裁剪尺寸 (448)
        scale_short_edge (int): 短边缩放尺寸 (512)
        flip_prob (float): 水平翻转概率 (0.0-1.0)
        center_crop (bool): 是否使用中心裁剪 (False=随机裁剪, True=中心裁剪)
        
    Returns:
        DataLoader: 优化后的数据加载器
            输出格式: (image_tensor, label) - 标准PyTorch格式
    """
    # 设置描述信息
    split_name = {'train': '训练集', 'val': '验证集', 'test': '测试集'}.get(split, split)
    crop_mode = "中心裁剪" if center_crop else "随机裁剪"
    flip_mode = f"翻转概率{flip_prob}" if flip_prob > 0 else "无翻转"
    
    print(f"\n🔄 加载{split_name} ({dataset_name}) - 新架构:")
    print(f"    📦 批次大小: {batch_size}, 进程数: {num_workers}")
    print(f"    📐 输出尺寸: {target_size}x{target_size} ({crop_mode}, {flip_mode})")
    print(f"    📊 标准化: CLIP参数 (直接在transform中完成)")
    print(f"    🏗️ 架构: 基础数据集 + MultiViewDataPreprocessor transform")
    
    # 创建简化的数据集助手  预处理
    dataset_helper = MultiViewDataset(
        data_root=data_path,
        dataset_name=dataset_name,
        split=split,
        target_size=target_size,
        scale_short_edge=scale_short_edge,
        flip_prob=flip_prob,
        center_crop=center_crop
    )
    
    # 直接创建带transform的基础数据集
    dataset = dataset_helper._create_base_dataset_with_transform()
    
    # 对大数据集使用鲁棒包装器，跳过损坏图像
    if dataset_name in ['imagenet', 'place365']:
        print(f"    🛡️ 对{dataset_name}启用损坏图像跳过机制")
        dataset = RobustDataset(dataset)
        collate_fn = safe_collate_fn
    else:
        collate_fn = None
    
    # 创建数据加载器 - 显式控制shuffle和drop_last + 性能优化
    is_train = (split == 'train')
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,  # 训练时打乱，验证/测试时保持顺序
        num_workers=num_workers,
        pin_memory=True,  # 加速GPU传输
        persistent_workers=True if num_workers > 0 else False,  # 🔥 保持worker进程活跃，避免重复创建
        prefetch_factor=2 if num_workers > 0 else None,  # 🔥 每个worker预取2个batch，减少等待时间
        drop_last=is_train,  # 训练时丢弃不完整批次，验证/测试时保留所有样本
        collate_fn=collate_fn  # 使用安全的collate函数
    )
    
    print(f"    📈 样本数: {len(dataset)}")
    print(f"    📊 批次数: {len(dataloader)}")
    print(f"    ⚡ 性能优化: persistent_workers={num_workers > 0}, prefetch_factor={2 if num_workers > 0 else None}")
    
    return dataloader





class MultiViewDataset:
    """
    
    不再继承Dataset，只作为创建基础数据集的助手
    负责参数管理和数据集创建逻辑
    
    Dataset creation helper for multi-view STN training.
    No longer inherits from Dataset - just creates base datasets with transforms.
    """
    
    def __init__(self, 
                 data_root: str,
                 dataset_name: str,
                 split: str = 'train',
                 target_size: int = 448,
                 scale_short_edge: int = 512,
                 flip_prob: float = 0.5,
                 center_crop: bool = False):
        """
        初始化数据集助手
        
        Args:
            data_root (str): 数据集根目录
            dataset_name (str): 数据集名称 (cub, imagenet, food101等)
            split (str): 数据集分割 ('train', 'val', 'test')
            target_size (int): 目标裁剪尺寸 (448)
            scale_short_edge (int): 短边缩放尺寸 (512)
            flip_prob (float): 水平翻转概率 (0.5)
            center_crop (bool): 是否使用中心裁剪 (False=随机裁剪, True=中心裁剪)
        """
        self.data_root = data_root
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.target_size = target_size
        self.scale_short_edge = scale_short_edge
        self.flip_prob = flip_prob
        self.center_crop = center_crop
        
    
    
    #创建数据集实例   通过 _create_base_dataset() 方法将分割参数 (split) 传递给各个具体的数据集类，每个数据集类都有自己的分割处理逻辑
    def _create_base_dataset_with_transform(self):
        """
        创建带预处理的基础数据集 - 新架构方案
        直接将MultiViewDataPreprocessor传给数据集的transform参数
        
        Returns:
            Dataset: 基础数据集实例（已包含预处理）
        """
        # 创建预处理器 - 使用显式传入的参数
        preprocessor = MultiViewDataPreprocessor(
            target_size=self.target_size,
            scale_short_edge=self.scale_short_edge,
            flip_prob=self.flip_prob,
            center_crop=self.center_crop
        )
        
        if self.dataset_name == 'cub':
            # CUB现在支持三种划分：train/val/test，使用split参数
            return CUBDataset(
                root=self.data_root,
                split=self.split,  # 直接传递split参数（train/val/test）
                transform=preprocessor  # 直接传递预处理器
            )
            
        elif self.dataset_name == 'food101':
            # Food101现在支持三种划分：train/val/test，直接传递split参数
            return Food101(
                root=self.data_root,
                split=self.split,  # 直接传递split参数（train/val/test）
                transform=preprocessor  # 直接传递预处理器
            )
            
        elif self.dataset_name == 'oxford_pets':
            # Oxford Pets现在支持三种划分：train/val/test，直接传递split参数
            return OxfordIIITPet(
                root=self.data_root,
                split=self.split,  # 直接传递split参数（train/val/test）
                transform=preprocessor  # 直接传递预处理器
            )
            
        elif self.dataset_name == 'dtd':
            return DTD(
                root=self.data_root,
                split=self.split,
                transform=preprocessor  # 直接传递预处理器
            )
            
        elif self.dataset_name == 'fgvc-aircraft':
            # FGVC Aircraft支持三种划分：train/val/test，使用split参数
            return FGVCAircraft(
                root=self.data_root,
                split=self.split,  # 直接传递split参数（train/val/test）
                annotation_level='variant',  # 使用变体级别分类（100类）
                transform=preprocessor  # 直接传递预处理器
            )
            
        elif self.dataset_name == 'stanford_cars':
            # Stanford Cars支持三种划分：train/val/test，使用split参数
            return StanfordCars(
                root=self.data_root,
                split=self.split,  # 直接传递split参数（train/val/test）
                transform=preprocessor  # 直接传递预处理器
            )
            
            
        elif self.dataset_name == 'stanford_dogs':
            # Stanford Dogs支持三种划分：train/val/test
            # train和val从splits文件夹读取，test从lists文件夹读取
            return Dogs(
                root=self.data_root,
                split=self.split,  # 直接传递split参数（train/val/test）
                transform=preprocessor  # 直接传递预处理器
            )
            
        elif self.dataset_name == 'flowers102':
            # Flowers102支持三种划分：train/val/test，使用split参数
            return Flowers102(
                root=self.data_root,
                split=self.split,  # 直接传递split参数（train/val/test）
                transform=preprocessor  # 直接传递预处理器
            )
            
        elif self.dataset_name == 'imagenet':
            # ImageNet使用标准的ImageFolder，支持train/val/test三个文件夹
            # 映射split参数到实际文件夹名
            if self.split == 'train':
                split_folder = 'train'
            elif self.split == 'val':
                split_folder = 'val'
            elif self.split == 'test':
                split_folder = 'test'
            else:
                # 默认回退逻辑
                split_folder = 'train' if self.split == 'train' else 'val'
            
            dataset_path = os.path.join(self.data_root, split_folder)
            
            # 验证路径是否存在
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"ImageNet数据集路径不存在: {dataset_path}")
            
            print(f"    📁 ImageNet数据路径: {dataset_path}")
            return ImageFolder(dataset_path, transform=preprocessor, loader=robust_image_loader)  # 使用安全加载器
        
        elif self.dataset_name in ['imagenetv2', 'imagenet-v2']:
            # ImageNet-V2: 领域泛化测试数据集
            # 数据结构: data_root/test/ (只有测试集)
            split_folder = 'test'  # ImageNet-V2只有测试集
            dataset_path = os.path.join(self.data_root, split_folder)
            
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"ImageNet-V2数据集路径不存在: {dataset_path}\n"
                                      f"请确保数据集结构为: {self.data_root}/test/class_folders/")
            
            print(f"    📁 ImageNet-V2数据路径: {dataset_path}")
            return ImageFolder(dataset_path, transform=preprocessor, loader=robust_image_loader)
        
        elif self.dataset_name in ['imagenet-r', 'imagenetr']:
            # ImageNet-R (Rendition): 艺术风格变体
            # 数据结构: data_root/test/ (只有测试集)
            split_folder = 'test'
            dataset_path = os.path.join(self.data_root, split_folder)
            
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"ImageNet-R数据集路径不存在: {dataset_path}\n"
                                      f"请确保数据集结构为: {self.data_root}/test/class_folders/")
            
            print(f"    📁 ImageNet-R数据路径: {dataset_path}")
            return ImageFolder(dataset_path, transform=preprocessor, loader=robust_image_loader)
        
        elif self.dataset_name in ['imagenet-s', 'imagenets', 'imagenet-sketch']:
            # ImageNet-Sketch: 素描风格变体
            # 数据结构: data_root/test/ (只有测试集)
            split_folder = 'test'
            dataset_path = os.path.join(self.data_root, split_folder)
            
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"ImageNet-Sketch数据集路径不存在: {dataset_path}\n"
                                      f"请确保数据集结构为: {self.data_root}/test/class_folders/")
            
            print(f"    📁 ImageNet-Sketch数据路径: {dataset_path}")
            return ImageFolder(dataset_path, transform=preprocessor, loader=robust_image_loader)
        
        elif self.dataset_name in ['imagenet-a', 'imageneta']:
            # ImageNet-A (Adversarial): 对抗样本变体
            # 数据结构: data_root/test/ (只有测试集)
            split_folder = 'test'
            dataset_path = os.path.join(self.data_root, split_folder)
            
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"ImageNet-A数据集路径不存在: {dataset_path}\n"
                                      f"请确保数据集结构为: {self.data_root}/test/class_folders/")
            
            print(f"    📁 ImageNet-A数据路径: {dataset_path}")
            return ImageFolder(dataset_path, transform=preprocessor, loader=robust_image_loader)
            
        elif self.dataset_name == 'place365':
            # 映射split参数
            places_split = 'train-standard' if self.split == 'train' else 'val'
            
            return Places365(
                root=self.data_root,
                split=places_split,
                transform=preprocessor,
                loader=robust_image_loader,  # 使用会抛出异常的加载器
                download=False
            )
        
        elif self.dataset_name == 'eurosat':
            from my_datasets.eurosat import EuroSAT
            return EuroSAT(
                root=self.data_root,
                split=self.split,
                transform=preprocessor
            )
            
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")
    
    

    


#数据预处理器，包括等比例缩放、随机裁剪、随机水平翻转、转换为PyTorch张量等步骤  最底层

class MultiViewDataPreprocessor:
    """
    多视角STN数据预处理器 - 优化版本

    
    Multi-view data preprocessor with delayed CLIP processing.
    """
    
    def __init__(self, 
                 target_size: int = 448,
                 scale_short_edge: int = 512,
                 flip_prob: float = 0.5,
                 interpolation: str = 'bicubic',
                 center_crop: bool = False):
        """
        
        Args:
            target_size (int): 目标裁剪尺寸 (448x448)
            scale_short_edge (int): 短边缩放目标尺寸 (512)
            flip_prob (float): 水平翻转概率 (0.5，验证时应设为0.0)
            interpolation (str): 插值算法 ('bilinear' 或 'bicubic')
            center_crop (bool): 是否使用中心裁剪 (验证时使用True，训练时使用False)
        """
        self.target_size = target_size
        self.scale_short_edge = scale_short_edge
        self.flip_prob = flip_prob
        self.center_crop = center_crop
        # 使用双三次插值作为默认（与 CLIP 预处理一致）；保留对 'bilinear' 的兼容
        self.interpolation = Image.BILINEAR if interpolation == 'bilinear' else Image.BICUBIC
        
        crop_mode = "中心裁剪" if center_crop else "随机裁剪"
        mode = "验证模式" if center_crop else "训练模式"
        
        # 简化输出，避免重复
        # print(f"🔧 多视角数据预处理器初始化 ({mode}):")
        # print(f"    📐 目标尺寸: {target_size}x{target_size}")
        # print(f"    📏 短边缩放: {scale_short_edge}px")
        # print(f"    ✂️  裁剪模式: {crop_mode}")
        # print(f"    🔄 翻转概率: {flip_prob}")
        # print(f"    🎨 插值算法: {interpolation}")
        # print(f"    📊 标准化: ImageNet统计值 (与CLIP相同)")
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        执行预处理流程 - 支持训练和验证模式
        
        训练模式：随机裁剪 + 随机翻转
        验证模式：中心裁剪 + 无翻转
        
        Args:
            image (PIL.Image): 输入PIL图像
            
        Returns:
            torch.Tensor: [3, 448, 448] 标准化后的图像张量
        """
        # 步骤1: 等比例缩放 (短边变为512)
        scaled_image = self._resize_short_edge(image)
        
        # 步骤2: 裁剪 (448x448) - 根据模式选择随机或中心
        if self.center_crop:
            cropped_image = self._center_crop(scaled_image)  # 验证模式：中心裁剪
        else:
            cropped_image = self._random_crop(scaled_image)  # 训练模式：随机裁剪
        
        # 步骤3: 随机水平翻转 (训练时50%概率，验证时0%概率)
        flipped_image = self._random_horizontal_flip(cropped_image)
        
        # 步骤4: 转换为PyTorch张量，归一化
        image_tensor = self._to_tensor(flipped_image)
        
        return image_tensor  # [3, 448, 448] 标准化后张量
    

    def _resize_short_edge(self, image: Image.Image) -> Image.Image:
        """
        等比例缩放，将较短边变为指定尺寸
        
        Args:
            image (PIL.Image): 输入图像
            
        Returns:
            PIL.Image: 缩放后的图像
        """
        width, height = image.size
        
        # 计算缩放比例 (短边变为target_short_edge)
        if width < height:
            scale_ratio = self.scale_short_edge / width
            new_width = self.scale_short_edge
            new_height = int(height * scale_ratio)
        else:
            scale_ratio = self.scale_short_edge / height
            new_width = int(width * scale_ratio)
            new_height = self.scale_short_edge
        
        # 使用双线性或双三次插值进行缩放
        resized_image = image.resize((new_width, new_height), self.interpolation)
        
        return resized_image
    
    def _random_crop(self, image: Image.Image) -> Image.Image:
        """
        随机裁剪出目标尺寸的方块（训练时使用）
        
        Args:
            image (PIL.Image): 输入图像
            
        Returns:
            PIL.Image: 裁剪后的图像 (target_size x target_size)
        """
        width, height = image.size
        
        # 确保图像尺寸足够进行裁剪
        if width < self.target_size or height < self.target_size:
            # 如果图像太小，先填充到最小尺寸
            min_size = max(self.target_size, max(width, height))
            image = image.resize((min_size, min_size), self.interpolation)
            width, height = min_size, min_size
        
        # 随机选择裁剪起始点，确保裁剪区域不会超过边界
        x_start = random.randint(0, width - self.target_size)
        y_start = random.randint(0, height - self.target_size)
        
        # 裁剪目标区域
        cropped_image = image.crop((
            x_start, 
            y_start, 
            x_start + self.target_size, 
            y_start + self.target_size
        ))
        
        return cropped_image
    
    def _center_crop(self, image: Image.Image) -> Image.Image:
        """
        中心裁剪出目标尺寸的方块（验证时使用）
        
        Args:
            image (PIL.Image): 输入图像
            
        Returns:
            PIL.Image: 中心裁剪后的图像 (target_size x target_size)
        """
        width, height = image.size
        
        # 确保图像尺寸足够进行裁剪
        if width < self.target_size or height < self.target_size:
            # 如果图像太小，先填充到最小尺寸
            min_size = max(self.target_size, max(width, height))
            image = image.resize((min_size, min_size), self.interpolation)
            width, height = min_size, min_size
        
        # 计算中心裁剪的起始点
        x_start = (width - self.target_size) // 2
        y_start = (height - self.target_size) // 2
        
        # 裁剪中心区域
        cropped_image = image.crop((
            x_start, 
            y_start, 
            x_start + self.target_size, 
            y_start + self.target_size
        ))
        
        return cropped_image
    
    def _random_horizontal_flip(self, image: Image.Image) -> Image.Image:
        """
        以指定概率进行水平翻转
        
        Args:
            image (PIL.Image): 输入图像
            
        Returns:
            PIL.Image: 可能翻转后的图像
        """
        if random.random() < self.flip_prob:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        return image
    
    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        """
        将PIL图像转换为PyTorch张量（归一化+标准化）
        
        STN优化策略：
        - STN几何变换对像素值大小不敏感
        - 可以直接在标准化后的图像上进行STN变换
        - 简化流程：归一化→标准化→STN变换→CLIP
        
        Args:
            image (PIL.Image): 输入图像
            
        Returns:
            torch.Tensor: [3, H, W] 标准化后的张量
        """
        # 转换为张量并调整维度顺序: [H, W, 3] -> [3, H, W]
        tensor = torch.from_numpy(np.array(image)).float() / 255.0
        if len(tensor.shape) == 3:
            tensor = tensor.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
        elif len(tensor.shape) == 2:
            # 灰度图像，扩展为3通道
            tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
        
        # 直接进行标准化（CLIP参数）；与张量 dtype/device 对齐，避免潜在的类型/设备不匹配
        mean = torch.tensor(CLIP_MEAN, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
        std = torch.tensor(CLIP_STD, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
        tensor = (tensor - mean) / std
        
        return tensor
    



# CLIP预处理函数  双三次插值
def prepare_clip_input(image_batch: torch.Tensor,
                      clip_size: int = 224) -> torch.Tensor:
    """
    将448x448标准化图像批次处理为CLIP输入格式
    
    STN优化方案：
    - 输入已经是标准化后的图像
    - 只需要下采样，无需重复标准化
    - 简化处理流程，提升效率
    
    Args:
        image_batch (torch.Tensor): [B, 3, 448, 448] 已标准化图像批次
        clip_size (int): CLIP输入尺寸，默认224
        
    Returns:
        torch.Tensor: [B, 3, 224, 224] CLIP输入批次（已标准化）
    
    Example:
        >>> images = torch.randn(32, 3, 448, 448).cuda()  # 已标准化
        >>> clip_input = prepare_clip_input(images)  # [32, 3, 224, 224]
        >>> # 直接可用于CLIP模型
    """
    # GPU上的下采样 (448x448 -> 224x224)
    # 使用双三次插值匹配CLIP官方预处理
    clip_batch = F.interpolate(
        image_batch,
        size=(clip_size, clip_size),
        mode='bicubic',  # ✅ 与CLIP官方保持一致
        align_corners=False
    )
    
    # 注意：不需要重复标准化，输入已经是标准化后的图像
    
    return clip_batch








