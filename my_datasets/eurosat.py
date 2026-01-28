import os
import json
import torch
from PIL import Image
from torchvision.datasets import VisionDataset

class EuroSAT(VisionDataset):
    """
    EuroSAT 数据集加载器
    基于 split_zhou_EuroSAT.json 文件进行划分 (train/val/test)
    """
    def __init__(self, root, split='train', transform=None, target_transform=None):
        """
        Args:
            root (str): 数据集根目录
                        结构应为:
                        root/
                          eurosat/
                            2750/ (或者是直接放类别的文件夹)
                            split_zhou_EuroSAT.json
            split (str): 'train', 'val', 'test' 之一
            transform (callable, optional): 图像预处理
            target_transform (callable, optional): 标签预处理
        """
        # 兼容处理路径：
        # 如果 root 是 .../DATA，则追加 eurosat
        # 如果 root 已经是 .../eurosat，则直接用
        
        target_path = os.path.join(root, 'eurosat')
        if os.path.exists(target_path):
            data_root = target_path
        else:
            data_root = root
            
        super().__init__(root=data_root, transform=transform, target_transform=target_transform)
        
        self.split = split
        # 假设 JSON 文件名为 split_zhou_EuroSAT.json，位于 data_root 下
        self.split_file = os.path.join(self.root, 'split_zhou_EuroSAT.json')
        
        # 图片基础路径，通常是在 2750 文件夹或者是 direct categories
        # 根据之前的 ls 输出，实际结构是 data_root/2750/Category/xxx.jpg
        # 但也有可能 json 里的路径已经包含了 2750/ 前缀，或者就在 data_root 下
        # 我们这里假设 json 中的路径是相对于 2750 文件夹的，或者 json 中的路径已经正确
        
        # 检查是否有一个名为 '2750' 的子文件夹，这是 EuroSAT 的官方命名习惯
        if os.path.exists(os.path.join(self.root, '2750')):
             self.image_base_path = os.path.join(self.root, '2750')
        else:
             self.image_base_path = self.root

        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"未找到划分文件: {self.split_file}")
            
        # 加载 JSON 数据
        with open(self.split_file, 'r') as f:
            full_data = json.load(f)
            
        if split not in full_data:
            raise ValueError(f"Split '{split}' 不在 JSON 文件中 (可用: {list(full_data.keys())})")
            
        self.samples = full_data[split] 
        # structure: [[rel_path, label_id, class_name], ...]

    def __getitem__(self, index):
        """
        Returns:
            tuple: (image, target) where target is class_index
        """
        item = self.samples[index]
        rel_path = item[0]
        label = int(item[1])
        
        # 拼接完整路径
        img_path = os.path.join(self.image_base_path, rel_path)
        
        # 加载图像
        try:
             img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
             # 回退尝试：也许 json 里的路径不包含 2750，但我们需要加，或者不需要加
             # 这里做一个简单的再次尝试
             if '2750' in self.image_base_path:
                 # 也许不需要 2750
                 alt_path = os.path.join(self.root, rel_path)
                 img = Image.open(alt_path).convert('RGB')
             else:
                 raise

        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
            
        return img, label

    def __len__(self):
        return len(self.samples)

