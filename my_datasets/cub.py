#数字标签是根据加载顺序来的   使用ImageFolder加载  每个子文件夹名作为类别名，按文件名进行排序，从0开始
#train_test_split.txt文件是官方分割文件，两列，第一列是图片的唯一ID 第二列是二进制标志 1表示训练集，0表示测试集
#images.txt文件是图片的唯一ID和文件名，两列，第一列是图片的唯一ID 第二列是文件名
#没有验证集

import os
import torch
from torchvision import datasets
from pathlib import Path

# 利用ImageFolder的便利性： 先让ImageFolder自动扫描所有图片，创建出包含全部11,788个样本的列表，并自动完成从文件夹名到类别索引的映射。
# 读取官方定义： 读取train_test_split.txt和images.txt，得到当前划分（训练或测试）应该包含的精确的文件名列表。
# 精确筛选： 使用这个文件名列表，从ImageFolder创建的完整样本列表中，过滤并保留出正确的样本子集，并用这个子集覆盖掉ImageFolder的内部状态。


class CUBDataset(datasets.ImageFolder):
    """
    Wrapper for the CUB-200-2011 dataset. 
    Method DatasetBirds.__getitem__() returns tuple of image and its corresponding label.    
    Dataset per https://github.com/slipnitskaya/caltech-birds-advanced-classification
    
    新增功能：
    - 支持使用自定义的三分类划分文件 train_val_test_split.txt
    - 可以加载训练集(1)、验证集(2)、测试集(0)
    - 简化为统一的三分类模式
    
    参数说明：
    - split: 'train', 'val', 'test' 必需参数，指定要加载的数据集分割
    - 统一使用自定义的三分类划分文件 train_val_test_split.txt
    """
    def __init__(self,
                 root,
                 split,  # 必需参数：'train', 'val', 'test'
                 transform=None,
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None,
                 bboxes=False):

        img_root = os.path.join(root, 'images')

        super(CUBDataset, self).__init__(
            root=img_root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )
        
        self.redefine_class_to_idx()

        self.transform_ = transform
        self.target_transform_ = target_transform
        self.split = split
        
        #原始的官方划分
        # obtain sample ids filtered by split  读取官方的分割文件  根据 self.train 参数选择训练集或测试集的图片ID
        # path_to_splits = os.path.join(root, 'train_test_split.txt')
        # indices_to_use = list()
        # with open(path_to_splits, 'r', encoding='utf-8') as in_file:
        #     for line in in_file:
        #         idx, use_train = line.strip('\n').split(' ', 2) #获取图片ID和属于训练集还是测试集
        #         if bool(int(use_train)) == self.train:  #如果self.train=True，只保留标志为1的图像ID
        #             indices_to_use.append(int(idx))   #筛选出所有属于训练集的图片ID (image_id)，存入indices_to_use列表


        # 使用自定义的三分类划分文件 train_val_test_split.txt
        path_to_splits = os.path.join(root, 'train_val_test_split.txt')
        indices_to_use = list()
        
        # 确定目标分割类型
        if self.split == 'train':
            target_split_type = 1  # 训练集
            split_name = "训练集"
        elif self.split == 'val':
            target_split_type = 2  # 验证集
            split_name = "验证集"
        elif self.split == 'test':
            target_split_type = 0  # 测试集
            split_name = "测试集"
        else:
            raise ValueError(f"无效的split参数: {self.split}，应为 'train', 'val' 或 'test'")
        
        # 读取自定义划分文件
        with open(path_to_splits, 'r', encoding='utf-8') as in_file:
            for line in in_file:
                line = line.strip()
                if line:
                    parts = line.split(' ')
                    if len(parts) == 2:
                        idx, split_type = int(parts[0]), int(parts[1])
                        if split_type == target_split_type:
                            indices_to_use.append(idx)
        
        print(f"✅ 使用自定义划分文件加载{split_name}，共{len(indices_to_use)}个样本")
        






        # obtain filenames of images  读取官方的图片文件名文件   根据上面得到的ID列表，获取对应的图片文件名
        #图片文件映射格式    1    001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg
        path_to_index = os.path.join(root, 'images.txt')
        filenames_to_use = list()
        with open(path_to_index, 'r', encoding='utf-8') as in_file:
            for line in in_file:
                idx, fn = line.strip('\n').split(' ', 2)
                if fn not in filenames_to_use and int(idx) in indices_to_use:
                    filenames_to_use.append(fn)

        # 标准化路径格式以匹配images.txt中的条目
        img_paths_cut = {'/'.join(img_path.rsplit('/', 2)[-2:]): idx for idx, (img_path, lb) in enumerate(self.imgs)}
        imgs_to_use = [self.imgs[img_paths_cut[fn]] for fn in filenames_to_use]

        _, targets_to_use = list(zip(*imgs_to_use))

        self.imgs = self.samples = imgs_to_use
        self.targets = targets_to_use

        
        
        # 输出类名到标签的映射，方便检查
        # split_name = "训练集" if self.train else "测试集"
        # print(f"\n🐦 CUB数据集 ({split_name}) 类别映射:")
        # print(f"   总类别数: {len(self.class_to_idx)}")
        # print(f"   样本数量: {len(self.imgs)}")
        # print(f"   完整类别映射字典:")
        # for class_name, class_idx in sorted(self.class_to_idx.items(), key=lambda x: x[1]):
        #     print(f"     '{class_name}' → {class_idx}")
        # print(f"   ✅ 全部 {len(self.class_to_idx)} 个鸟类类别映射完成")


        if bboxes:
            # get coordinates of a bounding box
            path_to_bboxes = os.path.join(root, 'bounding_boxes.txt')
            bounding_boxes = list()
            with open(path_to_bboxes, 'r', encoding='utf-8') as in_file:
                for line in in_file:
                    idx, x, y, w, h = map(lambda x: float(x), line.strip('\n').split(' '))
                    if int(idx) in indices_to_use:
                        bounding_boxes.append((x, y, w, h))

            self.bboxes = bounding_boxes
        else:
            self.bboxes = None

    def __getitem__(self, index):
        # generate one sample
        sample, target = super(CUBDataset, self).__getitem__(index)

        # if self.bboxes is not None:
        #     # squeeze coordinates of the bounding box to range [0, 1]
        #     width, height = sample.width, sample.height
        #     x, y, w, h = self.bboxes[index]

        #     scale_resize = 500 / width
        #     scale_resize_crop = scale_resize * (375 / 500)

        #     x_rel = scale_resize_crop * x / 375
        #     y_rel = scale_resize_crop * y / 375
        #     w_rel = scale_resize_crop * w / 375
        #     h_rel = scale_resize_crop * h / 375

        #     target = torch.tensor([target, x_rel, y_rel, w_rel, h_rel])

        if self.transform_ is not None:
            sample = self.transform_(sample)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return sample, target
    
    #类别名优化  将类别名中的点号和下划线替换为空格，并合并连续的空格
    def redefine_class_to_idx(self):
        adjusted_dict = {}
        for k, v in self.class_to_idx.items():
            k = k.split('.')[-1].replace('_', ' ')
            split_key = k.split(' ')
            if len(split_key) > 2: 
                k = '-'.join(split_key[:-1]) + " " + split_key[-1]
            adjusted_dict[k] = v
        self.class_to_idx = adjusted_dict