import os
import scipy.io
from os.path import join
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, list_dir, verify_str_arg

#train_list.mat - 训练集划分文件（官方，12,000张）
#test_list.mat - 测试集划分文件（官方，8,580张）
#splits/train_split.mat - 自定义训练集划分（9,600张）
#splits/val_split.mat - 自定义验证集划分（2,400张）

class Dogs(VisionDataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
            train (bool, optional): [DEPRECATED] Use split parameter instead. 
                                   If True, loads training set, otherwise loads test set.
            transform (callable, optional): A function/transform that takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    download_url_prefix = 'http://vision.stanford.edu/aditya86/ImageNetDogs'

    def __init__(self, root, split=None, train=None, transform=None, target_transform=None, download=False):
        super(Dogs, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        
        # 兼容旧接口：如果使用 train 参数，转换为 split 参数
        if split is None:
            if train is None:
                split = 'train'  # 默认为训练集
            else:
                split = 'train' if train else 'test'
        
        # 验证 split 参数
        self.split = verify_str_arg(split, "split", ("train", "val", "test"))

        if download:
            self.download()

        split_data = self.load_split()

        self.images_folder = join(self.root, 'images')  # 注意：改为小写 images
        self.annotations_folder = join(self.root, 'annotation')  # 注意：改为小写 annotation
        self._breeds = list_dir(self.images_folder)

        self._breed_images = [(annotation + '.jpg', idx) for annotation, idx in split_data]

        self._flat_breed_images = self._breed_images

    def __len__(self):
        return len(self._flat_breed_images)

    def __getitem__(self, index):
        image_name, target = self._flat_breed_images[index]
        image_path = join(self.images_folder, image_name)
        image = self.loader(image_path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def download(self):
        import tarfile

        # 检查是否已下载（兼容大小写文件夹名）
        images_folder = join(self.root, 'images') if os.path.exists(join(self.root, 'images')) else join(self.root, 'Images')
        annotation_folder = join(self.root, 'annotation') if os.path.exists(join(self.root, 'annotation')) else join(self.root, 'Annotation')
        
        if os.path.exists(images_folder) and os.path.exists(annotation_folder):
            if len(os.listdir(images_folder)) == len(os.listdir(annotation_folder)) == 120:
                print('Files already downloaded and verified')
                return

        for filename in ['images', 'annotation', 'lists']:
            tar_filename = filename + '.tar'
            url = self.download_url_prefix + '/' + tar_filename
            download_url(url, self.root, tar_filename, None)
            print('Extracting downloaded file: ' + join(self.root, tar_filename))
            with tarfile.open(join(self.root, tar_filename), 'r') as tar_file:
                tar_file.extractall(self.root)
            os.remove(join(self.root, tar_filename))

    def load_split(self):
        """
        加载数据集划分
        
        - train: 从 splits/train_split.mat 加载（9,600张）
        - val: 从 splits/val_split.mat 加载（2,400张）
        - test: 从 lists/test_list.mat 加载（8,580张）
        """
        if self.split == 'train':
            split_file = join(self.root, 'splits', 'train_split.mat')
        elif self.split == 'val':
            split_file = join(self.root, 'splits', 'val_split.mat')
        else:  # test
            split_file = join(self.root, 'lists', 'test_list.mat')
        
        # 读取 .mat 文件
        mat_data = scipy.io.loadmat(split_file)
        split = mat_data['annotation_list']
        labels = mat_data['labels']

        # 解析数据格式
        split = [item[0][0] for item in split]      # 提取文件路径（无.jpg后缀）
        labels = [item[0] - 1 for item in labels]   # 提取标签并转为0-based索引
        
        return list(zip(split, labels))

    def stats(self):
        counts = {}
        for index in range(len(self._flat_breed_images)):
            image_name, target_class = self._flat_breed_images[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print("%d samples spanning %d classes (avg %f per class)" % (len(self._flat_breed_images), len(counts.keys()),
                                                                     float(len(self._flat_breed_images)) / float(
                                                                         len(counts.keys()))))

        return counts


if __name__ == '__main__':
    # 测试新接口
    train_dataset = Dogs('./dogs', split='train', download=False)
    val_dataset = Dogs('./dogs', split='val', download=False)
    test_dataset = Dogs('./dogs', split='test', download=False)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")