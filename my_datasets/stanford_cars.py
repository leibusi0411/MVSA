import pathlib
import json
from typing import Any, Callable, Optional, Union
from torchvision import datasets
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg

# 使用 split_zhou_StanfordCars.json 统一分割文件    
# 包含 train/val/test 三个分割，每个条目格式: [文件路径, 类别标签, 类别名称]
# cars_train 文件夹 包含训练集和验证集的图片
# cars_test 文件夹 包含测试集的图片
    # - train: 6,509 images
    # - val: 1,635 images  
    # - test: 8,041 images


class StanfordCars(VisionDataset):
    """Stanford Cars Dataset

    The Cars dataset contains 16,185 images of 196 classes of cars. The data is
    split using split_zhou_StanfordCars.json into:


    The original URL is https://ai.stanford.edu/~jkrause/cars/car_dataset.html,
    the dataset isn't available online anymore.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        transform (callable, optional): A function/transform that takes in a PIL image or torch.Tensor, depends on the given loader,
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): This parameter exists for backward compatibility but it does not
            download the dataset, since the original URL is not available anymore.
        loader (callable, optional): A function to load an image given its path.
            By default, it uses PIL as its image loader, but users could also pass in
            ``torchvision.io.decode_image`` for decoding image data into tensors directly.
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        loader: Callable[[str], Any] = datasets.folder.default_loader,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        # 验证分割参数
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = pathlib.Path(root)
        
        # 分割文件路径 - 与 cars_train 文件夹处于同一级别
        self._split_file_path = self._base_folder / "split_zhou_StanfordCars.json"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. Please ensure split_zhou_StanfordCars.json exists in the root directory.")

        # 读取分割文件
        with open(self._split_file_path, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
        
        # 获取当前分割的数据   根据 split 参数 ("train"/"val"/"test") 获取对应的数据列表
        current_split_data = split_data[self._split]
        
        # 构建样本列表: (图像路径, 类别标签)  
        self._samples = []
        for item in current_split_data:   #遍历当前分割的数据列表   item 格式: ["cars_train/05266.jpg", 13, "2012 Audi TTS Coupe"]
            image_path = str(self._base_folder / item[0])  # item[0] 是相对路径
            class_label = item[1]  # item[1] 是类别标签 (0-195)   #这里没有使用 item[2] (类别名称)
            self._samples.append((image_path, class_label))
        
        # 注意：本项目不使用 dataset.classes 和 dataset.class_to_idx
        # 文本特征通过 dataset_utils.load_classes() 从外部JSON文件加载
        # 因此这里不需要构建类别映射，节省内存和计算时间
        self.loader = loader

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        """Returns pil_image and class_id for given index"""
        image_path, target = self._samples[idx]
        image = self.loader(image_path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def _check_exists(self) -> bool:
        # 检查分割文件是否存在
        if not self._split_file_path.exists():
            return False
        
        # 检查图像文件夹是否存在
        cars_train_dir = self._base_folder / "cars_train"
        cars_test_dir = self._base_folder / "cars_test"
        
        return cars_train_dir.is_dir() and cars_test_dir.is_dir()

    def download(self):
        raise ValueError("The original URL is broken so the StanfordCars dataset cannot be downloaded anymore.")