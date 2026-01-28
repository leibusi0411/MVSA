#分割文件中没有定义标签，分割文件中只有所含的图片文件名，是按字母排序分配的数字标签  从0开始
#有多个分割文件，每个分割文件中包含类名和图片的文件名    banded/banded_0001.jpg  类别名/文件名.jpg
#支持三种分割：train、val、test，有独立的验证集
import os
import pathlib
from typing import Optional, Callable

import PIL.Image

from torchvision.datasets.utils import (
    download_and_extract_archive,
    verify_str_arg,
)
from torchvision.datasets import VisionDataset
from torchvision import datasets


class DTD(VisionDataset):
    """`Describable Textures Dataset (DTD) <https://www.robots.ox.ac.uk/~vgg/data/dtd/>`_.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        partition (int, optional): The dataset partition. Should be ``1 <= partition <= 10``. Defaults to ``1``.

            .. note::

                The partition only changes which split each image belongs to. Thus, regardless of the selected
                partition, combining all splits will result in all images.

        transform (callable, optional): A function/transform that  takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _URL = (
        "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
    )
    _MD5 = "fff73e5086ae6bdbea199a49dfb8a4c1"

    def __init__(
        self,
        root: str,
        split: str = "train",
        partition: int = 1,  #使用第一个分区
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        loader=datasets.folder.default_loader,
    ) -> None:
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        if not isinstance(partition, int) and not (1 <= partition <= 10):
            raise ValueError(
                f"Parameter 'partition' should be an integer with `1 <= partition <= 10`, "
                f"but got {partition} instead"
            )
        self._partition = partition

        super().__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.loader = loader
        self._base_folder = (
            pathlib.Path(self.root) / type(self).__name__.lower()
        )
        self._data_folder = self._base_folder / "dtd"
        self._meta_folder = self._data_folder / "labels"
        self._images_folder = self._data_folder / "images"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self._image_files = []
        classes = []
        with open(  #读取分割文件
            self._meta_folder / f"{self._split}{self._partition}.txt"
        ) as file:
            for line in file:
                cls, name = line.strip().split("/")  #获取类别名和图片文件名  banded/banded_0001.jpg  类别名/文件名.jpg
                self._image_files.append(
                    self._images_folder.joinpath(cls, name)  #拼接路径  dtd/images/banded/banded_0001.jpg
                )
                classes.append(cls) #收集类别名
        
        #创建类别名列表和类别名到索引的映射
        self.classes = sorted(set(classes)) #按字母排序
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))  # 创建字典，将类名映射到数字标签
        self._labels = [self.class_to_idx[cls] for cls in classes]
        

        
        # # 输出类名到标签的映射，方便检查
        # print(f"\n📊 DTD数据集 ({self._split}{self._partition}) 类别映射:")
        # print(f"   总类别数: {len(self.classes)}")
        # print(f"   样本数量: {len(self._image_files)}")
        # print(f"   完整类别映射字典:")
        # for class_name, class_idx in self.class_to_idx.items():
        #     print(f"     '{class_name}' → {class_idx}")
        # print(f"   ✅ 全部 {len(self.classes)} 个类别映射完成")



    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx):
        image_file, label = self._image_files[idx], self._labels[idx]
        image = self.loader(image_file)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}, partition={self._partition}"

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_folder) and os.path.isdir(
            self._data_folder
        )

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(
            self._URL, download_root=str(self._base_folder), md5=self._MD5
        )
