#会加载json文件，JSON 文件的结构是一个字典，其中键（key）是类别名称（比如 "apple_pie"），
# 值（value）是属于该类别的图片路径列表（文件名）。
#获取了 JSON 文件中所有的类别名称（metadata.keys()），然后对它们进行字母排序。 将每个类别名称映射到一个从 0 开始的唯一整数ID
#没有验证集 

import json
from pathlib import Path
from typing import Any, Tuple, Callable, Optional

import PIL.Image

from torchvision.datasets.utils import (
    download_and_extract_archive,
    verify_str_arg,
)
from torchvision.datasets import VisionDataset
from torchvision import datasets


class Food101(VisionDataset):
    """`The Food-101 Data Set <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>`_.

    The Food-101 is a challenging data set of 101 food categories, with 101'000 images.
    For each class, 250 manually reviewed test images are provided as well as 750 training images.
    On purpose, the training images were not cleaned, and thus still contain some amount of noise.
    This comes mostly in the form of intense colors and sometimes wrong labels. All images were
    rescaled to have a maximum side length of 512 pixels.


    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, and ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    _MD5 = "85eeb15f3717b99a5da872d97d918f87"

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        loader=datasets.folder.default_loader,
    ) -> None:
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
        )
        self.loader = loader
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = Path(self.root) / "food-101"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "images"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self._labels = []
        self._image_files = []
        
        with open(self._meta_folder / f"{split}.json") as f: #meta/train.json  加载分割文件
            metadata = json.loads(f.read())

        self.classes = sorted(metadata.keys())   #创建类别名列表，按字母排序所有类别名
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))  #创建类别名到索引的映射

        for class_label, im_rel_paths in metadata.items():
            ## 为当前类别的所有图像分配相同的数字标签
            self._labels += [self.class_to_idx[class_label]] * len(im_rel_paths)
            #拼接路径  food-101/images/apple_pie/apple_pie_0001.jpg
            self._image_files += [
                self._images_folder.joinpath(*f"{im_rel_path}.jpg".split("/"))
                for im_rel_path in im_rel_paths
            ]
        
        # 输出类名到标签的映射，方便检查
        # print(f"\n🍕 Food101数据集 ({self._split}) 类别映射:")
        # print(f"   总类别数: {len(self.classes)}")
        # print(f"   样本数量: {len(self._image_files)}")
        # print(f"   完整类别映射字典:")
        # for class_name, class_idx in self.class_to_idx.items():
        #     print(f"     '{class_name}' → {class_idx}")
        # print(f"   ✅ 全部 {len(self.classes)} 个食物类别映射完成")


    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image, label = self.loader(self._image_files[idx]), self._labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}"
    

    def _check_exists(self) -> bool:
        return all(
            folder.exists() and folder.is_dir()
            for folder in (self._meta_folder, self._images_folder)
        )

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(
            self._URL, download_root=self.root, md5=self._MD5
        )
