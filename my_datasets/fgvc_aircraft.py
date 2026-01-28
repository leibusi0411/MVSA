from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable
from torchvision import datasets
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg

#data/images_variant_{train/val/test}.txt 文件 核心分割文件 根据txt文件进行分割  有验证集划分   官方划分 训练集 3,334   验证集  3,333  测试集   3,333

class FGVCAircraft(VisionDataset):
    """`FGVC Aircraft <https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.

    The dataset contains 10,000 images of aircraft, with 100 images for each of 100
    different aircraft model variants, most of which are airplanes.
    Aircraft models are organized in a three-levels hierarchy. The three levels, from
    finer to coarser, are:

    - ``variant``, e.g. Boeing 737-700. A variant collapses all the models that are visually
        indistinguishable into one class. The dataset comprises 100 different variants.
    - ``family``, e.g. Boeing 737. The dataset comprises 70 different families.
    - ``manufacturer``, e.g. Boeing. The dataset comprises 30 different manufacturers.

    Args:
        root (str or ``pathlib.Path``): Root directory of the FGVC Aircraft dataset.
        split (string, optional): The dataset split, supports ``train``, ``val``,
            ``trainval`` and ``test``.
        annotation_level (str, optional): The annotation level, supports ``variant``,
            ``family`` and ``manufacturer``.
        transform (callable, optional): A function/transform that takes in a PIL image or torch.Tensor, depends on the given loader,
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        loader (callable, optional): A function to load an image given its path.
            By default, it uses PIL as its image loader, but users could also pass in
            ``torchvision.io.decode_image`` for decoding image data into tensors directly.
    """

    _URL = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"

    def __init__(  #初始化参数
        self,
        root: str | Path,   #根目录
        split: str = "trainval",   #分割
        annotation_level: str = "variant",
        transform: Callable | None = None,   #变换    三级分类层次结构   细粒度使用variant
        target_transform: Callable | None = None,   #目标变换
        download: bool = False,   #下载数据
        loader: Callable[[str], Any] = datasets.folder.default_loader,   #加载器
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "val", "trainval", "test"))
        self._annotation_level = verify_str_arg(
            annotation_level, "annotation_level", ("variant", "family", "manufacturer")
        )
        #数据路径
        self._data_path = os.path.join(self.root, "fgvc-aircraft-2013b")
        if download:   #下载数据
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        annotation_file = os.path.join(   #分割文件   类别信息文件加载
            self._data_path,
            "data",
            {
                "variant": "variants.txt",
                "family": "families.txt",
                "manufacturer": "manufacturers.txt",
            }[self._annotation_level],
        )
        with open(annotation_file) as f:
            self.classes = [line.strip() for line in f]

        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))   #创建类别名到索引的映射
        
        #核心数据加载逻辑
        image_data_folder = os.path.join(self._data_path, "data", "images")   #图片数据文件夹
        labels_file = os.path.join(self._data_path, "data", f"images_{self._annotation_level}_{self._split}.txt")   #分割文件

        self._image_files = []
        self._labels = []

        with open(labels_file) as f:
            for line in f:
                image_name, label_name = line.strip().split(" ", 1)
                self._image_files.append(os.path.join(image_data_folder, f"{image_name}.jpg"))
                self._labels.append(self.class_to_idx[label_name])
        self.loader = loader

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = self.loader(image_file)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def _download(self) -> None:
        """
        Download the FGVC Aircraft dataset archive and extract it under root.
        """
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, self.root)

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_path) and os.path.isdir(self._data_path)