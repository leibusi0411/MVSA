from pathlib import Path
import json
from typing import Any, Callable, Optional, Union
from torchvision import datasets
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg

# 使用 split_zhou_OxfordFlowers102.json 文件进行分割  有验证集划分
# 官方划分: 训练集 (Training Set) 1,020 | 验证集 (Validation Set) 1,020 | 测试集 (Test Set) 6,149
# 自定义划分: 训练集 (train) 4,093 | 验证集 (val) 1,633 | 测试集 (test) 2,463

class Flowers102(VisionDataset):
    """`Oxford 102 Flower <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ Dataset.

    Oxford 102 Flower is an image classification dataset consisting of 102 flower categories.
    This implementation uses split_zhou_OxfordFlowers102.json for data splitting.

    The flowers were chosen to be flowers commonly occurring in the United Kingdom. Each class
    consists of between 40 and 258 images. The images have large scale, pose and light variations.

    Args:
        root (str or ``pathlib.Path``): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        transform (callable, optional): A function/transform that takes in a PIL image or torch.Tensor, depends on the given loader,
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): This parameter exists for backward compatibility but does not download the dataset.
        loader (callable, optional): A function to load an image given its path.
            By default, it uses PIL as its image loader, but users could also pass in
            ``torchvision.io.decode_image`` for decoding image data into tensors directly.
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        loader: Callable[[Union[str, Path]], Any] = datasets.folder.default_loader,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        # 验证分割参数
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = Path(root)
        
        # 分割文件路径 - 与 flowers-102 文件夹处于同一级别
        self._split_file_path = self._base_folder / "split_zhou_OxfordFlowers102.json"
        
        if download:
            self.download()
        
        if not self._check_exists():
            raise RuntimeError(
                f"Dataset not found. Please ensure split_zhou_OxfordFlowers102.json exists in {self._base_folder}."
            )
        
        # 读取分割文件
        with open(self._split_file_path, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
        
        # 获取当前分割的数据   根据 split 参数 ("train"/"val"/"test") 获取对应的数据列表
        current_split_data = split_data[self._split]
        
        # 构建样本列表: (图像路径, 类别标签)
        self._samples = []
        for item in current_split_data:  # 遍历当前分割的数据列表
            # item 格式: ["image_00001.jpg", 0, "pink primrose"]
            # 注意：JSON中只有文件名，需要添加 jpg/ 前缀
            image_filename = item[0]  # item[0] 是文件名
            image_path = str(self._base_folder / "jpg" / image_filename)
            class_label = item[1]  # item[1] 是类别标签 (0-101)
            self._samples.append((image_path, class_label))
        
        # 注意：本项目不使用 dataset.classes 和 dataset.class_to_idx
        # 类别名称由 item[2] 提供，但不在此处使用
        
        self.loader = loader

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        """
        获取一个样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            tuple: (image, label) where label is the class index (0-101)
        """
        image_path, label = self._samples[idx]
        image = self.loader(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

    def _check_exists(self) -> bool:
        """检查分割文件是否存在"""
        return self._split_file_path.exists() and self._split_file_path.is_file()

    def download(self):
        """
        下载方法（向后兼容，但不执行实际下载）
        
        Note:
            The dataset needs to be manually downloaded and the split file needs to be created.
        """
        if self._check_exists():
            print(f"Split file {self._split_file_path.name} already exists.")
            return
        
        raise NotImplementedError(
            "Automatic download is not implemented. Please:\n"
            "1. Download the dataset manually from https://www.robots.ox.ac.uk/~vgg/data/flowers/102/\n"
            "2. Ensure split_zhou_OxfordFlowers102.json exists in the root directory."
        )

    def extra_repr(self) -> str:
        """额外的字符串表示"""
        return f"split={self._split}"

    # 102个花卉类别名称（与官方实现保持一致）
    classes = [
        "pink primrose",
        "hard-leaved pocket orchid",
        "canterbury bells",
        "sweet pea",
        "english marigold",
        "tiger lily",
        "moon orchid",
        "bird of paradise",
        "monkshood",
        "globe thistle",
        "snapdragon",
        "colt's foot",
        "king protea",
        "spear thistle",
        "yellow iris",
        "globe-flower",
        "purple coneflower",
        "peruvian lily",
        "balloon flower",
        "giant white arum lily",
        "fire lily",
        "pincushion flower",
        "fritillary",
        "red ginger",
        "grape hyacinth",
        "corn poppy",
        "prince of wales feathers",
        "stemless gentian",
        "artichoke",
        "sweet william",
        "carnation",
        "garden phlox",
        "love in the mist",
        "mexican aster",
        "alpine sea holly",
        "ruby-lipped cattleya",
        "cape flower",
        "great masterwort",
        "siam tulip",
        "lenten rose",
        "barbeton daisy",
        "daffodil",
        "sword lily",
        "poinsettia",
        "bolero deep blue",
        "wallflower",
        "marigold",
        "buttercup",
        "oxeye daisy",
        "common dandelion",
        "petunia",
        "wild pansy",
        "primula",
        "sunflower",
        "pelargonium",
        "bishop of llandaff",
        "gaura",
        "geranium",
        "orange dahlia",
        "pink-yellow dahlia?",
        "cautleya spicata",
        "japanese anemone",
        "black-eyed susan",
        "silverbush",
        "californian poppy",
        "osteospermum",
        "spring crocus",
        "bearded iris",
        "windflower",
        "tree poppy",
        "gazania",
        "azalea",
        "water lily",
        "rose",
        "thorn apple",
        "morning glory",
        "passion flower",
        "lotus",
        "toad lily",
        "anthurium",
        "frangipani",
        "clematis",
        "hibiscus",
        "columbine",
        "desert-rose",
        "tree mallow",
        "magnolia",
        "cyclamen",
        "watercress",
        "canna lily",
        "hippeastrum",
        "bee balm",
        "ball moss",
        "foxglove",
        "bougainvillea",
        "camellia",
        "mallow",
        "mexican petunia",
        "bromelia",
        "blanket flower",
        "trumpet creeper",
        "blackberry lily",
    ]


# class Flowers102(VisionDataset):
#     """`Oxford 102 Flower <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ Dataset.

#     .. warning::

#         This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

#     Oxford 102 Flower is an image classification dataset consisting of 102 flower categories. The
#     flowers were chosen to be flowers commonly occurring in the United Kingdom. Each class consists of
#     between 40 and 258 images.

#     The images have large scale, pose and light variations. In addition, there are categories that
#     have large variations within the category, and several very similar categories.

#     Args:
#         root (str or ``pathlib.Path``): Root directory of the dataset.
#         split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
#         transform (callable, optional): A function/transform that takes in a PIL image or torch.Tensor, depends on the given loader,
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the target and transforms it.
#         download (bool, optional): If true, downloads the dataset from the internet and
#             puts it in root directory. If dataset is already downloaded, it is not
#             downloaded again.
#         loader (callable, optional): A function to load an image given its path.
#             By default, it uses PIL as its image loader, but users could also pass in
#             ``torchvision.io.decode_image`` for decoding image data into tensors directly.
#     """

#     _download_url_prefix = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
#     _file_dict = {  # filename, md5
#         "image": ("102flowers.tgz", "52808999861908f626f3c1f4e79d11fa"),
#         "label": ("imagelabels.mat", "e0620be6f572b9609742df49c70aed4d"),
#         "setid": ("setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c"),
#     }
#     _splits_map = {"train": "trnid", "val": "valid", "test": "tstid"}

#     def __init__(
#         self,
#         root: Union[str, Path],
#         split: str = "train",
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         download: bool = False,
#         loader: Callable[[Union[str, Path]], Any] = datasets.folder.default_loader,
#     ) -> None:
#         super().__init__(root, transform=transform, target_transform=target_transform)
#         self._split = verify_str_arg(split, "split", ("train", "val", "test"))
#         self._base_folder = Path(self.root) / "flowers-102"
#         self._images_folder = self._base_folder / "jpg"

#         if download:
#             self.download()

#         if not self._check_integrity():
#             raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

#         from scipy.io import loadmat

#         set_ids = loadmat(self._base_folder / self._file_dict["setid"][0], squeeze_me=True)
#         image_ids = set_ids[self._splits_map[self._split]].tolist()

#         labels = loadmat(self._base_folder / self._file_dict["label"][0], squeeze_me=True)
#         image_id_to_label = dict(enumerate((labels["labels"] - 1).tolist(), 1))

#         self._labels = []
#         self._image_files = []
#         for image_id in image_ids:
#             self._labels.append(image_id_to_label[image_id])
#             self._image_files.append(self._images_folder / f"image_{image_id:05d}.jpg")

#         self.loader = loader

#     def __len__(self) -> int:
#         return len(self._image_files)

#     def __getitem__(self, idx: int) -> tuple[Any, Any]:
#         image_file, label = self._image_files[idx], self._labels[idx]
#         image = self.loader(image_file)

#         if self.transform:
#             image = self.transform(image)

#         if self.target_transform:
#             label = self.target_transform(label)

#         return image, label

#     def extra_repr(self) -> str:
#         return f"split={self._split}"

#     def _check_integrity(self):
#         if not (self._images_folder.exists() and self._images_folder.is_dir()):
#             return False

#         for id in ["label", "setid"]:
#             filename, md5 = self._file_dict[id]
#             if not check_integrity(str(self._base_folder / filename), md5):
#                 return False
#         return True

#     def download(self):
#         if self._check_integrity():
#             return
#         download_and_extract_archive(
#             f"{self._download_url_prefix}{self._file_dict['image'][0]}",
#             str(self._base_folder),
#             md5=self._file_dict["image"][1],
#         )
#         for id in ["label", "setid"]:
#             filename, md5 = self._file_dict[id]
#             download_url(self._download_url_prefix + filename, str(self._base_folder), md5=md5)

#     classes = [
#         "pink primrose",
#         "hard-leaved pocket orchid",
#         "canterbury bells",
#         "sweet pea",
#         "english marigold",
#         "tiger lily",
#         "moon orchid",
#         "bird of paradise",
#         "monkshood",
#         "globe thistle",
#         "snapdragon",
#         "colt's foot",
#         "king protea",
#         "spear thistle",
#         "yellow iris",
#         "globe-flower",
#         "purple coneflower",
#         "peruvian lily",
#         "balloon flower",
#         "giant white arum lily",
#         "fire lily",
#         "pincushion flower",
#         "fritillary",
#         "red ginger",
#         "grape hyacinth",
#         "corn poppy",
#         "prince of wales feathers",
#         "stemless gentian",
#         "artichoke",
#         "sweet william",
#         "carnation",
#         "garden phlox",
#         "love in the mist",
#         "mexican aster",
#         "alpine sea holly",
#         "ruby-lipped cattleya",
#         "cape flower",
#         "great masterwort",
#         "siam tulip",
#         "lenten rose",
#         "barbeton daisy",
#         "daffodil",
#         "sword lily",
#         "poinsettia",
#         "bolero deep blue",
#         "wallflower",
#         "marigold",
#         "buttercup",
#         "oxeye daisy",
#         "common dandelion",
#         "petunia",
#         "wild pansy",
#         "primula",
#         "sunflower",
#         "pelargonium",
#         "bishop of llandaff",
#         "gaura",
#         "geranium",
#         "orange dahlia",
#         "pink-yellow dahlia?",
#         "cautleya spicata",
#         "japanese anemone",
#         "black-eyed susan",
#         "silverbush",
#         "californian poppy",
#         "osteospermum",
#         "spring crocus",
#         "bearded iris",
#         "windflower",
#         "tree poppy",
#         "gazania",
#         "azalea",
#         "water lily",
#         "rose",
#         "thorn apple",
#         "morning glory",
#         "passion flower",
#         "lotus",
#         "toad lily",
#         "anthurium",
#         "frangipani",
#         "clematis",
#         "hibiscus",
#         "columbine",
#         "desert-rose",
#         "tree mallow",
#         "magnolia",
#         "cyclamen",
#         "watercress",
#         "canna lily",
#         "hippeastrum",
#         "bee balm",
#         "ball moss",
#         "foxglove",
#         "bougainvillea",
#         "camellia",
#         "mallow",
#         "mexican petunia",
#         "bromelia",
#         "blanket flower",
#         "trumpet creeper",
#         "blackberry lily",
#     ]


