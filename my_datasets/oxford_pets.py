#分割文件中提供了数字标签  从1开始，代码中调整从0开始
#直接使用了数据集创建者提供的现成数字标签
#将 'train' 映射到 'trainval'（训练+验证合并）      官方分割：'trainval' 和 'test'
#trainval.txt中的数据需要手动分割

#第1列：图像ID（不含.jpg扩展名）     Abyssinian_1 1 1 1  图像ID/（宠物品种）类别标签/物种标签/品种内ID  图像ID也就是图片文件名
# 第2列：宠物品种标签（1-37，代表37个品种）
# 第3列：物种标签（1=猫，2=狗）
# 第4列：品种内ID
import os
import os.path
import pathlib
from typing import Any, Callable, Optional, Union, Tuple
from typing import Sequence

from PIL import Image

from torchvision.datasets.utils import (
    download_and_extract_archive,
    verify_str_arg,
)
from torchvision.datasets import VisionDataset
from torchvision import datasets


class OxfordIIITPet(VisionDataset):
    """`Oxford-IIIT Pet Dataset   <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"trainval"`` (default) or ``"test"``.
        target_types (string, sequence of strings, optional): Types of target to use. Can be ``category`` (default) or
            ``segmentation``. Can also be a list to output a tuple with all specified target types. The types represent:

                - ``category`` (int): Label for one of the 37 pet categories.
                - ``segmentation`` (PIL image): Segmentation trimap of the image.

            If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that  takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and puts it into
            ``root/oxford-iiit-pet``. If dataset is already downloaded, it is not downloaded again.
    """

    _RESOURCES = (
        (
            "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            "5c4f3ee8e5d25df40f4fd59a7f44e54c",
        ),
        (
            "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            "95a8c909bbe2e81eed6a22bccdf3f68f",
        ),
    )
    _VALID_TARGET_TYPES = ("category", "segmentation")

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_types: Union[Sequence[str], str] = "category",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        loader=datasets.folder.default_loader,
    ):
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        if isinstance(target_types, str):
            target_types = [target_types]
        self._target_types = [
            verify_str_arg(
                target_type, "target_types", self._VALID_TARGET_TYPES
            )
            for target_type in target_types
        ]

        super().__init__(
            root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        self.loader = loader
        self._base_folder = pathlib.Path(self.root) / "oxford-iiit-pet"
        self._images_folder = self._base_folder / "images"
        self._anns_folder = self._base_folder / "annotations"
        self._segs_folder = self._anns_folder / "trimaps"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        image_ids = []
        self._labels = []
        
        # 读取对应的划分文件（train.txt/val.txt/test.txt）
        split_file = self._anns_folder / f"{self._split}.txt"
        with open(split_file) as file:
            for line in file:
                line = line.strip()
                if line:
                    image_id, label, *_ = line.split()
                    image_ids.append(image_id)
                    self._labels.append(int(label) - 1)  # 标签从1开始，转换为从0开始



        self.classes = [   #从图像ID提取品种名称
            " ".join(part for part in raw_cls.split("_"))
            for raw_cls, _ in sorted(
                {
                    (image_id.rsplit("_", 1)[0], label)   #提取品种名称，提取类别名
                    for image_id, label in zip(image_ids, self._labels)
                },
                key=lambda image_id_and_label: image_id_and_label[1],   # 按标签值排序  ("品种名", 标签) 取标签值（第二个元素）作为排序键
            ) 
        ]
        
        # # 保存第一次排序后的映射（用于检查）
        # first_class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        # self.classes = sorted(self.classes)  #按字母排序
        # self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        


        # # 添加检查代码：输出两次排序结果并检查是否有改变
        # print(f"\n🐕 Oxford Pets数据集 ({self._split}) 标签映射检查:")
        # print(f"   总类别数: {len(self.classes)}")
        # print(f"   样本数量: {len(image_ids)}")
        
        # # 输出第一次排序结果（按原始标签排序）- 全部类别
        # print(f"\n   第一次排序（按原始标签）全部类别:")
        # for class_name, class_idx in first_class_to_idx.items():
        #     print(f"     '{class_name}' → {class_idx}")
        
        # # 输出第二次排序结果（按字母排序）- 全部类别
        # print(f"\n   第二次排序（按字母顺序）全部类别:")
        # for class_name, class_idx in self.class_to_idx.items():
        #     print(f"     '{class_name}' → {class_idx}")
        
        # # 检查标签映射是否发生改变
        # mapping_changed = first_class_to_idx != self.class_to_idx
        # if mapping_changed:
        #     print(f"\n   ⚠️  警告：两次排序导致标签映射发生改变！")
        #     print(f"       这可能导致self._labels与class_to_idx不一致")
            
        #     # 统计有多少类别的标签发生了改变
        #     changed_count = 0
        #     for class_name in self.classes:
        #         if first_class_to_idx[class_name] != self.class_to_idx[class_name]:
        #             changed_count += 1
            
        #     print(f"       共有 {changed_count}/{len(self.classes)} 个类别的标签发生了改变")
            
        #     # 显示所有发生改变的类别
        #     print(f"       标签映射变化详情（全部）:")
        #     for class_name in self.classes:
        #         old_idx = first_class_to_idx[class_name]
        #         new_idx = self.class_to_idx[class_name]
        #         if old_idx != new_idx:
        #             print(f"         '{class_name}': {old_idx} → {new_idx}")
            
        #     # 额外检查：验证标签一致性
        #     print(f"\n   🔍 标签一致性验证（检查前10个样本）:")
        #     for idx in range(min(10, len(image_ids))):
        #         image_id = image_ids[idx]
        #         actual_label = self._labels[idx]
        #         breed_from_id = image_id.rsplit("_", 1)[0].replace("_", " ")
                
        #         # 检查这个标签对应的类别名是否与从ID提取的品种名一致
        #         if actual_label < len(self.classes):
        #             predicted_breed = list(self.class_to_idx.keys())[list(self.class_to_idx.values()).index(actual_label)]
        #             consistency = "✅" if breed_from_id == predicted_breed else "❌"
        #             print(f"       样本{idx}: {image_id}")
        #             print(f"         标签: {actual_label}")
        #             print(f"         从ID提取: '{breed_from_id}'")
        #             print(f"         从标签映射: '{predicted_breed}'")
        #             print(f"         一致性: {consistency}")
        #         else:
        #             print(f"       样本{idx}: {image_id} - 标签超出范围: {actual_label}")
        # else:
        #     # 即使一致，也做一致性验证
        #     print(f"\n   🔍 标签一致性验证（检查前10个样本）:")
        #     for idx in range(min(10, len(image_ids))):
        #         image_id = image_ids[idx]
        #         actual_label = self._labels[idx]
        #         breed_from_id = image_id.rsplit("_", 1)[0].replace("_", " ")
                
        #         if actual_label < len(self.classes):
        #             predicted_breed = list(self.class_to_idx.keys())[list(self.class_to_idx.values()).index(actual_label)]
        #             consistency = "✅" if breed_from_id == predicted_breed else "❌"
        #             print(f"       样本{idx}: {image_id} - 标签{actual_label} - {consistency}")
        #             if breed_from_id != predicted_breed:
        #                 print(f"         从ID提取: '{breed_from_id}' vs 从标签映射: '{predicted_breed}'")
        #         else:
        #             print(f"       样本{idx}: {image_id} - 标签超出范围: {actual_label}")




        #拼接路径  oxford-iiit-pet/images/Abyssinian_1.jpg
        self._images = [
            self._images_folder / f"{image_id}.jpg" for image_id in image_ids
        ]

        self._segs = [
            self._segs_folder / f"{image_id}.png" for image_id in image_ids
        ]

    def __len__(self) -> int:
        return len(self._images)


    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        # image = Image.open(self._images[idx]).convert("RGB")
        image = self.loader(self._images[idx])

        target: Any = []
        for target_type in self._target_types:
            if target_type == "category":
                target.append(self._labels[idx])
            else:  # target_type == "segmentation"
                target.append(Image.open(self._segs[idx]))

        if not target:
            target = None
        elif len(target) == 1:
            target = target[0]
        else:
            target = tuple(target)

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    def _check_exists(self) -> bool:
        for folder in (self._images_folder, self._anns_folder):
            if not (os.path.exists(folder) and os.path.isdir(folder)):
                return False
        else:
            return True


    def _download(self) -> None:
        if self._check_exists():
            return

        for url, md5 in self._RESOURCES:
            download_and_extract_archive(
                url, download_root=str(self._base_folder), md5=md5
            )