from enum import Enum
from .cub import CUBDataset
from .dtd import DTD
from .food101 import Food101
from .fgvc_aircraft import FGVCAircraft
from .imagenetv2 import ImageNetV2Dataset
from .oxford_pets import OxfordIIITPet
from .eurosat import EuroSAT

class MyDataset(str, Enum):
    ImageNet = "imagenet"
    ImageNetV2 = "imagenetv2"
    ImageNetR = "imagenet-r"
    ImageNetS = "imagenet-s"
    ImageNetA = "imagenet-a"
    CUB = "cub"
    DTD = "dtd"
    Food101 = "food101"
    FGVCAircraft = "fgvc-aircraft"
    OxfordIIITPet = "oxford_pet"
    Place365 = "place365"
    EuroSAT = "eurosat"

    def __str__(self) -> str:
        return self.value
