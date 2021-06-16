"""Contains custom Pytorch styled datasets."""
import os
from pathlib import PosixPath
from typing import List, Tuple, Union

from PIL import Image

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class CustomDataset(Dataset):
    def __init__(
        self,
        paths: List[Union[str, PosixPath]],
        in_transforms: Compose,
        out_transforms: Compose,
    ):
        self.paths = paths
        self.in_transforms = in_transforms
        self.out_transforms = out_transforms

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        img = Image.open(self.paths[idx])
        return self.in_transforms(img), self.out_transforms(img)
