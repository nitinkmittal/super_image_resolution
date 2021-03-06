"""Contain PyTorch style dataset functions."""
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
        """
        Parameters
        ----------
        paths: path to dataset

        in_transforms: transforms for model input

        out_transforms: transforms for model output
        """
        self.paths = paths
        self.in_transforms = in_transforms
        self.out_transforms = out_transforms

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx) -> Tuple[Union[PosixPath, str], Tensor, Tensor]:
        fp = self.paths[idx]
        img = Image.open(fp)
        return fp, self.in_transforms(img), self.out_transforms(img)
