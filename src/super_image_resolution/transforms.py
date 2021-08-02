"""Contains Pytorch styled image transformations."""
from typing import Any, Sequence, Union

import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode, Resize


class CustomResize(object):
    """Added randomness to torchvision.transforms.Resize."""

    def __init__(self, size: Union[Sequence[int], int], interpolation=None):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        if self.interpolation is None:
            self.interpolation = np.random.choice(
                [
                    InterpolationMode.NEAREST,
                    InterpolationMode.BILINEAR,
                    InterpolationMode.BICUBIC,
                ]
            )
        return Resize(size=self.size, interpolation=self.interpolation)(img)

    def __repr(self):
        return self.__class__.__name__ + f"Custom Resize"


class InvertNormalize(object):
    """Perform invert normalization."""

    def __init__(
        self, num_channels: int, mean: Sequence[float], std: Sequence[float]
    ):
        self.num_channels = num_channels
        self.mean = self._fix_arg(mean)
        self._validate_arg(self.mean)
        self.std = self._fix_arg(std)
        self._validate_arg(self.std)

    def _fix_arg(self, arg: Any, dtype=torch.float32) -> Any:
        if not isinstance(arg, Tensor):
            arg = torch.tensor(arg, dtype=dtype)
        arg = arg.type(torch.float32)
        return arg

    def _validate_arg(self, arg: Tensor) -> Tensor:
        if arg.shape != (self.num_channels,):
            raise ValueError(f"Expected shape: {arg.shape}, got {type(arg)}")

    def __call__(self, img: Tensor) -> Tensor:
        ndim = img.ndim
        if ndim < 3:
            raise ValueError(
                f"Expected 3 or more dimensional tensor, "
                f"got {img.ndim} instead"
            )
        dim = torch.ones(ndim, dtype=torch.uint8)
        dim[1] = self.num_channels
        mean = self.mean.view(*dim)
        std = self.std.view(*dim)
        return (img * std) + mean

    def __repr__(self):
        return (
            self.__class__.__name__ + f"(mean: {self.mean}, std: {self.std})"
        )
