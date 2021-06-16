"""Contains Pytorch styled image transformations."""
from typing import Sequence, Union

import numpy as np

from torchvision.transforms import InterpolationMode, Resize
from torch import Tensor


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

    def __init__(self, num_channels, mean, std):
        self.mean = mean.view(1, num_channels, 1, 1)
        self.std = std.view(1, num_channels, 1, 1)

    def __call__(self, img: Tensor) -> Tensor:
        return (img * self.std) + self.mean

    def __repr(self):
        return self.__class__.__name__ + f"(mean: {self.mean}, std: {self.std})"
