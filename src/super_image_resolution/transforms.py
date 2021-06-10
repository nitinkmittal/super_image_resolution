from typing import Sequence, Union

import numpy as np

from torchvision.transforms import InterpolationMode, Resize


class CustomResize(object):
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
