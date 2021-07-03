"""Contains functions to generate plots."""
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from torch import Tensor

from .validate import validate_arg_type


def display_random_output(
    X: List[Tensor],
    Y: List[Tensor],
    Y_hat: List[Tensor],
    save_image: bool = False,
    fp: str = None,
    *args,
    **kwargs
):
    """Display random input, ground truth and predicted output."""
    idxs = np.arange(len(X))
    np.random.shuffle(idxs)
    i = np.random.choice(idxs)

    x = X[i].numpy().transpose(1, 2, 0)
    y = Y[i].numpy().transpose(1, 2, 0)
    y_hat = Y_hat[i].numpy().transpose(1, 2, 0)

    figsize = (12, 4)
    if "figsize" in kwargs.keys():
        figsize = kwargs["figsize"]

    _, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=figsize)
    ax1.imshow(x)
    ax1.set_axis_off()
    ax1.set_title("low-resolution input")

    ax2.imshow(y)
    ax2.set_axis_off()
    ax2.set_title("high-resolution ground truth")

    ax3.imshow(y_hat)
    ax3.set_axis_off()
    ax3.set_title("high-resolution output")
    plt.tight_layout()

    if save_image:
        validate_arg_type(fp, str)
        del kwargs["figsize"]
        if ".png" not in fp[-4:]:
            fp += ".png"
        plt.savefig(fp, *args, **kwargs)
    plt.show()
