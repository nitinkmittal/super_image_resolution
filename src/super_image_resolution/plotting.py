"""Contains functions to generate plots."""
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from torch import Tensor

from .validate import validate_arg_type
from PIL import ImageDraw, ImageFont, Image
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage


def plot_grid(x: Tensor, y: Tensor, y_hat: Tensor, **kwargs) -> Image.Image:
    """Plot grid."""
    padding = 2
    if "padding" in kwargs.keys():
        padding = kwargs["padding"]

    pad_value = 1.0
    if "pad_value" in kwargs.keys():
        pad_value = kwargs["pad_value"]

    font_color = "black"
    if "font_color" in kwargs.keys():
        font_color = kwargs["font_color"]

    img = make_grid(
        tensor=[x, y, y_hat], nrow=3, padding=padding, pad_value=pad_value
    )

    img = ToPILImage()(img)
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype(
        "/home/mittal.nit/projects/super_image_resolution/fonts/arial.ttf", 15
    )

    size = x.size()[-1]
    x_1, x_2, x_3 = (
        padding + size // 2,
        padding * 2 + size + size // 2,
        padding * 4 + size * 2 + size // 2,
    )
    x_1 -= 60
    x_2 -= 90
    x_3 -= 90

    y_1 = 0

    # drawing text size
    draw.text(
        (x_1, y_1),
        "low-resolution input",
        fill=font_color,
        font=font,
        align="left",
    )
    draw.text(
        (x_2, y_1),
        "high-resolution ground truth",
        fill=font_color,
        font=font,
        align="left",
    )
    draw.text(
        (x_3, y_1),
        "high-resolution output",
        fill=font_color,
        font=font,
        align="left",
    )

    return img


def plot_output(x: Tensor, y: Tensor, y_hat: Tensor, **kwargs):
    """Plot x, y and y_hat."""
    figsize = (12, 4)
    if "figsize" in kwargs.keys():
        figsize = kwargs["figsize"]

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=figsize)
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

    fig = plot_output(x, y, y_hat, kwargs)

    if save_image:
        validate_arg_type(fp, str)
        del kwargs["figsize"]
        if ".png" not in fp[-4:]:
            fp += ".png"
        plt.savefig(fp, *args, **kwargs)
    plt.show()
