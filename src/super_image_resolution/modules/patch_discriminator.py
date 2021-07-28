import torch
from torch import nn
from typing import Union, Tuple
from copy import deepcopy
from .modules import norm2d, activations


def convpatch2d(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]],
    dilation: Union[int, Tuple[int, int]],
    groups: int,
    bias: bool,
    padding_mode: str,
    norm: str,
    activation: str,
):

    out = nn.ModuleList()

    out.append(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
    )
    if norm is not None:
        out.append(norm2d(norm, num_features=out_channels))

    if activation is not None:
        out.append(
            activations[activation],
        )
    return out


class PatchDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        depth: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        norm: str = None,
        activation: str = None,
        final_activation: str = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.norm = norm
        self.activation = activation
        self.final_activation = final_activation
        self.out = nn.ModuleList()

        _in_channels_l = 32

        for i in range(self.depth):

            self.out.append(
                convpatch2d(
                    in_channels=self.in_channels if i == 0 else _in_channels_l,
                    out_channels=_in_channels_l * 2,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                    bias=self.bias,
                    padding_mode=self.padding_mode,
                    norm=self.norm,
                    activation=self.activation,
                )
            )
            _in_channels_l = _in_channels_l * 2

        self.out.append(
            convpatch2d(
                in_channels=_in_channels_l,
                out_channels=1,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias,
                padding_mode=self.padding_mode,
                norm=None,
                activation=self.final_activation,
            )
        )

    def forward(self, x):
        for l1 in self.out:
            for l2 in l1:
                x = l2(x)
        return x
