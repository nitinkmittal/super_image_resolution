from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .modules import norm2d, activations


class DoubleConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = "identity",
        activation: str = "relu",
    ):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                padding_mode="zeros",
            ),
            norm2d(norm, num_features=out_channels),
            activations[activation],
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                padding_mode="zeros",
            ),
            norm2d(norm, num_features=out_channels),
            activations[activation],
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down(x)


class UNetEncoder(nn.Module):
    def __init__(
        self, in_channels: int, norm: str = "identity", activation: str = "relu"
    ):
        super().__init__()
        self.downs = nn.ModuleList()
        self.downs.append(
            DoubleConv2d(
                in_channels=in_channels,
                out_channels=64,
                norm=norm,
                activation=activation,
            )
        )
        self.downs.append(
            DoubleConv2d(
                in_channels=64, out_channels=128, norm=norm, activation=activation
            )
        )
        self.downs.append(
            DoubleConv2d(
                in_channels=128, out_channels=256, norm=norm, activation=activation
            )
        )
        self.downs.append(
            DoubleConv2d(
                in_channels=256, out_channels=512, norm=norm, activation=activation
            )
        )
        self.depth = len(self.downs)

    def forward(self, x) -> Tuple[Tensor, List[Tensor]]:

        downs_x = []
        for i, layer in enumerate(self.downs):
            x = layer(x)
            if i < self.depth - 1:
                downs_x.append(x)
                x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x, downs_x


class UpSample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = "identity",
        activation: str = "relu",
    ):
        super().__init__()
        self.up = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                DoubleConv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    norm=norm,
                    activation=activation,
                ),
            ]
        )

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.up[0](x1)
        x1 = self.up[1](torch.cat([x1, x2], dim=1))
        return x1


class UNetDecoder(nn.Module):
    def __init__(self, norm="identity", activation="relu"):
        super().__init__()

        self.ups = nn.ModuleList()
        self.ups.append(
            UpSample(
                in_channels=512, out_channels=256, norm=norm, activation=activation
            )
        )
        self.ups.append(
            UpSample(
                in_channels=256, out_channels=128, norm=norm, activation=activation
            )
        )
        self.ups.append(
            UpSample(in_channels=128, out_channels=64, norm=norm, activation=activation)
        )
        self.depth = len(self.ups)

    def forward(self, x1: Tensor, x2: List[Tensor]) -> torch.Tensor:
        for i, layer in enumerate(self.ups):

            x1 = layer(x1, x2[-(i + 1)])
        return x1


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = "identity",
        activation: str = "relu",
        final_act: str = "tanh",
    ):
        super().__init__()

        self.encoder = UNetEncoder(
            in_channels=in_channels, norm=norm, activation=activation
        )
        self.decoder = UNetDecoder(norm=norm, activation=activation)
        self.out = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            activations[final_act],
        )

    def forward(self, x) -> Tensor:

        x1, x2 = self.encoder(x)
        x = self.decoder(x1, x2)
        x = self.out(x)

        return x
