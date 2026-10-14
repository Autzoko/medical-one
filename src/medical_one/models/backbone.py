"""3D residual encoder-decoder backbone and multi-scale neck.

Layer shapes follow the exported ABUS 3D detector (24 Conv nodes): a stride-2 stem, four residual
stages (s1..s4, max-pool between them), two nearest-upsampling stages with skip concatenation
(de3, de2), and 1^3 projections of (de2, de3, s4) to a shared 192-channel width with GroupNorm(24).
BatchNorm was folded into the convolutions at export, so the backbone norm type is configurable.
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(kind: str, channels: int) -> nn.Module:
    if kind == "bn":
        return nn.BatchNorm3d(channels)
    if kind == "gn":
        return nn.GroupNorm(8, channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm: {kind}")


class ConvModule(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, stride: int = 1, act: bool = True, norm: str = "gn"):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, kernel, stride, kernel // 2, bias=norm == "none")
        self.norm = _make_norm(norm, cout)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    """3^3 conv -> ReLU -> 3^3 conv, plus a 1^3 projection shortcut; add, then ReLU."""

    def __init__(self, cin: int, cout: int, norm: str):
        super().__init__()
        self.conv1 = ConvModule(cin, cout, 3, norm=norm)
        self.conv2 = ConvModule(cout, cout, 3, act=False, norm=norm)
        self.conv3 = ConvModule(cin, cout, 1, act=False, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv2(self.conv1(x)) + self.conv3(x), inplace=True)


class UpConcat(nn.Module):
    """1^3 channel reduction, nearest upsampling to the skip resolution, channel concat."""

    def __init__(self, cin: int, cout: int, norm: str):
        super().__init__()
        self.conv1 = ConvModule(cin, cout, 1, norm=norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(self.conv1(x), size=skip.shape[2:], mode="nearest")
        return torch.cat([x, skip], 1)


class ResUNet3D(nn.Module):
    """Returns [de2 (stride 4, 96ch), de3 (stride 8, 192ch), s4 (stride 16, 384ch)]."""

    def __init__(self, norm: str = "gn"):
        super().__init__()
        self.conv1 = ConvModule(1, 24, 3, stride=2, norm=norm)
        self.s1_0 = ResBlock(24, 48, norm)
        self.s2_0 = ResBlock(48, 96, norm)
        self.s3_0 = ResBlock(96, 192, norm)
        self.s4_0 = ResBlock(192, 384, norm)
        self.up3 = UpConcat(384, 192, norm)
        self.de3_1 = ResBlock(384, 192, norm)
        self.up2 = UpConcat(192, 96, norm)
        self.de2_1 = ResBlock(192, 96, norm)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        s1 = self.s1_0(self.conv1(x))
        s2 = self.s2_0(F.max_pool3d(s1, 2))
        s3 = self.s3_0(F.max_pool3d(s2, 2))
        s4 = self.s4_0(F.max_pool3d(s3, 2))
        de3 = self.de3_1(self.up3(s4, s3))
        de2 = self.de2_1(self.up2(de3, s2))
        return [de2, de3, s4]


class _ProjGN(nn.Module):
    def __init__(self, cin: int, dim: int):
        super().__init__()
        self.conv = nn.Conv3d(cin, dim, 1)
        self.gn = nn.GroupNorm(24, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gn(self.conv(x))


class Neck(nn.Module):
    def __init__(self, in_channels=(96, 192, 384), dim: int = 192):
        super().__init__()
        self.out_conv1 = _ProjGN(in_channels[0], dim)
        self.out_conv2 = _ProjGN(in_channels[1], dim)
        self.out_conv3 = _ProjGN(in_channels[2], dim)

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.out_conv1(feats[0]), self.out_conv2(feats[1]), self.out_conv3(feats[2])]
