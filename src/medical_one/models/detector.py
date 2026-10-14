"""ABUS 3D lesion detector: ResUNet3D backbone + neck + two-stage deformable transformer.

Input [B,1,176,96,176] normalized to [-1,1]. ``predict`` mirrors the exported outputs:
``out_c`` [B,900,1] logits and ``out_b`` [B,900,6] corners (d0,h0,w0,d1,h1,w1) in [0,1].
Module names follow the exported initializer names (``backbone.backbone.*``, ``transformer.*``).
"""
from typing import List, Tuple

import torch
import torch.nn as nn

from ..utils.boxes import cxcyczwhd_to_xyxy
from .backbone import Neck, ResUNet3D
from .transformer import Output, Transformer3D


class ABUSDet3D(nn.Module):
    def __init__(self, norm: str = "gn", dim: int = 192, num_queries: int = 900, drop: float = 0.0):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.backbone = ResUNet3D(norm)
        self.backbone.neck = Neck(dim=dim)
        self.transformer = Transformer3D(dim=dim, num_queries=num_queries, drop=drop)

    def forward(self, x: torch.Tensor) -> Tuple[List[Output], Output]:
        return self.transformer(self.backbone.neck(self.backbone.backbone(x)))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs, _ = self(x)
        return outputs[-1]["logits"], cxcyczwhd_to_xyxy(outputs[-1]["boxes"])
