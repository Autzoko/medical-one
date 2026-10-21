"""Expert 1 + Expert 2 in one network: CNN backbone + neck, a dense center-point head on the stride-4 neck feature
(expert 1, local), and the two-stage deformable transformer (expert 2, global set prediction) whose decoder
queries start from expert 1's top-K heat-map peaks: query content is the encoder memory at the peak cell and the
reference box is expert 1's box decoded there (instead of the encoder's own top-K proposals).

``forward`` returns (decoder outputs, encoder output, center-head output); ``predict(x, head)`` returns logits
[B,K,1] and corners [B,K,6] in [0,1] for ``head="detr"`` (expert 2) or ``head="center"`` (expert 1's peaks).
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from ..utils.boxes import cxcyczwhd_to_xyxy, xyxy_to_cxcyczwhd
from .backbone import Neck, ResUNet3D
from .center_head import CenterHead, decode_boxes, peaks
from .transformer import Output, Transformer3D


class CenterDETR3D(nn.Module):
    def __init__(self, norm: str = "gn", dim: int = 192, num_queries: int = 300, drop: float = 0.0,
                 center_hidden: int = 96):
        super().__init__()
        self.num_queries = num_queries
        self.backbone = nn.Module()
        self.backbone.backbone = ResUNet3D(norm)
        self.backbone.neck = Neck(dim=dim)
        self.center = CenterHead(dim, center_hidden)
        self.transformer = Transformer3D(dim=dim, num_queries=num_queries, drop=drop)

    def _queries(self, center: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expert-1 proposals: peak cells, their reference boxes (center/size in (0,1)) and heat logits."""
        idx, logits = peaks(center["heat"], self.num_queries)
        boxes = decode_boxes(center["size"], center["offset"])                           # [B, N, 6] corners
        boxes = torch.gather(boxes, 1, idx[..., None].expand(-1, -1, 6)).clamp(0, 1)
        ref = xyxy_to_cxcyczwhd(boxes).clamp(1e-4, 1 - 1e-4)
        return idx, ref, logits

    def forward(self, x: torch.Tensor) -> Tuple[List[Output], Output, Dict[str, torch.Tensor]]:
        feats = self.backbone.neck(self.backbone.backbone(x))
        center = self.center(feats[0])
        idx, ref, _ = self._queries({k: v.detach() for k, v in center.items()})
        dec_outputs, enc_output = self.transformer(feats, query_index=idx, query_ref=ref)
        return dec_outputs, enc_output, center

    @torch.no_grad()
    def predict(self, x: torch.Tensor, head: str = "detr") -> Tuple[torch.Tensor, torch.Tensor]:
        if head == "detr":
            outputs, _, _ = self(x)
            return outputs[-1]["logits"], cxcyczwhd_to_xyxy(outputs[-1]["boxes"])
        assert head == "center", head
        center = self.center(self.backbone.neck(self.backbone.backbone(x))[0])
        _, ref, logits = self._queries(center)
        return logits[..., None], cxcyczwhd_to_xyxy(ref)
