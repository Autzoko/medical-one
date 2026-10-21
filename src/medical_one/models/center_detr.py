"""Expert 1 + Expert 2 in one network: CNN backbone + neck, a dense center-point head on the stride-4 neck feature
(expert 1, local), and the two-stage deformable transformer (expert 2, global set prediction) whose decoder
queries start from expert 1's top heat-map peaks: query content is the encoder memory at the peak cell and the
reference box is expert 1's box decoded there (instead of the encoder's own top-K proposals).

Training-only query groups (see ``query_groups``): one-to-many queries (H-DETR, ``num_o2m`` > 0), denoising
queries (DINO CDN, ``dn_number`` > 0) and mined hard-negative queries (``num_neg_queries`` > 0). With all of
them off this is exactly the baseline CenterDETR3D.

``forward(x, targets, negs)`` returns a dict: ``o2o`` / ``o2m`` / ``dn`` / ``neg`` (lists of per-layer outputs or
None), ``dn_meta``, ``enc`` (encoder proposals) and ``center`` (expert-1 maps). ``predict(x, head)`` returns
logits [B,K,1] and corners [B,K,6] in [0,1] for ``head="detr"`` (expert 2) or ``head="center"`` (expert 1).
"""
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..utils.boxes import cxcyczwhd_to_xyxy, xyxy_to_cxcyczwhd
from .backbone import Neck, ResUNet3D
from .center_head import CenterHead, decode_boxes, peaks
from .query_groups import attention_mask, denoising_queries, negative_queries
from .transformer import Output, Transformer3D


def _slice(outputs: List[Output], start: int, end: int) -> List[Output]:
    return [{k: v[:, start:end] for k, v in o.items()} for o in outputs]


class CenterDETR3D(nn.Module):
    def __init__(self, norm: str = "gn", dim: int = 192, num_queries: int = 300, drop: float = 0.0,
                 center_hidden: int = 96, num_o2m: int = 0, dn_number: int = 0, dn_noise: float = 1.0,
                 num_neg_queries: int = 0, look_forward_twice: bool = False):
        super().__init__()
        self.num_queries, self.num_o2m = num_queries, num_o2m
        self.dn_number, self.dn_noise, self.num_neg_queries = dn_number, dn_noise, num_neg_queries
        self.backbone = nn.Module()
        self.backbone.backbone = ResUNet3D(norm)
        self.backbone.neck = Neck(dim=dim)
        self.center = CenterHead(dim, center_hidden)
        self.transformer = Transformer3D(dim=dim, num_queries=num_queries, drop=drop,
                                         look_forward_twice=look_forward_twice,
                                         extra_queries=dn_number > 0 or num_neg_queries > 0)

    def _queries(self, center: Dict[str, torch.Tensor], k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expert-1 proposals: top-k peak cells, their reference boxes (center/size in (0,1)) and heat logits."""
        idx, logits = peaks(center["heat"], k)
        boxes = decode_boxes(center["size"], center["offset"])                           # [B, N, 6] corners
        boxes = torch.gather(boxes, 1, idx[..., None].expand(-1, -1, 6)).clamp(0, 1)
        ref = xyxy_to_cxcyczwhd(boxes).clamp(1e-4, 1 - 1e-4)
        return idx, ref, logits

    def forward(self, x: torch.Tensor, targets: Optional[Sequence[torch.Tensor]] = None,
                negs: Optional[Sequence[torch.Tensor]] = None) -> Dict[str, object]:
        feats = self.backbone.neck(self.backbone.backbone(x))
        center = self.center(feats[0])
        train_groups = self.training and targets is not None
        k1, k2 = self.num_queries, (self.num_o2m if train_groups else 0)
        idx, ref, _ = self._queries({k: v.detach() for k, v in center.items()}, k1 + k2)
        k2 = idx.shape[1] - k1                                    # fewer cells than k1 + k2 (tiny inputs)
        dn_ref, dn_meta = denoising_queries(targets, self.dn_number, self.dn_noise, x.device) if train_groups \
            else (x.new_zeros(x.shape[0], 0, 6), {"groups": 0, "single": 0})
        neg_ref = negative_queries(negs, self.num_neg_queries, x.shape[0], x.device) if train_groups \
            else x.new_zeros(x.shape[0], 0, 6)
        extra = torch.cat([dn_ref, neg_ref], 1)
        mask = attention_mask(k1, k2, dn_meta["groups"], dn_meta["single"], neg_ref.shape[1], x.device)
        dec, enc = self.transformer(feats, query_index=idx, query_ref=ref, extra_ref=extra, attn_mask=mask)
        d0, d1 = k1 + k2, k1 + k2 + dn_ref.shape[1]
        return {"o2o": _slice(dec, 0, k1), "o2m": _slice(dec, k1, d0) if k2 else None,
                "dn": _slice(dec, d0, d1) if d1 > d0 else None, "dn_meta": dn_meta,
                "neg": _slice(dec, d1, d1 + neg_ref.shape[1]) if neg_ref.shape[1] else None,
                "enc": enc, "center": center}

    @torch.no_grad()
    def predict(self, x: torch.Tensor, head: str = "detr") -> Tuple[torch.Tensor, torch.Tensor]:
        if head == "detr":
            last = self(x)["o2o"][-1]
            return last["logits"], cxcyczwhd_to_xyxy(last["boxes"])
        assert head == "center", head
        center = self.center(self.backbone.neck(self.backbone.backbone(x))[0])
        _, ref, logits = self._queries(center, self.num_queries)
        return logits[..., None], cxcyczwhd_to_xyxy(ref)
