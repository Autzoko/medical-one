"""Dense detectors trained from boxes only.

``DenseDet3D``: CNN backbone + neck, a center heat-map head (stride 4) and a dense anchor-based box head on the three
neck levels (``dense_head``, nnDetection's Retina U-Net recipe without segmentation). Outputs:
  "dense"       the box head's anchors, scored by the classifier;
  "dense_heat"  the same boxes, scored by sqrt(classifier x heat at the box center) (heat as a centerness prior);
  "center"      the heat-map peaks with their decoded boxes.

``CoDETR3D`` (Co-DETR, ICCV 2023, collaborative hybrid assignments): the two-stage deformable DETR on the same
backbone, with the dense box head and the heat-map head as auxiliary one-to-many heads on the *encoder* memory
(reshaped to per-level maps). The aux box head's positive anchors (ATSS) become customised positive queries for a
second decoder pass with fixed targets (``positive_queries``); at inference only the one-to-one decoder ("detr") is
needed, the aux heads stay available ("dense", "dense_heat"). The decoder is trained with the Rank-DETR losses of
``SetCriterion`` (``iou_aware``, high-order matching).

``forward`` returns the raw head outputs; the trainer computes the losses (the positive queries need the aux
head's assignment, so ``CoDETR3D`` is driven through ``encode`` / ``aux`` / ``decode``).
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..utils.boxes import cxcyczwhd_to_xyxy, xyxy_to_cxcyczwhd
from .backbone import Neck, ResUNet3D
from .center_head import STRIDE, CenterHead, decode_boxes, peaks
from .dense_head import DenseHead, decode
from .transformer import Transformer3D


def _logit(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-6, 1 - 1e-6)
    return torch.log(p / (1 - p))


def _dense_predict(dense: Dict[str, object], heat: torch.Tensor, head: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Box-head outputs -> (logits [B, M, 1], corners [B, M, 6] in [0, 1])."""
    boxes = decode(dense["deltas"], dense["anchors"]).clamp(0, 1)
    logits = dense["logits"]
    if head == "dense_heat":
        B, _, D, H, W = heat.shape
        ctr = (boxes[..., :3] + boxes[..., 3:]) / 2                                 # normalised centers
        cell = [(ctr[..., a] * n).long().clamp(0, n - 1) for a, n in enumerate((D, H, W))]
        flat = (cell[0] * H + cell[1]) * W + cell[2]
        h = torch.gather(heat.flatten(1).sigmoid(), 1, flat)
        logits = _logit((logits.sigmoid() * h).sqrt())
    return logits[..., None], boxes


def _center_predict(center: Dict[str, torch.Tensor], k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    idx, logits = peaks(center["heat"], k)
    boxes = decode_boxes(center["size"], center["offset"])
    boxes = torch.gather(boxes, 1, idx[..., None].expand(-1, -1, 6)).clamp(0, 1)
    return logits[..., None], boxes


class DenseDet3D(nn.Module):
    def __init__(self, norm: str = "gn", dim: int = 192, center_hidden: int = 96, dense_hidden: int = 128,
                 dense_convs: int = 1, anchors_per_level: int = 3, center_k: int = 300):
        super().__init__()
        self.center_k = center_k
        self.backbone = nn.Module()
        self.backbone.backbone = ResUNet3D(norm)
        self.backbone.neck = Neck(dim=dim)
        self.center = CenterHead(dim, center_hidden)
        self.dense = DenseHead(dim, dense_hidden, dense_convs, per_level=anchors_per_level)

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        feats = self.backbone.neck(self.backbone.backbone(x))
        return {"dense": self.dense(feats), "center": self.center(feats[0])}

    @torch.no_grad()
    def predict(self, x: torch.Tensor, head: str = "dense") -> Tuple[torch.Tensor, torch.Tensor]:
        out = self(x)
        if head == "center":
            return _center_predict(out["center"], self.center_k)
        assert head in ("dense", "dense_heat"), head
        return _dense_predict(out["dense"], out["center"]["heat"], head)


class CoDETR3D(nn.Module):
    def __init__(self, norm: str = "gn", dim: int = 192, num_queries: int = 300, drop: float = 0.0,
                 center_hidden: int = 96, dense_hidden: int = 128, dense_convs: int = 1, anchors_per_level: int = 3,
                 max_pos_queries: int = 300, min_pos_queries: int = 9):
        super().__init__()
        self.max_pos_queries, self.min_pos_queries = max_pos_queries, min_pos_queries
        self.backbone = nn.Module()
        self.backbone.backbone = ResUNet3D(norm)
        self.backbone.neck = Neck(dim=dim)
        self.transformer = Transformer3D(dim=dim, num_queries=num_queries, drop=drop)
        self.center = CenterHead(dim, center_hidden)
        self.dense = DenseHead(dim, dense_hidden, dense_convs, per_level=anchors_per_level)

    def encode(self, x: torch.Tensor) -> Dict[str, object]:
        return self.transformer.encode(self.backbone.neck(self.backbone.backbone(x)))

    def aux(self, state: Dict[str, object]) -> Dict[str, object]:
        maps = self.transformer.memory_maps(state)
        return {"dense": self.dense(maps), "center": self.center(maps[0])}

    def decode(self, state: Dict[str, object], pos_ref: Optional[torch.Tensor] = None,
               pos_index: Optional[torch.Tensor] = None) -> List[Dict[str, torch.Tensor]]:
        """One-to-one decoder (encoder top-K proposals), or with ``pos_ref`` [B, P, 6] (center/size) and
        ``pos_index`` [B, P] (memory positions) a separate pass over customised positive queries."""
        if pos_ref is None:
            return self.transformer.decode(state)
        return self.transformer.decode(state, query_index=pos_index, query_ref=pos_ref)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, head: str = "detr") -> Tuple[torch.Tensor, torch.Tensor]:
        state = self.encode(x)
        if head == "detr":
            last = self.decode(state)[-1]
            return last["logits"], cxcyczwhd_to_xyxy(last["boxes"])
        out = self.aux(state)
        if head == "center":
            return _center_predict(out["center"], self.transformer.num_queries)
        assert head in ("dense", "dense_heat"), head
        return _dense_predict(out["dense"], out["center"]["heat"], head)


@torch.no_grad()
def positive_queries(dense: Dict[str, object], match: torch.Tensor, targets: Sequence[torch.Tensor],
                     max_pos: int = 300, min_pos: int = 9):
    """Co-DETR customised positive queries from the aux box head's assignment ``match`` [B, M] (-1 = background).

    Per image: its positive anchors (at most ``max_pos``, random subset) with their matched targets, padded to the
    batch maximum (at least ``min_pos``) with random negative anchors that are trained as background (Co-DETR's
    default, ``use_zero_padding=False``). Returns reference boxes [B, P, 6] (center/size), memory positions [B, P]
    of the anchors' cells (flattened multi-level memory), and per image (query idx, target idx) of the positives.
    """
    anchors, counts, per_loc = dense["anchors"], dense["counts"], dense["per_loc"]
    starts = torch.tensor([0] + list(counts[:-1]), device=anchors.device).cumsum(0)
    cell_starts = starts // per_loc                                   # level start in the flattened memory
    level = torch.zeros(len(anchors), dtype=torch.long, device=anchors.device)
    for lvl, s in enumerate(starts.tolist()):
        level[s:] = lvl
    cell = cell_starts[level] + (torch.arange(len(anchors), device=anchors.device) - starts[level]) // per_loc
    pos_lists = []
    for m in match:
        p = torch.where(m >= 0)[0]
        pos_lists.append(p[torch.randperm(len(p), device=p.device)[:max_pos]])
    P = max(min_pos, max(len(p) for p in pos_lists))
    refs, index, pairs = [], [], []
    for m, p in zip(match, pos_lists):
        neg = torch.where(m < 0)[0]
        pad = neg[torch.randperm(len(neg), device=neg.device)[:P - len(p)]]
        sel = torch.cat([p, pad])
        refs.append(xyxy_to_cxcyczwhd(anchors[sel]).clamp(1e-4, 1 - 1e-4))
        index.append(cell[sel])
        pairs.append((torch.arange(len(p), device=p.device), m[p]))
    return torch.stack(refs), torch.stack(index), pairs
