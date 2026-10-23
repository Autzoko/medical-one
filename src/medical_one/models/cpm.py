"""CPM3D: a whole-volume, anchor-free center-points matching detector (CPM-Net, Song et al. MICCAI 2020; the
TDSC-ABUS 2023 winner was CPM-Net based), trained from boxes only. The complementary detector of the cascade
(docs/prereg/cascade_cpm.md): it sees the whole working volume (336 x 96 x 336) at once, so large lesions are never
cut by a sliding window, and it scores cells instead of matched queries.

Model: ``ResUNet3D`` (stride-4 decoder feature de2, 96 ch) + three conv branches per cell (``CenterHead``):
  cls     lesion-point logit,
  size    log box size in cells (d, h, w),
  offset  box center minus cell center, in cells.

Matching (``CPMLoss.targets``): for every GT box the cells whose centers lie inside the box are candidates; the
``topk`` candidates nearest to the box center (distance normalised per axis by the box size, so elongated lesions
spread their positives along the long axis) are positives; a box with no cell center inside gets its nearest cell.
The other cells inside a GT box are ignored (``ignore_inside``) and every remaining cell is negative.

Loss: sigmoid focal loss (``alpha``, ``gamma``) on all non-ignored cells / number of positives; L1 on log size and
offset and 1 - GIoU of the decoded box at the positives. Hard negatives (boxes, e.g. out-of-fold false positives):
negative cells whose centers lie inside one get (1 + ``w_neg``) x their focal loss.
"""
import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.boxes import box_iou_giou
from .backbone import ResUNet3D
from .center_head import STRIDE, CenterHead, decode_boxes, peaks


class CPM3D(nn.Module):
    def __init__(self, norm: str = "gn", hidden: int = 96, top_k: int = 100):
        super().__init__()
        self.top_k = top_k
        self.backbone = ResUNet3D(norm)
        self.head = CenterHead(96, hidden)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.head(self.backbone(x)[0])
        return {"cls": out["heat"], "size": out["size"], "offset": out["offset"]}

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """-> logits [B, K, 1], corners [B, K, 6] in [0, 1] of the top-K local maxima of the point scores."""
        out = self(x)
        idx, logits = peaks(out["cls"], self.top_k)
        boxes = decode_boxes(out["size"], out["offset"])
        boxes = torch.gather(boxes, 1, idx[..., None].expand(-1, -1, 6)).clamp(0, 1)
        return logits[..., None], boxes


def _cell_centers(grid: Sequence[int], device) -> torch.Tensor:
    """Cell centers in normalised patch coordinates, [D*H*W, 3]."""
    axes = [(torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n for n in grid]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


def _inside(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """points [N, 3], boxes [M, 6] (normalised corners) -> [M, N] bool."""
    if not len(boxes):
        return torch.zeros(0, len(points), dtype=torch.bool, device=points.device)
    return ((points[None] >= boxes[:, None, :3]) & (points[None] < boxes[:, None, 3:])).all(-1)


class CPMLoss(nn.Module):
    def __init__(self, topk: int = 7, alpha: float = 0.75, gamma: float = 2.0, w_cls: float = 1.0,
                 w_size: float = 1.0, w_offset: float = 1.0, w_giou: float = 1.0, ignore_inside: bool = True,
                 w_neg: float = 0.0):
        super().__init__()
        self.topk, self.alpha, self.gamma, self.ignore_inside, self.w_neg = topk, alpha, gamma, ignore_inside, w_neg
        self.w_cls, self.w_size, self.w_offset, self.w_giou = w_cls, w_size, w_offset, w_giou

    @torch.no_grad()
    def targets(self, boxes: Sequence[torch.Tensor], grid: Sequence[int], device,
                negs: Optional[Sequence[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """cls target / weight [B, N] (weight 0 = ignored); positives: batch, cell, log size, offset, box."""
        D, H, W = grid
        n = D * H * W
        pts = _cell_centers(grid, device)
        cells = torch.tensor([D, H, W], device=device, dtype=torch.float32)
        tgt = torch.zeros(len(boxes), n, device=device)
        weight = torch.ones(len(boxes), n, device=device)
        pb, pi, psize, poff, pbox = [], [], [], [], []
        for b, tb in enumerate(boxes):
            tb = tb.to(device).float().reshape(-1, 6)
            inside = _inside(pts, tb)
            if self.ignore_inside and len(tb):
                weight[b, inside.any(0)] = 0.0
            for box, ins in zip(tb, inside):
                c, size = (box[:3] + box[3:]) / 2, (box[3:] - box[:3]).clamp(min=1e-6)
                dist = (((pts - c) / size) ** 2).sum(-1)
                cand = torch.nonzero(ins).flatten()
                if len(cand) == 0:
                    cand = dist.argmin()[None]
                chosen = cand[dist[cand].topk(min(self.topk, len(cand)), largest=False).indices]
                tgt[b, chosen], weight[b, chosen] = 1.0, 1.0
                for i in chosen.tolist():
                    pb.append(b)
                    pi.append(i)
                    psize.append((size * cells).clamp(min=0.5).log())
                    poff.append((c - pts[i]) * cells)
                    pbox.append(box)
            if self.w_neg > 0 and negs is not None and len(negs[b]):
                hard = _inside(pts, negs[b].to(device).float().reshape(-1, 6)).any(0) & (tgt[b] == 0)
                weight[b, hard] = weight[b, hard] * (1 + self.w_neg)
        pos = {"b": torch.tensor(pb, device=device, dtype=torch.long), "i": torch.tensor(pi, device=device, dtype=torch.long)}
        if pb:
            pos.update(size=torch.stack(psize), offset=torch.stack(poff), box=torch.stack(pbox))
        return {"cls": tgt, "weight": weight, "pos": pos}

    def forward(self, out: Dict[str, torch.Tensor], boxes: Sequence[torch.Tensor],
                negs: Optional[Sequence[torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits = out["cls"].flatten(1)
        t = self.targets(boxes, out["cls"].shape[2:], logits.device, negs)
        pos = t["pos"]
        npos = max(len(pos["b"]), 1)
        p = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, t["cls"], reduction="none")
        p_t = p * t["cls"] + (1 - p) * (1 - t["cls"])
        a_t = self.alpha * t["cls"] + (1 - self.alpha) * (1 - t["cls"])
        l_cls = (a_t * (1 - p_t) ** self.gamma * ce * t["weight"]).sum() / npos
        zero = logits.new_zeros(())
        l_size = l_off = l_giou = zero
        if len(pos["b"]):
            b, i = pos["b"], pos["i"]
            size = out["size"].flatten(2)[b, :, i]
            offset = out["offset"].flatten(2)[b, :, i]
            l_size = F.l1_loss(size, pos["size"], reduction="sum") / npos
            l_off = F.l1_loss(offset, pos["offset"], reduction="sum") / npos
            pred = decode_boxes(out["size"], out["offset"])[b, i]
            l_giou = (1 - box_iou_giou(pred, pos["box"])[1].diagonal()).sum() / npos
        total = self.w_cls * l_cls + self.w_size * l_size + self.w_offset * l_off + self.w_giou * l_giou
        return total, {"cls": l_cls.detach(), "size": l_size.detach(), "offset": l_off.detach(),
                       "giou": l_giou.detach(), "npos": torch.tensor(float(len(pos["b"])))}
