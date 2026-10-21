"""Expert 1: a dense 3D center-point head (CenterNet / CPM-Net style) trained from boxes only.

It reads the level-0 neck feature (stride 4; 44 x 24 x 44 cells for a 176 x 96 x 176 patch) and predicts per cell
  heat    lesion-center probability (sigmoid; bias initialised to the prior ``prior_prob``),
  size    log box size in cells, per axis (d, h, w),
  offset  sub-cell center offset, per axis, relative to the cell center.

Targets come from the boxes alone: the center cell of each box is 1 and the heat target around it is an
axis-aligned Gaussian whose sigma per axis is ``sigma_ratio`` x the box size in cells (at least ``min_sigma``),
so anisotropic lesions get ellipsoidal targets. The heat loss is the penalty-reduced focal loss of CenterNet
(alpha=2, beta=4); size and offset get L1 at the center cells, and the box decoded there gets 1 - GIoU.

Boxes everywhere are corners (d0, h0, w0, d1, h1, w1) normalised by the patch size, as in ``PatchDataset``; the
patch size is the heat-map grid x ``STRIDE`` (the backbone maps a 176 x 96 x 176 patch to 44 x 24 x 44 cells).
"""
import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.boxes import box_iou_giou

STRIDE = 4


def _branch(dim: int, hidden: int, out: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv3d(dim, hidden, 3, padding=1), nn.GroupNorm(24, hidden), nn.ReLU(inplace=True),
                         nn.Conv3d(hidden, out, 1))


class CenterHead(nn.Module):
    def __init__(self, dim: int = 192, hidden: int = 96, prior_prob: float = 0.01):
        super().__init__()
        self.heat, self.size, self.offset = _branch(dim, hidden, 1), _branch(dim, hidden, 3), _branch(dim, hidden, 3)
        nn.init.constant_(self.heat[-1].bias, -math.log((1 - prior_prob) / prior_prob))
        for branch in (self.size, self.offset):
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """feat [B,C,D,H,W] -> heat logits [B,1,D,H,W], log size [B,3,...], offset [B,3,...] (float32)."""
        return {"heat": self.heat(feat).float(), "size": self.size(feat).float(), "offset": self.offset(feat).float()}


def decode_boxes(size: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Per-cell boxes [B, D*H*W, 6] normalised corners from log size / offset maps [B,3,D,H,W]."""
    B, _, D, H, W = size.shape
    patch = [n * STRIDE for n in (D, H, W)]
    axes = [torch.arange(n, device=size.device, dtype=torch.float32) + 0.5 for n in (D, H, W)]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), 0)[None]                  # [1,3,D,H,W] cell centers
    center = (grid + offset) * STRIDE
    half = size.exp() * STRIDE / 2
    scale = torch.tensor(list(patch) * 2, device=size.device, dtype=torch.float32)
    corners = torch.cat([center - half, center + half], 1).flatten(2).transpose(1, 2)  # [B, N, 6] voxels
    return corners / scale


def peaks(heat_logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-k local maxima (3x3x3) of the heat map: (flat cell indices [B,k], heat logits there [B,k])."""
    flat = heat_logits.flatten(1)
    is_peak = (heat_logits == F.max_pool3d(heat_logits, 3, 1, 1)).flatten(1)
    ranked = flat.masked_fill(~is_peak, float("-inf"))
    k = min(k, flat.shape[1])
    idx = ranked.topk(k, dim=1).indices  # fewer peaks than k: non-peaks fill the tail (they score -inf -> low)
    return idx, torch.gather(flat, 1, idx)


class CenterLoss(nn.Module):
    def __init__(self, w_heat: float = 1.0, w_size: float = 1.0, w_offset: float = 1.0, w_giou: float = 1.0,
                 sigma_ratio: float = 1 / 6, min_sigma: float = 0.5, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.w_heat, self.w_size, self.w_offset, self.w_giou = w_heat, w_size, w_offset, w_giou
        self.sigma_ratio, self.min_sigma, self.alpha, self.beta = sigma_ratio, min_sigma, alpha, beta

    @torch.no_grad()
    def targets(self, boxes: Sequence[torch.Tensor], grid: Sequence[int], device) -> Dict[str, torch.Tensor]:
        """Heat target [B,1,D,H,W]; per positive: (batch, flat cell, log size [3], offset [3], box [6])."""
        D, H, W = grid
        heat = torch.zeros(len(boxes), 1, D, H, W, device=device)
        axes = [torch.arange(n, device=device, dtype=torch.float32) for n in (D, H, W)]
        scale = torch.tensor([n * STRIDE for n in grid] * 2, device=device, dtype=torch.float32)
        gmax = torch.tensor([D - 1, H - 1, W - 1], device=device)
        pos_b, pos_i, pos_size, pos_off, pos_box = [], [], [], [], []
        for b, tb in enumerate(boxes):
            for box in tb.to(device).float():
                vox = box * scale
                cg = (vox[:3] + vox[3:]) / 2 / STRIDE                 # center in cell units
                size = ((vox[3:] - vox[:3]) / STRIDE).clamp(min=0.5)  # size in cells
                cell = cg.floor().long().clamp(min=torch.zeros_like(gmax), max=gmax)
                sigma = (size * self.sigma_ratio).clamp(min=self.min_sigma)
                g = [torch.exp(-((a - c.float()) ** 2) / (2 * s ** 2)) for a, c, s in zip(axes, cell, sigma)]
                gauss = g[0][:, None, None] * g[1][None, :, None] * g[2][None, None, :]
                heat[b, 0] = torch.maximum(heat[b, 0], gauss)
                heat[b, 0, cell[0], cell[1], cell[2]] = 1.0
                pos_b.append(b)
                pos_i.append(int((cell[0] * H + cell[1]) * W + cell[2]))
                pos_size.append(size.log())
                pos_off.append(cg - (cell.float() + 0.5))
                pos_box.append(box)
        if pos_b:
            stack = lambda xs: torch.stack(xs)
            pos = {"b": torch.tensor(pos_b, device=device), "i": torch.tensor(pos_i, device=device),
                   "size": stack(pos_size), "offset": stack(pos_off), "box": stack(pos_box)}
        else:
            pos = {"b": torch.zeros(0, dtype=torch.long, device=device)}
        return {"heat": heat, "pos": pos}

    def forward(self, out: Dict[str, torch.Tensor], boxes: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        heat_logits = out["heat"]
        tgt = self.targets(boxes, heat_logits.shape[2:], heat_logits.device)
        t, pos = tgt["heat"], tgt["pos"]
        n = max(len(pos["b"]), 1)
        p = heat_logits.sigmoid().clamp(1e-4, 1 - 1e-4)
        is_pos = t.eq(1).float()
        pos_loss = -(1 - p) ** self.alpha * p.log() * is_pos
        neg_loss = -(1 - t) ** self.beta * p ** self.alpha * (1 - p).log() * (1 - is_pos)
        l_heat = (pos_loss.sum() + neg_loss.sum()) / n
        zero = heat_logits.new_zeros(())
        l_size = l_off = l_giou = zero
        if len(pos["b"]):
            b, i = pos["b"], pos["i"]
            size = out["size"].flatten(2)[b, :, i]
            offset = out["offset"].flatten(2)[b, :, i]
            l_size = F.l1_loss(size, pos["size"], reduction="sum") / n
            l_off = F.l1_loss(offset, pos["offset"], reduction="sum") / n
            pred = decode_boxes(out["size"], out["offset"])[b, i]
            l_giou = (1 - box_iou_giou(pred, pos["box"])[1].diagonal()).sum() / n
        total = self.w_heat * l_heat + self.w_size * l_size + self.w_offset * l_off + self.w_giou * l_giou
        parts = {"heat": l_heat.detach(), "size": l_size.detach(), "offset": l_off.detach(), "giou": l_giou.detach()}
        return total, parts
