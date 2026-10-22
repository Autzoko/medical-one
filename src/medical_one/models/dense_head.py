"""Dense anchor-based box head (Retina U-Net / nnDetection recipe, re-implemented; boxes only, no segmentation).

What is taken from nnDetection (commit 97a58f3, ``RetinaUNetV001`` + ``conf/train/v001.yaml``), re-implemented here:
  * one shared head on every pyramid level: classifier and regressor towers of ``num_convs`` conv3-GN-ReLU blocks,
    a learned per-level scale on the regression output, classifier bias at the prior 0.01;
  * ATSS matching (``num_candidates`` = 4 per level and anchor shape, IoU threshold = mean + std of the candidates,
    no center-in-box requirement, an anchor matched to several boxes takes the one with the highest IoU);
  * batched hard-negative sampling: per batch ``32 x B`` anchors, at most 1/3 positives, twice as many negatives
    drawn at random from the ``20 x`` highest-scoring negatives (at least one);
  * BCE classification averaged over the sampled anchors, GIoU regression summed over the sampled positives and
    divided by their number.
Anchor shapes come from our own rule (``plan_anchors``): k-means on the log box sizes of the training cases,
``per_level`` shapes per level, small to large; nnDetection's planner is not used.

Boxes are corners (d0, h0, w0, d1, h1, w1) normalised by the patch size, as in ``PatchDataset``.
"""
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.boxes import box_iou_giou

STRIDES = (4, 8, 16)
DELTA_CLAMP = math.log(1000 / 16)


def plan_anchors(sizes_vox: np.ndarray, levels: int = len(STRIDES), per_level: int = 3, iters: int = 50,
                 seed: int = 0) -> np.ndarray:
    """Anchor shapes [levels, per_level, 3] in voxels: k-means (log space) on training box sizes [N, 3], the
    ``levels * per_level`` centroids sorted by volume and dealt to the levels from small to large."""
    x = np.log(np.maximum(np.asarray(sizes_vox, np.float64), 1.0))
    k = levels * per_level
    rng = np.random.RandomState(seed)
    c = x[np.argsort(x.sum(1))[np.linspace(0, len(x) - 1, k).round().astype(int)]]  # spread init by volume
    for _ in range(iters):
        assign = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (assign == j).any():
                c[j] = x[assign == j].mean(0)
            else:
                c[j] = x[rng.randint(len(x))]
    c = c[np.argsort(c.sum(1))]
    return np.exp(c).reshape(levels, per_level, 3).astype(np.float32)


class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def _tower(dim: int, hidden: int, num_convs: int) -> nn.Sequential:
    layers, c = [], dim
    for _ in range(num_convs):
        layers += [nn.Conv3d(c, hidden, 3, padding=1), nn.GroupNorm(hidden // 16, hidden), nn.ReLU(inplace=True)]
        c = hidden
    return nn.Sequential(*layers)


class DenseHead(nn.Module):
    """Per level: classification logits [B, N_l*A] and box deltas [B, N_l*A, 6], concatenated over levels in the
    order level -> cell (d, h, w) -> anchor shape. ``anchor_sizes`` [L, A, 3] (voxels) is a buffer set by
    ``set_anchors`` before training and restored from checkpoints."""

    def __init__(self, dim: int = 192, hidden: int = 128, num_convs: int = 1, levels: int = len(STRIDES),
                 per_level: int = 3, prior_prob: float = 0.01):
        super().__init__()
        self.levels, self.per_level = levels, per_level
        self.cls_tower, self.reg_tower = _tower(dim, hidden, num_convs), _tower(dim, hidden, num_convs)
        self.cls_out = nn.Conv3d(hidden, per_level, 3, padding=1)
        self.reg_out = nn.Conv3d(hidden, per_level * 6, 3, padding=1)
        self.scales = nn.ModuleList([Scale() for _ in range(levels)])
        nn.init.constant_(self.cls_out.bias, -math.log((1 - prior_prob) / prior_prob))
        for conv in (self.cls_out, self.reg_out):
            nn.init.normal_(conv.weight, std=0.01)
        nn.init.zeros_(self.reg_out.bias)
        self.register_buffer("anchor_sizes", torch.zeros(levels, per_level, 3))
        self._cache: Dict[Tuple, Tuple[torch.Tensor, List[int]]] = {}

    def set_anchors(self, sizes: np.ndarray) -> None:
        self.anchor_sizes.copy_(torch.as_tensor(sizes, dtype=torch.float32))
        self._cache.clear()

    def anchors(self, shapes: Sequence[Sequence[int]], device) -> Tuple[torch.Tensor, List[int]]:
        """Anchor corners [M, 6] normalised by the patch (grid x stride of level 0) and anchors per level."""
        assert float(self.anchor_sizes.abs().sum()) > 0, "anchor shapes not set (set_anchors / checkpoint)"
        key = (tuple(map(tuple, shapes)), str(device), tuple(self.anchor_sizes.flatten().tolist()))
        if key not in self._cache:
            patch = [n * STRIDES[0] for n in shapes[0]]
            scale = torch.tensor(patch * 2, device=device, dtype=torch.float32)
            out, counts = [], []
            for lvl, shape in enumerate(shapes):
                s = STRIDES[lvl]
                axes = [(torch.arange(n, device=device, dtype=torch.float32) + 0.5) * s for n in shape]
                ctr = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).view(-1, 1, 3)
                half = self.anchor_sizes[lvl].to(device)[None] / 2                 # [1, A, 3]
                a = torch.cat([ctr - half, ctr + half], -1).view(-1, 6) / scale
                out.append(a)
                counts.append(len(a))
            self._cache = {key: (torch.cat(out), counts)}
        return self._cache[key]

    def forward(self, feats: Sequence[torch.Tensor]) -> Dict[str, object]:
        logits, deltas = [], []
        for lvl, f in enumerate(feats[:self.levels]):
            B = f.shape[0]
            c = self.cls_out(self.cls_tower(f)).float()                            # [B, A, D, H, W]
            r = self.scales[lvl](self.reg_out(self.reg_tower(f)).float())          # [B, A*6, D, H, W]
            logits.append(c.permute(0, 2, 3, 4, 1).reshape(B, -1))
            deltas.append(r.view(B, self.per_level, 6, *r.shape[2:]).permute(0, 3, 4, 5, 1, 2).reshape(B, -1, 6))
        anchors, counts = self.anchors([f.shape[2:] for f in feats[:self.levels]], feats[0].device)
        return {"logits": torch.cat(logits, 1), "deltas": torch.cat(deltas, 1), "anchors": anchors,
                "counts": counts, "per_loc": self.per_level}


def decode(deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """RetinaNet-style deltas (center offset / anchor size, log size ratio) [..., M, 6] -> corners [..., M, 6]."""
    ac, asz = (anchors[:, :3] + anchors[:, 3:]) / 2, (anchors[:, 3:] - anchors[:, :3]).clamp(min=1e-6)
    ctr = ac + deltas[..., :3] * asz
    size = asz * deltas[..., 3:].clamp(max=DELTA_CLAMP).exp()
    return torch.cat([ctr - size / 2, ctr + size / 2], -1)


def _center_dist(boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    cb, ca = (boxes[:, :3] + boxes[:, 3:]) / 2, (anchors[:, :3] + anchors[:, 3:]) / 2
    return torch.cdist(cb, ca)


@torch.no_grad()
def atss_match(boxes: torch.Tensor, anchors: torch.Tensor, counts: Sequence[int], per_loc: int,
               num_candidates: int = 4) -> torch.Tensor:
    """Matched box index per anchor [M] (-1 = background), nnDetection's ``ATSSMatcher`` (center_in_gt=False).
    ``boxes`` [G, 6] and ``anchors`` [M, 6] in voxels (center distances and IoUs are taken there)."""
    M = anchors.shape[0]
    match = torch.full((M,), -1, dtype=torch.long, device=anchors.device)
    if not len(boxes):
        return match
    dist = _center_dist(boxes, anchors)                                           # [G, M]
    cand, start = [], 0
    for n in counts:
        k = min(num_candidates * per_loc, n)
        cand.append(dist[:, start:start + n].topk(k, dim=1, largest=False).indices + start)
        start += n
    cand = torch.cat(cand, 1)                                                     # [G, C]
    iou = box_iou_giou(boxes, anchors)[0]                                         # [G, M]
    ci = iou.gather(1, cand)
    thr = ci.mean(1) + ci.std(1)
    is_pos = ci >= thr[:, None]
    best = torch.full((len(boxes), M), -1.0, device=anchors.device)
    rows = torch.arange(len(boxes), device=anchors.device)[:, None].expand_as(cand)
    best[rows[is_pos], cand[is_pos]] = iou[rows[is_pos], cand[is_pos]]
    val, idx = best.max(0)
    match[val >= 0] = idx[val >= 0]
    return match


class DenseLoss(nn.Module):
    """ATSS + batched hard-negative sampling + BCE / GIoU (nnDetection ``v001`` values by default)."""

    def __init__(self, num_candidates: int = 4, batch_per_image: int = 32, positive_fraction: float = 0.33,
                 pool_size: float = 20, min_neg: int = 1, w_cls: float = 1.0, w_reg: float = 1.0):
        super().__init__()
        self.num_candidates, self.batch_per_image, self.positive_fraction = num_candidates, batch_per_image, \
            positive_fraction
        self.pool_size, self.min_neg, self.w_cls, self.w_reg = pool_size, min_neg, w_cls, w_reg

    @torch.no_grad()
    def assign(self, out: Dict[str, object], targets: Sequence[torch.Tensor], vox_scale: torch.Tensor):
        """Per image: matched box index per anchor [B, M] (-1 = background). Matching runs in voxels."""
        anchors = out["anchors"] * vox_scale
        return torch.stack([atss_match(t.float() * vox_scale, anchors, out["counts"], out["per_loc"],
                                       self.num_candidates) for t in targets])

    @torch.no_grad()
    def sample(self, match: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """nnDetection's ``HardNegativeSamplerBatched`` over the whole batch -> (pos mask, neg mask) [B, M]."""
        lab, prob = match.flatten() >= 0, logits.detach().flatten().sigmoid()
        pos, neg = torch.where(lab)[0], torch.where(~lab)[0]
        size = self.batch_per_image * match.shape[0]
        num_pos = min(pos.numel(), int(size * self.positive_fraction))
        pos = pos[torch.randperm(pos.numel(), device=pos.device)[:num_pos]]
        num_neg = int(max(1, num_pos) * abs(1 - 1 / self.positive_fraction))
        num_neg = min(neg.numel(), max(num_neg, self.min_neg))
        pool = min(neg.numel(), int(num_neg * self.pool_size))
        neg = neg[prob[neg].topk(pool).indices]
        neg = neg[torch.randperm(neg.numel(), device=neg.device)[:num_neg]]
        pm, nm = torch.zeros_like(lab), torch.zeros_like(lab)
        pm[pos], nm[neg] = True, True
        return pm.view_as(match), nm.view_as(match)

    def forward(self, out: Dict[str, object], targets: Sequence[torch.Tensor], vox_scale: torch.Tensor):
        """-> (total loss, parts, match [B, M]); ``vox_scale`` [6] = patch size in voxels (for matching)."""
        match = self.assign(out, targets, vox_scale)
        logits, deltas = out["logits"], out["deltas"]
        pm, nm = self.sample(match, logits)
        sel = pm | nm
        cls = F.binary_cross_entropy_with_logits(logits[sel], pm[sel].float(), reduction="mean") if sel.any() \
            else logits.sum() * 0
        reg = logits.new_zeros(())
        if pm.any():
            b, i = torch.where(pm)
            tgt = torch.stack([targets[bb][match[bb, ii]] for bb, ii in zip(b.tolist(), i.tolist())]).float()
            pred = decode(deltas[b, i], out["anchors"][i])
            reg = (1 - box_iou_giou(pred * vox_scale, tgt * vox_scale)[1].diagonal()).sum() / pm.sum()
        total = self.w_cls * cls + self.w_reg * reg
        return total, {"cls": cls.detach(), "reg": reg.detach(), "pos": pm.sum().float()}, match

