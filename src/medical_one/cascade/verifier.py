"""Second stage of the cascade (docs/prereg/cascade_cpm.md): a 3D ROI verifier + box refiner, trained from boxes only.

For a candidate box (working-space corners) the ROI is the box enlarged ``enlarge`` x per axis (at least
``min_extent`` voxels), sampled from the *whole* working volume (zero outside) to ``ROI`` voxels, so a lesion
that a sliding window cut is seen whole. The network (small 3D ResNet + box-geometry scalars) predicts
  hit     logit of P(IoU(candidate, GT) > 0.3),
  deltas  (GT center - box center) / box size and log(GT size / box size), per axis (d, h, w),
  malig   benign / malignant logit (multi-task; the TDSC case label, not used for detection scores).
Refinement is iterative (``refine``): the same network is applied to its own refined box ``iters`` times.
Training candidates come from out-of-fold detector predictions (never memorised), jittered GT boxes (scale
0.4-2.5x, shifts) and random boxes; ``sample_weight`` implements the AdaBoost-style emphasis on the first
stage's out-of-fold errors.
"""
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.boxes import box_iou_giou

ROI = (48, 32, 48)
VOL = (336, 96, 336)
N_SCALARS = 7


def roi_grid(boxes: torch.Tensor, enlarge: float = 2.5, min_extent: Sequence[float] = (16, 8, 16),
             roi: Sequence[int] = ROI, vol: Sequence[int] = VOL) -> torch.Tensor:
    """boxes [N, 6] working-space corners -> grid_sample grid [N, rd, rh, rw, 3] (x = w, y = h, z = d)."""
    c = (boxes[:, :3] + boxes[:, 3:]) / 2
    ext = torch.maximum((boxes[:, 3:] - boxes[:, :3]) * enlarge, torch.tensor(min_extent, dtype=boxes.dtype))
    axes = []
    for a, n in enumerate(roi):
        t = (torch.arange(n, dtype=boxes.dtype) + 0.5) / n - 0.5                       # [-0.5, 0.5)
        p = c[:, a:a + 1] + t[None] * ext[:, a:a + 1]                                    # voxel coordinates [N, n]
        axes.append(p / vol[a] * 2 - 1)                                                  # align_corners=False
    gd, gh, gw = axes
    N = len(boxes)
    grid = torch.stack(torch.broadcast_tensors(gw[:, None, None, :], gh[:, None, :, None], gd[:, :, None, None]), -1)
    return grid.reshape(N, roi[0], roi[1], roi[2], 3)


def extract_rois(volume: torch.Tensor, boxes: torch.Tensor, **kw) -> torch.Tensor:
    """volume [D, H, W] float (normalised), boxes [N, 6] -> ROIs [N, 1, *ROI]."""
    grid = roi_grid(boxes.float().cpu(), **kw).to(volume.device)
    v = volume[None, None].expand(len(boxes), -1, -1, -1, -1)
    return F.grid_sample(v, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def box_scalars(boxes: torch.Tensor, vol: Sequence[int] = VOL) -> torch.Tensor:
    """[N, 6] -> [N, 7]: log size (/ 32 voxels) per axis, center per axis as a fraction of the volume, log volume."""
    size = (boxes[:, 3:] - boxes[:, :3]).clamp(min=1.0)
    c = (boxes[:, :3] + boxes[:, 3:]) / 2 / torch.tensor(vol, dtype=boxes.dtype)
    return torch.cat([(size / 32).log(), c, size.prod(1, keepdim=True).log()[:, :1] / 10], 1).float()


def apply_deltas(boxes: torch.Tensor, deltas: torch.Tensor, max_log: float = math.log(3.0)) -> torch.Tensor:
    size = boxes[:, 3:] - boxes[:, :3]
    c = (boxes[:, :3] + boxes[:, 3:]) / 2 + deltas[:, :3].clamp(-1, 1) * size
    new = size * deltas[:, 3:].clamp(-max_log, max_log).exp()
    return torch.cat([c - new / 2, c + new / 2], 1)


def target_deltas(boxes: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    size = (boxes[:, 3:] - boxes[:, :3]).clamp(min=1e-3)
    gsize = (gt[3:] - gt[:3]).clamp(min=1e-3)
    return torch.cat([((gt[:3] + gt[3:]) / 2 - (boxes[:, :3] + boxes[:, 3:]) / 2) / size, (gsize / size).log()], 1)


class _Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = nn.GroupNorm(8, cout)
        self.skip = nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), nn.GroupNorm(8, cout)) \
            if stride != 1 or cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.n1(self.c1(x)))
        return F.relu(self.n2(self.c2(y)) + self.skip(x))


class Verifier3D(nn.Module):
    """3D ResNet (4 stages, 2 blocks each, widths ``widths``) + box scalars -> hit, deltas, malignancy."""

    def __init__(self, widths: Sequence[int] = (16, 32, 64, 128), drop: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, widths[0], 3, 1, 1, bias=False), nn.GroupNorm(8, widths[0]), nn.ReLU())
        layers, cin = [], widths[0]
        for i, w in enumerate(widths):
            layers += [_Block(cin, w, 1 if i == 0 else 2), _Block(w, w, 1)]
            cin = w
        self.body = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Linear(cin + N_SCALARS, 128), nn.ReLU(), nn.Dropout(drop))
        self.hit, self.deltas, self.malig = nn.Linear(128, 1), nn.Linear(128, 6), nn.Linear(128, 1)
        nn.init.zeros_(self.deltas.weight)
        nn.init.zeros_(self.deltas.bias)

    def forward(self, rois: torch.Tensor, scalars: torch.Tensor) -> Dict[str, torch.Tensor]:
        f = self.body(self.stem(rois)).mean((2, 3, 4))
        h = self.fc(torch.cat([f, scalars], 1))
        return {"hit": self.hit(h)[:, 0], "deltas": self.deltas(h), "malig": self.malig(h)[:, 0]}


@torch.no_grad()
def refine(model: Verifier3D, volume: torch.Tensor, boxes: torch.Tensor, iters: int = 2, batch: int = 64,
           enlarge: float = 2.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Iterative refinement; returns (refined boxes [N, 6], hit probability [N] at the refined box,
    malignancy probability [N])."""
    model.eval()
    dev = next(model.parameters()).device
    boxes = boxes.float().cpu()
    hit = malig = torch.zeros(len(boxes))
    for it in range(iters + 1):
        outs = []
        for s in range(0, len(boxes), batch):
            b = boxes[s:s + batch]
            r = extract_rois(volume.to(dev), b, enlarge=enlarge)
            outs.append({k: v.float().cpu() for k, v in model(r, box_scalars(b).to(dev)).items()})
        o = {k: torch.cat([x[k] for x in outs]) if outs else torch.zeros(0) for k in ("hit", "deltas", "malig")}
        if it < iters and len(boxes):
            boxes = apply_deltas(boxes, o["deltas"].reshape(-1, 6))
            boxes[:, 3:] = torch.maximum(boxes[:, 3:], boxes[:, :3] + 1.0)
        else:
            hit, malig = o["hit"].sigmoid(), o["malig"].sigmoid()
    return boxes, hit, malig


def jitter_boxes(gt: torch.Tensor, n: int, rng: np.random.RandomState, scale=(0.4, 2.5), axis=0.2,
                 shift: float = 0.3) -> torch.Tensor:
    """n boxes around the GT box: joint log-uniform scale, per-axis +-``axis`` log scale, shift +-shift x size."""
    gt = gt.double()
    c, size = (gt[:3] + gt[3:]) / 2, gt[3:] - gt[:3]
    s = np.exp(rng.uniform(np.log(scale[0]), np.log(scale[1]), (n, 1)) + rng.uniform(-axis, axis, (n, 3)))
    new = torch.from_numpy(s) * size
    cc = c + torch.from_numpy(rng.uniform(-shift, shift, (n, 3))) * size
    return torch.cat([cc - new / 2, cc + new / 2], 1).float()


def random_boxes(gt: torch.Tensor, n: int, rng: np.random.RandomState, vol: Sequence[int] = VOL) -> torch.Tensor:
    """n boxes of GT-like size (x U(0.5, 2) per axis) anywhere in the volume."""
    size = (gt[3:] - gt[:3]).double().numpy() * rng.uniform(0.5, 2.0, (n, 3))
    size = np.minimum(size, np.array(vol) - 1)
    lo = rng.uniform(0, 1, (n, 3)) * (np.array(vol) - size)
    return torch.from_numpy(np.concatenate([lo, lo + size], 1)).float()


def labels_for(boxes: torch.Tensor, gt: torch.Tensor, thr: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
    """(hit [N] float, IoU [N]) of boxes against the GT box (working space)."""
    if not len(boxes):
        return torch.zeros(0), torch.zeros(0)
    iou = box_iou_giou(boxes.float(), gt.float()[None])[0][:, 0]
    return (iou > thr).float(), iou


def verifier_loss(out: Dict[str, torch.Tensor], hit: torch.Tensor, iou: torch.Tensor, deltas: torch.Tensor,
                  malig: torch.Tensor, weight: torch.Tensor, w_box: float = 1.0, w_malig: float = 0.2):
    """Weighted BCE on hit; smooth L1 on deltas for boxes that overlap the GT; BCE on malignancy for hits."""
    l_hit = (F.binary_cross_entropy_with_logits(out["hit"], hit, reduction="none") * weight).sum() / weight.sum()
    touch = iou > 0
    l_box = F.smooth_l1_loss(out["deltas"][touch], deltas[touch], beta=0.1) if touch.any() else out["deltas"].sum() * 0
    pos = hit > 0
    l_mal = F.binary_cross_entropy_with_logits(out["malig"][pos], malig[pos]) if pos.any() else out["malig"].sum() * 0
    return l_hit + w_box * l_box + w_malig * l_mal, {"hit": l_hit.item(), "box": l_box.item(), "malig": l_mal.item()}
