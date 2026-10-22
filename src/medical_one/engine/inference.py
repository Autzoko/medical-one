"""Whole-volume sliding-window inference on a 336x96x336 working-space volume.

Follows the deployed per-window logic (sigmoid, top300, score>0.01, NMS IoU>0.05, at most 100,
strict center keep with margin 22) but crops contiguously and reads every candidate row, i.e.
without the non-contiguous-buffer and repeated-first-row defects found in the original DLL.
Cross-window fusion uses NMS instead of the deployed single-source WBF; ``deployed_fusion=True`` instead
follows the deployed post-processing (inclusive-IoU window NMS, single-source WBF with max score, top 20;
see ``pipeline.fusion``). ``predict_volume_fused`` adds flip test-time augmentation and multi-model
ensembling on top, fused with WBF.
"""
import math
from typing import Sequence, Tuple

import numpy as np
import torch

from ..data.dataset import PATCH, normalize
from ..utils.boxes import nms_3d, weighted_box_fusion

WINDOW_STARTS = (0, 80, 160)
CENTER_MARGIN = 22


def window_starts(size: int, patch: int) -> Tuple[int, ...]:
    """Evenly spaced window starts covering ``size`` with at least 50% overlap (336 / 176 -> 0, 80, 160)."""
    if patch >= size:
        return (0,)
    n = math.ceil((size - patch) / (patch * 0.5)) + 1
    return tuple(int(round(i * (size - patch) / (n - 1))) for i in range(n))


def _center_keep(centers: torch.Tensor, sd: int, sw: int, starts_d: Sequence[int] = WINDOW_STARTS,
                 starts_w: Sequence[int] = WINDOW_STARTS, patch: Sequence[int] = PATCH) -> torch.Tensor:
    def bounds(start, starts, size):
        lo = 0 if start == starts[0] else CENTER_MARGIN
        hi = size if start == starts[-1] else size - CENTER_MARGIN
        return lo, hi

    (ld, hd), (lw, hw) = bounds(sd, starts_d, patch[0]), bounds(sw, starts_w, patch[2])
    return (centers[:, 0] > ld) & (centers[:, 0] < hd) & (centers[:, 2] > lw) & (centers[:, 2] < hw)


@torch.no_grad()
def predict_volume(model, volume: np.ndarray, device: str = "cuda", top_k: int = 300, score_thr: float = 0.01,
                   window_nms: float = 0.05, max_per_window: int = 100, fuse_nms: float = 0.1,
                   max_dets: int = 50, deployed_fusion: bool = False, head: str = "detr",
                   patch: Sequence[int] = PATCH) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (boxes [N,6] corners in working-space voxels, scores [N]) sorted by score. ``head`` selects the
    expert of a ``CenterDETR3D`` ("detr" or "center"); other models only have "detr". ``patch`` = the training
    patch (d, h, w); windows cover d and w (``window_starts``), h is taken whole."""
    patch = tuple(patch)
    assert patch[1] == volume.shape[1], "windows span the whole depth axis"
    starts_d, starts_w = window_starts(volume.shape[0], patch[0]), window_starts(volume.shape[2], patch[2])
    x = normalize(torch.from_numpy(volume).float().to(device))
    scale = torch.tensor(patch * 2, device=device, dtype=torch.float)
    all_boxes, all_scores = [], []
    for sd in starts_d:
        for sw in starts_w:
            window = x[sd:sd + patch[0], :, sw:sw + patch[2]].contiguous()[None, None]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                logits, corners = model.predict(window) if head == "detr" else model.predict(window, head=head)
            scores, boxes = logits[0, :, 0].float().sigmoid(), corners[0].float() * scale
            top = scores.topk(min(top_k, len(scores))).indices
            scores, boxes = scores[top], boxes[top]
            keep = scores > score_thr
            scores, boxes = scores[keep], boxes[keep]
            keep = nms_3d(boxes, scores, window_nms, inclusive=deployed_fusion)[:max_per_window]
            scores, boxes = scores[keep], boxes[keep]
            keep = _center_keep((boxes[:, :3] + boxes[:, 3:]) / 2, sd, sw, starts_d, starts_w, patch)
            offset = torch.tensor([sd, 0, sw, sd, 0, sw], device=device, dtype=torch.float)
            all_boxes.append(boxes[keep] + offset)
            all_scores.append(scores[keep])
    boxes, scores = torch.cat(all_boxes), torch.cat(all_scores)
    if deployed_fusion:
        from ..pipeline.fusion import window_fusion

        fb, fs = window_fusion(boxes.double().cpu().numpy(), scores.double().cpu().numpy())
        return torch.from_numpy(fb).float(), torch.from_numpy(fs).float()
    keep = nms_3d(boxes, scores, fuse_nms)[:max_dets]
    return boxes[keep].cpu(), scores[keep].cpu()


FLIP_AXES = ((), (0,), (2,), (0, 2))


def _unflip(boxes: torch.Tensor, axes: Sequence[int], shape: Sequence[int]) -> torch.Tensor:
    boxes = boxes.clone()
    for a in axes:
        boxes[:, [a, a + 3]] = shape[a] - boxes[:, [a + 3, a]]
    return boxes


@torch.no_grad()
def predict_volume_fused(models: Sequence, volume: np.ndarray, device: str = "cuda", tta_flips: bool = True,
                         wbf_iou: float = 0.5, max_dets: int = 50, **post) -> Tuple[torch.Tensor, torch.Tensor]:
    """Runs every model on the volume (and its d / w / dw flips if ``tta_flips``), then fuses with WBF.
    ``post`` is forwarded to ``predict_volume``."""
    sources = []
    for model in models:
        for axes in (FLIP_AXES if tta_flips else ((),)):
            vol = np.ascontiguousarray(np.flip(volume, axes)) if axes else volume
            boxes, scores = predict_volume(model, vol, device, max_dets=max_dets, **post)
            sources.append((_unflip(boxes, axes, volume.shape), scores))
    boxes, scores = weighted_box_fusion(sources, wbf_iou)
    return boxes[:max_dets], scores[:max_dets]
