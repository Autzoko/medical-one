"""Hard negatives: confident false positives of a detector, mined out of fold, used as extra training signal.

A negative of a case is a predicted box that does not touch the ground-truth box enlarged by ``grow`` of its
size on every side (so near-misses and duplicates of the lesion are never negatives), with score
>= ``min_score``; per case the ``top`` best survive a greedy 3D NMS. Boxes are working-space corners
(d0, h0, w0, d1, h1, w1) on the 336 x 112 x 336 grid; 2D-branch boxes live in V space (u, v, w of the
native volume) and differ from it only by a per-axis scale.

Negatives for training a fold must come from predictions of models that never trained on that case
(out-of-fold predictions), otherwise the miner sees the model's memorized training set.
"""
import json
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .data.tdsc import KEEP_DEPTH, RESIZE
from .utils.boxes import box_iou_giou, nms_3d

WORK = np.array(list(RESIZE) * 2, np.float64)
Key = Tuple[str, int]
Negatives = Dict[Key, Tuple[np.ndarray, np.ndarray]]  # (split, case) -> (boxes [N, 6], scores [N])


def v_to_working(boxes: np.ndarray, v_shape: Sequence[int]) -> np.ndarray:
    return np.asarray(boxes, np.float64).reshape(-1, 6) / np.array(list(v_shape) * 2) * WORK


def working_to_v(boxes: np.ndarray, v_shape: Sequence[int]) -> np.ndarray:
    return np.asarray(boxes, np.float64).reshape(-1, 6) / WORK * np.array(list(v_shape) * 2)


def far_from_gt(boxes: np.ndarray, gt: Sequence[float], grow: float = 0.25) -> np.ndarray:
    """True where a box has no overlap with the GT box enlarged by ``grow`` x its size per side."""
    gt = np.asarray(gt, np.float64)
    pad = (gt[3:] - gt[:3]) * grow
    lo, hi = gt[:3] - pad, gt[3:] + pad
    b = np.asarray(boxes, np.float64).reshape(-1, 6)
    overlap = (np.minimum(b[:, 3:], hi) > np.maximum(b[:, :3], lo)).all(1)
    return ~overlap


def mine(preds: Sequence[Tuple[torch.Tensor, torch.Tensor]], metas: Sequence[dict], min_score: float = 0.3,
         top: int = 5, nms_iou: float = 0.3, grow: float = 0.25) -> Negatives:
    out: Negatives = {}
    for (boxes, scores), m in zip(preds, metas):
        b, s = np.asarray(boxes, np.float64).reshape(-1, 6), np.asarray(scores, np.float64)
        b[:, [1, 4]] = np.clip(b[:, [1, 4]], 0, KEEP_DEPTH)  # the working volume keeps the first 96 depth rows
        keep = (s >= min_score) & far_from_gt(b, m["box"], grow) & ((b[:, 3:] - b[:, :3]) > 0).all(1)
        b, s = b[keep], s[keep]
        if len(s):
            order = nms_3d(torch.from_numpy(b), torch.from_numpy(s), nms_iou)[:top]
            b, s = b[order], s[order]
        out[(m["split"], m["case"])] = (b, s)
    return out


def union(*sources: Negatives, top: int = 10, nms_iou: float = 0.3) -> Negatives:
    """Negatives of several miners merged per case (greedy NMS on score, at most ``top``)."""
    out: Negatives = {}
    for key in sorted(set().union(*sources)):
        parts = [src[key] for src in sources if key in src]
        b = np.concatenate([p[0] for p in parts]).reshape(-1, 6)
        s = np.concatenate([p[1] for p in parts])
        if len(s):
            order = nms_3d(torch.from_numpy(b), torch.from_numpy(s), nms_iou)[:top]
            b, s = b[order], s[order]
        out[key] = (b, s)
    return out


def iou_with(boxes: torch.Tensor, negs: torch.Tensor) -> torch.Tensor:
    """[Q, 6] x [N, 6] corners -> IoU [Q, N]."""
    return box_iou_giou(boxes, negs)[0]


def save_negatives(path: str, negs: Negatives, info: Dict[str, object]) -> None:
    cases = [{"split": k[0], "case": k[1], "boxes": np.round(b, 3).tolist(), "scores": np.round(s, 5).tolist()}
             for k, (b, s) in sorted(negs.items())]
    with open(path, "w") as f:
        json.dump({"info": info, "cases": cases}, f)


def load_negatives(path: str) -> Negatives:
    with open(path) as f:
        cases = json.load(f)["cases"]
    return {(c["split"], c["case"]): (np.asarray(c["boxes"], np.float64).reshape(-1, 6),
                                      np.asarray(c["scores"], np.float64)) for c in cases}


def summary(negs: Negatives) -> List[str]:
    n = np.array([len(s) for _, s in negs.values()])
    return [f"{len(negs)} cases, {n.sum()} negatives, {(n > 0).mean():.0%} of cases have one, "
            f"max {n.max() if len(n) else 0} per case"]
