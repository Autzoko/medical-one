"""Box fusion of the deployed pipeline (audit ``deep_fusion_cls.md``, ``final_fusion_boundary_audit.md``).

``wbf`` is the DLL's weighted boxes fusion on normalized 3D corners: per box, swap reversed ends, clamp to
[0, 1], drop zero volume, weight the score; visit by descending score and join the cluster whose current
fused box has the highest IoU strictly above the threshold (continuous volume, no +1); fused corners are
score-weighted means; the cluster score is the mean ('avg') or max of member scores, times
min(M, N) / M where M is the number of sources and N the number of member boxes (not distinct sources).

The pipeline uses it twice: 3D windows (one source, 'max', IoU > 0.05, top 20), then 2D + 3D (two
sources, 'avg', IoU > 0.05). Between them the 3D list keeps original ranks < 5, rank 0 always and the rest
only with score > 0.15, and boosts scores to min(1.5 s, 0.95). Final boxes need score > 0.2; the public
API returns the 20 best.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np

D3_SHAPE = np.array([336, 112, 336] * 2, np.float64)  # 3D boxes are normalized by the pre-crop grid


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    lo = np.maximum(box[:3], boxes[:, :3])
    hi = np.minimum(box[3:], boxes[:, 3:])
    inter = np.clip(hi - lo, 0, None).prod(1)
    vol = lambda b: (b[..., 3:] - b[..., :3]).prod(-1)  # noqa: E731
    union = vol(box) + vol(boxes) - inter
    return np.where(inter > 0, inter / np.where(union > 0, union, 1), 0.0)


def wbf(sources: Sequence[Tuple[np.ndarray, np.ndarray]], iou_thr: float, conf_type: str = "avg",
        skip_thr: float = 0.0, weights: Sequence[float] = ()) -> Tuple[np.ndarray, np.ndarray]:
    """sources: per source (boxes [N, 6] normalized corners, scores [N]). Returns fused (boxes, scores)
    sorted by score."""
    weights = list(weights) or [1.0] * len(sources)
    recs = []
    for (boxes, scores), wgt in zip(sources, weights):
        for b, s in zip(np.asarray(boxes, np.float64).reshape(-1, 6), np.asarray(scores, np.float64)):
            if s < skip_thr:
                continue
            lo, hi = np.minimum(b[:3], b[3:]), np.maximum(b[:3], b[3:])
            b = np.clip(np.concatenate([lo, hi]), 0, 1)
            if (b[3:] - b[:3]).prod() == 0:
                continue
            recs.append((s * wgt, b))
    recs.sort(key=lambda r: -r[0])
    fused: List[np.ndarray] = []
    members: List[List[Tuple[float, np.ndarray]]] = []
    for s, b in recs:
        best, best_iou = -1, iou_thr
        if fused:
            ious = _iou(b, np.stack(fused))
            for k, v in enumerate(ious):
                if v > best_iou:
                    best, best_iou = k, v
        if best < 0:
            fused.append(b.copy())
            members.append([(s, b)])
            continue
        members[best].append((s, b))
        w = np.array([m[0] for m in members[best]])
        fused[best] = (w[:, None] * np.stack([m[1] for m in members[best]])).sum(0) / w.sum()
    m = len(sources)
    scores = []
    for mem in members:
        ss = np.array([x[0] for x in mem])
        c = ss.mean() if conf_type == "avg" else ss.max()
        scores.append(c * min(m, len(mem)) / m)
    if not fused:
        return np.zeros((0, 6)), np.zeros(0)
    scores = np.array(scores)
    order = np.argsort(-scores, kind="stable")
    return np.stack(fused)[order], scores[order]


def window_fusion(boxes: np.ndarray, scores: np.ndarray, top: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """3D cross-window fusion on working-space corners (cropped grid 336 x 96 x 336)."""
    shape = np.array([336, 96, 336] * 2, np.float64)
    fb, fs = wbf([(np.asarray(boxes) / shape, np.asarray(scores))], 0.05, "max")
    return fb[:top] * shape, fs[:top]


def select_3d(boxes: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Top-5 by original rank (rank 0 exempt from score > 0.15), boosted to min(1.5 s, 0.95)."""
    keep = [r for r in range(min(len(scores), 5)) if r == 0 or scores[r] > 0.15]
    s = np.minimum(np.asarray(scores, np.float64)[keep] * 1.5, 0.95).astype(np.float32).astype(np.float64)
    return np.asarray(boxes, np.float64)[keep], s


def fuse_case(d2: Dict[str, object], v_shape: Sequence[int], d3_boxes: np.ndarray, d3_scores: np.ndarray,
              use_2d: bool = True, use_3d: bool = True, final_thr: float = 0.2, top: int = 20,
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Final detections of one case in working-space corners (h normalized by 112, as the 3D grid).

    ``d2`` is the 2D-branch record (``yolo2d.branch_2d``); ``d3_*`` the window-fused 3D list before the
    top-5 selection. Disabling a branch keeps M = 2, as the DLL does when a branch is empty."""
    if not d2.get("gate", True):
        return np.zeros((0, 6)), np.zeros(0)
    vs = np.array(list(v_shape) * 2, np.float64)
    rows = np.asarray(d2["boxes"], np.float64).reshape(-1, 7) if use_2d else np.zeros((0, 7))
    src2 = (rows[:, :6] / vs, rows[:, 6])
    b3, s3 = select_3d(d3_boxes, d3_scores) if use_3d else (np.zeros((0, 6)), np.zeros(0))
    src3 = (b3 / D3_SHAPE, s3)
    fb, fs = wbf([src2, src3], 0.05, "avg")
    keep = fs > final_thr
    return (fb[keep] * D3_SHAPE)[:top], fs[keep][:top]
