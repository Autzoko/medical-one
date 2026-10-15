"""TDSC-ABUS 2023 detection metric and AP, on boxes in raw-array voxel coordinates.

``official_froc`` re-implements the organizers' ``Final_Evaluation/det_score.py``
(github.com/PerceptionComputingLab/TDSC-ABUS2023, Apache-2.0, derived from the MELA challenge):
a prediction hits when its 3D IoU with a GT box is > 0.3; recall counts distinct GT boxes hit and
every prediction whose IoU is <= 0.3 is a false positive (duplicate hits are not penalized).
Recall and FPs per scan are swept over probability thresholds 0, 0.005, ..., 0.995, recall at
0.125-8 FPs per scan is linearly interpolated, and the score is their mean.

The official CSV stores a box as center and length with ``center = (lo + hi_incl) / 2`` and
``length = hi_incl - lo``, i.e. the interval [lo, hi - 1] of a half-open voxel box [lo, hi);
``to_official`` applies that convention to continuous half-open boxes so IoUs match the script.
"""
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from ..utils.boxes import box_iou_giou

KEY_FP = (0.125, 0.25, 0.5, 1, 2, 4, 8)
THRESHOLDS = np.arange(0, 1, 0.005)
EPS = 1e-8

PerVolume = Sequence[Tuple[np.ndarray, np.ndarray]]  # (scores [N], IoU with the volume's GT box [N])


def to_official(boxes: torch.Tensor) -> torch.Tensor:
    """Half-open raw-array corners -> the closed interval convention of ``bbx_labels.csv``."""
    out = boxes.clone()
    out[..., 3:] = torch.maximum(out[..., 3:] - 1, out[..., :3])
    return out


def official_ious(boxes_raw: torch.Tensor, mask_lo: Sequence[int], mask_hi: Sequence[int]) -> np.ndarray:
    if not len(boxes_raw):
        return np.zeros(0)
    gt = torch.tensor([list(mask_lo) + list(mask_hi)], dtype=torch.float64)
    return box_iou_giou(to_official(boxes_raw.double()), to_official(gt))[0][:, 0].numpy()


def _interpolate(fp_recall: pd.DataFrame, key_fp: float) -> float:
    less, more = fp_recall.loc[fp_recall.fp <= key_fp], fp_recall.loc[fp_recall.fp >= key_fp]
    if len(less) == 0:
        return 0.0
    if len(more) == 0:
        return float(fp_recall.recall.max())
    fp0, fp1 = less["fp"].values[-1], more["fp"].values[0]
    r0, r1 = less["recall"].values[-1], more["recall"].values[0]
    return float(r0 + (r1 - r0) * ((key_fp - fp0) / (fp1 - fp0 + 1e-8)))


def official_froc(per_volume: PerVolume, iou_thr: float = 0.3) -> Tuple[float, List[float], List[float], List[float]]:
    """Returns (mean key recall, key recalls, FP-per-scan curve, recall curve); one GT box per volume."""
    n = len(per_volume)
    fps, recalls = [], []
    for t in THRESHOLDS:
        tp = fp = 0
        for scores, ious in per_volume:
            hit = ious[scores >= t] > iou_thr
            tp += bool(hit.any())
            fp += int((~hit).sum())
        fps.append((fp + EPS) / (n + EPS))
        recalls.append((tp + EPS) / (n + EPS))
    # Same pandas sort as the official script, so ties resolve identically.
    fp_recall = pd.DataFrame({"fp": fps, "recall": recalls}).sort_values("fp")
    key = [_interpolate(fp_recall, k) for k in KEY_FP]
    return float(np.mean(key)), key, fps, recalls


def average_precision(per_volume: PerVolume, iou_thr: float) -> float:
    """COCO-style 101-point AP; per volume the highest-scoring hit is the TP, other boxes are FPs."""
    events = []
    for scores, ious in per_volume:
        found = False
        for s, iou in sorted(zip(scores.tolist(), ious.tolist()), key=lambda e: -e[0]):
            tp = iou >= iou_thr and not found
            found |= tp
            events.append((s, tp))
    if not events:
        return 0.0
    events.sort(key=lambda e: -e[0])
    tp = np.cumsum([e[1] for e in events])
    precision = tp / np.arange(1, len(events) + 1)
    recall = tp / len(per_volume)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    return float(np.mean([envelope[i] if i < len(envelope) else 0.0 for i in idx]))
