"""Lesion-level detection metrics (our protocol; not yet checked against the official TDSC script).

A detection hits a lesion when IoU >= threshold. Per volume, the highest-scoring hit is a TP;
every other detection (including duplicate hits) is an FP. FROC = mean sensitivity at the
average-FP-per-volume points ``FP_POINTS``.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

FP_POINTS = (0.125, 0.25, 0.5, 1, 2, 4, 8)


def froc(per_volume: Sequence[Tuple[torch.Tensor, torch.Tensor]], iou_thr: float,
         fp_points: Sequence[float] = FP_POINTS) -> Tuple[float, List[float]]:
    """per_volume: (scores [N] sorted desc, iou_with_gt [N]); one lesion per volume."""
    events = []
    for scores, ious in per_volume:
        found = False
        for s, iou in zip(scores.tolist(), ious.tolist()):
            tp = iou >= iou_thr and not found
            found |= tp
            events.append((s, tp))
    events.sort(key=lambda e: -e[0])
    n = len(per_volume)
    tp = fp = 0
    curve = []
    for _, is_tp in events:
        tp += is_tp
        fp += not is_tp
        curve.append((fp / n, tp / n))
    sens = [max([se for f, se in curve if f <= x] or [0.0]) for x in fp_points]
    return float(np.mean(sens)), [round(s, 3) for s in sens]


def detection_metrics(per_volume: Sequence[Tuple[torch.Tensor, torch.Tensor]],
                      iou_thrs: Sequence[float] = (0.1, 0.3)) -> Dict[str, object]:
    res: Dict[str, object] = {}
    for thr in iou_thrs:
        res[f"top1_hit@{thr}"] = float(np.mean([len(s) > 0 and float(i[0]) >= thr for s, i in per_volume]))
        res[f"recall_any@{thr}"] = float(np.mean([bool((i >= thr).any()) for _, i in per_volume]))
        res[f"froc@{thr}"], res[f"froc_sens@{thr}"] = froc(per_volume, thr)
    return res
