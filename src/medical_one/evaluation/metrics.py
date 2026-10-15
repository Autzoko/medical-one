"""All detection metrics for a split from working-space predictions."""
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from ..data.tdsc import boxes_to_original
from ..utils.boxes import box_iou_giou
from .froc import detection_metrics
from .tdsc_official import average_precision, official_froc, official_ious

Prediction = Tuple[torch.Tensor, torch.Tensor]  # working-space corner boxes [N,6], scores [N] (desc)


def compute_metrics(preds: Sequence[Prediction], metas: Sequence[dict],
                    iou_thrs: Sequence[float] = (0.1, 0.3)) -> Dict[str, object]:
    """``tdsc_froc`` is the official challenge score (raw-array boxes, IoU > 0.3). ``ap@t`` is COCO-style AP.
    ``froc@t`` / ``top1_hit@t`` / ``recall_any@t`` are the stricter in-house protocol on working-space boxes
    (duplicate hits count as FPs; GT clipped by the depth crop), kept for comparison with earlier runs."""
    work, raw = [], []
    for (boxes, scores), meta in zip(preds, metas):
        s = scores.float().numpy()
        gt = torch.tensor(meta["box"])[None]
        work.append((scores, box_iou_giou(boxes, gt)[0][:, 0] if len(boxes) else torch.zeros(0)))
        raw.append((s, official_ious(boxes_to_original(boxes.double(), meta["shape"]), meta["mask_lo"], meta["mask_hi"])))
    res = detection_metrics(work, iou_thrs)
    res["tdsc_froc"], key, _, _ = official_froc(raw)
    res["tdsc_sens"] = [round(k, 4) for k in key]
    res["tdsc_max_recall"] = float(np.mean([bool((i > 0.3).any()) for _, i in raw]))
    for thr in iou_thrs:
        res[f"ap@{thr}"] = average_precision(raw, thr)
    res["top1_iou_mean"] = float(np.mean([i[np.argmax(s)] if len(s) else 0.0 for s, i in raw]))
    return res


def official_curve(preds: Sequence[Prediction], metas: Sequence[dict]) -> Tuple[List[float], List[float]]:
    raw = [(s.float().numpy(), official_ious(boxes_to_original(b.double(), m["shape"]), m["mask_lo"], m["mask_hi"]))
           for (b, s), m in zip(preds, metas)]
    _, _, fps, recalls = official_froc(raw)
    return fps, recalls
