"""2D branch of the deployed pipeline: 2.5D YOLOv5m per slice, then cross-slice aggregation.

Follows the audited DLL (``deep_2d.md``, ``data_association_deep.md``): centers u = 0, 4, 8, ...; batches
of 8 letterboxed images scaled to [0, 1]; YOLOv5 NMS (objectness > 0.05, conf = obj * cls > 0.05, at most
30000 candidates, continuous-area IoU > 0.4 suppressed, at most 100 per image); boxes mapped back to the
slice and rounded; each box copied to slices u..u+3 (no clamp); seed-centred cross-slice aggregation with
the inclusive (+1) 2D IoU > 0.1; rows (u0, v0, w0, u1, v1, w1, score) with score > 0.3 are kept.
The DLL's 10 s NMS time limit is not reproduced.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torchvision

from .native import letterbox, three_slice, unletterbox

CENTER_STEP = 4
BATCH = 8
CONF_THR = 0.05
NMS_IOU = 0.4
MAX_DET = 100
MAX_NMS = 30000
MAX_WH = 4096
AGG_IOU = 0.1
KEEP_SCORE = 0.3


def yolo_nms(pred: torch.Tensor, conf_thr: float = CONF_THR, iou_thr: float = NMS_IOU,
             max_det: int = MAX_DET) -> List[torch.Tensor]:
    """pred [B, N, 5 + nc] (xywh, obj, cls...) -> per image [K, 6] (x1, y1, x2, y2, conf, cls)."""
    out = []
    for x in pred:
        x = x[x[:, 4] > conf_thr].clone()
        if not len(x):
            out.append(torch.zeros(0, 6, device=pred.device))
            continue
        x[:, 5:] *= x[:, 4:5]
        box = torch.cat([x[:, :2] - x[:, 2:4] / 2, x[:, :2] + x[:, 2:4] / 2], 1)
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat([box, conf, j.float()], 1)[conf.view(-1) > conf_thr]
        if len(x) > MAX_NMS:
            x = x[x[:, 4].argsort(descending=True)[:MAX_NMS]]
        keep = torchvision.ops.nms(x[:, :4] + x[:, 5:6] * MAX_WH, x[:, 4], iou_thr)[:max_det]
        out.append(x[keep])
    return out


@torch.no_grad()
def detect_slices(model, v: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Per-slice detections expanded to 4 slices: rows (x1, y1, x2, y2, score, slice), x = w, y = v."""
    centers = list(range(0, v.shape[0], CENTER_STEP))
    rows = []
    for b in range(0, len(centers), BATCH):
        batch = centers[b:b + BATCH]
        imgs = []
        for i in batch:
            img, _, _ = letterbox(three_slice(v, i))
            imgs.append(torch.from_numpy(img).permute(2, 0, 1))
        x = torch.stack(imgs).to(device).float() / 255
        pred = model(x)
        pred = pred[0] if isinstance(pred, (list, tuple)) else pred
        for i, det in zip(batch, yolo_nms(pred.float())):
            if not len(det):
                continue
            boxes = unletterbox(det[:, :4].cpu().numpy(), x.shape[2:], v.shape[1:])
            scores = det[:, 4].cpu().numpy()
            for box, s in zip(boxes, scores):
                for t in range(4):
                    rows.append([*box, s, i + t])
    return np.asarray(rows, np.float64).reshape(-1, 6)


def _inclusive_iou(seed: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    iw = np.clip(np.minimum(seed[2], boxes[:, 2]) - np.maximum(seed[0], boxes[:, 0]) + 1, 0, None)
    ih = np.clip(np.minimum(seed[3], boxes[:, 3]) - np.maximum(seed[1], boxes[:, 1]) + 1, 0, None)
    inter = iw * ih
    area = lambda b: (b[..., 2] - b[..., 0] + 1) * (b[..., 3] - b[..., 1] + 1)  # noqa: E731
    return inter / (area(seed) + area(boxes) - inter)


def aggregate_slices(rows: np.ndarray, iou_thr: float = AGG_IOU) -> List[Tuple[int, float, float]]:
    """Seed-centred cross-slice grouping (DLL 0x180041840). Returns (seed row, u_min - 1, u_max + 1)."""
    order = np.argsort(-rows[:, 4], kind="stable")
    groups = []
    while len(order):
        seed = order[0]
        positions = np.nonzero(_inclusive_iou(rows[seed, :4], rows[order, :4]) > iou_thr)[0]
        slices = rows[order[positions], 5]
        s0, lo, hi = rows[seed, 5], slices.min(), slices.max()
        present = set(slices.astype(int).tolist())
        missing_up = [s for s in range(int(s0), int(hi)) if s not in present]
        missing_down = [s for s in range(int(lo), int(s0)) if s not in present]
        upper = min(missing_up) if missing_up else hi
        lower = max(missing_down) if missing_down else lo
        take = positions[(slices >= lower) & (slices <= upper)]
        chosen = rows[order[take], 5]
        groups.append((int(seed), float(chosen.min() - 1), float(chosen.max() + 1)))
        order = np.delete(order, take)
    return groups


def branch_2d(model, v: np.ndarray, device: str = "cuda") -> Dict[str, object]:
    """Runs the whole 2D branch on V. ``gate`` is False when the DLL would return no result at all
    (no per-slice detection, so the 3D branch never runs)."""
    rows = detect_slices(model, v, device)
    if not len(rows):
        return {"gate": False, "n_slice_dets": 0, "boxes": []}
    boxes = []
    for seed, u0, u1 in aggregate_slices(rows):
        x1, y1, x2, y2, s, _ = rows[seed]
        boxes.append([u0, y1, x1, u1, y2, x2, s])
    kept = [b for b in boxes if b[6] > KEEP_SCORE]
    return {"gate": True, "n_slice_dets": int(len(rows) // 4), "n_groups": len(boxes), "boxes": kept,
            "boxes_all": boxes}


def v_boxes_to_working(boxes: Sequence[Sequence[float]], v_shape: Sequence[int]) -> np.ndarray:
    """V-space corners -> working space (V resized to 336 x 112 x 336), for scoring with ``boxes_to_original``."""
    b = np.asarray(boxes, np.float64).reshape(-1, 6)
    scale = np.array([336 / v_shape[0], 112 / v_shape[1], 336 / v_shape[2]] * 2)
    return b * scale
