"""Prediction files: per-case working-space boxes and scores, shared by every detector we compare."""
import json
from typing import Dict, List, Sequence, Tuple

import torch

from ..data.tdsc import boxes_to_original
from ..utils.boxes import weighted_box_fusion
from .tdsc_official import to_official

Prediction = Tuple[torch.Tensor, torch.Tensor]


def save_predictions(path: str, preds: Sequence[Prediction], metas: Sequence[dict], info: Dict[str, object]) -> None:
    cases = [{"split": m["split"], "case": m["case"], "boxes": [[round(v, 3) for v in b] for b in boxes.tolist()],
              "scores": [round(v, 5) for v in scores.tolist()]} for (boxes, scores), m in zip(preds, metas)]
    with open(path, "w") as f:
        json.dump({"info": info, "cases": cases}, f)


def load_predictions(path: str, metas: Sequence[dict]) -> List[Prediction]:
    """Predictions aligned with ``metas``; a case missing from the file has no detections."""
    with open(path) as f:
        cases = {(c["split"], c["case"]): c for c in json.load(f)["cases"]}
    out = []
    for m in metas:
        c = cases.get((m["split"], m["case"]), {"boxes": [], "scores": []})
        out.append((torch.tensor(c["boxes"], dtype=torch.float).reshape(-1, 6), torch.tensor(c["scores"], dtype=torch.float)))
    return out


def fuse_predictions(sources: Sequence[Sequence[Prediction]], wbf_iou: float = 0.5, max_dets: int = 50) -> List[Prediction]:
    """Per-case WBF across several prediction sets (models / runs), as in ``predict_volume_fused``."""
    fused = []
    for per_case in zip(*sources):
        boxes, scores = weighted_box_fusion(list(per_case), wbf_iou)
        fused.append((boxes[:max_dets], scores[:max_dets]))
    return fused


def to_official_rows(preds: Sequence[Prediction], metas: Sequence[dict]) -> List[Dict[str, float]]:
    """Rows in the challenge submission format (x = raw axis 2, y = axis 1, z = axis 0)."""
    rows = []
    for (boxes, scores), m in zip(preds, metas):
        if not len(boxes):
            continue
        b = to_official(boxes_to_original(boxes.double(), m["shape"]))
        for (z0, y0, x0, z1, y1, x1), s in zip(b.tolist(), scores.tolist()):
            rows.append({"public_id": m["case"], "coordX": (x0 + x1) / 2, "coordY": (y0 + y1) / 2,
                         "coordZ": (z0 + z1) / 2, "x_length": x1 - x0, "y_length": y1 - y0,
                         "z_length": z1 - z0, "probability": s})
    return rows
