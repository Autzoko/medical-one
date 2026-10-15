"""Segmentation -> detection: connected components of a lesion probability map become scored boxes."""
from typing import Tuple

import numpy as np
import torch
from scipy import ndimage


def prob_to_boxes(prob: np.ndarray, threshold: float = 0.5, score: str = "max", min_voxels: int = 10,
                  max_dets: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """``prob`` is a foreground probability volume on the working grid. Each 26-connected component of
    ``prob >= threshold`` gives a half-open voxel box; its score is the max or mean probability inside."""
    labels, n = ndimage.label(prob >= threshold, structure=np.ones((3, 3, 3), bool))
    boxes, scores = [], []
    for idx, sl in enumerate(ndimage.find_objects(labels), 1):
        if sl is None:
            continue
        inside = labels[sl] == idx
        if inside.sum() < min_voxels:
            continue
        p = prob[sl][inside]
        boxes.append([s.start for s in sl] + [s.stop for s in sl])
        scores.append(float(p.max() if score == "max" else p.mean()))
    if not boxes:
        return torch.zeros(0, 6), torch.zeros(0)
    b, s = torch.tensor(boxes, dtype=torch.float), torch.tensor(scores, dtype=torch.float)
    order = s.argsort(descending=True)[:max_dets]
    return b[order], s[order]
