"""Axis-aligned 3D box utilities. Corner format is (d0, h0, w0, d1, h1, w1), continuous (no +1)."""
from typing import List, Sequence, Tuple

import torch


def cxcyczwhd_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    return torch.cat([b[..., :3] - b[..., 3:] / 2, b[..., :3] + b[..., 3:] / 2], -1)


def xyxy_to_cxcyczwhd(b: torch.Tensor) -> torch.Tensor:
    return torch.cat([(b[..., :3] + b[..., 3:]) / 2, b[..., 3:] - b[..., :3]], -1)


def _prod3(x: torch.Tensor) -> torch.Tensor:
    """x[..., 0] * x[..., 1] * x[..., 2]; avoids the torch.prod CUDA kernel, which fails on V100 with torch 2.14."""
    return x[..., 0] * x[..., 1] * x[..., 2]


def box_volume(b: torch.Tensor) -> torch.Tensor:
    return _prod3((b[..., 3:] - b[..., :3]).clamp(min=0))


def box_iou_giou(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pairwise IoU and GIoU between corner boxes a [N,6] and b [M,6]."""
    lo = torch.max(a[:, None, :3], b[None, :, :3])
    hi = torch.min(a[:, None, 3:], b[None, :, 3:])
    inter = _prod3((hi - lo).clamp(min=0))
    union = box_volume(a)[:, None] + box_volume(b)[None] - inter
    iou = inter / union.clamp(min=1e-7)
    hull = _prod3((torch.max(a[:, None, 3:], b[None, :, 3:]) - torch.min(a[:, None, :3], b[None, :, :3])).clamp(min=0))
    return iou, iou - (hull - union) / hull.clamp(min=1e-7)


def _inclusive_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU of one box a [6] with b [N,6] counting both end voxels (+1 per axis), as the deployed 3D NMS."""
    inter = _prod3((torch.min(a[3:], b[:, 3:]) - torch.max(a[:3], b[:, :3]) + 1).clamp(min=0))
    vol = lambda x: _prod3(x[..., 3:] - x[..., :3] + 1)  # noqa: E731
    return inter / (vol(a) + vol(b) - inter)


def nms_3d(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float, inclusive: bool = False) -> List[int]:
    """Greedy NMS; suppresses boxes with IoU strictly greater than the threshold. ``inclusive`` uses the
    deployed (+1) edge lengths."""
    order, keep = scores.argsort(descending=True), []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        if inclusive:
            iou = _inclusive_iou(boxes[i], boxes[order[1:]])
        else:
            iou = box_iou_giou(boxes[i][None], boxes[order[1:]])[0][0]
        order = order[1:][iou <= iou_threshold]
    return keep


def weighted_box_fusion(sources: Sequence[Tuple[torch.Tensor, torch.Tensor]], iou_threshold: float = 0.5,
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse (boxes [N,6], scores [N]) from several predictors (TTA passes / models).

    Boxes are visited by descending score and joined to the best-overlapping cluster if its fused box has
    IoU > ``iou_threshold``. A cluster's box is the score-weighted mean of its members and its score
    is sum(scores) / len(sources), so a box found by only one predictor is down-weighted.
    """
    boxes = torch.cat([b for b, _ in sources]) if sources else torch.zeros(0, 6)
    scores = torch.cat([s for _, s in sources]) if sources else torch.zeros(0)
    fused: List[torch.Tensor] = []
    members: List[List[int]] = []
    for i in scores.argsort(descending=True).tolist():
        if fused:
            iou = box_iou_giou(boxes[i][None], torch.stack(fused))[0][0]
            j = int(iou.argmax())
            if iou[j] > iou_threshold:
                members[j].append(i)
                w = scores[members[j]]
                fused[j] = (boxes[members[j]] * w[:, None]).sum(0) / w.sum()
                continue
        fused.append(boxes[i].clone())
        members.append([i])
    if not fused:
        return boxes, scores
    out_scores = torch.stack([scores[m].sum() for m in members]) / max(len(sources), 1)
    order = out_scores.argsort(descending=True)
    return torch.stack(fused)[order], out_scores[order]
