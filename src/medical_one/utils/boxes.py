"""Axis-aligned 3D box utilities. Corner format is (d0, h0, w0, d1, h1, w1), continuous (no +1)."""
from typing import List, Tuple

import torch


def cxcyczwhd_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    return torch.cat([b[..., :3] - b[..., 3:] / 2, b[..., :3] + b[..., 3:] / 2], -1)


def xyxy_to_cxcyczwhd(b: torch.Tensor) -> torch.Tensor:
    return torch.cat([(b[..., :3] + b[..., 3:]) / 2, b[..., 3:] - b[..., :3]], -1)


def box_volume(b: torch.Tensor) -> torch.Tensor:
    return (b[..., 3:] - b[..., :3]).clamp(min=0).prod(-1)


def box_iou_giou(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pairwise IoU and GIoU between corner boxes a [N,6] and b [M,6]."""
    lo = torch.max(a[:, None, :3], b[None, :, :3])
    hi = torch.min(a[:, None, 3:], b[None, :, 3:])
    inter = (hi - lo).clamp(min=0).prod(-1)
    union = box_volume(a)[:, None] + box_volume(b)[None] - inter
    iou = inter / union.clamp(min=1e-7)
    hull = (torch.max(a[:, None, 3:], b[None, :, 3:]) - torch.min(a[:, None, :3], b[None, :, :3])).clamp(min=0).prod(-1)
    return iou, iou - (hull - union) / hull.clamp(min=1e-7)


def nms_3d(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> List[int]:
    """Greedy NMS; suppresses boxes with IoU strictly greater than the threshold."""
    order, keep = scores.argsort(descending=True), []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        iou = box_iou_giou(boxes[i][None], boxes[order[1:]])[0][0]
        order = order[1:][iou <= iou_threshold]
    return keep
