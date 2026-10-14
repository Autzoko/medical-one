"""DETR-style set loss: Hungarian matching, sigmoid focal classification, L1 + 3D GIoU boxes.

The original training recipe is not recoverable from the export; this is our choice. The exported
class bias (-4.57) is close to the focal-loss prior init for p=0.01 (-4.595), which is consistent with it.
"""
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ..utils.boxes import box_iou_giou, cxcyczwhd_to_xyxy, xyxy_to_cxcyczwhd


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25,
                       gamma: float = 2.0) -> torch.Tensor:
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1 - p) * (1 - targets)
    return (ce * (1 - pt) ** gamma * (alpha * targets + (1 - alpha) * (1 - targets))).sum()


class SetCriterion(nn.Module):
    def __init__(self, w_cls: float = 2.0, w_l1: float = 5.0, w_giou: float = 2.0,
                 alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.w_cls, self.w_l1, self.w_giou = w_cls, w_l1, w_giou
        self.alpha, self.gamma = alpha, gamma

    @torch.no_grad()
    def match(self, logits: torch.Tensor, boxes: torch.Tensor, target: torch.Tensor):
        """logits [Q,1], boxes [Q,6] (center/size), target [T,6] corners -> (query_idx, target_idx)."""
        p = logits[:, 0].sigmoid()
        pos = self.alpha * (1 - p) ** self.gamma * -(p + 1e-8).log()
        neg = (1 - self.alpha) * p ** self.gamma * -(1 - p + 1e-8).log()
        cost = (self.w_cls * (pos - neg)[:, None]
                + self.w_l1 * torch.cdist(boxes, xyxy_to_cxcyczwhd(target), p=1)
                - self.w_giou * box_iou_giou(cxcyczwhd_to_xyxy(boxes), target)[1])
        q, t = linear_sum_assignment(cost.cpu().numpy())
        return torch.as_tensor(q, device=boxes.device), torch.as_tensor(t, device=boxes.device)

    def forward(self, output: dict, targets: Sequence[torch.Tensor]) -> torch.Tensor:
        logits, boxes = output["logits"], output["boxes"]
        num = max(sum(len(t) for t in targets), 1)
        cls_target = torch.zeros_like(logits)
        l1 = giou = logits.new_zeros(())
        for b, target in enumerate(targets):
            if not len(target):
                continue
            q, t = self.match(logits[b], boxes[b], target)
            cls_target[b, q, 0] = 1
            l1 = l1 + F.l1_loss(boxes[b, q], xyxy_to_cxcyczwhd(target)[t], reduction="sum")
            giou = giou + (1 - box_iou_giou(cxcyczwhd_to_xyxy(boxes[b, q]), target[t])[1].diagonal()).sum()
        cls = sigmoid_focal_loss(logits, cls_target, self.alpha, self.gamma)
        return (self.w_cls * cls + self.w_l1 * l1 + self.w_giou * giou) / num

    def total(self, decoder_outputs: List[dict], encoder_output: dict, targets: Sequence[torch.Tensor]):
        """Sum over every decoder layer (aux losses) plus the encoder proposals; also returns the last layer."""
        layer_losses = [self(o, targets) for o in decoder_outputs]
        return sum(layer_losses) + self(encoder_output, targets), layer_losses[-1]
