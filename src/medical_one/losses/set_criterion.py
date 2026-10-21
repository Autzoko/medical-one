"""DETR-style set loss: Hungarian matching, sigmoid focal classification, L1 + 3D GIoU boxes.

The original training recipe is not recoverable from the export; this is our choice. The exported
class bias (-4.57) is close to the focal-loss prior init for p=0.01 (-4.595), which is consistent with it.

Optional hard negatives (``w_neg`` > 0): unmatched predictions whose IoU with a hard-negative box exceeds
``neg_iou`` get (1 + ``w_neg``) x their classification loss, i.e. confident false positives of earlier
models are pushed down harder than ordinary background.

Rank-DETR options (``rankdetr_criterion.py`` / ``high_order_matcher.py`` of LeapLabTHU/Rank-DETR):
``iou_aware``: GIoU-aware classification loss, the positive target is t = (GIoU(pred, gt) + 1) / 2 (detached) and
the loss is BCE(logit, t) |t - p|^gamma for positives, BCE(logit, 0) p^gamma for negatives, alpha-weighted.
``high_order`` (set by the trainer from ``high_order_start`` of training on): matching cost -p * IoU^alpha_ho.

Also: one-to-many targets (``repeat`` in ``forward``, H-DETR) and the losses of denoising / hard-negative query
groups (``dn_loss``, ``background_loss``; DINO CDN: positives regress their GT, the rest are background).
"""
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ..utils.boxes import box_iou_giou, cxcyczwhd_to_xyxy, xyxy_to_cxcyczwhd


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25,
                       gamma: float = 2.0, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1 - p) * (1 - targets)
    loss = ce * (1 - pt) ** gamma * (alpha * targets + (1 - alpha) * (1 - targets))
    return (loss if weight is None else loss * weight).sum()


def iou_aware_focal_loss(logits: torch.Tensor, soft: torch.Tensor, pos: torch.Tensor, alpha: float = 0.25,
                         gamma: float = 2.0, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Rank-DETR GIoU-aware classification loss; ``soft`` = (GIoU + 1) / 2 at positives (``pos`` = 1), else 0."""
    p = logits.sigmoid()
    pos_term = F.binary_cross_entropy_with_logits(logits, soft, reduction="none") * (soft - p * pos).abs() ** gamma
    neg_term = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction="none") * p ** gamma
    loss = alpha * pos * pos_term + (1 - alpha) * (1 - pos) * neg_term
    return (loss if weight is None else loss * weight).sum()


class SetCriterion(nn.Module):
    def __init__(self, w_cls: float = 2.0, w_l1: float = 5.0, w_giou: float = 2.0,
                 alpha: float = 0.25, gamma: float = 2.0, w_neg: float = 0.0, neg_iou: float = 0.3,
                 iou_aware: bool = False, high_order_alpha: float = 4.0):
        super().__init__()
        self.w_cls, self.w_l1, self.w_giou = w_cls, w_l1, w_giou
        self.alpha, self.gamma = alpha, gamma
        self.w_neg, self.neg_iou = w_neg, neg_iou
        self.iou_aware, self.high_order_alpha = iou_aware, high_order_alpha
        self.high_order = False  # switched on by the trainer

    @torch.no_grad()
    def match(self, logits: torch.Tensor, boxes: torch.Tensor, target: torch.Tensor):
        """logits [Q,1], boxes [Q,6] (center/size), target [T,6] corners -> (query_idx, target_idx)."""
        p = logits[:, 0].sigmoid()
        if self.high_order:
            iou = box_iou_giou(cxcyczwhd_to_xyxy(boxes), target)[0]
            q, t = linear_sum_assignment((-(p[:, None] * iou ** self.high_order_alpha)).cpu().numpy())
            return torch.as_tensor(q, device=boxes.device), torch.as_tensor(t, device=boxes.device)
        pos = self.alpha * (1 - p) ** self.gamma * -(p + 1e-8).log()
        neg = (1 - self.alpha) * p ** self.gamma * -(1 - p + 1e-8).log()
        cost = (self.w_cls * (pos - neg)[:, None]
                + self.w_l1 * torch.cdist(boxes, xyxy_to_cxcyczwhd(target), p=1)
                - self.w_giou * box_iou_giou(cxcyczwhd_to_xyxy(boxes), target)[1])
        q, t = linear_sum_assignment(cost.cpu().numpy())
        return torch.as_tensor(q, device=boxes.device), torch.as_tensor(t, device=boxes.device)

    @torch.no_grad()
    def neg_weight(self, boxes: torch.Tensor, negs: Optional[Sequence[torch.Tensor]]) -> Optional[torch.Tensor]:
        """[B, Q, 1] classification weights: 1 + w_neg where a predicted box overlaps a hard negative."""
        if not self.w_neg or negs is None or not any(len(n) for n in negs):
            return None
        weight = boxes.new_ones(boxes.shape[:2] + (1,))
        for b, neg in enumerate(negs):
            if len(neg):
                hit = box_iou_giou(cxcyczwhd_to_xyxy(boxes[b].float()), neg.float())[0].max(1)[0] > self.neg_iou
                weight[b, hit, 0] += self.w_neg
        return weight

    def _cls(self, logits: torch.Tensor, pos: torch.Tensor, soft: torch.Tensor,
             weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.iou_aware:
            return iou_aware_focal_loss(logits, soft, pos, self.alpha, self.gamma, weight)
        return sigmoid_focal_loss(logits, pos, self.alpha, self.gamma, weight)

    def _positives(self, logits, boxes, pairs, targets):
        """pairs: per sample (query idx, target idx) -> (pos mask, soft targets, L1 sum, 1 - GIoU sum)."""
        pos, soft = torch.zeros_like(logits), torch.zeros_like(logits)
        l1 = giou = logits.new_zeros(())
        for b, ((q, t), target) in enumerate(zip(pairs, targets)):
            if not len(q):
                continue
            g = box_iou_giou(cxcyczwhd_to_xyxy(boxes[b, q]), target[t])[1].diagonal()
            pos[b, q, 0] = 1
            soft[b, q, 0] = ((g + 1) / 2).detach().to(soft.dtype)
            l1 = l1 + F.l1_loss(boxes[b, q], xyxy_to_cxcyczwhd(target)[t], reduction="sum")
            giou = giou + (1 - g).sum()
        return pos, soft, l1, giou

    def forward(self, output: dict, targets: Sequence[torch.Tensor],
                negs: Optional[Sequence[torch.Tensor]] = None, repeat: int = 1) -> torch.Tensor:
        """Hungarian matching loss; ``repeat`` > 1 repeats every target (one-to-many, H-DETR)."""
        logits, boxes = output["logits"], output["boxes"]
        targets = [t.repeat(repeat, 1) for t in targets] if repeat > 1 else targets
        num = max(sum(len(t) for t in targets), 1)
        weight = self.neg_weight(boxes, negs)
        empty = torch.zeros(0, dtype=torch.long, device=logits.device)
        pairs = [self.match(logits[b], boxes[b], t) if len(t) else (empty, empty) for b, t in enumerate(targets)]
        pos, soft, l1, giou = self._positives(logits, boxes, pairs, targets)
        if weight is not None:
            weight = torch.where(pos > 0, torch.ones_like(weight), weight)  # a matched query is a positive
        cls = self._cls(logits, pos, soft, weight)
        return (self.w_cls * cls + self.w_l1 * l1 + self.w_giou * giou) / num

    def dn_loss(self, output: dict, targets: Sequence[torch.Tensor], meta: dict) -> torch.Tensor:
        """Denoising queries (layout of ``query_groups.denoising_queries``): positive copy t of group g sits at
        g * 2 * single + t and regresses target t; negative copies and padding are background."""
        logits, boxes = output["logits"], output["boxes"]
        groups, single = meta["groups"], meta["single"]
        num = max(sum(len(t) for t in targets) * groups, 1)
        pairs = []
        for t in targets:
            ti = torch.arange(len(t), device=logits.device)
            q = (torch.arange(groups, device=logits.device)[:, None] * 2 * single + ti[None]).flatten()
            pairs.append((q, ti.repeat(groups)))
        pos, soft, l1, giou = self._positives(logits, boxes, pairs, targets)
        return (self.w_cls * self._cls(logits, pos, soft) + self.w_l1 * l1 + self.w_giou * giou) / num

    def background_loss(self, output: dict, targets: Sequence[torch.Tensor]) -> torch.Tensor:
        """Hard-negative queries: classification towards background only."""
        logits = output["logits"]
        num = max(sum(len(t) for t in targets), 1)
        zero = torch.zeros_like(logits)
        return self.w_cls * self._cls(logits, zero, zero) / num

    def total(self, decoder_outputs: List[dict], encoder_output: dict, targets: Sequence[torch.Tensor],
              negs: Optional[Sequence[torch.Tensor]] = None):
        """Sum over every decoder layer (aux losses) plus the encoder proposals; also returns the last layer."""
        layer_losses = [self(o, targets, negs) for o in decoder_outputs]
        return sum(layer_losses) + self(encoder_output, targets, negs), layer_losses[-1]
