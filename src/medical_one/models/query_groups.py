"""Extra decoder query groups used only in training, and the self-attention mask that keeps them apart.

Layout of the decoder queries (all from one transformer call):
  [ one-to-one (K1) | one-to-many (K2) | denoising (D) | hard-negative (N) ]
- one-to-one: expert-1 peaks 1..K1, Hungarian matching; the only queries used at inference.
- one-to-many (H-DETR hybrid matching): expert-1 peaks K1+1..K1+K2, each ground truth repeated k times.
- denoising (DINO contrastive denoising, ``dn_components.prepare_for_cdn``): ``groups`` groups, each
  [positive copy of every GT | negative copy of every GT], padded to the largest GT count in the batch. Each
  corner moves by sign * U(0,1) * half-size * ``noise`` (positive) or sign * U(1,2) * half-size * ``noise``
  (negative); positives regress their GT, negatives and padding are background.
- hard-negative: confident false positives mined from the model itself (``hardneg``), background targets.
Mask (True = blocked), as in H-DETR / DINO: one-to-one and one-to-many never see each other or the training-only
groups; a denoising group sees the one-to-one queries and itself only; hard-negative queries likewise.
"""
from typing import Dict, Optional, Sequence, Tuple

import torch

from ..utils.boxes import xyxy_to_cxcyczwhd

PAD_BOX = 0.5  # padded slots: sigmoid(0) like DINO's zero padding (center 0.5, size 0.5)


def _to_ref(corners: torch.Tensor) -> torch.Tensor:
    ref = xyxy_to_cxcyczwhd(corners.clamp(0, 1))
    return ref.clamp(1e-4, 1 - 1e-4)


def denoising_queries(targets: Sequence[torch.Tensor], number: int, noise: float,
                      device) -> Tuple[torch.Tensor, Dict[str, int]]:
    """targets: per sample [T,6] normalised corners -> (refs [B, D, 6] center/size, meta). D = 0 without GT.
    ``number`` follows DINO's dn_number: groups = number // (2 * max GT count), at least 1."""
    B = len(targets)
    max_t = max((len(t) for t in targets), default=0)
    if number <= 0 or max_t == 0:
        return torch.zeros(B, 0, 6, device=device), {"groups": 0, "single": 0}
    groups = max(1, number // (2 * max_t))
    refs = torch.full((B, groups, 2, max_t, 6), PAD_BOX, device=device)
    for b, t in enumerate(targets):
        if not len(t):
            continue
        t = t.to(device).float()
        half = ((t[:, 3:] - t[:, :3]) / 2).repeat(1, 2)                         # [T, 6] per-corner scale
        for g in range(groups):
            for k, base in ((0, 0.0), (1, 1.0)):                                  # 0 positive, 1 negative
                sign = torch.randint(0, 2, t.shape, device=device).float() * 2 - 1
                mag = torch.rand(t.shape, device=device) + base
                noisy = t + sign * mag * half * noise
                lo, hi = torch.minimum(noisy[:, :3], noisy[:, 3:]), torch.maximum(noisy[:, :3], noisy[:, 3:])
                refs[b, g, k, :len(t)] = _to_ref(torch.cat([lo, hi], 1))
    return refs.view(B, groups * 2 * max_t, 6), {"groups": groups, "single": max_t}


def negative_queries(negs: Optional[Sequence[torch.Tensor]], number: int, batch: int, device) -> torch.Tensor:
    """Up to ``number`` mined hard-negative boxes per sample as queries [B, N, 6] (padded with PAD_BOX)."""
    if number <= 0 or negs is None or not any(len(n) for n in negs):
        return torch.zeros(batch, 0, 6, device=device)
    n = min(number, max(len(x) for x in negs))
    refs = torch.full((batch, n, 6), PAD_BOX, device=device)
    for b, neg in enumerate(negs):
        if len(neg):
            refs[b, :min(n, len(neg))] = _to_ref(neg[:n].to(device).float())
    return refs


def attention_mask(k1: int, k2: int, dn_groups: int, dn_single: int, n_neg: int, device) -> Optional[torch.Tensor]:
    """[L, L] bool mask over the query layout above (None when there is a single group)."""
    d = dn_groups * 2 * dn_single
    total = k1 + k2 + d + n_neg
    if total == k1:
        return None
    mask = torch.ones(total, total, dtype=torch.bool, device=device)
    mask[:k1, :k1] = False                                       # one-to-one: itself only
    mask[k1:k1 + k2, k1:k1 + k2] = False                         # one-to-many: itself only
    start = k1 + k2
    for g in range(dn_groups):                                   # each denoising group: one-to-one + itself
        s, e = start + g * 2 * dn_single, start + (g + 1) * 2 * dn_single
        mask[s:e, :k1] = False
        mask[s:e, s:e] = False
    s = start + d
    mask[s:, :k1] = False                                        # hard negatives: one-to-one + themselves
    mask[s:, s:] = False
    return mask
