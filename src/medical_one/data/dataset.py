"""Random 176x96x176 training patches from preprocessed working-space volumes."""
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .tdsc import load_volume

PATCH = (176, 96, 176)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """uint8 range -> [-1, 1], as in the deployed pipeline."""
    return (x / 255 - 0.5) / 0.5


class PatchDataset(torch.utils.data.Dataset):
    """Samples a case per item; with ``pos_prob`` the patch keeps the lesion center >= ``margin`` from the
    d/w patch borders. A target is emitted only if the lesion center lies inside the patch (the deployed
    inference also keeps detections by center); the box is clipped to the patch and normalized.

    ``scale_range`` > 0 crops a lateral (d, w) window of size PATCH * U(1-r, 1+r) per axis and resizes it
    to PATCH; ``noise_std`` adds Gaussian noise in uint8 units. Both are off by default.

    ``extra_aug`` (probabilities, all 0 by default; see ``DataConfig``): ``rot90`` swaps the lateral axes d and w
    (box-exact; the patch is square in d/w), ``blur`` / ``speckle`` / ``lowres`` / ``inv_gamma`` are intensity
    transforms after nnDetection's ``base_more`` and nnU-Net's low-resolution simulation.

    ``negatives`` maps (split, case) to hard-negative boxes (``hardneg``); then an item also returns the
    negatives whose center lies inside the patch (same transform as the target), and with ``neg_prob`` a
    patch is centred on one of the case's negatives instead of being random (``neg_prob`` <= 1 - ``pos_prob``).
    Without ``negatives`` the random draws are those of the plain sampler."""

    def __init__(self, prep_dir: str, metas: Sequence[dict], length: int, pos_prob: float = 0.7,
                 margin: int = 24, augment: bool = True, scale_range: float = 0.0, noise_std: float = 0.0,
                 negatives: Optional[Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]]] = None,
                 neg_prob: float = 0.0, extra_aug: Optional[Dict[str, float]] = None):
        self.cases = [(load_volume(prep_dir, m), np.asarray(m["box"], np.float32)) for m in metas]
        self.keys = [(m["split"], m["case"]) for m in metas]
        self.length, self.pos_prob, self.margin, self.augment = length, pos_prob, margin, augment
        self.scale_range, self.noise_std = scale_range, noise_std
        assert neg_prob <= 1 - pos_prob + 1e-9, "negative patches are taken from the random share"
        self.set_negatives(negatives)
        self.neg_prob = neg_prob
        self.extra_aug = {k: 0.0 for k in ("rot90", "blur", "speckle", "lowres", "inv_gamma")}
        self.extra_aug.update(extra_aug or {})
        assert PATCH[0] == PATCH[2] or not self.extra_aug["rot90"], "rot90 needs a square lateral patch"

    def set_negatives(self, negatives) -> None:
        """Replace the hard negatives (workers forked afterwards see the new list)."""
        self.negatives = None if negatives is None else [
            np.asarray(negatives.get(k, (np.zeros((0, 6)), None))[0], np.float32).reshape(-1, 6) for k in self.keys]
        self.scale = np.asarray(PATCH * 2, np.float32)

    def __len__(self) -> int:
        return self.length

    def _start(self, center: float, size: int, patch: int, positive: bool) -> int:
        if not positive:
            return random.randint(0, size - patch)
        margin = min(self.margin, patch // 2)
        lo = int(np.clip(center - patch + margin, 0, size - patch))
        hi = int(np.clip(center - margin, 0, size - patch))
        return random.randint(lo, hi)

    def _crop_size(self, axis: int, size: int) -> int:
        if not (self.augment and self.scale_range > 0):
            return PATCH[axis]
        return min(size, int(round(PATCH[axis] * random.uniform(1 - self.scale_range, 1 + self.scale_range))))

    def __getitem__(self, _: int):
        idx = random.randrange(len(self.cases))
        vol, box = self.cases[idx]
        negs = self.negatives[idx] if self.negatives is not None else np.zeros((0, 6), np.float32)
        center = (box[:3] + box[3:]) / 2
        positive = random.random() < self.pos_prob
        if not positive and self.neg_prob > 0 and len(negs) and random.random() < self.neg_prob / (1 - self.pos_prob):
            n = negs[random.randrange(len(negs))]
            center, positive = (n[:3] + n[3:]) / 2, True  # centred like a positive patch, on the negative
        cd, cw = self._crop_size(0, vol.shape[0]), self._crop_size(2, vol.shape[2])
        sd = self._start(center[0], vol.shape[0], cd, positive)
        sw = self._start(center[2], vol.shape[2], cw, positive)
        x = torch.from_numpy(vol[sd:sd + cd, :, sw:sw + cw].astype(np.float32))
        b = np.concatenate([box[None], negs]) - np.asarray([sd, 0, sw, sd, 0, sw], np.float32)
        if (cd, cw) != (PATCH[0], PATCH[2]):
            x = F.interpolate(x[None, None], size=PATCH, mode="trilinear", align_corners=False)[0, 0]
            b = b * np.asarray([PATCH[0] / cd, 1, PATCH[2] / cw] * 2, np.float32)
        c = (b[:, :3] + b[:, 3:]) / 2
        inside = ((c >= 0) & (c < np.asarray(PATCH, np.float32))).all(1)
        b = np.clip(b, 0, self.scale)
        if self.augment:
            x, b = self._augment(x, b)
            x, b = self._extra(x, b)
            if self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
        b = torch.from_numpy(b / self.scale)
        target = b[:1] if inside[0] else torch.zeros(0, 6)
        if self.negatives is None:
            return normalize(x)[None], target
        return normalize(x)[None], target, b[1:][torch.from_numpy(inside[1:])]

    def _extra(self, x: torch.Tensor, b: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """Optional box-exact spatial + intensity transforms on a uint8-range patch [D,H,W]; b: boxes [N,6]."""
        p = self.extra_aug
        if random.random() < p["rot90"]:  # lateral axis swap; with the two flips this spans all 8 symmetries
            x, b = x.permute(2, 1, 0).contiguous(), b[:, [2, 1, 0, 5, 4, 3]].copy()
        if random.random() < p["inv_gamma"]:
            x = 255 - ((255 - x) / 255).clamp(1e-6, 1) ** math.exp(random.uniform(-0.3, 0.3)) * 255
        if random.random() < p["blur"]:
            from scipy.ndimage import gaussian_filter

            x = torch.from_numpy(gaussian_filter(x.numpy(), [random.uniform(0.5, 1.0) for _ in range(3)]))
        if random.random() < p["speckle"]:
            x = x * (1 + torch.randn_like(x) * random.uniform(0.0, 0.1))
        if random.random() < p["lowres"]:
            z = random.uniform(0.5, 1.0)
            small = [max(1, int(round(n * z))) for n in x.shape]
            x = F.interpolate(F.interpolate(x[None, None], size=small, mode="nearest"), size=tuple(x.shape),
                              mode="trilinear", align_corners=False)[0, 0]
        return x, b

    @staticmethod
    def _augment(x: torch.Tensor, b: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """b: boxes [..., 6]."""
        # Flip only the two lateral axes; the depth axis carries acquisition physics.
        if random.random() < 0.5:
            x, b = x.flip(0), b.copy()
            b[..., [0, 3]] = PATCH[0] - b[..., [3, 0]]
        if random.random() < 0.5:
            x, b = x.flip(2), b.copy()
            b[..., [2, 5]] = PATCH[2] - b[..., [5, 2]]
        x = (x / 255).clamp(1e-6, 1) ** math.exp(random.uniform(-0.3, 0.3))
        x = x * random.uniform(0.85, 1.15) + random.uniform(-0.08, 0.08)
        return x * 255, b


def collate(batch: List[Tuple[torch.Tensor, ...]]):
    """-> (images, targets) or, for items with negatives, (images, targets, negatives)."""
    return (torch.stack([b[0] for b in batch]),) + tuple([b[k] for b in batch] for k in range(1, len(batch[0])))
