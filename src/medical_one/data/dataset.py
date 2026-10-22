"""Random training patches (default 176x96x176) from preprocessed working-space volumes.

Working-space axes are (d, h, w) with h the depth axis: h = 0 is the deep side (attenuated, often zero-filled) and
h = 95 the skin side (mean intensity profile over the Train volumes, 2026-10-22), so acoustic shadows run towards
smaller h."""
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

    ``patch`` sets the patch size (d, h, w); ``scale_prob`` > 0 replaces the lateral zoom by an nnU-Net-style zoom
    of all three axes: with that probability the crop is ``patch / (z * u_a)`` per axis, z ~ U(``scale_iso``) and
    u_a ~ U(1 -/+ ``scale_axis``), zero-padded where it exceeds the volume and resized to ``patch`` (box-exact).

    ``paste_prob``: copy a lesion of another training case (its box enlarged by ``paste_context`` per side, flipped
    laterally at random) into the patch at a random lateral position and the donor's own depth, blended with an
    ellipsoid weight that is 1 inside the box's inscribed ellipsoid and falls to 0 at the enlarged box; its box
    becomes a second target. ``small_oversample`` = a > 0: lesion-centred patches pick their case with weight
    (box volume / median)^-a, clipped to [0.5, 2]. ``extra_aug["shadow"]``: a synthetic acoustic shadow (lateral
    Gaussian footprint, intensity x (1 - s) from a random depth towards the deep side, s ~ U(0.25, 0.8)).

    ``negatives`` maps (split, case) to hard-negative boxes (``hardneg``); then an item also returns the
    negatives whose center lies inside the patch (same transform as the target), and with ``neg_prob`` a
    patch is centred on one of the case's negatives instead of being random (``neg_prob`` <= 1 - ``pos_prob``).
    Without ``negatives`` the random draws are those of the plain sampler."""

    def __init__(self, prep_dir: str, metas: Sequence[dict], length: int, pos_prob: float = 0.7,
                 margin: int = 24, augment: bool = True, scale_range: float = 0.0, noise_std: float = 0.0,
                 negatives: Optional[Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]]] = None,
                 neg_prob: float = 0.0, extra_aug: Optional[Dict[str, float]] = None,
                 patch: Sequence[int] = PATCH, scale_prob: float = 0.0, scale_iso: Sequence[float] = (0.7, 1.4),
                 scale_axis: float = 0.0, paste_prob: float = 0.0, paste_context: float = 0.2,
                 small_oversample: float = 0.0):
        self.patch = tuple(int(p) for p in patch)
        self.cases = [(load_volume(prep_dir, m), np.asarray(m["box"], np.float32)) for m in metas]
        self.keys = [(m["split"], m["case"]) for m in metas]
        self.length, self.pos_prob, self.margin, self.augment = length, pos_prob, margin, augment
        self.scale_range, self.noise_std = scale_range, noise_std
        self.scale_prob, self.scale_iso, self.scale_axis = scale_prob, tuple(scale_iso), scale_axis
        self.paste_prob, self.paste_context = paste_prob, paste_context
        vol = np.array([np.prod(b[3:] - b[:3]) for _, b in self.cases])
        self.small_weights = np.clip((vol / np.median(vol)) ** -small_oversample, 0.5, 2.0) if small_oversample > 0 else None
        assert neg_prob <= 1 - pos_prob + 1e-9, "negative patches are taken from the random share"
        self.set_negatives(negatives)
        self.neg_prob = neg_prob
        self.extra_aug = {k: 0.0 for k in ("rot90", "blur", "speckle", "lowres", "inv_gamma", "shadow")}
        self.extra_aug.update(extra_aug or {})
        assert self.patch[0] == self.patch[2] or not self.extra_aug["rot90"], "rot90 needs a square lateral patch"

    def set_negatives(self, negatives) -> None:
        """Replace the hard negatives (workers forked afterwards see the new list)."""
        self.negatives = None if negatives is None else [
            np.asarray(negatives.get(k, (np.zeros((0, 6)), None))[0], np.float32).reshape(-1, 6) for k in self.keys]
        self.scale = np.asarray(self.patch * 2, np.float32)

    def __len__(self) -> int:
        return self.length

    def _start(self, center: float, size: int, patch: int, positive: bool) -> int:
        """Crop start on one axis; a crop larger than the volume starts at a random negative offset (zero pad)."""
        if patch == size:
            return 0
        lo_all, hi_all = min(0, size - patch), max(0, size - patch)
        if not positive:
            return random.randint(lo_all, hi_all)
        margin = min(self.margin, patch // 2)
        lo = int(np.clip(center - patch + margin, lo_all, hi_all))
        hi = int(np.clip(center - margin, lo_all, hi_all))
        return random.randint(lo, hi)

    def _crop_size(self, axis: int, size: int) -> int:
        if not (self.augment and self.scale_range > 0):
            return self.patch[axis]
        return min(size, int(round(self.patch[axis] * random.uniform(1 - self.scale_range, 1 + self.scale_range))))

    def _crop_sizes(self, shape: Sequence[int]) -> Tuple[int, int, int]:
        if self.augment and self.scale_prob > 0:
            if random.random() >= self.scale_prob:
                return self.patch
            z = random.uniform(*self.scale_iso)
            return tuple(max(8, int(round(p / (z * random.uniform(1 - self.scale_axis, 1 + self.scale_axis)))))
                         for p in self.patch)
        return self._crop_size(0, shape[0]), self.patch[1], self._crop_size(2, shape[2])

    @staticmethod
    def _extract(vol: np.ndarray, starts: Sequence[int], crop: Sequence[int]) -> np.ndarray:
        """vol[start:start + crop] per axis as float32, zero where the crop leaves the volume."""
        src = tuple(slice(max(s, 0), min(s + c, n)) for s, c, n in zip(starts, crop, vol.shape))
        if all(s >= 0 and s + c <= n for s, c, n in zip(starts, crop, vol.shape)):
            return vol[src].astype(np.float32)
        out = np.zeros(tuple(crop), np.float32)
        out[tuple(slice(sl.start - s, sl.stop - s) for sl, s in zip(src, starts))] = vol[src]
        return out

    def _pick_case(self) -> Tuple[int, bool]:
        if self.small_weights is None:
            idx = random.randrange(len(self.cases))
            return idx, random.random() < self.pos_prob
        positive = random.random() < self.pos_prob
        if positive:
            return random.choices(range(len(self.cases)), weights=self.small_weights)[0], True
        return random.randrange(len(self.cases)), False

    def _paste(self, x: np.ndarray, starts: Sequence[int], idx: int, avoid: np.ndarray):
        """Pastes another case's lesion into the crop ``x`` (crop coordinates, before resizing); returns the
        pasted box or None (no room, or every try overlapped ``avoid`` = the patch's own lesion box)."""
        j = random.randrange(len(self.cases) - 1)
        dvol, dbox = self.cases[j + (j >= idx)]
        half = (dbox[3:] - dbox[:3]) / 2 * (1 + self.paste_context)
        c = (dbox[:3] + dbox[3:]) / 2
        lo = np.maximum(np.floor(c - half), 0).astype(int)
        hi = np.minimum(np.ceil(c + half), dvol.shape).astype(int)
        size = hi - lo
        if (size > np.asarray(x.shape)).any():
            return None
        region = dvol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
        box = dbox - np.concatenate([lo, lo]).astype(np.float32)
        for a in (0, 2):  # lateral flips of the donor
            if random.random() < 0.5:
                region = np.flip(region, a)
                box[[a, a + 3]] = size[a] - box[[a + 3, a]]
        ph = int(np.clip(lo[1] - starts[1], 0, x.shape[1] - size[1]))  # the donor's own depth
        for _ in range(10):
            pd = random.randint(0, x.shape[0] - size[0])
            pw = random.randint(0, x.shape[2] - size[2])
            new = box + np.asarray([pd, ph, pw] * 2, np.float32)
            if (new[3:] <= avoid[:3] - 4).any() or (new[:3] >= avoid[3:] + 4).any():
                break
        else:
            return None
        bc, bh = (box[:3] + box[3:]) / 2, np.maximum((box[3:] - box[:3]) / 2, 0.5)
        grid = np.ogrid[:size[0], :size[1], :size[2]]
        r = np.sqrt(sum(((g + 0.5 - bc[a]) / bh[a]) ** 2 for a, g in enumerate(grid)))
        alpha = np.clip((1 + self.paste_context - r) / self.paste_context, 0, 1).astype(np.float32)
        dst = x[pd:pd + size[0], ph:ph + size[1], pw:pw + size[2]]
        dst[...] = alpha * region + (1 - alpha) * dst
        return new

    def __getitem__(self, _: int):
        idx, positive = self._pick_case()
        vol, box = self.cases[idx]
        negs = self.negatives[idx] if self.negatives is not None else np.zeros((0, 6), np.float32)
        center = (box[:3] + box[3:]) / 2
        if not positive and self.neg_prob > 0 and len(negs) and random.random() < self.neg_prob / (1 - self.pos_prob):
            n = negs[random.randrange(len(negs))]
            center, positive = (n[:3] + n[3:]) / 2, True  # centred like a positive patch, on the negative
        crop = self._crop_sizes(vol.shape)
        starts = [self._start(center[a], vol.shape[a], crop[a], positive) for a in range(3)]
        x = self._extract(vol, starts, crop)
        b = np.concatenate([box[None], negs]) - np.asarray(starts * 2, np.float32)
        n_pos = 1
        if self.augment and self.paste_prob > 0 and len(self.cases) > 1 and random.random() < self.paste_prob:
            new = self._paste(x, starts, idx, b[0])
            if new is not None:
                b, n_pos = np.concatenate([b[:1], new[None], b[1:]]), 2
        x = torch.from_numpy(x)
        if tuple(crop) != self.patch:
            x = F.interpolate(x[None, None], size=self.patch, mode="trilinear", align_corners=False)[0, 0]
            b = b * np.asarray([p / c for p, c in zip(self.patch, crop)] * 2, np.float32)
        c = (b[:, :3] + b[:, 3:]) / 2
        inside = ((c >= 0) & (c < np.asarray(self.patch, np.float32))).all(1)
        b = np.clip(b, 0, self.scale)
        if self.augment:
            x, b = self._augment(x, b)
            x, b = self._extra(x, b)
            if self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
        b = torch.from_numpy(b / self.scale)
        target = b[:n_pos][torch.from_numpy(inside[:n_pos])]
        if self.negatives is None:
            return normalize(x)[None], target
        return normalize(x)[None], target, b[n_pos:][torch.from_numpy(inside[n_pos:])]

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
        if random.random() < p.get("shadow", 0.0):
            x = self._shadow(x)
        return x, b

    @staticmethod
    def _shadow(x: torch.Tensor) -> torch.Tensor:
        """Acoustic shadow: lateral Gaussian footprint (half-widths 3-25 voxels), from a depth h0 in the upper 70%
        towards the deep side (smaller h, 4-voxel onset ramp), intensity x (1 - s), s ~ U(0.25, 0.8)."""
        D, H, W = x.shape
        cd, cw = random.uniform(0, D), random.uniform(0, W)
        rd, rw = random.uniform(3, 25), random.uniform(3, 25)
        h0, s = random.uniform(0.3 * H, H), random.uniform(0.25, 0.8)
        lat = torch.exp(-((torch.arange(D)[:, None] + 0.5 - cd) / rd) ** 2 - ((torch.arange(W)[None] + 0.5 - cw) / rw) ** 2)
        depth = ((h0 - torch.arange(H) - 0.5) / 4).clamp(0, 1)
        return x * (1 - s * lat[:, None, :] * depth[None, :, None])

    def _augment(self, x: torch.Tensor, b: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """b: boxes [..., 6]."""
        # Flip only the two lateral axes; the depth axis carries acquisition physics.
        if random.random() < 0.5:
            x, b = x.flip(0), b.copy()
            b[..., [0, 3]] = self.patch[0] - b[..., [3, 0]]
        if random.random() < 0.5:
            x, b = x.flip(2), b.copy()
            b[..., [2, 5]] = self.patch[2] - b[..., [5, 2]]
        x = (x / 255).clamp(1e-6, 1) ** math.exp(random.uniform(-0.3, 0.3))
        x = x * random.uniform(0.85, 1.15) + random.uniform(-0.08, 0.08)
        return x * 255, b


def collate(batch: List[Tuple[torch.Tensor, ...]]):
    """-> (images, targets) or, for items with negatives, (images, targets, negatives)."""
    return (torch.stack([b[0] for b in batch]),) + tuple([b[k] for b in batch] for k in range(1, len(batch[0])))
