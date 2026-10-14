"""Random 176x96x176 training patches from preprocessed working-space volumes."""
import math
import random
from typing import List, Sequence, Tuple

import numpy as np
import torch

from .tdsc import load_volume

PATCH = (176, 96, 176)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """uint8 range -> [-1, 1], as in the deployed pipeline."""
    return (x / 255 - 0.5) / 0.5


class PatchDataset(torch.utils.data.Dataset):
    """Samples a case per item; with ``pos_prob`` the patch keeps the lesion center >= ``margin`` from the
    d/w patch borders. A target is emitted only if the lesion center lies inside the patch (the deployed
    inference also keeps detections by center); the box is clipped to the patch and normalized."""

    def __init__(self, prep_dir: str, metas: Sequence[dict], length: int, pos_prob: float = 0.7,
                 margin: int = 24, augment: bool = True):
        self.cases = [(load_volume(prep_dir, m), np.asarray(m["box"], np.float32)) for m in metas]
        self.length, self.pos_prob, self.margin, self.augment = length, pos_prob, margin, augment
        self.scale = np.asarray(PATCH * 2, np.float32)

    def __len__(self) -> int:
        return self.length

    def _start(self, center: float, size: int, patch: int, positive: bool) -> int:
        if not positive:
            return random.randint(0, size - patch)
        lo = int(np.clip(center - patch + self.margin, 0, size - patch))
        hi = int(np.clip(center - self.margin, 0, size - patch))
        return random.randint(lo, hi)

    def __getitem__(self, _: int) -> Tuple[torch.Tensor, torch.Tensor]:
        vol, box = random.choice(self.cases)
        center = (box[:3] + box[3:]) / 2
        positive = random.random() < self.pos_prob
        sd = self._start(center[0], vol.shape[0], PATCH[0], positive)
        sw = self._start(center[2], vol.shape[2], PATCH[2], positive)
        x = torch.from_numpy(vol[sd:sd + PATCH[0], :, sw:sw + PATCH[2]].astype(np.float32))
        b = box - np.asarray([sd, 0, sw, sd, 0, sw], np.float32)
        c = (b[:3] + b[3:]) / 2
        inside = all(0 <= c[k] < PATCH[k] for k in range(3))
        b = np.clip(b, 0, self.scale)
        if self.augment:
            x, b = self._augment(x, b)
        target = torch.from_numpy(b / self.scale)[None] if inside else torch.zeros(0, 6)
        return normalize(x)[None], target

    @staticmethod
    def _augment(x: torch.Tensor, b: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        # Flip only the two lateral axes; the depth axis carries acquisition physics.
        if random.random() < 0.5:
            x, b = x.flip(0), b.copy()
            b[[0, 3]] = PATCH[0] - b[[3, 0]]
        if random.random() < 0.5:
            x, b = x.flip(2), b.copy()
            b[[2, 5]] = PATCH[2] - b[[5, 2]]
        x = (x / 255).clamp(1e-6, 1) ** math.exp(random.uniform(-0.3, 0.3))
        x = x * random.uniform(0.85, 1.15) + random.uniform(-0.08, 0.08)
        return x * 255, b


def collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]
