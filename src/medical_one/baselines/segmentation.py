"""Patch sampling and sliding-window inference for segmentation baselines (e.g. MONAI SwinUNETR).

Sampling mirrors ``PatchDataset``: with ``pos_prob`` the patch keeps the lesion center >= ``margin`` from
the d/w borders, lateral flips and the same intensity jitter; the mask is cropped and flipped alike.
"""
import math
import os
import random
from typing import Sequence, Tuple

import numpy as np
import torch

from ..data.dataset import normalize
from ..data.tdsc import load_volume


class SegPatchDataset(torch.utils.data.Dataset):
    def __init__(self, prep_dir: str, masks_dir: str, metas: Sequence[dict], length: int, patch: Sequence[int],
                 pos_prob: float = 0.7, margin: int = 24):
        self.cases = []
        for m in metas:
            mask = np.load(os.path.join(masks_dir, m["split"], f"{m['case']:03d}.npy"))
            self.cases.append((load_volume(prep_dir, m), mask, np.asarray(m["box"], np.float32)))
        self.length, self.patch, self.pos_prob, self.margin = length, tuple(patch), pos_prob, margin

    def __len__(self) -> int:
        return self.length

    def _start(self, center: float, size: int, patch: int, positive: bool) -> int:
        if not positive:
            return random.randint(0, size - patch)
        margin = min(self.margin, patch // 2)
        lo = int(np.clip(center - patch + margin, 0, size - patch))
        hi = int(np.clip(center - margin, 0, size - patch))
        return random.randint(lo, hi)

    def __getitem__(self, _: int) -> Tuple[torch.Tensor, torch.Tensor]:
        vol, mask, box = random.choice(self.cases)
        center = (box[:3] + box[3:]) / 2
        positive = random.random() < self.pos_prob
        pd_, ph, pw = self.patch
        sd = self._start(center[0], vol.shape[0], pd_, positive)
        sh = random.randint(0, vol.shape[1] - ph)
        sw = self._start(center[2], vol.shape[2], pw, positive)
        x = torch.from_numpy(vol[sd:sd + pd_, sh:sh + ph, sw:sw + pw].astype(np.float32))
        y = torch.from_numpy(mask[sd:sd + pd_, sh:sh + ph, sw:sw + pw].astype(np.float32))
        for axis in (0, 2):  # lateral flips only, as for the detectors
            if random.random() < 0.5:
                x, y = x.flip(axis), y.flip(axis)
        x = (x / 255).clamp(1e-6, 1) ** math.exp(random.uniform(-0.3, 0.3))
        x = (x * random.uniform(0.85, 1.15) + random.uniform(-0.08, 0.08)) * 255
        return normalize(x)[None], y[None]


@torch.no_grad()
def predict_prob(model, volume: np.ndarray, roi: Sequence[int], device: str = "cuda", overlap: float = 0.5) -> np.ndarray:
    """Foreground probability on the whole working-space volume (Gaussian-weighted sliding window)."""
    from monai.inferers import sliding_window_inference

    model.eval()
    x = normalize(torch.from_numpy(volume).float().to(device))[None, None]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        logits = sliding_window_inference(x, tuple(roi), 2, model, overlap=overlap, mode="gaussian")
    return torch.softmax(logits.float(), 1)[0, 1].cpu().numpy()
