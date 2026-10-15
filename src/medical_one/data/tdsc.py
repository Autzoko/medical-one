"""TDSC-ABUS -> model working space, following the deployed preprocessing.

Raw NRRD array R is [N0, N1, N2] with N0 the depth axis (0.073 mm). As in the deployed pipeline,
V = R.transpose(1, 0, 2) flipped along depth -> [N1, N0, N2], trilinearly resized to
``RESIZE`` (align_corners=False), then only the first ``KEEP_DEPTH`` depth slices are kept.
Lesion boxes come from the masks (half-open voxel boxes) and are mapped the same way.
"""
import json
import os
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from typing import Dict, List, Sequence, Tuple

import nrrd
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SPLITS = ("Train", "Validation", "Test")
RESIZE = (336, 112, 336)
KEEP_DEPTH = 96


@dataclass
class CaseMeta:
    split: str
    case: int
    label: str  # B / M
    shape: List[int]  # raw [N0, N1, N2]
    mask_lo: List[int]
    mask_hi: List[int]
    box: List[float]  # working space (d0, h0, w0, d1, h1, w1)
    depth_kept_frac: float


def case_paths(root: str, split: str, case: int) -> Tuple[str, str]:
    base = os.path.join(root, split)
    return os.path.join(base, "DATA", f"DATA_{case:03d}.nrrd"), os.path.join(base, "MASK", f"MASK_{case:03d}.nrrd")


def to_working_space(img: np.ndarray) -> np.ndarray:
    v = torch.from_numpy(np.ascontiguousarray(img.transpose(1, 0, 2)[:, ::-1, :])).float()[None, None]
    v = F.interpolate(v, size=RESIZE, mode="trilinear", align_corners=False)[0, 0, :, :KEEP_DEPTH]
    return v.round().clamp(0, 255).to(torch.uint8).numpy()


def box_to_working_space(lo: Sequence[int], hi: Sequence[int], shape: Sequence[int]) -> Tuple[List[float], float]:
    n0, n1, n2 = shape
    s = (RESIZE[0] / n1, RESIZE[1] / n0, RESIZE[2] / n2)
    box = [lo[1] * s[0], (n0 - hi[0]) * s[1], lo[2] * s[2], hi[1] * s[0], (n0 - lo[0]) * s[1], hi[2] * s[2]]
    full_depth = box[4] - box[1]
    box[1], box[4] = min(box[1], KEEP_DEPTH), min(box[4], KEEP_DEPTH)
    return [round(b, 3) for b in box], round((box[4] - box[1]) / full_depth, 4)


def boxes_to_original(boxes: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """Inverse of ``box_to_working_space``: working-space corners [N,6] -> raw-array corners [N,6]
    (lo0, lo1, lo2, hi0, hi1, hi2) in continuous half-open voxel units of the raw [N0, N1, N2] array."""
    n0, n1, n2 = shape
    s = (RESIZE[0] / n1, RESIZE[1] / n0, RESIZE[2] / n2)
    d0, h0, w0, d1, h1, w1 = boxes.unbind(-1)
    return torch.stack([n0 - h1 / s[1], d0 / s[0], w0 / s[2], n0 - h0 / s[1], d1 / s[0], w1 / s[2]], -1)


def _process(args: Tuple[Dict[str, str], str, int, str]) -> CaseMeta:
    roots, split, case, label = args
    torch.set_num_threads(4)
    img, _ = nrrd.read(case_paths(roots.get(split, roots["default"]), split, case)[0])
    mask, _ = nrrd.read(case_paths(roots.get(split, roots["default"]), split, case)[1])
    nz = np.nonzero(mask)
    lo = [int(a.min()) for a in nz]
    hi = [int(a.max()) + 1 for a in nz]
    box, kept = box_to_working_space(lo, hi, img.shape)
    os.makedirs(os.path.join(roots["out"], split), exist_ok=True)
    np.save(os.path.join(roots["out"], split, f"{case:03d}.npy"), to_working_space(img))
    return CaseMeta(split, case, label, list(img.shape), lo, hi, box, kept)


def preprocess_tdsc(root: str, out_dir: str, extracted_train: str = "", workers: int = 10) -> List[CaseMeta]:
    """``extracted_train`` points at an unzipped copy of Train/ (the official release zips it)."""
    roots = {"default": root, "out": out_dir}
    if extracted_train:
        roots["Train"] = extracted_train
    jobs = []
    for split in SPLITS:
        df = pd.read_csv(os.path.join(root, split, "labels.csv"))
        jobs += [(roots, split, int(r.case_id), r.label) for r in df.itertuples()]
    with Pool(workers) as pool:
        metas = pool.map(_process, jobs, chunksize=1)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump([asdict(m) for m in metas], f, indent=1)
    return metas


def load_meta(prep_dir: str) -> Dict[str, List[dict]]:
    with open(os.path.join(prep_dir, "meta.json")) as f:
        metas = json.load(f)
    return {s: [m for m in metas if m["split"] == s] for s in SPLITS}


def load_volume(prep_dir: str, meta: dict) -> np.ndarray:
    return np.load(os.path.join(prep_dir, meta["split"], f"{meta['case']:03d}.npy"))
