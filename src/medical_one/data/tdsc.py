"""TDSC-ABUS -> model working space, following the deployed preprocessing.

Raw NRRD array R is [N0, N1, N2] with N0 the depth axis (0.073 mm). As in the deployed pipeline,
V = R.transpose(1, 0, 2) flipped along depth -> [N1, N0, N2], trilinearly resized to
``RESIZE`` (align_corners=False), then only the first ``KEEP_DEPTH`` depth slices are kept.
Lesion boxes come from the masks (half-open voxel boxes) and are mapped the same way.
Baselines get only these boxes: ``export_box_masks`` writes cuboid masks for segmentation-based detectors.
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


def mask_to_working_space(mask: np.ndarray) -> np.ndarray:
    """Binary lesion mask on the working grid (trilinear resize of the 0/1 mask, threshold 0.5)."""
    m = torch.from_numpy(np.ascontiguousarray((mask > 0).transpose(1, 0, 2)[:, ::-1, :])).float()[None, None]
    m = F.interpolate(m, size=RESIZE, mode="trilinear", align_corners=False)[0, 0, :, :KEEP_DEPTH]
    return (m >= 0.5).to(torch.uint8).numpy()


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


def _process_mask(args: Tuple[Dict[str, str], str, int, str]) -> Tuple[str, int, int]:
    roots, split, case, _ = args
    torch.set_num_threads(2)
    mask, _ = nrrd.read(case_paths(roots.get(split, roots["default"]), split, case)[1])
    out = mask_to_working_space(mask)
    os.makedirs(os.path.join(roots["out"], "masks", split), exist_ok=True)
    np.save(os.path.join(roots["out"], "masks", split, f"{case:03d}.npy"), out)
    return split, case, int(out.sum())


def export_masks(root: str, out_dir: str, extracted_train: str = "", workers: int = 4) -> List[Tuple[str, int, int]]:
    """Working-space lesion masks under ``out_dir/masks/<split>/`` (for segmentation-based baselines)."""
    roots = {"default": root, "out": out_dir}
    if extracted_train:
        roots["Train"] = extracted_train
    jobs = []
    for split in SPLITS:
        df = pd.read_csv(os.path.join(root, split, "labels.csv"))
        jobs += [(roots, split, int(r.case_id), r.label) for r in df.itertuples()]
    with Pool(workers) as pool:
        return pool.map(_process_mask, jobs, chunksize=1)


def box_mask(box: Sequence[float], shape: Sequence[int] = (RESIZE[0], KEEP_DEPTH, RESIZE[2])) -> np.ndarray:
    """Cuboid mask of a working-space box (d0, h0, w0, d1, h1, w1): voxels [round(lo), round(hi)) per axis, at
    least one voxel. The only "mask" a bbox-only baseline (nnDetection, nnU-Net) is given."""
    lo = [min(max(int(np.floor(b + 0.5)), 0), n - 1) for b, n in zip(box[:3], shape)]
    hi = [min(max(int(np.floor(b + 0.5)), l + 1), n) for b, l, n in zip(box[3:], lo, shape)]
    mask = np.zeros(tuple(shape), np.uint8)
    mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 1
    return mask


def export_box_masks(prep_dir: str, out_name: str = "box_masks") -> List[Tuple[str, int, int]]:
    """Cuboid masks from the boxes in meta.json under ``prep_dir/<out_name>/<split>/`` (Train and Validation only:
    the training cases of the baselines)."""
    res = []
    for split, metas in load_meta(prep_dir).items():
        if split == "Test":
            continue
        os.makedirs(os.path.join(prep_dir, out_name, split), exist_ok=True)
        for m in metas:
            mask = box_mask(m["box"])
            np.save(os.path.join(prep_dir, out_name, split, f"{m['case']:03d}.npy"), mask)
            res.append((split, m["case"], int(mask.sum())))
    return res


def preprocess_tdsc(root: str, out_dir: str, extracted_train: str = "", workers: int = 10) -> List[CaseMeta]:
    """``extracted_train`` is the parent of an unzipped copy of Train/ (the official release zips it)."""
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


def cv_split(splits: Dict[str, List[dict]], folds: int, fold: int, seed: int = 0) -> Tuple[List[dict], List[dict]]:
    """K-fold split of Train + Validation, stratified by benign / malignant; returns (train, held-out)."""
    pool = sorted(splits["Train"] + splits["Validation"], key=lambda m: m["case"])
    rng = np.random.RandomState(seed)
    assign = {}
    for label in sorted({m["label"] for m in pool}):
        cases = [m["case"] for m in pool if m["label"] == label]
        for i, case in enumerate(rng.permutation(cases)):
            assign[int(case)] = i % folds
    return [m for m in pool if assign[m["case"]] != fold], [m for m in pool if assign[m["case"]] == fold]


def train_val_metas(splits: Dict[str, List[dict]], train_splits: Sequence[str], cv_folds: int = 0,
                    cv_fold: int = 0) -> Tuple[List[dict], List[dict]]:
    if cv_folds > 0:
        return cv_split(splits, cv_folds, cv_fold)
    return [m for s in train_splits for m in splits[s]], splits["Validation"]


def load_volume(prep_dir: str, meta: dict) -> np.ndarray:
    return np.load(os.path.join(prep_dir, meta["split"], f"{meta['case']:03d}.npy"))
