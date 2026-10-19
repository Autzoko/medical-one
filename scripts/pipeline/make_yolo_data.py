"""Build a YOLOv5 dataset of 2.5D slices from the raw TDSC volumes (native resolution, deployed layout).

  python scripts/pipeline/make_yolo_data.py --raw RAW --out OUT [--workers 8]

Each image is the deployed 2D-branch input before letterboxing: the slices (u-1, u, u+1) of
V = R.permute(1, 0, 2).flip(1) as channels, rows = v (depth), columns = w. The label is the lesion
mask's bounding box on the center slice u. YOLOv5 reads images with OpenCV and flips BGR -> RGB, so the
file stores the channels reversed (u+1, u, u-1) and the network sees (u-1, u, u+1), as in deployment.

train (Train split): every 2nd lesion slice (center-slice lesion area >= 4 px) plus every 8th lesion-free
slice as background. val (Validation split): the deployed sampling, every 4th slice from 0, all kept.

``--cv-folds K --cv-fold k`` pools Train + Validation and uses the 3D detector's stratified split
(``cv_split``): train = the other folds, val = fold k. ``--negatives`` (working-space hard negatives,
``scripts/hardneg/mine.py``) adds, for train cases, every 2nd slice crossing a negative box and writes the
negative's in-plane box as class 1 on every kept slice it crosses; the patched YOLOv5 loss
(``train_yolo.py``) never treats class 1 as an object and up-weights its objectness instead.
"""
import argparse
import os
from multiprocessing import Pool

import cv2
import nrrd
import numpy as np
import pandas as pd
import yaml

from medical_one.data.tdsc import cv_split
from medical_one.hardneg import load_negatives, working_to_v
from medical_one.pipeline.native import native_view, three_slice

MIN_AREA = 4


def slice_box(mask_slice: np.ndarray):
    rows = np.nonzero(mask_slice.any(1))[0]
    cols = np.nonzero(mask_slice.any(0))[0]
    if len(rows) == 0 or mask_slice.sum() < MIN_AREA:
        return None
    return cols[0], rows[0], cols[-1] + 1, rows[-1] + 1  # x0, y0, x1, y1 half-open pixels


def cv_cases(raw: str, folds: int, fold: int):
    """(train, held-out) lists of (split, case) for fold ``fold`` of the 3D detector's K-fold split."""
    metas = []
    for split in ("Train", "Validation"):
        df = pd.read_csv(os.path.join(raw, split, "labels.csv"))
        metas += [{"split": split, "case": int(r.case_id), "label": r.label} for r in df.itertuples()]
    train, held = cv_split({"Train": [m for m in metas if m["split"] == "Train"],
                            "Validation": [m for m in metas if m["split"] == "Validation"]}, folds, fold)
    return [(m["split"], m["case"]) for m in train], [(m["split"], m["case"]) for m in held]


def neg_slices(neg_boxes: np.ndarray, n: int):
    """V-space negative boxes -> {slice u: [(v0, w0, v1, w1), ...]} for the slices each box crosses."""
    out = {}
    for u0, v0, w0, u1, v1, w1 in np.asarray(neg_boxes, np.float64).reshape(-1, 6):
        for u in range(max(int(np.ceil(u0)), 0), min(int(np.ceil(u1)), n)):
            out.setdefault(u, []).append((v0, w0, v1, w1))
    return out


def process(job):
    raw, out, split, sub, case, negs = job
    base = os.path.join(raw, split)
    v = native_view(nrrd.read(os.path.join(base, "DATA", f"DATA_{case:03d}.nrrd"))[0])
    m = native_view((nrrd.read(os.path.join(base, "MASK", f"MASK_{case:03d}.nrrd"))[0] > 0).astype(np.uint8))
    h, w = v.shape[1:]
    hard = neg_slices(working_to_v(negs, v.shape), v.shape[0]) if sub == "train" and negs is not None else {}
    n_pos = n_neg = n_hard = 0
    for i in range(v.shape[0]):
        box = slice_box(m[i])
        if sub == "train":
            keep = (box is not None and i % 2 == 0) or (box is None and i % 8 == 4 and not m[i].any())
            keep = keep or (i in hard and i % 2 == 0)
        else:
            keep = i % 4 == 0
        if not keep:
            continue
        name = f"{split}_{case:03d}_{i:04d}"
        img = three_slice(v, i)[:, :, ::-1]
        cv2.imwrite(os.path.join(out, "images", sub, name + ".png"), np.ascontiguousarray(img))
        with open(os.path.join(out, "labels", sub, name + ".txt"), "w") as f:
            if box is not None:
                x0, y0, x1, y1 = box
                f.write(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n")
                n_pos += 1
            else:
                n_neg += 1
            for v0, w0, v1, w1 in hard.get(i, []):
                v0, v1, w0, w1 = max(v0, 0), min(v1, h), max(w0, 0), min(w1, w)
                if v1 - v0 >= 2 and w1 - w0 >= 2:
                    f.write(f"1 {(w0 + w1) / 2 / w:.6f} {(v0 + v1) / 2 / h:.6f} {(w1 - w0) / w:.6f} {(v1 - v0) / h:.6f}\n")
                    n_hard += 1
    return split, case, n_pos, n_neg, n_hard, v.shape


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, help="dir with Train/ Validation/ Test/ each holding DATA/ MASK/ labels.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--splits", nargs="+", default=["Train:train", "Validation:val"])
    ap.add_argument("--cv-folds", type=int, default=0, help="> 0: K-fold split of Train + Validation (ignores --splits)")
    ap.add_argument("--cv-fold", type=int, default=0)
    ap.add_argument("--negatives", default="", help="working-space hard negatives JSON (train cases only)")
    args = ap.parse_args()
    negs = load_negatives(args.negatives) if args.negatives else {}
    neg_of = lambda split, case: negs.get((split, case), (np.zeros((0, 6)),))[0] if args.negatives else None  # noqa: E731
    if args.cv_folds > 0:
        train, held = cv_cases(args.raw, args.cv_folds, args.cv_fold)
        pairs = [(s, c, "train") for s, c in train] + [(s, c, "val") for s, c in held]
    else:
        pairs = []
        for item in args.splits:
            split, sub = item.split(":")
            df = pd.read_csv(os.path.join(args.raw, split, "labels.csv"))
            pairs += [(split, int(c), sub) for c in df.case_id]
    for sub in {p[2] for p in pairs}:
        for d in ("images", "labels"):
            os.makedirs(os.path.join(args.out, d, sub), exist_ok=True)
    jobs = [(args.raw, args.out, split, sub, case, neg_of(split, case)) for split, case, sub in pairs]
    with Pool(args.workers) as pool:
        stats = pool.map(process, jobs, chunksize=1)
    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        yaml.safe_dump({"path": args.out, "train": "images/train", "val": "images/val", "nc": 1,
                        "names": ["lesion"]}, f)
    for sub in sorted({j[3] for j in jobs}):
        s = [x for x, j in zip(stats, jobs) if j[3] == sub]
        print(f"{sub}: {len(s)} cases, {sum(x[2] for x in s)} lesion slices, {sum(x[3] for x in s)} background, "
              f"{sum(x[4] for x in s)} hard-negative boxes", flush=True)


if __name__ == "__main__":
    main()
