"""Build a YOLOv5 dataset of 2.5D slices from the raw TDSC volumes (native resolution, deployed layout).

  python scripts/pipeline/make_yolo_data.py --raw RAW --out OUT [--workers 8]

Each image is the deployed 2D-branch input before letterboxing: the slices (u-1, u, u+1) of
V = R.permute(1, 0, 2).flip(1) as channels, rows = v (depth), columns = w. The label is the lesion
mask's bounding box on the center slice u. YOLOv5 reads images with OpenCV and flips BGR -> RGB, so the
file stores the channels reversed (u+1, u, u-1) and the network sees (u-1, u, u+1), as in deployment.

train (Train split): every 2nd lesion slice (center-slice lesion area >= 4 px) plus every 8th lesion-free
slice as background. val (Validation split): the deployed sampling, every 4th slice from 0, all kept.
"""
import argparse
import os
from multiprocessing import Pool

import cv2
import nrrd
import numpy as np
import pandas as pd
import yaml

from medical_one.pipeline.native import native_view, three_slice

MIN_AREA = 4


def slice_box(mask_slice: np.ndarray):
    rows = np.nonzero(mask_slice.any(1))[0]
    cols = np.nonzero(mask_slice.any(0))[0]
    if len(rows) == 0 or mask_slice.sum() < MIN_AREA:
        return None
    return cols[0], rows[0], cols[-1] + 1, rows[-1] + 1  # x0, y0, x1, y1 half-open pixels


def process(job):
    raw, out, split, sub, case = job
    base = os.path.join(raw, split)
    v = native_view(nrrd.read(os.path.join(base, "DATA", f"DATA_{case:03d}.nrrd"))[0])
    m = native_view((nrrd.read(os.path.join(base, "MASK", f"MASK_{case:03d}.nrrd"))[0] > 0).astype(np.uint8))
    h, w = v.shape[1:]
    n_pos = n_neg = 0
    for i in range(v.shape[0]):
        box = slice_box(m[i])
        if sub == "train":
            keep = (box is not None and i % 2 == 0) or (box is None and i % 8 == 4 and not m[i].any())
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
    return split, case, n_pos, n_neg, v.shape


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, help="dir with Train/ Validation/ Test/ each holding DATA/ MASK/ labels.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--splits", nargs="+", default=["Train:train", "Validation:val"])
    args = ap.parse_args()
    jobs = []
    for item in args.splits:
        split, sub = item.split(":")
        for d in ("images", "labels"):
            os.makedirs(os.path.join(args.out, d, sub), exist_ok=True)
        df = pd.read_csv(os.path.join(args.raw, split, "labels.csv"))
        jobs += [(args.raw, args.out, split, sub, int(c)) for c in df.case_id]
    with Pool(args.workers) as pool:
        stats = pool.map(process, jobs, chunksize=1)
    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        yaml.safe_dump({"path": args.out, "train": "images/train", "val": "images/val", "nc": 1,
                        "names": ["lesion"]}, f)
    for sub in sorted({j[3] for j in jobs}):
        s = [x for x, j in zip(stats, jobs) if j[3] == sub]
        print(f"{sub}: {len(s)} cases, {sum(x[2] for x in s)} lesion slices, {sum(x[3] for x in s)} background",
              flush=True)


if __name__ == "__main__":
    main()
