"""Run the deployed 2D branch (YOLOv5 per 2.5D slice + cross-slice aggregation) on raw TDSC volumes.

  python scripts/pipeline/infer_2d.py --weights best.pt --raw RAW --out d2.json [--splits Validation Test]

Writes per case: the V shape, whether the 2D gate passed, the number of per-slice detections, and the
aggregated V-space boxes (u0, v0, w0, u1, v1, w1, score) before (``boxes_all``) and after (``boxes``) the
score > 0.3 filter. Fusion with the 3D branch is done offline (``fuse.py``).
``--cv-folds K --cv-fold k`` keeps only fold k's held-out cases of Train + Validation (Test is kept), so a
K-fold YOLO CV yields out-of-fold 2D boxes.
"""
import argparse
import json
import os
import sys
import time

import nrrd
import pandas as pd
import torch

from medical_one.pipeline.native import native_view
from medical_one.pipeline.yolo2d import branch_2d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_yolo_data import cv_cases  # noqa: E402
from train_yolo import clone_yolov5  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["Validation", "Test"])
    ap.add_argument("--local", default=os.environ.get("LOCAL_DIR", "/tmp"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cv-folds", type=int, default=0)
    ap.add_argument("--cv-fold", type=int, default=0)
    args = ap.parse_args()
    held = set(cv_cases(args.raw, args.cv_folds, args.cv_fold)[1]) if args.cv_folds > 0 else None
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    repo = clone_yolov5(os.path.join(args.local, "yolov5"))
    sys.path.insert(0, repo)
    from models.experimental import attempt_load

    model = attempt_load(args.weights, device=torch.device(args.device), fuse=True).float().eval()
    cases = []
    for split in args.splits:
        df = pd.read_csv(os.path.join(args.raw, split, "labels.csv"))
        for case in df.case_id:
            if held is not None and split != "Test" and (split, int(case)) not in held:
                continue
            t = time.time()
            raw = nrrd.read(os.path.join(args.raw, split, "DATA", f"DATA_{int(case):03d}.nrrd"))[0]
            v = native_view(raw)
            rec = branch_2d(model, v, args.device)
            rec.update({"split": split, "case": int(case), "v_shape": list(v.shape)})
            cases.append(rec)
            print(f"{split} {int(case):03d} slices={v.shape[0]} dets={rec['n_slice_dets']} "
                  f"boxes={len(rec['boxes'])} {time.time() - t:.1f}s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"weights": args.weights, "cases": cases}, f)


if __name__ == "__main__":
    main()
