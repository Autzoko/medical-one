"""Fuse the 2D branch with 3D predictions as the deployed pipeline does, and write scorable predictions.

  python scripts/pipeline/fuse.py --meta meta.json --d2 d2.json --d3 d3_f0.json [d3_f1.json ...] \
      --split Test --out-dir OUT [--tag name]

``--d3`` files hold 3D predictions made with ``eval.deployed_fusion=true`` (window WBF, top 20); several
files are first combined with WBF (IoU 0.5) into one 3D list (an ensemble). Writes, in working-space
corners for ``score_preds.py`` / ``compare_methods.py``:
  <tag>_2d.json        2D branch alone (aggregated boxes with score > 0.3, YOLO scores)
  <tag>_pipeline.json  full deployed fusion: 2D + top-5 boosted 3D, two-source WBF, score > 0.2, top 20
  <tag>_pipe3d.json    the same fusion with the 2D source left empty (3D branch through the deployed path)
"""
import argparse
import json
import os

import numpy as np
import torch

from medical_one.evaluation.preds import fuse_predictions, load_predictions, save_predictions
from medical_one.pipeline.fusion import fuse_case
from medical_one.pipeline.yolo2d import v_boxes_to_working


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--d2", required=True)
    ap.add_argument("--d3", nargs="+", required=True)
    ap.add_argument("--split", default="Test")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="pipe")
    args = ap.parse_args()
    with open(args.meta) as f:
        metas = [m for m in json.load(f) if m["split"] == args.split]
    with open(args.d2) as f:
        d2 = {(c["split"], c["case"]): c for c in json.load(f)["cases"]}
    sources = [load_predictions(p, metas) for p in args.d3]
    d3 = sources[0] if len(sources) == 1 else fuse_predictions(sources, 0.5, 20)
    out = {"2d": [], "pipeline": [], "pipe3d": []}
    gate_closed = 0
    for meta, (b3, s3) in zip(metas, d3):
        rec = d2[(meta["split"], meta["case"])]
        gate_closed += not rec["gate"]
        rows = np.asarray(rec["boxes"], np.float64).reshape(-1, 7)
        out["2d"].append((torch.from_numpy(v_boxes_to_working(rows[:, :6], rec["v_shape"])).float(),
                          torch.from_numpy(rows[:, 6]).float()))
        for key, use_2d in (("pipeline", True), ("pipe3d", False)):
            fb, fs = fuse_case(rec, rec["v_shape"], b3.double().numpy(), s3.double().numpy(), use_2d=use_2d)
            out[key].append((torch.from_numpy(fb).float(), torch.from_numpy(fs).float()))
    os.makedirs(args.out_dir, exist_ok=True)
    for key, preds in out.items():
        save_predictions(os.path.join(args.out_dir, f"{args.tag}_{key}.json"), preds, metas,
                         {"d2": args.d2, "d3": args.d3, "variant": key})
    print(f"{args.split}: {len(metas)} cases, 2D gate closed for {gate_closed}", flush=True)


if __name__ == "__main__":
    main()
