"""Classify the final boxes of the pipeline with the trained ResNet-34, as the deployed stage does (one
ROI per kept box; classification does not filter or re-rank boxes).

  python scripts/pipeline/classify.py --ckpt best.pt --raw RAW --meta meta.json --preds pipe_pipeline.json \
      --out cls.json [--split Test --top 20]
"""
import argparse
import json
import os
import sys

import nrrd
import numpy as np
import torch

from medical_one.evaluation.preds import load_predictions
from medical_one.pipeline.native import native_view, roi_image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_classifier import build_model, predict  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="Test")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    with open(args.meta) as f:
        metas = [m for m in json.load(f) if m["split"] == args.split]
    preds = load_predictions(args.preds, metas)
    model = build_model(pretrained=False).to(args.device)
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device, weights_only=False)["model"])
    out = []
    for meta, (boxes, scores) in zip(metas, preds):
        rec = {"case": meta["case"], "label": meta["label"], "scores": scores[:args.top].tolist(), "p_malignant": []}
        if len(boxes):
            raw = nrrd.read(os.path.join(args.raw, meta["split"], "DATA", f"DATA_{meta['case']:03d}.nrrd"))[0]
            v = native_view(raw)
            scale = np.array([v.shape[0] / 336, v.shape[1] / 112, v.shape[2] / 336] * 2)
            rois = np.stack([roi_image(v, b * scale) for b in boxes[:args.top].double().numpy()])
            rec["p_malignant"] = predict(model, rois, args.device).tolist()
        out.append(rec)
        print(meta["case"], len(rec["p_malignant"]), flush=True)
    with open(args.out, "w") as f:
        json.dump({"preds": args.preds, "ckpt": args.ckpt, "cases": out}, f)


if __name__ == "__main__":
    main()
