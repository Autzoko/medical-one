"""Apply the cascade to one split: DETR (+ optional CPM, fixed WBF) -> top-K candidates -> verifier refinement and
re-scoring (stacking weights from ``stack.json``) -> NMS -> predictions file.

  python scripts/cascade/apply_cascade.py --prep-dir PREP --split Validation --detr preds_val_swa.json \
      [--cpm cpm/preds_val_swa.json] --verifier VDIR --out cascade_val.json [--mode full|fuse|refine|rescore]

``--mode``: ``fuse`` = DETR (+ CPM) only; ``rescore`` = verifier scores, boxes unchanged; ``refine`` = refined boxes,
detector scores; ``full`` = refined boxes + stacked scores. Fixed settings (pre-registered): WBF IoU 0.3, equal weights;
top 30 candidates; 2 refinement iterations; NMS 0.1; at most 50 boxes.
"""
import argparse

import numpy as np
import torch

from medical_one.cascade.verifier import Verifier3D, refine
from medical_one.data.dataset import normalize
from medical_one.data.tdsc import load_meta, load_volume
from medical_one.evaluation.preds import load_predictions, save_predictions
from medical_one.utils.boxes import nms_3d, weighted_box_fusion


def logit(p):
    p = p.clamp(1e-4, 1 - 1e-4)
    return torch.log(p / (1 - p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--split", default="Validation")
    ap.add_argument("--detr", required=True)
    ap.add_argument("--cpm", default="")
    ap.add_argument("--verifier", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="full", choices=["fuse", "rescore", "refine", "full"])
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--wbf-iou", type=float, default=0.3)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    metas = load_meta(args.prep_dir)[args.split]
    srcs = [load_predictions(args.detr, metas)] + ([load_predictions(args.cpm, metas)] if args.cpm else [])
    model = stack = None
    if args.mode != "fuse":
        state = torch.load(f"{args.verifier}/verifier.pt", map_location=dev, weights_only=False)
        model = Verifier3D().to(dev)
        model.load_state_dict(state["model"])
        stack = state["stack"]
    out = []
    for i, m in enumerate(metas):
        per = [(b.float(), s.float()) for b, s in (src[i] for src in srcs)]
        b, s = weighted_box_fusion(per, args.wbf_iou) if len(per) > 1 else per[0]
        o = s.argsort(descending=True)[:args.top]
        b, s = b[o].float(), s[o].float()
        if model is not None and len(b):
            vol = normalize(torch.from_numpy(load_volume(args.prep_dir, m)).float())
            rb, p, _ = refine(model, vol, b, iters=args.iters if args.mode in ("refine", "full") else 0)
            if args.mode in ("refine", "full"):
                b = rb
            if args.mode in ("rescore", "full"):
                a = stack["coef"]
                s = torch.sigmoid(a[0] * logit(s) + a[1] * logit(p) + stack["intercept"])
            keep = nms_3d(b, s, 0.1)[:50]
            b, s = b[keep], s[keep]
        out.append((b, s))
    save_predictions(args.out, out, metas, vars(args))
    print("wrote", args.out, len(out))


if __name__ == "__main__":
    main()
