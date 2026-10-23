"""AdaBoost-style case weights for training a single DETR from out-of-fold errors (docs/prereg/detr_boost.md).

  python scripts/boost/boost_weights.py --meta META --oof f0/preds_oof.json ... --out weights.json

Error of a training case (out of fold, official IoU > 0.3 in raw space): e = 1 - 1 / r, r = rank of the first
hit among the case's boxes sorted by score (hit at rank 1 -> 0, rank 2 -> 0.5, no hit -> 1), so false positives
ranked above the lesion and misses both count. AdaBoost update with a real-valued margin 1 - 2e:
eps = mean(e) (clipped to [0.05, 0.95]), alpha = 0.5 ln((1 - eps) / eps), w = exp(alpha (2e - 1)), normalised to
mean 1, clipped to [w_min, w_max], mixed with the uniform weight (``--mix`` boosted share) and normalised again.
"""
import argparse
import json

import numpy as np

from medical_one.data.tdsc import boxes_to_original
from medical_one.evaluation.preds import load_predictions
from medical_one.evaluation.tdsc_official import official_ious


def case_errors(preds, metas, iou_thr: float = 0.3):
    out = []
    for (boxes, scores), m in zip(preds, metas):
        order = scores.argsort(descending=True)
        ious = official_ious(boxes_to_original(boxes[order].double(), m["shape"]), m["mask_lo"], m["mask_hi"]) \
            if len(boxes) else np.zeros(0)
        hits = np.nonzero(ious > iou_thr)[0]
        rank = int(hits[0]) + 1 if len(hits) else None
        out.append((1.0 - 1.0 / rank if rank else 1.0, rank))
    return out


def boost_weights(errors: np.ndarray, w_min: float = 0.5, w_max: float = 3.0, mix: float = 0.5):
    eps = float(np.clip(errors.mean(), 0.05, 0.95))
    alpha = 0.5 * np.log((1 - eps) / eps)
    w = np.exp(alpha * (2 * errors - 1))
    w = np.clip(w / w.mean(), w_min, w_max)
    w = mix * w / w.mean() + (1 - mix)
    return w / w.mean(), eps, float(alpha)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--oof", nargs="+", required=True, help="preds_oof.json of the inner-fold runs (disjoint cases)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--w-min", type=float, default=0.5)
    ap.add_argument("--w-max", type=float, default=3.0)
    ap.add_argument("--mix", type=float, default=0.5)
    args = ap.parse_args()
    with open(args.meta) as f:
        metas = [m for m in json.load(f) if m["split"] == "Train"]
    covered = set()
    for p in args.oof:
        with open(p) as f:
            covered |= {(c["split"], c["case"]) for c in json.load(f)["cases"]}
    assert covered == {(m["split"], m["case"]) for m in metas}, "the OOF files must cover every Train case once"
    errs = case_errors(load_predictions(args.oof, metas), metas)
    e = np.array([x[0] for x in errs])
    w, eps, alpha = boost_weights(e, args.w_min, args.w_max, args.mix)
    info = {"oof": args.oof, "eps": round(eps, 4), "alpha": round(alpha, 4), "w_min": args.w_min,
            "w_max": args.w_max, "mix": args.mix, "top1_hit": round(float(np.mean([r == 1 for _, r in errs])), 3),
            "miss": round(float(np.mean([r is None for _, r in errs])), 3),
            "weight_range": [round(float(w.min()), 3), round(float(w.max()), 3)]}
    cases = [{"split": m["split"], "case": m["case"], "weight": round(float(wi), 4), "error": round(float(ei), 4),
              "first_hit_rank": r} for m, wi, ei, (_, r) in zip(metas, w, e, errs)]
    with open(args.out, "w") as f:
        json.dump({"info": info, "cases": cases}, f, indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
