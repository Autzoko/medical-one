"""Compare methods on one split with 95% case-bootstrap confidence intervals.

  python scripts/compare_methods.py --meta meta.json --split Test --ref "DETR" \
      --method "DETR=a.json,b.json" --method "nnU-Net=c.json" --out results.json

Several files for one method are WBF-fused per case (an ensemble); files joined with "+" are instead
concatenated (disjoint cases, e.g. out-of-fold predictions: --split Train,Validation --method
"DETR OOF=f0.json+f1.json+..."). The bootstrap resamples cases (with replacement) jointly for all
methods, so differences to ``--ref`` are paired.
"""
import argparse
import json

import numpy as np

from medical_one.data.tdsc import boxes_to_original
from medical_one.evaluation import compute_metrics
from medical_one.evaluation.preds import fuse_predictions, load_predictions
from medical_one.evaluation.tdsc_official import average_precision, official_froc, official_ious

KEYS = ("tdsc_froc", "ap@0.1", "ap@0.3", "froc@0.3", "top1_hit@0.3", "tdsc_max_recall", "top1_iou_mean")


def per_volume(preds, metas):
    return [(s.float().numpy(), official_ious(boxes_to_original(b.double(), m["shape"]), m["mask_lo"], m["mask_hi"]))
            for (b, s), m in zip(preds, metas)]


def boot_stats(pv, idx):
    sample = [pv[i] for i in idx]
    return official_froc(sample)[0], average_precision(sample, 0.1), average_precision(sample, 0.3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--split", default="Test", help="one split or several, comma-separated")
    ap.add_argument("--method", action="append", required=True, help="name=file1[,file2,...]")
    ap.add_argument("--ref", default="")
    ap.add_argument("--wbf-iou", type=float, default=0.5)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    with open(args.meta) as f:
        metas = [m for m in json.load(f) if m["split"] in args.split.split(",")]
    methods, results = {}, {}
    for item in args.method:
        name, files = item.split("=", 1)
        sources = [load_predictions(p.split("+"), metas) for p in files.split(",")]
        preds = sources[0] if len(sources) == 1 else fuse_predictions(sources, args.wbf_iou)
        methods[name] = per_volume(preds, metas)
        res = compute_metrics(preds, metas)
        results[name] = {k: res[k] for k in KEYS}
        results[name]["tdsc_sens"] = res["tdsc_sens"]
        results[name]["files"] = files.split(",")
    rng = np.random.RandomState(0)
    draws = [rng.randint(0, len(metas), len(metas)) for _ in range(args.boot)]
    boots = {name: np.array([boot_stats(pv, idx) for idx in draws]) for name, pv in methods.items()}
    for name, b in boots.items():
        for j, key in enumerate(("tdsc_froc", "ap@0.1", "ap@0.3")):
            results[name][f"{key}_ci"] = [float(np.percentile(b[:, j], 2.5)), float(np.percentile(b[:, j], 97.5))]
        if args.ref and name != args.ref:
            d = b[:, 0] - boots[args.ref][:, 0]
            results[name]["tdsc_froc_diff_vs_ref"] = [float(np.mean(d)), float(np.percentile(d, 2.5)),
                                                      float(np.percentile(d, 97.5)), float(np.mean(d <= 0))]
    print(f"{'method':38s} {'TDSC FROC (95% CI)':24s} {'AP@0.1':>7s} {'AP@0.3':>7s} {'top1':>6s} {'maxrec':>6s}")
    for name, r in results.items():
        ci = r["tdsc_froc_ci"]
        print(f"{name:38s} {r['tdsc_froc']:.3f} ({ci[0]:.3f}-{ci[1]:.3f})      {r['ap@0.1']:7.3f} {r['ap@0.3']:7.3f} "
              f"{r['top1_hit@0.3']:6.3f} {r['tdsc_max_recall']:6.3f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"split": args.split, "ref": args.ref, "n_cases": len(metas), "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
