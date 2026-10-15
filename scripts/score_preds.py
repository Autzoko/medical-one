"""Score prediction files offline; several files are fused per case with WBF (an ensemble).

  python scripts/score_preds.py runs/a/preds_val.json runs/b/preds_val.json --meta data/prep/meta.json
  --csv out.csv writes the challenge submission format for the official det_score.py.
"""
import argparse
import json

import pandas as pd

from medical_one.data.tdsc import SPLITS
from medical_one.evaluation import compute_metrics
from medical_one.evaluation.preds import fuse_predictions, load_predictions, to_official_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--meta", required=True, help="meta.json written by preprocess_tdsc.py")
    ap.add_argument("--split", default="", help="default: the split stored in the first file")
    ap.add_argument("--wbf-iou", type=float, default=0.5)
    ap.add_argument("--max-dets", type=int, default=50)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()
    with open(args.meta) as f:
        metas = json.load(f)
    split = args.split
    if not split:
        with open(args.preds[0]) as f:
            split = json.load(f)["cases"][0]["split"]
    assert split in SPLITS
    metas = [m for m in metas if m["split"] == split]
    sources = [load_predictions(p, metas) for p in args.preds]
    preds = sources[0] if len(sources) == 1 else fuse_predictions(sources, args.wbf_iou, args.max_dets)
    if args.csv:
        pd.DataFrame(to_official_rows(preds, metas)).to_csv(args.csv, index=False)
    print(json.dumps({"split": split, "preds": args.preds, **compute_metrics(preds, metas)}))


if __name__ == "__main__":
    main()
