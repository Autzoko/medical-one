"""Mine hard negatives from out-of-fold predictions of the 3D DETR and/or the 2D YOLO branch.

  python scripts/hardneg/mine.py --meta META --out negs.json [--detr OOF.json ...] [--d2 D2.json ...]
                                 [--min-score 0.3 --top 5] [--random]

``--detr``: working-space prediction files (e.g. the five ``preds_val_best.json`` of a 5-fold CV).
``--d2``: 2D-branch files of ``infer_2d.py`` (aggregated V-space boxes before the 0.3 filter,
``boxes_all``), mapped to working space. Several sources are merged per case. Only Train + Validation
cases are mined. ``--random`` writes the control: per case as many boxes as mined, with the mined boxes'
sizes, at random positions away from the lesion.
"""
import argparse
import json
import os

import numpy as np
import torch

from medical_one.data.tdsc import KEEP_DEPTH, RESIZE
from medical_one.evaluation.preds import load_predictions
from medical_one.hardneg import far_from_gt, mine, save_negatives, summary, union, v_to_working


def d2_predictions(paths, metas):
    cases = {}
    for p in paths:
        with open(p) as f:
            for c in json.load(f)["cases"]:
                key = (c["split"], c["case"])
                assert key not in cases, f"case {key} appears in several files"
                cases[key] = c
    out = []
    for m in metas:
        c = cases.get((m["split"], m["case"]))
        rows = np.asarray(c["boxes_all"] if c else [], np.float64).reshape(-1, 7)
        out.append((torch.from_numpy(v_to_working(rows[:, :6], c["v_shape"]) if c else np.zeros((0, 6))),
                    torch.from_numpy(rows[:, 6])))
    return out


def randomize(negs, metas, seed=0, grow=0.25):
    rng = np.random.RandomState(seed)
    grid = np.array([RESIZE[0], KEEP_DEPTH, RESIZE[2]], np.float64)
    out = {}
    for m in metas:
        b, s = negs[(m["split"], m["case"])]
        new = []
        for box in b:
            size = np.minimum(box[3:] - box[:3], grid)
            for _ in range(1000):
                lo = rng.uniform(0, grid - size)
                cand = np.concatenate([lo, lo + size])
                if far_from_gt(cand[None], m["box"], grow)[0]:
                    new.append(cand)
                    break
        out[(m["split"], m["case"])] = (np.asarray(new, np.float64).reshape(-1, 6), s[:len(new)])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", required=True, help="meta.json of the preprocessed working-space volumes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--detr", nargs="*", default=[])
    ap.add_argument("--d2", nargs="*", default=[])
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--top", type=int, default=5, help="negatives per case and source")
    ap.add_argument("--grow", type=float, default=0.25, help="GT box margin (fraction of its size per side)")
    ap.add_argument("--random", action="store_true", help="control: random boxes with the mined sizes and counts")
    args = ap.parse_args()
    with open(args.meta) as f:
        metas = [m for m in json.load(f) if m["split"] in ("Train", "Validation")]
    sources, info = [], {"min_score": args.min_score, "top": args.top, "grow": args.grow}
    if args.detr:
        sources.append(mine(load_predictions(args.detr, metas), metas, args.min_score, args.top, grow=args.grow))
        info["detr"] = [os.path.abspath(p) for p in args.detr]
    if args.d2:
        sources.append(mine(d2_predictions(args.d2, metas), metas, args.min_score, args.top, grow=args.grow))
        info["d2"] = [os.path.abspath(p) for p in args.d2]
    assert sources, "give --detr and/or --d2"
    for name, src in zip([k for k in ("detr", "d2") if k in info], sources):
        print(name, *summary(src), flush=True)
    negs = sources[0] if len(sources) == 1 else union(*sources, top=2 * args.top)
    if args.random:
        negs, info["random"] = randomize(negs, metas, grow=args.grow), True
    print("out", *summary(negs), flush=True)
    save_negatives(args.out, negs, info)


if __name__ == "__main__":
    main()
