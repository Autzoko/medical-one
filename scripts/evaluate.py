"""Evaluate one checkpoint, or an ensemble of several, on a TDSC split with sliding-window inference.

Flip TTA, the WBF IoU and the sliding-window post-processing come from the config, e.g.
``--set eval.tta_flips=true eval.wbf_iou=0.5``. ``--save-preds`` writes the per-case working-space
boxes and scores so metrics, ensembles and figures can be recomputed offline (``score_preds.py``).
"""
import argparse
import json
import os

import torch

from medical_one.config import load_config
from medical_one.data.tdsc import cv_split, load_meta
from medical_one.engine import build_model, predict_split
from medical_one.evaluation import compute_metrics
from medical_one.evaluation.preds import save_predictions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoints", nargs="+", help="one or more .pt files (several = ensemble)")
    ap.add_argument("--config", default="configs/abus3d_tdsc.yaml")
    ap.add_argument("--split", default="Validation", choices=["Train", "Validation", "Test", "OOF"],
                    help="OOF: the held-out cases of fold --cv-fold of the Train + Validation CV")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--cv-fold", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-preds", default="", help="write predictions to this .json")
    ap.add_argument("--set", dest="overrides", nargs="*", action="extend", default=[],
                    help="key.sub=value (may be repeated)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    models = []
    for path in args.checkpoints:
        model = build_model(cfg).to(args.device)
        model.load_state_dict(torch.load(path, map_location=args.device, weights_only=False)["model"])
        models.append(model)
    splits = load_meta(cfg.paths.prep_dir)
    metas = cv_split(splits, args.cv_folds, args.cv_fold)[1] if args.split == "OOF" else splits[args.split]
    preds = predict_split(models, cfg, metas, args.device)
    res = compute_metrics(preds, metas, cfg.eval.iou_thrs)
    if args.save_preds:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_preds)), exist_ok=True)
        save_predictions(args.save_preds, preds, metas, {"checkpoints": args.checkpoints, "overrides": args.overrides})
    print(json.dumps({"split": args.split, "checkpoints": args.checkpoints, "tta_flips": cfg.eval.tta_flips, **res}))


if __name__ == "__main__":
    main()
