"""Evaluate one checkpoint, or an ensemble of several, on a TDSC split with sliding-window inference.

Flip TTA and the WBF IoU come from the config, e.g. ``eval.tta_flips=true eval.wbf_iou=0.5``.
"""
import argparse
import json

import torch

from medical_one.config import load_config
from medical_one.data.tdsc import load_meta
from medical_one.engine import build_model, evaluate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoints", nargs="+", help="one or more .pt files (several = ensemble)")
    ap.add_argument("--config", default="configs/abus3d_tdsc.yaml")
    ap.add_argument("--split", default="Validation", choices=["Train", "Validation", "Test"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[], help="key.sub=value")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    models = []
    for path in args.checkpoints:
        model = build_model(cfg).to(args.device)
        model.load_state_dict(torch.load(path, map_location=args.device, weights_only=False)["model"])
        models.append(model)
    res = evaluate(models, cfg, load_meta(cfg.paths.prep_dir)[args.split], args.device)
    print(json.dumps({"split": args.split, "checkpoints": args.checkpoints, "tta_flips": cfg.eval.tta_flips, **res}))


if __name__ == "__main__":
    main()
