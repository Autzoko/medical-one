"""Evaluate a checkpoint on a TDSC split with whole-volume sliding-window inference."""
import argparse
import json

import torch

from medical_one.config import load_config
from medical_one.data.tdsc import load_meta
from medical_one.engine import build_model, evaluate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--config", default="configs/abus3d_tdsc.yaml")
    ap.add_argument("--split", default="Validation", choices=["Train", "Validation", "Test"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("overrides", nargs="*", help="key.sub=value")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    model = build_model(cfg).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device)["model"])
    res = evaluate(model, cfg, load_meta(cfg.paths.prep_dir)[args.split], args.device)
    print(json.dumps({"split": args.split, **res}))


if __name__ == "__main__":
    main()
