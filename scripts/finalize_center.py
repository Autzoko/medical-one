"""After training a CenterDETR3D / DenseDet3D / CoDETR3D run: the pre-registered post-processing grid on Validation,
plus Test predictions for every grid cell (saved, NOT scored here; Test is scored once, for the cell chosen on
Validation).

Grids:
  center_detr (docs/prereg/center_detr_v1.md): checkpoint swa.pt (fallback best.pt) x output {E2 = expert 2,
    E1 = expert 1, E1+E2 = WBF IoU 0.5 with equal weights, fused score = mean so it stays in [0, 1]}
    x per-window NMS IoU {0.05, 0.1, 0.2};
  dense / co_detr (docs/prereg/dense_codetr.md): checkpoint best.pt x output {dense, dense_heat} / {detr, dense_heat}
    x per-window NMS IoU {0.1, 0.2}.
Writes <run>/final/val_grid.json and <run>/final/<cell>_{Validation,Test}.json.

  python scripts/finalize_center.py RUN_DIR --set paths.prep_dir=... [--device cuda]
"""
import argparse
import json
import os

import torch

from medical_one.config import Config, _update
from medical_one.data.tdsc import load_meta
from medical_one.engine import build_model, predict_split
from medical_one.evaluation import compute_metrics
from medical_one.evaluation.preds import fuse_predictions, save_predictions

GRIDS = {"center_detr": (("E2", "E1", "E1+E2"), (0.05, 0.1, 0.2), ("swa.pt", "best.pt")),
         "dense": (("dense", "dense_heat"), (0.1, 0.2), ("best.pt",)),
         "co_detr": (("detr", "dense_heat"), (0.1, 0.2), ("best.pt",))}


def _outputs(model, cfg, metas, device, arch, names):
    if arch == "center_detr":
        out = {"E2": predict_split(model, cfg, metas, device, head="detr"),
               "E1": predict_split(model, cfg, metas, device, head="center")}
        out["E1+E2"] = fuse_predictions([out["E2"], out["E1"]], 0.5, cfg.eval.max_dets)
        return out
    return {n: predict_split(model, cfg, metas, device, head=n) for n in names}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[], help="key.sub=value (e.g. paths)")
    ap.add_argument("--splits", nargs="+", default=["Validation", "Test"])
    args = ap.parse_args()
    with open(os.path.join(args.run_dir, "config.json")) as f:
        saved = json.load(f)
    cfg = Config()
    _update(cfg, saved)
    for item in args.overrides:
        key, _, raw = item.partition("=")
        node = {}
        cur = node
        for p in key.split(".")[:-1]:
            cur[p] = {}
            cur = cur[p]
        cur[key.split(".")[-1]] = raw
        _update(cfg, node)
    cfg.expand_paths()
    arch = cfg.model.arch
    outputs, nms_grid, ckpts = GRIDS[arch]
    ckpt = next(c for c in ckpts if os.path.exists(os.path.join(args.run_dir, c)))
    model = build_model(cfg).to(args.device)
    state = torch.load(os.path.join(args.run_dir, ckpt), map_location=args.device, weights_only=False)
    model.load_state_dict(state["model"])
    splits = load_meta(cfg.paths.prep_dir)
    out_dir = os.path.join(args.run_dir, "final")
    os.makedirs(out_dir, exist_ok=True)
    grid = {"checkpoint": ckpt, "it": state["it"], "cells": {}}
    for nms in nms_grid:
        cfg.eval.window_nms = nms
        for split in args.splits:
            metas = splits[split]
            heads = _outputs(model, cfg, metas, args.device, arch, outputs)
            for name in outputs:
                cell = f"{name}_nms{nms}"
                save_predictions(os.path.join(out_dir, f"{cell}_{split}.json"), heads[name], metas,
                                 {"checkpoint": ckpt, "it": state["it"], "output": name, "window_nms": nms})
                if split == "Validation":
                    r = compute_metrics(heads[name], metas, cfg.eval.iou_thrs)
                    grid["cells"][cell] = {k: r[k] for k in ("tdsc_froc", "tdsc_sens", "ap@0.3", "top1_hit@0.3",
                                                            "tdsc_max_recall")}
                    print(json.dumps({"cell": cell, "val_froc": round(r["tdsc_froc"], 4)}), flush=True)
    with open(os.path.join(out_dir, "val_grid.json"), "w") as f:
        json.dump(grid, f, indent=1)


if __name__ == "__main__":
    main()
