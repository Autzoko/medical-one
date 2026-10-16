"""Train a MONAI SwinUNETR lesion segmenter and evaluate it as a detector (connected components -> boxes).

Same working-space volumes, Train/Validation split, patch sampling and augmentation as the detectors;
DiceCE loss, AdamW, warmup + cosine. Selection on Validation uses the official TDSC score of the default
box variant (threshold 0.5, max probability). At the end, predictions of best.pt for Validation and Test
are written for every box variant (preds_<split>_<variant>.json); the variant is chosen on Validation.
--predict-only skips training and only writes these predictions from an existing best.pt.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from medical_one.baselines.seg_to_det import prob_to_boxes
from medical_one.baselines.segmentation import SegPatchDataset, predict_prob
from medical_one.data.tdsc import load_meta, load_volume
from medical_one.evaluation import compute_metrics
from medical_one.evaluation.preds import save_predictions

PATCH = (160, 96, 160)  # SwinUNETR needs multiples of 32
VARIANTS = {"t0.5_max": (0.5, "max"), "t0.3_max": (0.3, "max"), "t0.1_max": (0.1, "max"), "t0.5_mean": (0.5, "mean")}


def build_swinunetr(feature_size: int = 48):
    from monai.networks.nets import SwinUNETR

    try:
        return SwinUNETR(in_channels=1, out_channels=2, feature_size=feature_size, use_checkpoint=True)
    except TypeError:  # MONAI < 1.5 also requires img_size
        return SwinUNETR(img_size=PATCH, in_channels=1, out_channels=2, feature_size=feature_size,
                         use_checkpoint=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--prep-dir", default=os.environ.get("PREP_DIR", ""))
    ap.add_argument("--runs-dir", default=os.environ.get("RUNS_DIR", ""))
    ap.add_argument("--masks-dir", required=True)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--feature-size", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--predict-only", action="store_true")
    args = ap.parse_args()
    from monai.losses import DiceCELoss

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_dir = os.path.join(args.runs_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)
    if not args.predict_only:
        with open(os.path.join(run_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=1)
    log = open(os.path.join(run_dir, "log.jsonl"), "a")

    def write(record, prefix=""):
        print(prefix + json.dumps(record), flush=True)
        log.write(json.dumps(record) + "\n")
        log.flush()

    def predict_split(metas):
        return {m["case"]: predict_prob(model, load_volume(args.prep_dir, m), PATCH, args.device) for m in metas}

    splits = load_meta(args.prep_dir)
    model = build_swinunetr(args.feature_size).to(args.device)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    if not args.predict_only:
        loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
        dataset = SegPatchDataset(args.prep_dir, args.masks_dir, splits["Train"], args.iters * args.batch_size, PATCH)
        loader = torch.utils.data.DataLoader(dataset, args.batch_size, num_workers=args.workers,
                                             persistent_workers=args.workers > 0)
        best, start = -1.0, time.time()
        model.train()
        for it, (x, y) in enumerate(loader, 1):
            factor = min(1.0, it / args.warmup) * 0.5 * (1 + math.cos(math.pi * it / args.iters))
            for g in opt.param_groups:
                g["lr"] = args.lr * factor
            x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if it % 50 == 0:
                write({"it": it, "loss": round(loss.item(), 4), "lr": opt.param_groups[0]["lr"],
                       "sec": round(time.time() - start, 1)})
            if it % args.eval_every == 0 or it == args.iters:
                probs = predict_split(splits["Validation"])
                preds = [prob_to_boxes(probs[m["case"]], 0.5, "max") for m in splits["Validation"]]
                res = compute_metrics(preds, splits["Validation"])
                res["it"] = it
                write({"eval": res}, prefix="EVAL ")
                torch.save({"model": model.state_dict(), "it": it}, os.path.join(run_dir, "last.pt"))
                if res["tdsc_froc"] > best:
                    best = res["tdsc_froc"]
                    torch.save({"model": model.state_dict(), "it": it, "val": res}, os.path.join(run_dir, "best.pt"))
                model.train()
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), weights_only=False)["model"])
    for split in ("Validation", "Test"):
        probs = predict_split(splits[split])
        for variant, (thr, score) in VARIANTS.items():
            preds = [prob_to_boxes(probs[m["case"]], thr, score) for m in splits[split]]
            if split == "Validation":
                print(variant, json.dumps({k: compute_metrics(preds, splits[split])[k]
                                           for k in ("tdsc_froc", "ap@0.1", "top1_hit@0.3")}), flush=True)
            save_predictions(os.path.join(run_dir, f"preds_{split}_{variant}.json"), preds, splits[split],
                             {"model": "monai_swinunetr", "checkpoint": "best.pt", "threshold": thr, "score": score})
    print("done", flush=True)


if __name__ == "__main__":
    main()
