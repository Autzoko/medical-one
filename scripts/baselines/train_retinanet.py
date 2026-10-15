"""Train the MONAI 3D RetinaNet baseline on the same patches as the DETR model; select on Validation.

Writes <runs>/<name>/{log.jsonl,best.pt,last.pt} and, at the end, predictions of best.pt for Validation and
Test (preds_<split>.json, same format as evaluate.py --save-preds). Optimizer and sampler follow the MONAI
LUNA16 tutorial (SGD + Nesterov, lr 1e-2, ATSS, hard negative sampling); the schedule is warmup + cosine.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from medical_one.baselines.retinanet import build_retinanet, predict_volume_retinanet, to_targets
from medical_one.data.dataset import PATCH, PatchDataset, collate
from medical_one.data.tdsc import load_meta, load_volume
from medical_one.evaluation import compute_metrics
from medical_one.evaluation.preds import save_predictions


def predict(detector, prep_dir, metas, device):
    return [predict_volume_retinanet(detector, load_volume(prep_dir, m), device) for m in metas]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--prep-dir", default=os.environ.get("PREP_DIR", ""))
    ap.add_argument("--runs-dir", default=os.environ.get("RUNS_DIR", ""))
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--pos-prob", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_dir = os.path.join(args.runs_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    log = open(os.path.join(run_dir, "log.jsonl"), "a")

    def write(record, prefix=""):
        print(prefix + json.dumps(record), flush=True)
        log.write(json.dumps(record) + "\n")
        log.flush()

    splits = load_meta(args.prep_dir)
    detector = build_retinanet().to(args.device)
    params = [p for p in detector.network.parameters() if p.requires_grad]
    print(f"params {sum(p.numel() for p in params) / 1e6:.2f}M", flush=True)
    opt = torch.optim.SGD(params, args.lr, momentum=0.9, weight_decay=3e-5, nesterov=True)
    dataset = PatchDataset(args.prep_dir, splits["Train"], args.iters * args.batch_size, args.pos_prob)
    loader = torch.utils.data.DataLoader(dataset, args.batch_size, num_workers=args.workers, collate_fn=collate,
                                         persistent_workers=args.workers > 0)
    best, start = -1.0, time.time()
    detector.train()
    for it, (x, targets) in enumerate(loader, 1):
        factor = min(1.0, it / args.warmup) * 0.5 * (1 + math.cos(math.pi * it / args.iters))
        for g in opt.param_groups:
            g["lr"] = args.lr * factor
        x = x.to(args.device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = detector(x, to_targets(targets, PATCH, args.device))
        loss = losses[detector.cls_key] + losses[detector.box_reg_key]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 50 == 0:
            write({"it": it, "loss": round(loss.item(), 4), "cls": round(losses[detector.cls_key].item(), 4),
                   "box": round(losses[detector.box_reg_key].item(), 4), "lr": opt.param_groups[0]["lr"],
                   "sec": round(time.time() - start, 1)})
        if it % args.eval_every == 0 or it == args.iters:
            res = compute_metrics(predict(detector, args.prep_dir, splits["Validation"], args.device),
                                  splits["Validation"])
            res["it"] = it
            write({"eval": res}, prefix="EVAL ")
            torch.save({"model": detector.network.state_dict(), "it": it}, os.path.join(run_dir, "last.pt"))
            if res["tdsc_froc"] > best:
                best = res["tdsc_froc"]
                torch.save({"model": detector.network.state_dict(), "it": it, "val": res},
                           os.path.join(run_dir, "best.pt"))
            detector.train()
    detector.network.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), weights_only=False)["model"])
    for split in ("Validation", "Test"):
        preds = predict(detector, args.prep_dir, splits[split], args.device)
        save_predictions(os.path.join(run_dir, f"preds_{split}.json"), preds, splits[split],
                         {"model": "monai_retinanet", "checkpoint": "best.pt"})
    print("done", flush=True)


if __name__ == "__main__":
    main()
