"""Train the cascade's ROI verifier + refiner (``medical_one.cascade.verifier``) on Train only.

  python scripts/cascade/train_verifier.py --prep-dir PREP --oof f0/preds_oof.json ... --weights boost_weights.json \
      --out OUT [--seed 0 --epochs 40]

Training candidates per Train case and epoch: the case's out-of-fold detector candidates (top ``--top`` of the inner-
fold DETRs' ``preds_oof.json``), ``--jitter`` jittered GT boxes and ``--random`` random boxes. AdaBoost-style weights:
case weight (``boost_weights.json``) x clip(exp(|hit - detector score|), 0.5, 3) for detector candidates (the first
stage's out-of-fold errors count more); jittered / random boxes get the case weight.

Stacking (the boosting step): the verifier is first trained on the inner folds of Train (same folds as the DETRs);
its out-of-fold outputs on the held-out fold's detector candidates give, after refinement, the features
[logit(detector score), logit(verifier hit probability)] and the label hit(refined box); a 2-feature logistic
regression on them is the final score (``stack.json``). Then one verifier is trained on all of Train
(``verifier.pt``). Validation / Test are not touched.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from medical_one.cascade.verifier import (Verifier3D, apply_deltas, box_scalars, extract_rois, jitter_boxes,
                                          labels_for, random_boxes, refine, target_deltas, verifier_loss)
from medical_one.data.dataset import normalize
from medical_one.data.tdsc import inner_split, load_meta, load_volume
from medical_one.evaluation.preds import load_predictions


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50):
    """L2-regularised logistic regression by Newton's method (intercept not penalised)."""
    A = np.concatenate([X, np.ones((len(X), 1))], 1)
    w = np.zeros(A.shape[1])
    reg = np.full(A.shape[1], l2)
    reg[-1] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-A @ w))
        g = A.T @ (p - y) + reg * w
        H = (A * (p * (1 - p))[:, None]).T @ A + np.diag(reg) + 1e-9 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w[:-1], w[-1]


def flip(vol: torch.Tensor, boxes: torch.Tensor, axes):
    for a in axes:
        vol = vol.flip(a)
        n = vol.shape[a]
        boxes = boxes.clone()
        boxes[:, [a, a + 3]] = n - boxes[:, [a + 3, a]]
    return vol, boxes


def train(cases, args, dev, log, tag):
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    random.seed(args.seed)
    model = Verifier3D().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps = args.epochs * len(cases)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    probs = np.array([c["w"] for c in cases], np.float64)
    probs /= probs.sum()
    model.train()
    t0, step = time.time(), 0
    for ep in range(args.epochs):
        agg = []
        for _ in range(len(cases)):
            c = cases[rng.choice(len(cases), p=probs)]  # AdaBoost case weights as sampling probabilities
            gt = c["gt"]
            boxes = torch.cat([c["cand"], jitter_boxes(gt, args.jitter, rng), random_boxes(gt, args.random, rng)])
            sw = torch.cat([c["cand_w"], torch.ones(args.jitter + args.random)])
            vol = c["vol"].to(dev).float()
            axes = [a for a in (0, 2) if rng.rand() < 0.5]
            vol, boxes_f = flip(vol, torch.cat([boxes, gt[None]]), axes)
            boxes, gt_f = boxes_f[:-1], boxes_f[-1]
            hit, iou = labels_for(boxes, gt_f)
            rois = extract_rois(normalize(vol), boxes)
            g = float(np.exp(rng.uniform(-0.3, 0.3)))
            rois = ((rois + 1) / 2).clamp(1e-6, 1) ** g * 2 - 1
            rois = rois * float(rng.uniform(0.85, 1.15)) + float(rng.uniform(-0.1, 0.1)) + 0.05 * torch.randn_like(rois)
            out = model(rois, box_scalars(boxes).to(dev))
            loss, parts = verifier_loss(out, hit.to(dev), iou.to(dev), target_deltas(boxes, gt_f).to(dev),
                                        torch.full_like(hit, c["malig"]).to(dev), sw.to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            agg.append([parts["hit"], parts["box"], parts["malig"]])
        if (ep + 1) % 5 == 0 or ep == 0:
            m = np.mean(agg, 0)
            log({"tag": tag, "epoch": ep + 1, "hit": round(m[0], 4), "box": round(m[1], 4), "malig": round(m[2], 4),
                 "sec": round(time.time() - t0, 1)})
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--oof", nargs="+", required=True)
    ap.add_argument("--weights", default="", help="boost_weights.json (case weights); empty = uniform")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--jitter", type=int, default=16)
    ap.add_argument("--random", type=int, default=8)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--folds", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    logf = open(os.path.join(args.out, "log.jsonl"), "a")

    def log(rec):
        print(json.dumps(rec), flush=True)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

    train_metas = load_meta(args.prep_dir)["Train"]
    preds = load_predictions(args.oof, train_metas)
    cw = {}
    if args.weights:
        with open(args.weights) as f:
            cw = {(c["split"], c["case"]): c["weight"] for c in json.load(f)["cases"]}
    cases = []
    for m, (b, s) in zip(train_metas, preds):
        o = s.argsort(descending=True)[:args.top]
        b, s = b[o].float(), s[o].float()
        gt = torch.tensor(m["box"], dtype=torch.float32)
        hit, _ = labels_for(b, gt)
        cand_w = (torch.exp((hit - s).abs())).clamp(0.5, 3.0)
        cases.append({"meta": m, "vol": torch.from_numpy(load_volume(args.prep_dir, m)), "gt": gt, "cand": b,
                      "score": s, "cand_w": cand_w / cand_w.mean() if len(cand_w) else cand_w,
                      "w": cw.get((m["split"], m["case"]), 1.0), "malig": float(m["label"] == "M")})
    log({"cases": len(cases), "candidates": int(sum(len(c["cand"]) for c in cases)),
         "cand_hits": int(sum(labels_for(c["cand"], c["gt"])[0].sum() for c in cases)), "args": vars(args)})

    # inner folds -> out-of-fold verifier features for the stacking weights
    feats, labels, base_hits = [], [], []
    for k in range(args.folds):
        tr, ho = inner_split([c["meta"] for c in cases], args.folds, k)
        keep = {(m["split"], m["case"]) for m in tr}
        model = train([c for c in cases if (c["meta"]["split"], c["meta"]["case"]) in keep], args, dev, log, f"fold{k}")
        for c in cases:
            if (c["meta"]["split"], c["meta"]["case"]) in keep or not len(c["cand"]):
                continue
            boxes, p, _ = refine(model, normalize(c["vol"].float()), c["cand"], iters=args.iters)
            hit, _ = labels_for(boxes, c["gt"])
            feats.append(np.stack([logit(c["score"].numpy()), logit(p.numpy())], 1))
            labels.append(hit.numpy())
            base_hits.append(labels_for(c["cand"], c["gt"])[0].numpy())
    X, y = np.concatenate(feats), np.concatenate(labels)
    coef, intercept = fit_logistic(X, y)
    stack = {"coef": np.round(coef, 5).tolist(), "intercept": round(float(intercept), 5),
             "oof_candidates": int(len(y)), "hit_rate_before_refine": round(float(np.concatenate(base_hits).mean()), 4),
             "hit_rate_after_refine": round(float(y.mean()), 4)}
    json.dump(stack, open(os.path.join(args.out, "stack.json"), "w"), indent=1)
    log({"stack": stack})

    model = train(cases, args, dev, log, "full")
    torch.save({"model": model.state_dict(), "args": vars(args), "stack": stack}, os.path.join(args.out, "verifier.pt"))
    log({"done": True})


if __name__ == "__main__":
    main()
