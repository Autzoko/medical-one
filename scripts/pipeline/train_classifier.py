"""Train the lesion classifier of the deployed pipeline: ResNet-34 on 2.5D lesion ROIs (benign / malignant).

  python scripts/pipeline/train_classifier.py --name cls_r34 --raw RAW [--epochs 30]

The ROI is built exactly as the deployed stage (``pipeline.native.roi_image``: center slice of the box with
its two neighbours as channels, +32 px context, zero-padded square, 224 x 224, (x/255 - 0.5)/0.5). The
deployed head has 3 classes of unknown meaning; TDSC only has benign / malignant, so the head here has 2.
Training ROIs come from ground-truth boxes with random shifts (+-15% of the box size per axis) and scales
(0.8-1.25), which mimic detection error; ImageNet initialization is our choice. Selection: case-level
ROC AUC on Validation with the exact ground-truth ROI. Writes best.pt and per-case Validation/Test
probabilities for the ground-truth ROIs.
"""
import argparse
import json
import math
import os
import random
from multiprocessing import Pool

import nrrd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import roc_auc_score

from medical_one.pipeline.native import native_view, roi_image

JITTERS = 48


def gt_box(raw_mask: np.ndarray) -> np.ndarray:
    nz = np.nonzero(raw_mask)
    lo = [int(a.min()) for a in nz]
    hi = [int(a.max()) + 1 for a in nz]
    n0 = raw_mask.shape[0]
    return np.array([lo[1], n0 - hi[0], lo[2], hi[1], n0 - lo[0], hi[2]], np.float64)


def jitter(box: np.ndarray, rng: random.Random) -> np.ndarray:
    c, s = (box[:3] + box[3:]) / 2, box[3:] - box[:3]
    c = c + np.array([rng.uniform(-0.15, 0.15) for _ in range(3)]) * s
    s = s * np.array([rng.uniform(0.8, 1.25) for _ in range(3)])
    return np.concatenate([c - s / 2, c + s / 2])


def load_case(job):
    raw, split, case, label, n_jitter, seed = job
    base = os.path.join(raw, split)
    v = native_view(nrrd.read(os.path.join(base, "DATA", f"DATA_{case:03d}.nrrd"))[0])
    box = gt_box(nrrd.read(os.path.join(base, "MASK", f"MASK_{case:03d}.nrrd"))[0] > 0)
    rng = random.Random(seed)
    rois = [roi_image(v, box)] + [roi_image(v, jitter(box, rng)) for _ in range(n_jitter)]
    return split, case, int(label == "M"), np.stack(rois), box.tolist()


def to_input(rois: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(rois).permute(0, 3, 1, 2).float() / 255
    return (x - 0.5) / 0.5


def build_model(pretrained: bool = True) -> nn.Module:
    weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.resnet34(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


@torch.no_grad()
def predict(model, rois: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    out = []
    for b in range(0, len(rois), 64):
        logits = model(to_input(rois[b:b + 64]).to(device)).float()
        out.append(torch.softmax(logits, 1)[:, 1].cpu())
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--runs-dir", default=os.environ.get("RUNS_DIR", ""))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_dir = os.path.join(args.runs_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)
    jobs = []
    for split in ("Train", "Validation", "Test"):
        df = pd.read_csv(os.path.join(args.raw, split, "labels.csv"))
        n = JITTERS if split == "Train" else 0
        jobs += [(args.raw, split, int(r.case_id), r.label, n, int(r.case_id)) for r in df.itertuples()]
    with Pool(args.workers) as pool:
        data = pool.map(load_case, jobs, chunksize=1)
    train = [d for d in data if d[0] == "Train"]
    x_train = np.concatenate([d[3] for d in train])
    y_train = np.concatenate([[d[2]] * len(d[3]) for d in train])
    val = [d for d in data if d[0] == "Validation"]
    test = [d for d in data if d[0] == "Test"]
    print(f"train ROIs {len(x_train)} ({y_train.mean():.2f} malignant), val {len(val)}, test {len(test)}", flush=True)

    model = build_model().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(x_train) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
    loss_fn = nn.CrossEntropyLoss()
    best, log = -1.0, open(os.path.join(run_dir, "log.jsonl"), "a")
    x_val = np.stack([d[3][0] for d in val])
    y_val = np.array([d[2] for d in val])
    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(x_train))
        total = 0.0
        for b in range(0, len(perm), args.batch):
            idx = perm[b:b + args.batch]
            x = to_input(x_train[idx])
            flip = torch.rand(len(idx)) < 0.5
            x[flip] = x[flip].flip(-1)  # columns = w, a lateral axis
            x = x * torch.empty(len(idx), 1, 1, 1).uniform_(0.85, 1.15) + torch.empty(len(idx), 1, 1, 1).uniform_(-0.1, 0.1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x.to(args.device))
            loss = loss_fn(logits.float(), torch.from_numpy(y_train[idx]).to(args.device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        auc = roc_auc_score(y_val, predict(model, x_val, args.device))
        rec = {"epoch": epoch + 1, "loss": round(total / len(perm), 4), "val_auc": round(float(auc), 4)}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if auc > best:
            best = auc
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val_auc": auc}, os.path.join(run_dir, "best.pt"))
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), weights_only=False)["model"])
    out = {}
    for name, ds in (("Validation", val), ("Test", test)):
        p = predict(model, np.stack([d[3][0] for d in ds]), args.device)
        out[name] = [{"case": d[1], "malignant": d[2], "p_malignant": float(q), "v_box": d[4]} for d, q in zip(ds, p)]
        print(name, "GT-ROI AUC", round(float(roc_auc_score([d[2] for d in ds], p)), 4), flush=True)
    with open(os.path.join(run_dir, "gt_roi_probs.json"), "w") as f:
        json.dump(out, f)
    print("done", flush=True)


if __name__ == "__main__":
    main()
