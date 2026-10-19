"""Train the 2D branch: Ultralytics YOLOv5 v7.0 yolov5m, one class, on 2.5D TDSC slices.

  python scripts/pipeline/train_yolo.py --name y5m --raw RAW [--epochs 60 --batch 16]

Matches the deployed model where it is known (audit: yolov5m.yaml depth/width, nc=1, the default
anchors, SiLU, SPPF, 960 long-side letterbox input): ``--noautoanchor`` keeps the default anchors.
Unknown in the original and chosen here: COCO-pretrained yolov5m.pt initialization, hyp.scratch-low with
the hue/saturation jitter off (channels are neighbouring slices, not colour) and no vertical flip (the
image rows are ultrasound depth). Data are built on node-local disk (``make_yolo_data.py``); weights,
results.csv and the options are copied to ``<runs>/<name>``.

Hard negatives (``--negatives``, optional): label rows of class 1 mark regions where an earlier model fired
falsely. The patched ``ComputeLoss`` removes them from the targets and adds ``hyp['neg']`` (``--neg-weight``)
x the objectness BCE (target 0) of the anchor cells YOLOv5 would assign to such a box. Without class-1 rows
the loss is unchanged.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

import yaml

YOLOV5_TAG = "v7.0"


NEG_EDITS = {  # hard-negative objectness penalty; inert without class-1 labels
    "utils/loss.py": [
        ("        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets\n",
         "        neg = targets[targets[:, 1] == 1]  # hard negatives (medical-one)\n"
         "        targets = targets[targets[:, 1] != 1]\n"
         "        neg_idx = self.build_targets(p, neg)[2] if len(neg) and self.hyp.get('neg', 0) > 0 else None\n"
         "        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets\n"),
        ("            obji = self.BCEobj(pi[..., 4], tobj)\n",
         "            obji = self.BCEobj(pi[..., 4], tobj)\n"
         "            if neg_idx is not None:  # hard negatives (medical-one)\n"
         "                nb, na_, nj, ni = neg_idx[i]\n"
         "                free = tobj[nb, na_, nj, ni] == 0\n"
         "                if free.any():\n"
         "                    obji = obji + self.hyp['neg'] * self.BCEobj(pi[nb, na_, nj, ni, 4][free], tobj[nb, na_, nj, ni][free])\n"),
    ],
    "train.py": [("    assert mlc < nc, f", "    assert mlc <= nc, f")],  # class nc (= 1) marks hard negatives
}


def patch_repo(repo: str) -> None:
    """Port v7.0 to the current stack without changing its behaviour: torch.cuda.amp.* -> torch.amp, and
    the numpy aliases removed in numpy 2 (np.trapz -> np.trapezoid, np.float -> float). Then the
    hard-negative edits (``NEG_EDITS``)."""
    for root, _, files in os.walk(repo):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            src = open(path).read()
            new = re.sub(r"(?:torch\.cuda\.)?\bamp\.autocast\(([^)=]+)\)", r'torch.amp.autocast("cuda", enabled=\1)', src)
            new = new.replace("torch.cuda.amp.GradScaler(", 'torch.amp.GradScaler("cuda", ')
            new = new.replace("np.trapz(", "np.trapezoid(").replace(".astype(np.float)", ".astype(float)")
            if new != src:
                open(path, "w").write(new)
    for rel, edits in NEG_EDITS.items():
        path = os.path.join(repo, rel)
        src = open(path).read()
        for old, new in edits:
            if new not in src:
                assert src.count(old) == 1, (rel, old)
                src = src.replace(old, new)
        open(path, "w").write(src)


def clone_yolov5(dest: str) -> str:
    if not os.path.isdir(dest):
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", YOLOV5_TAG,
                        "https://github.com/ultralytics/yolov5.git", dest], check=True)
        patch_repo(dest)
    return dest


def yolo_env() -> dict:
    env = dict(os.environ)
    env.update({"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "YOLOv5_AUTOINSTALL": "false"})
    return env


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--runs-dir", default=os.environ.get("RUNS_DIR", ""))
    ap.add_argument("--local", default=os.environ.get("LOCAL_DIR", "/tmp"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--img", type=int, default=960)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--weights", default="yolov5m.pt")
    ap.add_argument("--cache", default="ram")
    ap.add_argument("--cv-folds", type=int, default=0, help="> 0: train on the other folds, validate on --cv-fold")
    ap.add_argument("--cv-fold", type=int, default=0)
    ap.add_argument("--negatives", default="", help="working-space hard negatives JSON")
    ap.add_argument("--neg-weight", type=float, default=0.0, help="hyp['neg']; 0 = negative slices only")
    args = ap.parse_args()
    out_dir = os.path.join(args.runs_dir, args.name)
    os.makedirs(out_dir, exist_ok=True)
    data = os.path.join(args.local, "yolo_data")
    here = os.path.dirname(os.path.abspath(__file__))
    repo = clone_yolov5(os.path.join(args.local, "yolov5"))
    preflight = ("import numpy as np, train, val; from utils.metrics import ap_per_class; "
                 "ap_per_class(np.ones((4, 10), bool), np.random.rand(4), np.zeros(4), np.zeros(3), names={0: \"lesion\"})")
    subprocess.run([sys.executable, "-c", preflight], cwd=repo, env=yolo_env(), check=True)  # fail fast
    subprocess.run([sys.executable, os.path.join(here, "make_yolo_data.py"), "--raw", args.raw, "--out", data,
                    "--workers", str(args.workers), "--cv-folds", str(args.cv_folds), "--cv-fold", str(args.cv_fold),
                    "--negatives", args.negatives], check=True)
    with open(os.path.join(repo, "data", "hyps", "hyp.scratch-low.yaml")) as f:
        hyp = yaml.safe_load(f)
    hyp.update({"hsv_h": 0.0, "hsv_s": 0.0, "flipud": 0.0, "fliplr": 0.5})
    if args.neg_weight > 0:
        hyp["neg"] = args.neg_weight
    hyp_path = os.path.join(args.local, "hyp_tdsc.yaml")
    with open(hyp_path, "w") as f:
        yaml.safe_dump(hyp, f)
    project = os.path.join(args.local, "yolo_runs")
    cmd = [sys.executable, "train.py", "--img", str(args.img), "--batch", str(args.batch), "--epochs",
           str(args.epochs), "--data", os.path.join(data, "data.yaml"), "--cfg", "models/yolov5m.yaml",
           "--weights", args.weights, "--hyp", hyp_path, "--project", project, "--name", args.name,
           "--exist-ok", "--workers", str(args.workers), "--noautoanchor", "--noplots", "--seed", "0"]
    if args.cache:
        cmd += ["--cache", args.cache]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=repo, env=yolo_env(), check=True)
    run = os.path.join(project, args.name)
    for rel in ("weights/best.pt", "weights/last.pt", "results.csv", "opt.yaml", "hyp.yaml"):
        src = os.path.join(run, rel)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(out_dir, os.path.basename(rel)))  # CIFS refuses chmod
    shutil.copyfile(hyp_path, os.path.join(out_dir, "hyp_tdsc.yaml"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
