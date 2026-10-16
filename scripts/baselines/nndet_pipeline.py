"""nnDetection (Retina U-Net, default RetinaUNetV001 / D3V001 plan) baseline on the same working-space volumes.

Steps (each skipped when its output exists, so a rerun or a follow-up job resumes):
  1. setup    clone nnDetection at NNDET_COMMIT (run through nndet_launch.py, which holds the torch 2 shims).
  2. export   working-space volumes and lesion masks as NIfTI (det_data/Task101_TDSC/raw_splitted) with the
              real per-case voxel spacing; each mask is one instance of class 0. Train + Validation are
              training cases; Validation + Test go to imagesTs for prediction.
  3. prep     nndet_prep (crop, analyse, plan, preprocess), then splits_final.pkl with one fold:
              train = Train, val = Validation (the split every model here uses), then nndet_unpack.
  4. train    nndet_train -o exp.fold=0 --sweep: training (resumed from model_last.ckpt when present), then
              nnDetection's empirical post-processing sweep on the validation fold.
  5. predict  nndet_predict --fold 0 (default test-time mirroring) for Validation and Test.
  6. boxes    nnDetection boxes -> preds_<split>.json in working-space corners.
Preprocessed data live on node-local disk; models and predictions go to <runs>/<name>/.
"""
import argparse
import json
import os
import pickle
import subprocess
import sys

import numpy as np
import torch

from medical_one.data.tdsc import load_meta
from medical_one.evaluation.preds import save_predictions

NNDET_REPO = "https://github.com/MIC-DKFZ/nnDetection.git"
NNDET_COMMIT = "97a58f3110b71caf1b4bcc1851e67cf11e987fc5"
TASK_ID, TASK = "101", "Task101_TDSC"
MODEL = "RetinaUNetV001_D3V001_3d"
RAW_SPACING = (0.073, 0.2, 0.476)  # mm, raw axes N0 (depth), N1, N2
LAUNCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nndet_launch.py")


def working_spacing(shape):
    """Working axes are (N1, flipped N0, N2) resized to RESIZE."""
    from medical_one.data.tdsc import RESIZE

    n0, n1, n2 = shape
    return (n1 * RAW_SPACING[1] / RESIZE[0], n0 * RAW_SPACING[0] / RESIZE[1], n2 * RAW_SPACING[2] / RESIZE[2])


def case_id(meta):
    return f"TDSC_{meta['case']:03d}"


def from_nndet_boxes(boxes: np.ndarray) -> np.ndarray:
    """nnDetection boxes (x1, y1, x2, y2, z1, z2) on array axes (0, 1, 2), with ground truth written as
    [min - 1, max + 1] of the voxel indices, -> half-open working-space corners (d0, h0, w0, d1, h1, w1)."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 6)
    return np.stack([b[:, 0] + 1, b[:, 1] + 1, b[:, 4] + 1, b[:, 2], b[:, 3], b[:, 5]], 1)


def write_nifti(arr, spacing, path):
    """SimpleITK keeps the numpy axis order as nnDetection's array order; its spacing is listed reversed."""
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(tuple(float(s) for s in reversed(spacing)))
    sitk.WriteImage(img, path)


def export(task_dir, prep_dir, masks_dir, splits):
    raw = os.path.join(task_dir, "raw_splitted")
    for sub in ("imagesTr", "labelsTr", "imagesTs"):
        os.makedirs(os.path.join(raw, sub), exist_ok=True)
    for split in ("Train", "Validation", "Test"):
        for m in splits[split]:
            vol = np.load(os.path.join(prep_dir, split, f"{m['case']:03d}.npy"))
            sp = working_spacing(m["shape"])
            if split != "Test":
                write_nifti(vol, sp, os.path.join(raw, "imagesTr", f"{case_id(m)}_0000.nii.gz"))
                mask = (np.load(os.path.join(masks_dir, split, f"{m['case']:03d}.npy")) > 0).astype(np.uint8)
                assert mask.shape == vol.shape and mask.any(), (m["case"], mask.shape, vol.shape)
                write_nifti(mask, sp, os.path.join(raw, "labelsTr", f"{case_id(m)}.nii.gz"))
                with open(os.path.join(raw, "labelsTr", f"{case_id(m)}.json"), "w") as f:
                    json.dump({"instances": {"1": 0}}, f)
            if split != "Train":
                write_nifti(vol, sp, os.path.join(raw, "imagesTs", f"{case_id(m)}_0000.nii.gz"))
    with open(os.path.join(task_dir, "dataset.json"), "w") as f:
        json.dump({"task": TASK, "name": "TDSC", "target_class": None, "test_labels": False,
                   "labels": {"0": "lesion"}, "modalities": {"0": "US"}, "dim": 3}, f, indent=1)


def check_box_convention(task_dir, metas):
    """nnDetection's own ground-truth boxes of the exported labels must map back onto the mask extents."""
    import SimpleITK as sitk

    from nndet.io.transforms.instances import instances_to_boxes_np

    for m in metas:
        seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(task_dir, "raw_splitted", "labelsTr",
                                                                  f"{case_id(m)}.nii.gz")))
        nz = np.nonzero(seg)
        ref = [a.min() for a in nz] + [a.max() + 1 for a in nz]
        got = from_nndet_boxes(instances_to_boxes_np(seg, dim=3)[0])[0]
        assert np.allclose(got, ref), (m["case"], got, ref)


def run(cmd, env):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=50, help="nnDetection default; 10 SWA epochs follow")
    ap.add_argument("--batches", type=int, default=2500, help="train batches per epoch (nnDetection default)")
    ap.add_argument("--prep-dir", default=os.environ.get("PREP_DIR", ""))
    ap.add_argument("--runs-dir", default=os.environ.get("RUNS_DIR", ""))
    ap.add_argument("--masks-dir", required=True, help="working-space masks (preprocess_tdsc.py --masks-only)")
    ap.add_argument("--local-dir", default=os.environ.get("LOCAL_DIR", "/tmp"))
    ap.add_argument("--nndet-dir", default="", help="existing nnDetection checkout (default: clone into local dir)")
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--threads", type=int, default=12, help="det_num_threads (augmentation workers)")
    ap.add_argument("--train-overrides", nargs="*", default=[], help="extra nnDetection config overrides")
    args = ap.parse_args()
    run_dir = os.path.abspath(os.path.join(args.runs_dir, args.name))
    os.makedirs(run_dir, exist_ok=True)
    splits = load_meta(args.prep_dir)
    root = args.nndet_dir or os.path.join(args.local_dir, "nnDetection")
    if not os.path.isdir(root):
        subprocess.run(["git", "clone", "-q", NNDET_REPO, root], check=True)
        subprocess.run(["git", "-C", root, "checkout", "-q", NNDET_COMMIT], check=True)
    sys.path.insert(0, os.path.dirname(LAUNCH))
    from nndet_launch import patch_repo

    patch_repo(root)
    sys.path.insert(0, root)
    data = os.path.join(args.local_dir, "nndet_data")
    models = os.path.join(run_dir, "nndet_models")
    env = dict(os.environ, det_data=data, det_models=models, OMP_NUM_THREADS="1",
               det_num_threads=str(args.threads), det_verbose="0")
    os.environ.update(env)
    launch = [sys.executable, LAUNCH, root]
    task_dir = os.path.join(data, TASK)
    fold_dir = os.path.join(models, TASK, MODEL, "fold0")
    pred_dir = os.path.join(fold_dir, "test_predictions")
    eval_metas = splits["Validation"] + splits["Test"]
    done = lambda: all(os.path.exists(os.path.join(pred_dir, f"{case_id(m)}_boxes.pkl")) for m in eval_metas)

    if not done():
        export(task_dir, args.prep_dir, args.masks_dir, splits)
        check_box_convention(task_dir, splits["Train"] + splits["Validation"])
        run(launch + ["prep", TASK_ID, "-np", str(args.processes), "-npp", str(args.processes)], env)
        prep_out = os.path.join(task_dir, "preprocessed")
        with open(os.path.join(prep_out, "splits_final.pkl"), "wb") as f:
            pickle.dump([{"train": [case_id(m) for m in splits["Train"]],
                          "val": [case_id(m) for m in splits["Validation"]]}], f)
        run(launch + ["unpack", os.path.join(prep_out, "D3V001_3d", "imagesTr"), str(args.processes)], env)
        if not os.path.exists(os.path.join(fold_dir, "plan_inference.pkl")):
            mode = "resume" if os.path.exists(os.path.join(fold_dir, "model_last.ckpt")) else "overwrite"
            run(launch + ["train", TASK_ID, "--sweep", "-o", "exp.fold=0", f"train.mode={mode}",
                          f"trainer_cfg.max_num_epochs={args.epochs}",
                          f"trainer_cfg.num_train_batches_per_epoch={args.batches}"] + args.train_overrides, env)
        run(launch + ["predict", TASK_ID, MODEL, "--fold", "0", "-npp", str(args.processes)], env)

    for split in ("Validation", "Test"):
        preds = []
        for m in splits[split]:
            with open(os.path.join(pred_dir, f"{case_id(m)}_boxes.pkl"), "rb") as f:
                p = pickle.load(f)
            boxes, scores = from_nndet_boxes(p["pred_boxes"]), np.asarray(p["pred_scores"], np.float64)
            order = np.argsort(-scores)[:100]
            preds.append((torch.from_numpy(boxes[order]).float(), torch.from_numpy(scores[order]).float()))
        save_predictions(os.path.join(run_dir, f"preds_{split}.json"), preds, splits[split],
                         {"model": f"nndetection_{MODEL}", "commit": NNDET_COMMIT, "epochs": args.epochs,
                          "batches_per_epoch": args.batches})
    print("done", flush=True)


if __name__ == "__main__":
    main()
