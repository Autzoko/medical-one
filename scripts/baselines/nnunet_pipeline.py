"""nnU-Net v2 (3d_fullres) baseline on the same working-space volumes, evaluated as a detector.

Steps (each skipped when its output exists, so a rerun resumes):
  1. export   working-space volumes + lesion masks as NIfTI (nnUNet_raw/Dataset501_TDSCABUS), with the
              real per-case voxel spacing; Train + Validation are training cases, Test goes to imagesTs.
  2. plan     nnUNetv2_plan_and_preprocess, then splits_final.json with one fold: train = Train,
              val = Validation (the same split every model here uses).
  3. train    nnUNetv2_train ... 0 -tr nnUNetTrainer_<E>epochs
  4. predict  nnUNetv2_predict --save_probabilities for Validation and Test (checkpoint_final.pth).
  5. boxes    connected components of the lesion probability -> preds_<split>_<variant>.json for several
              (threshold, score) variants; the variant is chosen on Validation.
Raw/preprocessed data live on node-local disk; results and predictions go to <runs>/<name>/.
"""
import argparse
import json
import os
import subprocess

import numpy as np

from medical_one.baselines.seg_to_det import prob_to_boxes
from medical_one.data.tdsc import KEEP_DEPTH, RESIZE, load_meta
from medical_one.evaluation.preds import save_predictions

DATASET = "Dataset501_TDSCABUS"
RAW_SPACING = (0.073, 0.2, 0.476)  # mm, raw axes N0 (depth), N1, N2
VARIANTS = {"t0.5_max": (0.5, "max"), "t0.3_max": (0.3, "max"), "t0.1_max": (0.1, "max"), "t0.5_mean": (0.5, "mean")}


def working_spacing(shape):
    """Working axes are (N1, flipped N0, N2) resized to RESIZE."""
    n0, n1, n2 = shape
    return (n1 * RAW_SPACING[1] / RESIZE[0], n0 * RAW_SPACING[0] / RESIZE[1], n2 * RAW_SPACING[2] / RESIZE[2])


def case_id(meta):
    return f"TDSC_{meta['case']:03d}"


def export(prep_dir, masks_dir, raw_dir, splits):
    import nibabel as nib

    base = os.path.join(raw_dir, DATASET)
    for sub in ("imagesTr", "labelsTr", "imagesVal", "imagesTs"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    groups = {"Train": ("imagesTr", True), "Validation": ("imagesTr", True), "Test": ("imagesTs", False)}
    for split, (sub, with_label) in groups.items():
        for m in splits[split]:
            affine = np.diag(list(working_spacing(m["shape"])) + [1.0])
            vol = np.load(os.path.join(prep_dir, split, f"{m['case']:03d}.npy"))
            nib.save(nib.Nifti1Image(vol, affine), os.path.join(base, sub, f"{case_id(m)}_0000.nii.gz"))
            if split == "Validation":  # also as a separate input folder for prediction
                nib.save(nib.Nifti1Image(vol, affine), os.path.join(base, "imagesVal", f"{case_id(m)}_0000.nii.gz"))
            if with_label:
                mask = np.load(os.path.join(masks_dir, split, f"{m['case']:03d}.npy")).astype(np.uint8)
                nib.save(nib.Nifti1Image(mask, affine), os.path.join(base, "labelsTr", f"{case_id(m)}.nii.gz"))
    with open(os.path.join(base, "dataset.json"), "w") as f:
        json.dump({"channel_names": {"0": "ABUS"}, "labels": {"background": 0, "lesion": 1},
                   "numTraining": len(splits["Train"]) + len(splits["Validation"]), "file_ending": ".nii.gz",
                   "overwrite_image_reader_writer": "NibabelIO"}, f, indent=1)


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_prob(npz_path, seg_path):
    """Lesion probability on the working grid. nnU-Net's NibabelIO reverses the axis order internally, so
    the stored probabilities are transposed back and checked against the written segmentation."""
    import nibabel as nib

    prob = np.load(npz_path)["probabilities"][1].transpose(2, 1, 0)
    seg = np.asarray(nib.load(seg_path).dataobj)
    agree = ((prob > 0.5) == (seg > 0)).mean()
    assert prob.shape == seg.shape == (RESIZE[0], KEEP_DEPTH, RESIZE[2]) and agree > 0.999, (prob.shape, agree)
    return prob


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--prep-dir", default=os.environ.get("PREP_DIR", ""))
    ap.add_argument("--runs-dir", default=os.environ.get("RUNS_DIR", ""))
    ap.add_argument("--masks-dir", required=True, help="working-space masks (preprocess_tdsc.py --masks-only)")
    ap.add_argument("--local-dir", default=os.environ.get("LOCAL_DIR", "/tmp"))
    ap.add_argument("--processes", type=int, default=8)
    args = ap.parse_args()
    run_dir = os.path.join(args.runs_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)
    env = {"nnUNet_raw": os.path.join(args.local_dir, "nnUNet_raw"),
           "nnUNet_preprocessed": os.path.join(args.local_dir, "nnUNet_preprocessed"),
           "nnUNet_results": os.path.join(run_dir, "nnUNet_results"), "nnUNet_compile": "false",
           "nnUNet_n_proc_DA": "12"}
    os.environ.update(env)
    splits = load_meta(args.prep_dir)
    trainer = f"nnUNetTrainer_{args.epochs}epochs"

    export(args.prep_dir, args.masks_dir, env["nnUNet_raw"], splits)
    run(["nnUNetv2_plan_and_preprocess", "-d", "501", "-c", "3d_fullres", "-np", str(args.processes),
         "--verify_dataset_integrity"])
    with open(os.path.join(env["nnUNet_preprocessed"], DATASET, "splits_final.json"), "w") as f:
        json.dump([{"train": [case_id(m) for m in splits["Train"]],
                    "val": [case_id(m) for m in splits["Validation"]]}], f)
    fold_dir = os.path.join(env["nnUNet_results"], DATASET, f"{trainer}__nnUNetPlans__3d_fullres", "fold_0")
    if not os.path.exists(os.path.join(fold_dir, "checkpoint_final.pth")):
        resume = ["--c"] if os.path.exists(os.path.join(fold_dir, "checkpoint_latest.pth")) else []
        run(["nnUNetv2_train", "501", "3d_fullres", "0", "-tr", trainer] + resume)
    for split, sub in (("Validation", "imagesVal"), ("Test", "imagesTs")):
        out = os.path.join(run_dir, f"seg_{split}")
        if len([f for f in os.listdir(out) if f.endswith(".npz")] if os.path.isdir(out) else []) < len(splits[split]):
            run(["nnUNetv2_predict", "-i", os.path.join(env["nnUNet_raw"], DATASET, sub), "-o", out, "-d", "501",
                 "-c", "3d_fullres", "-f", "0", "-tr", trainer, "--save_probabilities"])
        probs = {m["case"]: load_prob(os.path.join(out, f"{case_id(m)}.npz"), os.path.join(out, f"{case_id(m)}.nii.gz"))
                 for m in splits[split]}
        for variant, (thr, score) in VARIANTS.items():
            preds = [prob_to_boxes(probs[m["case"]], thr, score) for m in splits[split]]
            save_predictions(os.path.join(run_dir, f"preds_{split}_{variant}.json"), preds, splits[split],
                             {"model": "nnunet_3d_fullres", "trainer": trainer, "threshold": thr, "score": score})
    print("done", flush=True)


if __name__ == "__main__":
    main()
