"""Read-out of the small-lesion diagnostics (docs/prereg/small_lesion_diag.md) from the files of small_lesion_diag.py.

  python scripts/diag/small_lesion_readout.py --meta META --diag DIAG_DIR [--stored RUNS_DIR]

``--stored``: directory holding detr_v3_p256_s<i>/preds_{val,Test}_swa.json (sanity check of the base variant).
"""
import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np
from scipy.stats import wilcoxon

from medical_one.data.tdsc import boxes_to_original
from medical_one.evaluation.preds import fuse_predictions, load_predictions
from medical_one.evaluation.tdsc_official import official_froc, official_ious

SEEDS = (0, 1, 2)


def per_case(preds, metas):
    """[(scores sorted desc, official IoUs in the same order)] per case."""
    out = []
    for (b, s), m in zip(preds, metas):
        o = s.argsort(descending=True)
        iou = official_ious(boxes_to_original(b[o].double(), m["shape"]), m["mask_lo"], m["mask_hi"]) if len(s) else np.zeros(0)
        out.append((s[o].numpy(), np.asarray(iou)))
    return out


def summary(pc, terc) -> Dict[str, object]:
    froc, sens, *_ = official_froc(pc)
    top1 = np.array([len(s) > 0 and i[0] > 0.3 for s, i in pc])
    lesion = np.array([s[i > 0.3].max() if (i > 0.3).any() else 0.0 for s, i in pc])
    first = np.array([s[np.nonzero(i > 0.3)[0][0]] if (i > 0.3).any() else np.nan for s, i in pc])
    small = terc == 0
    return {"froc": froc, "sens125": sens[0], "top1": top1.mean(), "top1_small": top1[small].mean(),
            "anyhit_small": np.isfinite(first[small]).mean(), "first_med_small": np.nanmedian(first[small]),
            "first_lt03_small": np.mean(first[small][np.isfinite(first[small])] < 0.3), "_lesion": lesion, "_top1": top1}


def paired(a: List[np.ndarray], b: List[np.ndarray], sel: np.ndarray) -> str:
    """Change b - a of the per-lesion score, pooled over seeds, on the selected lesions."""
    d = np.concatenate([bb[sel] - aa[sel] for aa, bb in zip(a, b)])
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return f"median {np.median(d):+.3f} mean {np.mean(d):+.3f} up {np.mean(d > 0):.2f} down {np.mean(d < 0):.2f} p {p:.3g} (n {len(d)})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--stored", default="")
    args = ap.parse_args()
    allm = json.load(open(args.meta))
    for split in ("Train", "Validation", "Test"):
        metas = [m for m in allm if m["split"] == split]
        files = {}
        for f in glob.glob(os.path.join(args.diag, f"preds_{split}_*_s*.part*.json")):
            v, s = os.path.basename(f)[len(f"preds_{split}_"):].rsplit("_s", 1)
            files.setdefault(v, {}).setdefault(int(s.split(".")[0]), []).append(f)
        if not files:
            continue
        vol = np.array([np.prod(np.subtract(m["box"][3:], m["box"][:3])) for m in metas])
        terc = np.digitize(vol, np.quantile(vol, [1 / 3, 2 / 3]))
        print(f"\n===== {split} (n={len(metas)}, small n={int((terc == 0).sum())}) =====")
        R = {v: {s: summary(per_case(load_predictions(fs[s], metas), metas), terc) for s in SEEDS if s in fs}
             for v, fs in sorted(files.items())}
        keys = ("froc", "sens125", "top1", "top1_small", "anyhit_small", "first_med_small", "first_lt03_small")
        print("variant".ljust(11) + "".join(k.rjust(16) for k in keys))
        for v, rs in R.items():
            print(v.ljust(11) + "".join(f"{np.mean([r[k] for r in rs.values()]):16.3f}" for k in keys))
        if args.stored and "base" in R:
            tag = {"Validation": "val", "Test": "Test"}.get(split)
            for s in SEEDS:
                p = os.path.join(args.stored, f"detr_v3_p256_s{s}", f"preds_{tag}_swa.json")
                if tag and os.path.exists(p):
                    st = summary(per_case(load_predictions(p, metas), metas), terc)["froc"]
                    print(f"  sanity seed {s}: base FROC {R['base'][s]['froc']:.4f} vs stored {st:.4f}")
        if "base" not in R:
            continue
        base = [R["base"][s]["_lesion"] for s in SEEDS]
        lesion = lambda v: [R[v][s]["_lesion"] for s in SEEDS]
        sd_of = lambda vs: np.mean([np.stack([R[v][s]["_lesion"] for v in vs]).std(0) for s in SEEDS], 0)
        phases = [v for v in ("base", "phase1", "phase2", "phase3") if v in R]
        if len(phases) == 4:
            sd = sd_of(phases)
            print(f"D2 phase std of lesion score: small {np.median(sd[terc == 0]):.3f} mid {np.median(sd[terc == 1]):.3f} "
                  f"large {np.median(sd[terc == 2]):.3f}")
            fused = []
            for s in SEEDS:
                src = [load_predictions(files[v][s], metas) for v in phases]
                fused.append(summary(per_case(fuse_predictions(src, 0.5), metas), terc))
            print(f"D2 4-phase WBF: FROC {np.mean([f['froc'] for f in fused]):.3f} top1_small "
                  f"{np.mean([f['top1_small'] for f in fused]):.3f}; small lesion score vs base: "
                  f"{paired(base, [f['_lesion'] for f in fused], terc == 0)}")
        shifts = [v for v in R if v.startswith("shift_")]
        if len(shifts) == 6:
            sd = sd_of(["base"] + shifts)
            print(f"D2b shift std of lesion score: small {np.median(sd[terc == 0]):.3f} mid {np.median(sd[terc == 1]):.3f} "
                  f"large {np.median(sd[terc == 2]):.3f}")
        for v in ("aa0.5", "aa1", "aa1.5", "z2", "z1n", "z1"):
            if v in R:
                print(f"{v:6s} vs base, small: {paired(base, lesion(v), terc == 0)} | all: {paired(base, lesion(v), terc >= 0)}")
        if "z2" in R and "z1n" in R:
            print(f"z1n vs z2, small: {paired(lesion('z2'), lesion('z1n'), terc == 0)}")
        if "z1n" in R and "z1" in R:
            print(f"z1 vs z1n, small: {paired(lesion('z1n'), lesion('z1'), terc == 0)}")
        d5 = []
        for f in glob.glob(os.path.join(args.diag, f"d5_{split}.part*.json")):
            d5 += json.load(open(f))
        if d5:
            by = {(r["split"], r["case"]): r for r in d5}
            recs = [by.get((m["split"], m["case"]), {}) for m in metas]
            print(f"D5 resampler check: max diff {max(r.get('base_rs_maxdiff', 0) for r in recs)}, "
                  f"max frac {max(r.get('base_rs_frac_diff', 0) for r in recs):.5f}")
            sp = [r.get("spacings") or r.get("space directions") for r in recs if r]
            print(f"D5 header spacing examples: {sp[:2]}")
            fw = np.array([[r.get("speckle", {}).get(f"fwhm_ax{a}", np.nan) for a in range(3)] for r in recs], float)
            print(f"D5 speckle FWHM (raw voxels) median N0/N1/N2: {np.nanmedian(fw, 0).round(2)} (n {int(np.isfinite(fw[:, 0]).sum())})")
            ratio = np.array([r.get("gt_aa1", {}).get("cnr", np.nan) / max(r.get("gt_prep", {}).get("cnr", np.nan), 1e-6)
                              if r.get("gt_prep") else np.nan for r in recs])
            for t, name in enumerate(("small", "mid", "large")):
                print(f"D5 CNR(aa1)/CNR(prep) {name}: median {np.nanmedian(ratio[terc == t]):.3f}")
            # appearance of the seed-0 top-1 box when it is a false positive, vs the lesion (small tertile)
            pc0 = per_case(load_predictions(files["base"][0], metas), metas)
            rows = []
            for r, (s, i), t in zip(recs, pc0, terc):
                if t == 0 and len(s) and i[0] <= 0.3 and r.get("top_boxes") and r.get("gt_prep"):
                    fp = r["top_boxes"][0]
                    rows.append([r["gt_prep"].get(k, np.nan) for k in ("mean_in", "cnr", "size")]
                                + [fp.get(k, np.nan) for k in ("mean_in", "cnr", "size")])
            if rows:
                a = np.nanmedian(np.array(rows, float), 0).round(2)
                print(f"D5 small lesions losing top-1 (n {len(rows)}): lesion mean/CNR/size {a[:3]} vs top-1 FP {a[3:]}")


if __name__ == "__main__":
    main()
