"""Detection figures from prediction files: official FROC curves and per-case slice montages.

  python scripts/visualize_detections.py --prep-dir P --out D --split Test \
      --preds "DETR (ours)=a.json" "nnU-Net=b.json" ...

Montage rows are cases, columns are two planes through the GT lesion center: the coronal d-w plane
(fixed depth) and the transverse h-w plane (fixed d, ultrasound depth downwards). The GT box is drawn
in white, each method's highest-scoring box in its color with score and official IoU. Images contain
patient data: keep the output out of version control.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from medical_one.data.tdsc import boxes_to_original, load_meta, load_volume  # noqa: E402
from medical_one.evaluation import official_curve  # noqa: E402
from medical_one.evaluation.preds import load_predictions  # noqa: E402
from medical_one.evaluation.tdsc_official import KEY_FP, official_ious  # noqa: E402

COLORS = ["#e4572e", "#29a3d6", "#f3a712", "#a8c256", "#b07bd6", "#ef8fb4"]


def top1_iou(pred, meta):
    boxes, scores = pred
    if not len(boxes):
        return 0.0, 0.0
    iou = official_ious(boxes_to_original(boxes[:1].double(), meta["shape"]), meta["mask_lo"], meta["mask_hi"])
    return float(iou[0]), float(scores[0])


def froc_figure(methods, metas, out):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for (name, preds), color in zip(methods.items(), COLORS):
        fps, rec = official_curve(preds, metas)
        order = np.argsort(fps)
        ax.plot(np.asarray(fps)[order], np.asarray(rec)[order], color=color, lw=2, label=name)
    for k in KEY_FP:
        ax.axvline(k, color="#999", lw=0.5, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.1, 10)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positives per scan")
    ax.set_ylabel("Sensitivity (IoU > 0.3)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def draw_box(ax, x0, y0, x1, y1, color, label=None, lw=1.6, ls="-"):
    ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=lw, ls=ls))
    if label:
        ax.text(x0, y0 - 2, label, color=color, fontsize=6, va="bottom")


def montage(methods, metas, cases, prep_dir, out):
    rows = len(cases)
    fig, axes = plt.subplots(rows, 2, figsize=(9, 2.6 * rows), gridspec_kw={"width_ratios": [1, 1]})
    axes = np.atleast_2d(axes)
    index = {m["case"]: i for i, m in enumerate(metas)}
    for r, case in enumerate(cases):
        i = index[case]
        meta = metas[i]
        vol = load_volume(prep_dir, meta).astype(np.float32)
        g = np.asarray(meta["box"])
        c = ((g[:3] + g[3:]) / 2).astype(int)
        c[1] = min(c[1], vol.shape[1] - 1)
        coronal, transverse = vol[:, c[1], :], vol[c[0], :, :]  # (d, w) and (h, w)
        for col, (img, (ya, xa)) in enumerate(((coronal, (0, 2)), (transverse, (1, 2)))):
            ax = axes[r, col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="auto" if col else "equal")
            draw_box(ax, g[xa], g[ya], g[xa + 3], g[ya + 3], "white", lw=1.4, ls="--")
            for (name, preds), color in zip(methods.items(), COLORS):
                boxes, _ = preds[i]
                if len(boxes):
                    b = boxes[0].numpy()
                    iou, score = top1_iou(preds[i], meta)
                    label = f"{name}: {score:.2f} / IoU {iou:.2f}" if col == 0 else None
                    draw_box(ax, b[xa], b[ya], b[xa + 3], b[ya + 3], color, label)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(f"case {case} ({meta['label']})", fontsize=8)
    axes[0, 0].set_title("coronal (d-w) at lesion depth", fontsize=9)
    axes[0, 1].set_title("transverse (h-w), depth downwards", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prep-dir", default=os.environ.get("PREP_DIR", ""))
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="Test")
    ap.add_argument("--preds", nargs="+", required=True, help="name=path.json")
    ap.add_argument("--cases", type=int, nargs="*", default=[], help="default: picked from the first method")
    ap.add_argument("--per-figure", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    metas = load_meta(args.prep_dir)[args.split]
    methods = {}
    for item in args.preds:
        name, path = item.split("=", 1)
        methods[name] = load_predictions(path, metas)
    froc_figure(methods, metas, os.path.join(args.out, f"froc_{args.split}.png"))
    first = next(iter(methods.values()))
    ious = [top1_iou(p, m)[0] for p, m in zip(first, metas)]
    order = np.argsort(ious)[::-1]
    picks = {"hits": [metas[i]["case"] for i in order[:args.per_figure]],
             "median": [metas[i]["case"] for i in order[len(order) // 2 - args.per_figure // 2:][:args.per_figure]],
             "misses": [metas[i]["case"] for i in order[-args.per_figure:]]}
    if args.cases:
        picks = {"selected": args.cases}
    for tag, cases in picks.items():
        montage(methods, metas, cases, args.prep_dir, os.path.join(args.out, f"cases_{args.split}_{tag}.png"))
    with open(os.path.join(args.out, f"top1_{args.split}.json"), "w") as f:
        json.dump({name: {m["case"]: top1_iou(p, m) for p, m in zip(preds, metas)} for name, preds in methods.items()},
                  f)
    print("figures ->", args.out, flush=True)


if __name__ == "__main__":
    main()
