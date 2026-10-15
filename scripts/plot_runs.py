"""Validation curves and a summary table for groups of runs (seeds of one configuration).

  python scripts/plot_runs.py --runs-dir R --out fig.png --metric tdsc_froc \
      --group "baseline=w1_base,w2_tr_s1,w2_tr_s2" --group "300 queries=h3_q300"

Per run: the best score, its iteration, and the mean over the second half of training (a steadier
estimate than the max on a 30-case validation set). Iterations are scaled to samples seen
(iteration x batch size / 2) so runs with different batch sizes share the x-axis.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = ["#555555", "#e4572e", "#29a3d6", "#f3a712", "#a8c256", "#b07bd6", "#ef8fb4", "#3bb273"]


def load_curve(run_dir, metric):
    with open(os.path.join(run_dir, "log.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    evals = [r["eval"] for r in rows if "eval" in r and metric in r["eval"]]
    batch = 2
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            batch = json.load(f)["train"]["batch_size"]
    return np.array([e["it"] * batch / 2 for e in evals]), np.array([e[metric] for e in evals])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="tdsc_froc")
    ap.add_argument("--group", action="append", required=True, help="label=run1,run2,...")
    args = ap.parse_args()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    summary = {}
    for (label, runs), color in zip((g.split("=", 1) for g in args.group), COLORS):
        stats = []
        for i, run in enumerate(runs.split(",")):
            x, y = load_curve(os.path.join(args.runs_dir, run), args.metric)
            if not len(y):
                continue
            ax.plot(x, y, color=color, lw=1.2, alpha=0.8, label=label if i == 0 else None)
            half = y[x >= x.max() / 2]
            stats.append({"run": run, "best": float(y.max()), "best_it": float(x[y.argmax()]),
                          "late_mean": float(half.mean()), "last": float(y[-1])})
        if stats:
            summary[label] = {"runs": stats, "best_mean": float(np.mean([s["best"] for s in stats])),
                              "late_mean": float(np.mean([s["late_mean"] for s in stats]))}
    ax.set_xlabel("iterations (batch-2 equivalent)")
    ax.set_ylabel(f"Validation {args.metric}")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
