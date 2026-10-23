#!/bin/bash
# Flip TTA on Validation for DETR v2-round runs (docs/prereg/detr_nnaug.md, N1; no training): swa.pt of each run with
# its own config, eval.tta_flips=true (d / w / dw flips, WBF IoU 0.5) -> runs/<name>/preds_val_swa_tta.json.
#   bash scripts/slurm/tta_v2_jubail.sh [arm ...]   default: v2 paste shadow patch256, seeds 0-2, one job
set -euo pipefail
cd "$(dirname "$0")/../.."
git pull -q
C=$(git rev-parse HEAD)
S=/scratch/ll5582/langtian/abus3d
arms=("$@")
[ ${#arms[@]} -gt 0 ] || arms=(v2 paste shadow patch256)
cmd="true"
for arm in "${arms[@]}"; do
    for s in ${SEEDS:-0 1 2}; do
        r=$S/runs/detr_${arm}_s$s
        cmd+=" && python scripts/evaluate.py $r/swa.pt --config $r/config.json --split Validation"
        cmd+=" --save-preds $r/preds_val_swa_tta.json --set eval.tta_flips=true"
    done
done
id=$(COMMIT=$C sbatch --parsable -J tta_v2 -t ${HOURS:-6}:00:00 -o "$S/slurm_logs/tta_v2-%j.out" \
     scripts/slurm/jubail_job.sbatch sh "$cmd")
echo "tta_v2 $id commit $C"
