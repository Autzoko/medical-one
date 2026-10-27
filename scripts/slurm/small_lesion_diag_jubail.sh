#!/bin/bash
# Small-lesion diagnostics D1-D5 (docs/prereg/small_lesion_diag.md): v3_p256 swa.pt seeds 0-2, no training.
#   bash scripts/slurm/small_lesion_diag_jubail.sh            # Train (base only, D1) + Validation + Test (all variants)
#   SMOKE=1 bash scripts/slurm/small_lesion_diag_jubail.sh    # one Validation case, all variants, into $OUT/smoke
# Raw volumes are read inside the jobs only; outputs (boxes, scores, summary numbers) go to $S/diag/small_lesion.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -n "${PULLED:-}" ] || { git pull -q; PULLED=1 exec bash "scripts/slurm/$(basename "$0")" "$@"; }  # rerun the pulled script text
C=$(git rev-parse HEAD)
S=/scratch/ll5582/langtian/abus3d
OUT=$S/diag/small_lesion
RUNS="$S/runs/detr_v3_p256_s0 $S/runs/detr_v3_p256_s1 $S/runs/detr_v3_p256_s2"
SET="--set paths.prep_dir=$S/data/prep 'data.patch=[256,96,256]'"
DIAG="python scripts/diag/small_lesion_diag.py $RUNS --raw-root $S/data/raw"
mkdir -p "$OUT"
submit() {  # name hours command
    id=$(COMMIT=$C sbatch --parsable -J "$1" -t "$2":00:00 -o "$S/slurm_logs/$1-%j.out" scripts/slurm/jubail_job.sbatch sh "$3")
    echo "$1 $id commit $C"
}
if [ -n "${SMOKE:-}" ]; then
    submit sld_smoke 1 "$DIAG --split Validation --max-cases 1 --out $OUT/smoke $SET"
    exit 0
fi
submit sld_train 2 "$DIAG --split Train --variants base --out $OUT $SET"
for split in Validation Test; do
    parts=$([ $split = Validation ] && echo 2 || echo 4)
    for k in $(seq 0 $((parts - 1))); do
        submit sld_${split}_p$k ${HOURS:-8} "$DIAG --split $split --part $k --parts $parts --out $OUT $SET"
    done
done
