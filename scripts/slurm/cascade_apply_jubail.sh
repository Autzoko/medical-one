#!/bin/bash
# Cascade read-out on Validation (docs/prereg/cascade_cpm.md): per DETR seed s, verifier_s<s> applied to the swa.pt
# predictions in each mode of the ablation chain (fuse = DETR alone through top-30 / NMS 0.1, rescore, refine, full).
#   bash scripts/slurm/cascade_apply_jubail.sh [arm ...]   default: v3_p256 v3 paste, seeds 0-2, one job
# Writes $S/cascade/apply/detr_<arm>_s<s>_<mode>_Validation.json.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -n "${PULLED:-}" ] || { git pull -q; PULLED=1 exec bash "scripts/slurm/$(basename "$0")" "$@"; }  # rerun the pulled script text
C=$(git rev-parse HEAD)
S=/scratch/ll5582/langtian/abus3d
SPLIT=${SPLIT:-Validation}
mkdir -p "$S/cascade/apply"
arms=("$@")
[ ${#arms[@]} -gt 0 ] || arms=(v3_p256 v3 paste)
tag=$([ "$SPLIT" = Validation ] && echo val || echo Test)
cmd="true"
for arm in "${arms[@]}"; do
    for s in ${SEEDS:-0 1 2}; do
        for mode in ${MODES:-fuse rescore refine full}; do
            cmd+=" && python scripts/cascade/apply_cascade.py --prep-dir $S/data/prep --split $SPLIT"
            cmd+=" --detr $S/runs/detr_${arm}_s$s/preds_${tag}_swa.json --verifier $S/cascade/verifier_s$s --mode $mode"
            cmd+=" --out $S/cascade/apply/detr_${arm}_s${s}_${mode}_$SPLIT.json"
        done
    done
done
id=$(COMMIT=$C sbatch --parsable -J cascade_apply -t ${HOURS:-6}:00:00 -o "$S/slurm_logs/cascade_apply-%j.out" \
     scripts/slurm/jubail_job.sbatch sh "$cmd")
echo "cascade_apply $id commit $C"
