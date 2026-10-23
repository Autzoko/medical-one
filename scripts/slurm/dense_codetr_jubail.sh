#!/bin/bash
# DenseDet3D and CoDETR3D (docs/prereg/dense_codetr.md), official TDSC split, on Jubail V100 (fp32): each job
# trains, then runs the pre-registered post-processing grid (Validation scored, Test only predicted).
#   bash scripts/slurm/dense_codetr_jubail.sh [arm ...]    arms: dense_det co_detr (default both); SEEDS="0 1 2"
#   EXTRA="train.stop_iter=40 train.eval_every=20" NAME_SUFFIX=_smoke SEEDS=0 bash ... (smoke test)
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -n "${PULLED:-}" ] || { git pull -q; PULLED=1 exec bash "scripts/slurm/$(basename "$0")" "$@"; }  # rerun the pulled script text
C=$(git rev-parse HEAD)
S=/scratch/ll5582/langtian/abus3d
arms=("$@")
[ ${#arms[@]} -gt 0 ] || arms=(dense_det co_detr)
for arm in "${arms[@]}"; do
    for s in ${SEEDS:-0 1 2}; do
        name=${arm}_s$s${NAME_SUFFIX:-}
        cmd="python scripts/train.py --config configs/$arm.yaml paths.prep_dir=$S/data/prep paths.runs_dir=$S/runs"
        cmd+=" train.max_gpu_mem_gb=0 train.amp=false data.num_workers=5 name=$name train.seed=$s ${EXTRA:-}"
        cmd+=" && python scripts/finalize_center.py $S/runs/$name --set paths.prep_dir=$S/data/prep"
        id=$(COMMIT=$C sbatch --parsable -J "$name" -t ${HOURS:-24}:00:00 -o "$S/slurm_logs/$name-%j.out" \
             scripts/slurm/jubail_job.sbatch sh "$cmd")
        echo "$name $id commit $C"
    done
done
