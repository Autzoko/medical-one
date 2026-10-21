#!/bin/bash
# Expert 1 + expert 2 (CenterDETR3D), official TDSC split, seeds 0-2 on Jubail V100 (fp32), one job per seed.
# Pre-registration: docs/prereg/center_detr_e1e2.md (same commit).
#   bash scripts/slurm/center_detr_jubail.sh [seed ...]      (default: 0 1 2)
set -euo pipefail
cd "$(dirname "$0")/../.."
git pull -q
C=$(git rev-parse HEAD)
S=/scratch/ll5582/langtian/abus3d
seeds=("$@")
[ ${#seeds[@]} -gt 0 ] || seeds=(0 1 2)
for s in "${seeds[@]}"; do
    name=center_detr_s$s
    id=$(COMMIT=$C sbatch --parsable -J "$name" -t 12:00:00 -o "$S/slurm_logs/$name-%j.out" \
         scripts/slurm/jubail_job.sbatch run scripts/train.py --config configs/center_detr.yaml \
         paths.prep_dir="$S/data/prep" paths.runs_dir="$S/runs" train.max_gpu_mem_gb=0 train.amp=false \
         data.num_workers=5 name="$name" train.seed="$s")
    echo "$name $id commit $C"
done
