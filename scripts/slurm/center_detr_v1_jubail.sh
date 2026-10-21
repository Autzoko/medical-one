#!/bin/bash
# CenterDETR3D v1 (docs/prereg/center_detr_v1.md), official TDSC split, seeds 0-2 on Jubail V100 (fp32):
# each job trains, then runs the pre-registered post-processing grid (Validation scored, Test only predicted).
#   bash scripts/slurm/center_detr_v1_jubail.sh [seed ...]      (default: 0 1 2)
set -euo pipefail
cd "$(dirname "$0")/../.."
git pull -q
C=$(git rev-parse HEAD)
S=/scratch/ll5582/langtian/abus3d
seeds=("$@")
[ ${#seeds[@]} -gt 0 ] || seeds=(0 1 2)
for s in "${seeds[@]}"; do
    name=center_detr_v1_s$s
    cmd="python scripts/train.py --config configs/center_detr_v1.yaml paths.prep_dir=$S/data/prep paths.runs_dir=$S/runs"
    cmd+=" train.max_gpu_mem_gb=0 train.amp=false data.num_workers=5 name=$name train.seed=$s"
    cmd+=" && python scripts/finalize_center.py $S/runs/$name --set paths.prep_dir=$S/data/prep"
    id=$(COMMIT=$C sbatch --parsable -J "$name" -t 24:00:00 -o "$S/slurm_logs/$name-%j.out" \
         scripts/slurm/jubail_job.sbatch sh "$cmd")
    echo "$name $id commit $C"
done
