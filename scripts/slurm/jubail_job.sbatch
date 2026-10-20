#!/bin/bash
# Run train.py or a script on one Jubail (NYUAD HPC) V100 GPU. Everything lives under $B on /scratch.
#   COMMIT=$(git -C $B/medical-one rev-parse HEAD) sbatch -J exp scripts/slurm/jubail_job.sbatch train name=exp ...
#   COMMIT=... sbatch -J x scripts/slurm/jubail_job.sbatch run scripts/hardneg/mine.py ...
# V100 has no bf16, so training runs in fp32 (train.amp=false); compare arms only within one hardware.
#SBATCH -p nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH -c 10
#SBATCH --mem=60G
#SBATCH -t 12:00:00
#SBATCH -o /scratch/ll5582/langtian/abus3d/slurm_logs/%x-%j.out
set -euo pipefail
B=/scratch/ll5582/langtian
SHARED=$B/abus3d
ENV=$B/envs/abus3d-cu126
MODE=$1
shift
L=$(mktemp -d "${TMPDIR:-/tmp}/abus3d_${SLURM_JOB_ID}_XXXX")
trap 'rm -rf "$L"' EXIT
echo "[$(date)] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) commit=$COMMIT mode=$MODE $*"
git clone -q https://github.com/Autzoko/medical-one.git "$L/code"
git -C "$L/code" checkout -q "$COMMIT"
cd "$L/code"
export PYTHONPATH="$L/code/src" PYTHONNOUSERSITE=1 PREP_DIR="$SHARED/data/prep" RUNS_DIR="$SHARED/runs" LOCAL_DIR="$L"
case "$MODE" in
    train) "$ENV/bin/python" scripts/train.py --config configs/abus3d_tdsc.yaml paths.prep_dir="$SHARED/data/prep" \
               paths.runs_dir="$SHARED/runs" train.max_gpu_mem_gb=0 train.amp=false "$@" ;;
    run) "$ENV/bin/python" "$@" ;;
    *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
echo "[$(date)] done"
