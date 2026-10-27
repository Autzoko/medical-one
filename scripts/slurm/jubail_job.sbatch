#!/bin/bash
# Run train.py or a script on one Jubail (NYUAD HPC) V100 GPU. Everything lives under $B on /scratch.
#   COMMIT=$(git -C $B/medical-one rev-parse HEAD) sbatch -J exp scripts/slurm/jubail_job.sbatch train name=exp ...
#   COMMIT=... sbatch -J x scripts/slurm/jubail_job.sbatch run scripts/hardneg/mine.py ...
# V100 has no bf16, so training runs in fp32 (train.amp=false); compare arms only within one hardware.
# Partition visual (QOS normal): the nvidia QOS caps the user at 12 GPUs, shared with the user's other projects;
# visual allows at most 6 CPUs per node, hence 5 data workers.
#SBATCH -p visual
#SBATCH --gres=gpu:v100:1
#SBATCH -c 6
#SBATCH --mem=40G
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
git clone -q https://github.com/Autzoko/medical-one.git "$L/code" \
    || { echo "GitHub unreachable, cloning $B/medical-one instead"; rm -rf "$L/code"; git clone -q "$B/medical-one" "$L/code"; }
git -C "$L/code" checkout -q "$COMMIT"
cd "$L/code"
export PYTHONPATH="$L/code/src" PYTHONNOUSERSITE=1 PREP_DIR="$SHARED/data/prep" RUNS_DIR="$SHARED/runs" LOCAL_DIR="$L"
case "$MODE" in
    train) "$ENV/bin/python" scripts/train.py --config configs/abus3d_tdsc.yaml paths.prep_dir="$SHARED/data/prep" \
               paths.runs_dir="$SHARED/runs" train.max_gpu_mem_gb=0 train.amp=false data.num_workers=5 "$@" ;;
    run) "$ENV/bin/python" "$@" ;;
    sh) PATH="$ENV/bin:$PATH" bash -c "$1" ;;  # a shell command line run with the env's python first on PATH
    *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
echo "[$(date)] done"
