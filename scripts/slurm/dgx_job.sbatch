#!/bin/bash
# Run train.py / evaluate.py on one DGX Spark node (GB10, aarch64, 1 GPU).
#
# Compute nodes cannot see the login node's /home, so everything shared lives on the CIFS mount
# $SHARED (data/prep, runs/, slurm_logs/). Code is checked out from GitHub at $COMMIT; data is
# copied to node-local /tmp; the Python env is built once per node under the node-local $HOME.
#
# Usage, from the repo on dgx-login:
#   COMMIT=$(git rev-parse HEAD) sbatch -J exp1 scripts/slurm/dgx_job.sbatch train name=exp1 train.iters=10000
#   COMMIT=$(git rev-parse HEAD) sbatch -J ev scripts/slurm/dgx_job.sbatch evaluate run.pt --split Test
#SBATCH -p spark
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=100G
#SBATCH -t 12:00:00
#SBATCH -o /media/cvpr/langtian/abus3d/slurm_logs/%x-%j.out
set -euo pipefail

SHARED=${SHARED:-/media/cvpr/langtian/abus3d}
REPO=${REPO:-https://github.com/Autzoko/medical-one.git}
: "${COMMIT:?set COMMIT to the git sha to run}"
MODE=$1; shift
ENV=$HOME/langtian/envs/abus3d-cu130
L=/tmp/abus3d_${SLURM_JOB_ID}
mkdir -p "$L"
trap 'rm -rf "$L"' EXIT
echo "[$(date)] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) commit=$COMMIT mode=$MODE $*"

mkdir -p "$(dirname "$ENV")"
(
    flock 9
    if [ ! -x "$ENV/bin/python" ]; then
        echo "[$(date)] building env $ENV"
        python3 -m venv "$ENV"
        "$ENV/bin/pip" install -q --upgrade pip
        "$ENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cu130
        "$ENV/bin/pip" install -q numpy scipy pandas pynrrd pyyaml pytest
    fi
) 9>"$ENV.lock"

git clone -q "$REPO" "$L/code"
git -C "$L/code" checkout -q "$COMMIT"
cp -r "$SHARED/data/prep" "$L/prep"
echo "[$(date)] staged code and data"

cd "$L/code"
export PYTHONPATH="$L/code/src" PYTHONNOUSERSITE=1
PATHS=(paths.prep_dir="$L/prep" paths.runs_dir="$SHARED/runs" train.max_gpu_mem_gb=0)
case "$MODE" in
    train) "$ENV/bin/python" scripts/train.py --config configs/abus3d_tdsc.yaml "${PATHS[@]}" "$@" ;;
    evaluate) "$ENV/bin/python" scripts/evaluate.py --config configs/abus3d_tdsc.yaml "$@" --set "${PATHS[@]}" ;;
    *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
echo "[$(date)] done"
