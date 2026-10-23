#!/bin/bash
# DETR v2 round (docs/prereg/detr_v2.md) on Jubail V100: the v2 recipe and its single-factor arms, seeds 0-2.
#   bash scripts/slurm/detr_v2_jubail.sh [arm ...]   arms: v2 pos033 paste patch256 small shadow iouaware nnaug
#   (default: the first seven, the v2 round)
#   EXTRA="train.stop_iter=40 train.eval_every=20 train.swa_last=2" NAME_SUFFIX=_smoke SEEDS=0 bash ... (smoke)
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -n "${PULLED:-}" ] || { git pull -q; PULLED=1 exec bash "scripts/slurm/$(basename "$0")" "$@"; }  # rerun the pulled script text
C=$(git rev-parse HEAD)
S=/scratch/ll5582/langtian/abus3d
declare -A ARM=(
    [v2]=""
    [pos033]="data.pos_prob=0.33"
    [paste]="data.paste_prob=0.3"
    [patch256]="data.patch=[256,96,256]"
    [small]="data.small_oversample=0.5"
    [shadow]="data.shadow_prob=0.3"
    [iouaware]="loss.iou_aware_cls=true"
    [nnaug]="data.rot_prob=0.2 data.contrast_prob=0.15"   # docs/prereg/detr_nnaug.md
)
arms=("$@")
[ ${#arms[@]} -gt 0 ] || arms=(v2 pos033 paste patch256 small shadow iouaware)
for arm in "${arms[@]}"; do
    [ -n "${ARM[$arm]+x}" ] || { echo "unknown arm $arm"; exit 2; }
    for s in ${SEEDS:-0 1 2}; do
        name=detr_${arm}_s$s${NAME_SUFFIX:-}
        cmd="python scripts/train.py --config configs/detr_v2.yaml paths.prep_dir=$S/data/prep paths.runs_dir=$S/runs"
        cmd+=" name=$name train.seed=$s ${ARM[$arm]} ${EXTRA:-}"
        id=$(COMMIT=$C sbatch --parsable -J "$name" -t ${HOURS:-24}:00:00 -o "$S/slurm_logs/$name-%j.out" \
             scripts/slurm/jubail_job.sbatch sh "$cmd")
        echo "$name $id commit $C"
    done
done
