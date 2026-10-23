#!/bin/bash
# Hard-negative experiments, round 1. Run on the DGX login node from the repo checkout:
#   bash scripts/slurm/hardneg_round1.sh
#
# Negatives are confident false positives mined from OUT-OF-FOLD predictions (scripts/hardneg/mine.py) on
# the same stratified 5-fold split of Train + Validation that the DETR CV uses; Test is never touched.
#   y5cv   YOLOv5m 5-fold CV without negatives (baseline, and the source of the YOLO negatives)
#   DETR arms (main cv_aug config, + data.neg_prob=0.2 loss.w_neg=1.0):
#     hn_d_self  <- DETR negatives     hn_d_cross <- YOLO negatives     hn_d_rand <- random boxes (control)
#   YOLO arms (y5cv settings, + --neg-weight 0.1):
#     hn_y_self  <- YOLO negatives     hn_y_cross <- DETR negatives     hn_y_rand <- random boxes (control)
# The DETR baseline is the existing cv_aug_f* runs. Throttle: every job runs in one of $LANES singleton
# lanes, so at most $LANES of our nodes are busy at once.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -n "${PULLED:-}" ] || { git pull -q; PULLED=1 exec bash "scripts/slurm/$(basename "$0")" "$@"; }  # rerun the pulled script text
C=$(git rev-parse HEAD)
S=${SHARED:-/media/cvpr/langtian/abus3d}
R=$S/runs
H=$S/hardneg
RAW=$S/data/raw
META=$S/data/prep/meta.json
LANES=${LANES:-4}
YOLO_EPOCHS=${YOLO_EPOCHS:-20}
DETR_CV=(data.cv_folds=5 data.scale_range=0.2 data.noise_std=8.0 train.iters=10000 train.eval_every=500)
DETR_NEG=(data.neg_prob=0.2 loss.w_neg=1.0)
YOLO_NEG=(--neg-weight 0.1)
mkdir -p "$H" "$S/pipeline"

# sb NAME TIME DEPS EXTRAS MODE ARGS... -> job id (DEPS: "" or "afterok:1:2")
sb() {
    local name=$1 time=$2 deps=$3 extras=$4
    shift 4
    local d="singleton"
    [ -n "$deps" ] && d="$deps,singleton"
    local id lane  # sb runs in $(...) subshells, so the lane comes from the job list kept on disk
    lane=$(( $(wc -l < "$H/jobs_round1.txt") % LANES ))
    id=$(COMMIT=$C EXTRAS=$extras NO_PREP=${NO_PREP:-} sbatch --parsable -J "hn_L$lane" --comment "$name" \
         -o "$S/slurm_logs/$name-%j.out" -t "$time" --dependency="$d" scripts/slurm/dgx_job.sbatch "$@")
    echo "$name $id" >> "$H/jobs_round1.txt"
    echo "$id"
}

detr_arm() {  # detr_arm NAME NEG_FILE DEPS
    local ids=()
    for k in 0 1 2 3 4; do
        ids+=("$(sb "$1_f$k" 6:00:00 "$3" "" train name="$1_f$k" "${DETR_CV[@]}" data.cv_fold=$k \
                 "${DETR_NEG[@]}" data.neg_file="$2")")
    done
    echo "${ids[*]}"
}

yolo_arm() {  # yolo_arm NAME DEPS [train_yolo args...] -> ids of the 2D inference jobs
    local name=$1 deps=$2
    shift 2
    local ids=()
    for k in 0 1 2 3 4; do
        local t
        t=$(sb "${name}_f$k" 10:00:00 "$deps" yolov5 run scripts/pipeline/train_yolo.py --name "${name}_f$k" \
            --raw "$RAW" --epochs "$YOLO_EPOCHS" --cv-folds 5 --cv-fold $k "$@")
        ids+=("$(sb "${name}_d2_f$k" 3:00:00 "afterok:$t" yolov5 run scripts/pipeline/infer_2d.py \
                 --weights "$R/${name}_f$k/best.pt" --raw "$RAW" --out "$S/pipeline/d2_${name}_f$k.json" \
                 --splits Train Validation Test --cv-folds 5 --cv-fold $k)")
    done
    echo "${ids[*]}"
}

colon() { local IFS=:; echo "$*"; }

: > "$H/jobs_round1.txt"
echo "commit $C" >> "$H/jobs_round1.txt"

# 1. DETR negatives (the cv_aug OOF predictions exist) and their random control
DETR_OOF=$(printf "$R/cv_aug_f%d/preds_val_best.json " 0 1 2 3 4)
m_detr=$(NO_PREP=1 sb mine_detr 0:30:00 "" "" run scripts/hardneg/mine.py --meta "$META" --out "$H/r1_detr.json" \
         --detr $DETR_OOF)
m_detr_rand=$(NO_PREP=1 sb mine_detr_rand 0:30:00 "" "" run scripts/hardneg/mine.py --meta "$META" \
              --out "$H/r1_detr_rand.json" --detr $DETR_OOF --random)

# 2. YOLO CV baseline -> YOLO OOF 2D boxes -> YOLO negatives and their control
y5cv=$(yolo_arm y5cv "")
YOLO_OOF=$(printf "$S/pipeline/d2_y5cv_f%d.json " 0 1 2 3 4)
m_yolo=$(NO_PREP=1 sb mine_yolo 0:30:00 "afterok:$(colon $y5cv)" "" run scripts/hardneg/mine.py --meta "$META" \
         --out "$H/r1_yolo.json" --d2 $YOLO_OOF)
m_yolo_rand=$(NO_PREP=1 sb mine_yolo_rand 0:30:00 "afterok:$(colon $y5cv)" "" run scripts/hardneg/mine.py \
              --meta "$META" --out "$H/r1_yolo_rand.json" --d2 $YOLO_OOF --random)

# 3. arms whose negatives are ready first, then the ones waiting on the YOLO CV
detr_arm hn_d_self "$H/r1_detr.json" "afterok:$m_detr" > /dev/null
yolo_arm hn_y_cross "afterok:$m_detr" --negatives "$H/r1_detr.json" "${YOLO_NEG[@]}" > /dev/null
detr_arm hn_d_rand "$H/r1_detr_rand.json" "afterok:$m_detr_rand" > /dev/null
detr_arm hn_d_cross "$H/r1_yolo.json" "afterok:$m_yolo" > /dev/null
yolo_arm hn_y_self "afterok:$m_yolo" --negatives "$H/r1_yolo.json" "${YOLO_NEG[@]}" > /dev/null
yolo_arm hn_y_rand "afterok:$m_yolo_rand" --negatives "$H/r1_yolo_rand.json" "${YOLO_NEG[@]}" > /dev/null

echo "submitted $(($(wc -l < "$H/jobs_round1.txt") - 1)) jobs in $LANES lanes; list: $H/jobs_round1.txt"
