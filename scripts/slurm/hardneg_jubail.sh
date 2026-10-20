#!/bin/bash
# Hard-negative study for the 3D DETR on Jubail V100s (fp32), all arms on the same hardware:
#   jb_base        cv_aug config, no negatives (hardware-matched baseline)
#   jb_oof         + negatives mined from the DGX cv_aug out-of-fold predictions (offline, self)
#   jb_online      + online bootstrapping: every 2000 iterations the model re-mines its own false positives
#   jb_oof_online  + both
#   jb_rand        + random boxes with the mined sizes and counts (control)
# 5 folds each; every job runs in one of $LANES singleton lanes (at most $LANES GPUs at once).
#   bash scripts/slurm/hardneg_jubail.sh [arm ...]     (default: all arms)
set -euo pipefail
cd "$(dirname "$0")/../.."
git pull -q
C=$(git rev-parse HEAD)
S=/scratch/ll5582/langtian/abus3d
H=$S/hardneg
LANES=${LANES:-10}
COMMON=(data.cv_folds=5 data.scale_range=0.2 data.noise_std=8.0 train.iters=10000 train.eval_every=500)
NEG=(data.neg_prob=0.2 loss.w_neg=1.0)
LIST=$H/jobs_jubail.txt
touch "$LIST"
echo "commit $C $(date)" >> "$LIST"

arm_args() {
    case "$1" in
        jb_base) ;;
        jb_oof) echo "${NEG[*]} data.neg_file=$H/r1_detr.json" ;;
        jb_online) echo "${NEG[*]} data.neg_refresh_every=2000" ;;
        jb_oof_online) echo "${NEG[*]} data.neg_file=$H/r1_detr.json data.neg_refresh_every=2000" ;;
        jb_rand) echo "${NEG[*]} data.neg_file=$H/r1_detr_rand.json" ;;
        *) echo "unknown arm $1" >&2; exit 2 ;;
    esac
}

arms=("$@")
[ ${#arms[@]} -gt 0 ] || arms=(jb_base jb_oof jb_online jb_oof_online jb_rand)
for arm in "${arms[@]}"; do
    extra=$(arm_args "$arm")
    for k in 0 1 2 3 4; do
        lane=$(( $(grep -vc "^commit" "$LIST" || true) % LANES ))
        # shellcheck disable=SC2086
        id=$(COMMIT=$C sbatch --parsable -J "jb_L$lane" --comment "${arm}_f$k" --dependency=singleton -t 8:00:00 \
             -o "$S/slurm_logs/${arm}_f$k-%j.out" scripts/slurm/jubail_job.sbatch train name="${arm}_f$k" \
             "${COMMON[@]}" data.cv_fold=$k $extra)
        echo "${arm}_f$k $id" >> "$LIST"
    done
done
tail -n $(( ${#arms[@]} * 5 )) "$LIST"
