#!/bin/bash
# Cascade round (docs/prereg/cascade_cpm.md) on Jubail V100: DETR v3 + CPM3D (complementary) + 3D ROI verifier.
#   bash scripts/slurm/cascade_jubail.sh [part ...]   parts: cpm cpm_boost detr_v3 verifier (default all)
# The boost parts and the verifier wait for the inner-fold DETRs of the AdaBoost round (detr_boostfold_k0-3):
# DEP=<jobid:jobid:...> (afterok); their out-of-fold predictions give case weights, hard negatives and the
# verifier's training candidates.
#   EXTRA="train.stop_iter=40 train.eval_every=20 train.swa_last=2" VEXTRA="--epochs 1 --folds 4" NAME_SUFFIX=_smoke \
#     SEEDS=0 FOLDS_SUFFIX=_smoke bash scripts/slurm/cascade_jubail.sh     (smoke)
set -euo pipefail
cd "$(dirname "$0")/../.."
git pull -q
C=$(git rev-parse HEAD)
S=/scratch/ll5582/langtian/abus3d
X=${NAME_SUFFIX:-}
FX=${FOLDS_SUFFIX:-}
mkdir -p "$S/boost" "$S/cascade"
OOF=""
for k in 0 1 2 3; do OOF+=" $S/runs/detr_boostfold_k$k$FX/preds_oof.json"; done
W=$S/boost/weights_cascade$X.json
N=$S/boost/negatives_cascade$X.json
BOOST="python scripts/boost/boost_weights.py --meta $S/data/prep/meta.json --oof$OOF --out $W.\$SEED"
BOOST+=" && python scripts/hardneg/mine.py --meta $S/data/prep/meta.json --out $N.\$SEED --detr$OOF"
DEPARG=${DEP:+--dependency=afterok:$DEP}
submit() {  # name hours cmd [dependency]
    id=$(COMMIT=$C sbatch --parsable ${4:-} -J "$1" -t "$2":00:00 -o "$S/slurm_logs/$1-%j.out" \
         scripts/slurm/jubail_job.sbatch sh "$3")
    echo "$1 $id commit $C"
}
parts=("$@")
[ ${#parts[@]} -gt 0 ] || parts=(cpm cpm_boost detr_v3 verifier)
for part in "${parts[@]}"; do
    for s in ${SEEDS:-0 1 2}; do
        TR="python scripts/train.py paths.prep_dir=$S/data/prep paths.runs_dir=$S/runs train.seed=$s"
        b=${BOOST//\$SEED/s$s}
        case "$part" in
            cpm) submit cpm_s$s$X 24 "$TR --config configs/cpm.yaml name=cpm_s$s$X ${EXTRA:-}" ;;
            cpm_boost) submit cpm_boost_s$s$X 24 "$b && $TR --config configs/cpm.yaml name=cpm_boost_s$s$X \
data.case_weight_file=$W.s$s data.neg_file=$N.s$s loss.w_neg=1.0 ${EXTRA:-}" "$DEPARG" ;;
            detr_v3) submit detr_v3_s$s$X 24 "$b && $TR --config configs/detr_v2.yaml name=detr_v3_s$s$X \
data.paste_prob=0.3 data.shadow_prob=0.3 data.case_weight_file=$W.s$s data.neg_file=$N.s$s data.neg_prob=0.2 \
loss.w_neg=1.0 ${EXTRA:-}" "$DEPARG" ;;
            verifier) submit verifier_s$s$X 12 "$b && python scripts/cascade/train_verifier.py --prep-dir $S/data/prep \
--oof$OOF --weights $W.s$s --out $S/cascade/verifier_s$s$X --seed $s ${VEXTRA:-}" "$DEPARG" ;;
            *) echo "unknown part $part"; exit 2 ;;
        esac
    done
done
