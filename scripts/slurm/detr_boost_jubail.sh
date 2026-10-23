#!/bin/bash
# DETR + AdaBoost-style reweighting (docs/prereg/detr_boost.md) on Jubail V100; the deliverable is one DETR.
#   stage 1: 4 inner-fold DETRs (v2 recipe, 12k iterations) on 75 Train cases each -> preds_oof.json of the other 25
#   stage 2 (after all of stage 1): case weights (scripts/boost/boost_weights.py) + out-of-fold hard negatives
#            (scripts/hardneg/mine.py), then the v2 recipe with weighted case sampling and negative-centred patches,
#            seeds 0-2 (36k iterations, swa.pt).
#   bash scripts/slurm/detr_boost_jubail.sh
#   FOLD_EXTRA="train.stop_iter=40 train.eval_every=20 train.swa_last=2" EXTRA="train.stop_iter=40 train.eval_every=20 \
#     train.swa_last=2" NAME_SUFFIX=_smoke SEEDS=0 bash scripts/slurm/detr_boost_jubail.sh      (smoke)
set -euo pipefail
cd "$(dirname "$0")/../.."
git pull -q
C=$(git rev-parse HEAD)
S=/scratch/ll5582/langtian/abus3d
X=${NAME_SUFFIX:-}
mkdir -p "$S/boost"
TRAIN="python scripts/train.py --config configs/detr_v2.yaml paths.prep_dir=$S/data/prep paths.runs_dir=$S/runs"
ids=()
oof=""
for k in 0 1 2 3; do
    name=detr_boostfold_k$k$X
    cmd="$TRAIN name=$name train.seed=0 data.inner_folds=4 data.inner_fold=$k train.iters=12000 train.eval_every=1500"
    cmd+=" train.swa_last=3 train.predict_test=false ${FOLD_EXTRA:-}"
    id=$(COMMIT=$C sbatch --parsable -J "$name" -t 12:00:00 -o "$S/slurm_logs/$name-%j.out" \
         scripts/slurm/jubail_job.sbatch sh "$cmd")
    ids+=("$id")
    oof+=" $S/runs/$name/preds_oof.json"
    echo "$name $id commit $C"
done
dep=$(IFS=:; echo "${ids[*]}")
for s in ${SEEDS:-0 1 2}; do
    name=detr_boost_s$s$X
    W=$S/boost/weights$X.s$s.json
    N=$S/boost/negatives$X.s$s.json
    cmd="python scripts/boost/boost_weights.py --meta $S/data/prep/meta.json --oof$oof --out $W"
    cmd+=" && python scripts/hardneg/mine.py --meta $S/data/prep/meta.json --out $N --detr$oof"
    cmd+=" && $TRAIN name=$name train.seed=$s data.case_weight_file=$W data.neg_file=$N data.neg_prob=0.2 ${EXTRA:-}"
    id=$(COMMIT=$C sbatch --parsable --dependency=afterok:$dep -J "$name" -t ${HOURS:-24}:00:00 \
         -o "$S/slurm_logs/$name-%j.out" scripts/slurm/jubail_job.sbatch sh "$cmd")
    echo "$name $id after $dep commit $C"
done
