#!/bin/bash
# Flip-TTA predictions of trained CV runs on Jubail (no training): for every <arm>_f<k>, best.pt is run on its
# held-out fold (OOF) and on Test with eval.tta_flips=true; written next to the run as preds_{oof,Test}_tta.json.
#   bash scripts/slurm/tta_jubail.sh arm [arm ...]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -n "${PULLED:-}" ] || { git pull -q; PULLED=1 exec bash "scripts/slurm/$(basename "$0")" "$@"; }  # rerun the pulled script text
C=$(git rev-parse HEAD)
S=/scratch/ll5582/langtian/abus3d
LANES=${LANES:-8}
LIST=$S/tta_jobs.txt
touch "$LIST"
for arm in "$@"; do
    for k in 0 1 2 3 4; do
        r=$S/runs/${arm}_f$k
        lane=$(( $(wc -l < "$LIST") % LANES ))
        cmd="for sp in OOF Test; do python scripts/evaluate.py $r/best.pt --split \$sp --cv-fold $k \
--save-preds $r/preds_\${sp}_tta.json --set paths.prep_dir=$S/data/prep eval.tta_flips=true; done"
        id=$(COMMIT=$C sbatch --parsable -J "tta_L$lane" --comment "tta_${arm}_f$k" --dependency=singleton -t 6:00:00 \
             -o "$S/slurm_logs/tta_${arm}_f$k-%j.out" scripts/slurm/jubail_job.sbatch sh "$cmd")
        echo "tta_${arm}_f$k $id" >> "$LIST"
    done
done
tail -n $(( $# * 5 )) "$LIST"
