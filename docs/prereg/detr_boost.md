# Pre-registration: DETR with AdaBoost-style reweighting (official TDSC split, boxes only, one model)

Written 2026-10-23, before any of these runs (smoke runs only), while the DETR v2 round (`detr_v2.md`) is still
running. User request (10-23): "试一下DETR用AdaBoost"; chosen variant: single-model reweighting (no ensemble; the
deliverable stays one DETR).

## Why not classic AdaBoost
- A boosted ensemble of K DETRs violates the single-model rule.
- Boosting on the model's own training errors does not work here: the DETR memorises Train (in-sample FROC ~0.95,
  E15; boost weights computed in sample on detr_bbox_s0 Train: top-1 hit 0.95, 95 of 100 cases get the same
  weight), the same reason online hard-negative mining found almost nothing.
- So the errors come from **out-of-fold predictions inside Train**: 4 folds of the 100 Train cases (stratified by
  benign / malignant, `medical_one.data.tdsc.inner_split`). Validation and Test are not used; the fold models are
  only used to compute the weights and are never used for inference or selection.

## Procedure (`scripts/slurm/detr_boost_jubail.sh`)
1. Fold models `detr_boostfold_k{0..3}`: v2 recipe (`configs/detr_v2.yaml`), 75 Train cases, 12000 iterations,
   evaluation every 1500, swa.pt = average of the EMA weights at 9k / 10.5k / 12k; `preds_oof.json` = swa.pt on the
   25 held-out Train cases (inference as v2).
2. Case weights (`scripts/boost/boost_weights.py`): error e = 1 - 1 / r with r the rank of the first official hit
   (IoU > 0.3, raw space) among the case's boxes by score (no hit: e = 1); AdaBoost with a real-valued margin:
   eps = mean(e) clipped [0.05, 0.95], alpha = 0.5 ln((1 - eps) / eps), w = exp(alpha (2e - 1)), normalised to mean
   1, clipped to [0.5, 3], mixed 50 / 50 with uniform, normalised again. Every training patch (lesion-centred and
   random) picks its case with these weights.
3. Hard negatives from the same out-of-fold predictions (`scripts/hardneg/mine.py`, score >= 0.3, <= 5 per case,
   away from the GT box enlarged by 25%); 20% of all patches are centred on one (taken from the random share, so
   lesion-centred stays 50%). The loss is unchanged (w_neg = 0): only the sampling distribution changes, as in
   AdaBoost.
4. Final arm `detr_boost_s{0,1,2}`: v2 + steps 2-3, 36000 iterations, swa.pt (identical to the v2 arms otherwise).

## Read-out and adoption
As `detr_v2.md` (Validation only, swa.pt, local FROC with pandas 2.3.3, paired case bootstrap vs v2 same seed,
top-1, max recall, top-1 error types, size tertiles). Adoption: 3-seed mean at least +0.03 over v2 and the paired
bootstrap above 0 in >= 80% of resamples. Also reported: the out-of-fold top-1 / miss rates of the fold models,
the weight distribution, and whether the up-weighted cases are small / shadowed. Test is not scored in this round.

## Risks
| # | Risk | Handling |
|---|---|---|
| A1 | AdaBoost over-weights outliers / label noise (a few unlearnable cases dominate 100 cases) | weights clipped to [0.5, 3] and mixed 50 / 50 with uniform; weight list reported |
| A2 | 75-case fold models err differently from the 100-case final model | weights are a soft sampling prior only; no hard selection |
| A3 | Out-of-fold negatives include unlabelled lesions or the lesion's own shadow | mine.py keeps boxes away from the GT (+25%); loss unchanged, only patch centring |
| A4 | Effect confounded with v2 seed noise (v2 seeds 0.546-0.639) | 3 seeds, paired bootstrap per seed |
| A5 | Compute / queue (4 x ~3 h + 3 x ~7 h on Jubail V100) | Jubail visual partition only; nothing on the 3090 / DGX |
