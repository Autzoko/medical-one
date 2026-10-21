# Pre-registration: expert 1 + expert 2 (CenterDETR3D), official TDSC split

Written and committed before any of these runs started; nothing below is changed after Test is scored.

## Data and protocol
- TDSC-ABUS official split only: train = Train (100), selection = Validation (30), Test (70) scored once.
- Boxes only: training uses the 3D boxes (`meta["box"]`); no mask enters training.
- Same preprocessing as all earlier models (working space 336 x 96 x 336, `preprocess_tdsc.py`).

## Model and training
- `configs/center_detr.yaml` at the commit of this file: CenterDETR3D (ResUNet3D + neck, stride-4 center head,
  two-stage deformable DETR with 300 queries seeded by center peaks), 20000 iterations, batch 2, AdamW 2e-4,
  EMA 0.999, augmentation = lateral zoom 0.2 + Gaussian noise 8 + flips/gamma/contrast (as the earlier aug runs).
- Seeds 0, 1, 2 (`train.seed`), all on Jubail V100 (fp32, `train.amp=false`); one GPU per run.

## Selection (Validation only)
- Checkpoint: every 1000 iterations the EMA model is evaluated on Validation; `best.pt` = highest official
  TDSC FROC of expert 2 (`select_metric: tdsc_froc`). The trainer then also saves Test predictions of
  `best.pt` and `last.pt` for both experts; they are not scored until the Validation decisions below are logged.
- Three outputs per run, all from the same `best.pt`:
  - E2 = expert 2 (DETR) boxes;
  - E1 = expert 1 (center peaks) boxes;
  - E1+E2 = WBF of E1 and E2, IoU 0.5, equal weights (fused score = mean, so it stays in [0, 1]); no tuning.
- Deliverable output = the one of {E2, E1, E1+E2} with the highest mean Validation FROC over the 3 seeds.

## Test reporting
- For each output: Test official FROC per seed, mean and range over seeds, plus AP@0.1/0.3/0.5, top-1 hit,
  max recall and per-FP sensitivities (the checks of `outputs/research/e12_final.py`: score range, sorted,
  valid boxes, independent FROC re-implementation).
- Baseline: standard nnDetection single model (`scripts/baselines/nndet_pipeline.py --docker`, same split).
  Comparison: paired case bootstrap (`scripts/compare_methods.py`) of each seed's deliverable vs nnDetection.
  "Better" = higher mean Test FROC over seeds; "significantly better" = paired 95% CI lower bound > 0.
- References (already known, not targets): single DETR pilot6k 0.581, single nnU-Net 0.644, DETR + nnU-Net
  pre-registered WBF 0.699 (all official split).

## Decision for the next step
- If E1 alone has mean Validation FROC < 0.55, strengthen expert 1 before adding expert 3.
- If E1+E2 does not beat both E1 and E2 on Validation, the experts are not complementary as built; revisit
  before adding the gate.
