# Pre-registration: CenterDETR3D v1 (expert 1 + expert 2 with training / model tricks), official TDSC split

Written and committed before the runs start; nothing below changes after Test is scored.

## Data and protocol
- Official split only: train = Train (100), selection = Validation (30), Test (70) scored once.
- Boxes only: training reads the 3D boxes, never the masks.

## Model and training (`configs/center_detr_v1.yaml` at the commit of this file)
- Base: CenterDETR3D (ResUNet3D + neck, stride-4 center head = expert 1, two-stage deformable DETR = expert 2,
  300 one-to-one queries seeded by expert-1 peaks; peaks ranked first, then non-peak cells by heat).
- Expert 2: Rank-DETR GIoU-aware classification loss and high-order matching (-p IoU^4) from 75% of training;
  H-DETR one-to-many branch (600 queries, k = 6, weight 1); DINO contrastive denoising (dn_number 100, noise 1.0)
  and look-forward-twice; up to 10 mined hard negatives per patch as background queries (weight 1).
- Expert 1: loss weights x2 (heat / size / offset / GIoU).
- Data: lateral zoom 0.2, Gaussian noise 8, lateral flips, gamma / contrast, box-exact lateral axis swap
  (p 0.5), Gaussian blur (p 0.2), speckle (p 0.2), low-resolution simulation (p 0.25), inverted gamma (p 0.1);
  online hard-negative mining every 2000 iterations with the EMA model on Train (score >= 0.3, 5 per case,
  merged with earlier ones), 20% of patches centred on a negative.
- 20000 iterations, batch 2, AdamW 2e-4, EMA 0.999, evaluation every 1000 iterations;
  swa.pt = average of the EMA weights at the last 5 evaluations (16k..20k).
- Seeds 0, 1, 2 on Jubail V100 (fp32), one GPU each.

## Selection (Validation only)
- Checkpoint: swa.pt (no Validation peak picking). best.pt (highest Validation FROC of expert 2 among the 20
  evaluations) is kept and reported as secondary only.
- Post-processing grid on swa.pt (`scripts/finalize_center.py`, 9 cells): output {E2, E1, E1+E2 (WBF IoU 0.5,
  equal weights, mean score)} x per-window NMS IoU {0.05, 0.1, 0.2}. The chosen cell is the one with the highest
  mean Validation official FROC over the 3 seeds (ties: E2 before E1+E2 before E1, then smaller NMS).
- Deliverable model: the seed with the highest Validation FROC in the chosen cell.

## Test reporting
- Test predictions of every cell are written by the finalize step but only the chosen cell is scored: per seed,
  mean and range, plus AP@0.1/0.3/0.5, top-1 hit, max recall, per-FP sensitivities, with the checks of
  `outputs/research/e12_final.py` (score range, sorted, valid boxes, independent FROC re-implementation).
- Main comparison: deliverable vs the standard nnDetection single model (same split), paired case bootstrap.
  "Better" = higher Test FROC; "significantly better" = paired 95% CI lower bound > 0. Also the mean over seeds.
- Secondary: vs the baseline CenterDETR3D run center_detr_s0 (commit f92521f; note v1 also contains the peak
  ranking fix of commit b5ab7c5) on Validation and Test.
