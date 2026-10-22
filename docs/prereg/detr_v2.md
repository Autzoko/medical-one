# Pre-registration: DETR v2 recipe and single-factor arms (official TDSC split, boxes only)

Written 2026-10-22 before any of these runs (40-iteration smoke runs only). Motivation:
- E15 (DETR rerun vs CoDETR3D, same inference, Validation): no detectable difference (+0.023, 95% CI -0.083..0.107);
  both memorise Train (FROC 0.95 / 0.96 vs Validation 0.62 / 0.65); CoDETR3D is worse on small lesions (top-1 0.37 vs
  0.70 in the smallest tertile) and costs ~3x. Decision (user, 10-22): continue with the original DETR.
- Literature review (`_docs/reports/框监督3D病灶检测训练方法.md`, outside the repo): our training sees ~24k patches vs
  ~600k for nnDetection; frameworks train a fixed long schedule with strong augmentation and use averaged weights.
- Top-1 errors are mostly far false positives (~2/3); in-sample mining finds nothing because Train is memorised.

## Base recipe "v2" (`configs/detr_v2.yaml`)
Original two-stage deformable DETR (`configs/abus3d_tdsc.yaml`, 900 queries, focal + L1 + GIoU) with, together
(one bundled change, ablated later only if it helps): 36000 iterations (was 6000), three-axis zoom
z ~ U(0.7, 1.4) x U(0.9, 1.1) per axis with p 0.25 (box-exact, zero-padded), the intensity set of DenseDet3D /
CoDETR3D (noise 8, blur 0.2, speckle 0.2, low-res 0.25, inverted gamma 0.1), positive-patch share 0.5 (was 0.7),
transformer dropout 0.1, EMA 0.999, and **swa.pt = average of the EMA weights at 30k, 31.5k, 33k, 34.5k, 36k**.
Inference: score threshold 0, window NMS 0.1, cross-window NMS 0.1, <= 50 boxes (as CoDETR3D / E15).

## Single-factor arms (each = v2 + one change)
| Arm | Change | Targets |
|---|---|---|
| pos033 | positive-patch share 0.5 -> 0.33 | far false positives (train/test patch prior mismatch) |
| paste | copy-paste of another training lesion (box + 20% context, ellipsoid blend, donor depth, lateral flips), p 0.3 | over-fitting, small lesions |
| patch256 | patch / window 176x96x176 -> 256x96x256 (2 x 2 windows) | far false positives (more context per window) |
| small | lesion-centred patches pick cases with weight (volume / median)^-0.5, clipped [0.5, 2] | small lesions |
| shadow | synthetic acoustic shadow, p 0.3 (lateral Gaussian footprint, x (1 - s) towards the deep side, s ~ U(0.25, 0.8)) | shadow false positives |
| iouaware | Rank-DETR GIoU-aware classification | diagnostic: expected to hurt small lesions (E15 hypothesis) |

Not in this round (user decision pending / later): negative mining from folds inside Train, size-normalised L1,
box-supervised segmentation, pre-training, CPM-Net.

## Runs
7 arms x seeds 0, 1, 2 = 21 runs on Jubail V100 (fp32, batch 2), `scripts/slurm/detr_v2_jubail.sh`; Test predictions are
written at the end and not scored.

## Read-out (Validation only; deliverable checkpoint = swa.pt)
- Per run: Validation official FROC of swa.pt (recomputed locally, pandas 2.3.3), top-1 hit, max recall, top-1 error
  types (hit / near 0 < IoU <= 0.3 / far IoU = 0), top-1 hit by size tertile; smoothed Validation curve (mean of the
  last 5 evaluations); Train - Validation FROC gap for v2 and adopted arms (separate inference job).
- v2 vs the 6k DETR rerun (E15 numbers, best.pt): reported, not a selection.
- Arm adoption: mean over seeds of swa.pt Validation FROC at least +0.03 over v2 AND the paired case bootstrap
  (same seed vs same seed, 30 cases) puts the difference above 0 in >= 80% of resamples; an arm that improves its
  target metric (small-tertile top-1, far-FP share) without lowering FROC is reported as "neutral-positive".
  Adopted arms are combined in the next round (a new pre-registration); no Test scoring in this round.

## Risks
| # | Risk | Handling |
|---|---|---|
| V1 | 36k iterations still over-fit (or under-fit with strong augmentation) | 24 evaluations logged; smoothed curve; SWA over the last 21% |
| V2 | Zero padding in depth (zoom-out) creates unrealistic black slabs | p 0.25 only; deep side is often zero-filled anyway (40% zeros at h = 0) |
| V3 | Pasted lesions look artificial (seams, mismatched depth, shadows of the donor) | donor depth kept, context + ellipsoid blend; a test checks box exactness; read out far-FP share |
| V4 | 256 patch exceeds V100 memory or 24 h | smoke run logs peak memory; batch 2 kept; job limit 24 h |
| V5 | Shadow direction wrong | measured: h = 0 is the deep side (intensity / zero profile); unit test: deeper never brighter |
| V6 | Validation noise (SD ~0.09, seed spread ~0.1) | 3 seeds, fixed length, averaged weights, paired bootstrap, size strata |
| V7 | Shared Jubail GPUs (user's own jobs) | visual partition only; jobs queue, no preemption |
