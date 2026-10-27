# Pre-registration: zero-training diagnostics of the low small-lesion scores (DETR v3_p256)

Written 2026-10-27, before any of these diagnostics was run. Background: `_docs/reports/频域特征与小病灶检测.md`.
User approval (10-27): "好的" to running D1-D5 first, then deciding on training arms.

## Question
Small lesions (smallest volume tertile, cube-root ~21 working voxels, ~10 mm) are found by v3_p256 (any hit 96% on
Test) but ranked low (top-1 0.42; first-hit score median 0.26 vs 0.81 / 0.90 for the mid / large tertiles; the
false positive that outranks them is ~1.8x larger). Which of these causes dominates?
- **H-alias (preprocessing)**: `to_working_space` shrinks the axial (depth) axis 865 -> 112 (x7.7) with one trilinear
  `F.interpolate` and no low-pass, i.e. near point sampling; unaveraged speckle and grid-phase-dependent pseudo
  texture leave small lesions with noisy evidence.
- **H-alias (network)**: the un-filtered `max_pool3d` stages and nearest upsampling of the ResUNet3D make features
  shift-variant.
- **H-resolution**: real axial detail lost at 0.56 mm (the lateral axis is ~2x box-averaged, the sweep axis ~native).
- **Size prior / matching hedging**: not a signal problem; frequency or resampling fixes would not help.

## Models and data
v3_p256 `swa.pt`, seeds 0-2 (`$S/runs/detr_v3_p256_s{0,1,2}`), inference as in training (patch 256x96x256, eval
post-processing of `configs/detr_v2.yaml`), no TTA unless stated. Raw NRRD volumes are read only inside Slurm jobs on
Jubail; only predictions (boxes + scores) and per-case summary numbers leave the cluster. No training, no Test-based
selection. Script: `scripts/diag/small_lesion_diag.py`, launcher `scripts/slurm/small_lesion_diag_jubail.sh`,
read-out `scripts/diag/small_lesion_readout.py`.

Every input variant is built either from the prepared working volume (`prep`) or from the raw volume by our own
separable linear resampler, which reproduces `to_working_space` exactly when no shift / filter / zoom is applied
(checked per case: `base_rs` vs `prep`, reported). Predicted boxes are mapped back to the base working grid before
scoring, so every variant is scored with the official raw-space IoU > 0.3.

| ID | Variant(s) | Built from | Split(s) |
|---|---|---|---|
| D1 | `base` | prep | Train (in-sample), Validation, Test |
| D2 | `phase1..3`: axial sampling grid shifted by q/4 working voxel (q = 1, 2, 3); `base` = q 0 | raw | Validation, Test |
| D2b | `shift_{d,h,w}{+1,-1}`: whole working volume shifted by one voxel (edge replicated) | prep | Validation, Test |
| D3 | `aa0.5`, `aa1`, `aa1.5`: Gaussian pre-filter before resampling, sigma = c (f - 1) / 2 input voxels per axis (f = shrink factor; skimage's anti-aliasing default at c = 1), same grid | raw | Validation, Test |
| D4 | `z2`: prep upsampled x1.4 (trilinear); `z1n`: raw resampled to a x1.4 grid, no pre-filter; `z1`: same with the pre-filter; windows on all three axes | prep / raw | Validation, Test |
| D5 | CPU signal measurements: NRRD header spacing; speckle autocorrelation FWHM per axis in a parenchyma block beside the lesion (native); lesion CNR (inscribed ellipsoid of the GT box x0.8 vs shell 1.3-1.8) in `prep` vs `aa1`; the same appearance numbers for seed 0's top-5 boxes | raw | Validation, Test |

## Read-outs
Per variant, seed and size tertile (tertiles within each split, as before): top-1 hit, any hit, first-hit score
(median, share < 0.3), FROC and sensitivity at 0.125 FP/scan. Per lesion: the **lesion score** = highest score of a
box with official IoU > 0.3 (0 if none). Paired per-lesion changes are pooled over the 3 seeds (Wilcoxon
signed-rank, two-sided). Sanity: `base` FROC must equal the stored `preds_*_swa.json` (difference <= 0.005).

## Decision rules (fixed now; decisions use Validation, Test is reported as a descriptive replication)
- **D1**: Train small-tertile first-hit score median < 0.5 -> systematic size bias (training-level). Median >= 0.7 and
  Train small top-1 >= 0.8 -> generalisation gap (memorised in sample).
- **D2 (H-alias, preprocessing)** supported if both: (a) the median per-lesion standard deviation of the lesion score
  across the 4 phases is >= 0.05 for the small tertile and >= 2x that of the large tertile; (b) the 4-phase WBF
  (IoU 0.5) raises the small-tertile lesion score (Wilcoxon p < 0.05, median change > 0).
- **D2b (H-alias, network)**: same rule (a) across `base` + the 6 one-voxel shifts.
- **D3**: positive if some c raises the small-tertile lesion score (Wilcoxon p < 0.05, median change > 0) without
  lowering the 3-seed mean FROC by more than 0.02. A drop is **not** evidence against H-alias (the model never saw
  filtered inputs; only 20% blur augmentation).
- **D4**: `z2` vs `base` significant rise -> apparent-size prior; `z1n` vs `z2` significant rise -> real detail
  (H-resolution); `z1` vs `z1n` -> anti-aliasing at the finer grid (descriptive).
- **D5**: median CNR(aa1) / CNR(prep) >= 1.3 for small lesions -> physical premise of H-alias holds.

Next step, proposed to the user with the read-out (no run is started automatically):
- anti-aliased preprocessing arm (+ matched-smoothness control) if D2 or D3 is positive, or D5 holds with D2(a);
- 3D Haar / BlurPool downsampling arms if D2b is positive;
- 2x axial input arm only if D4 shows a real-detail effect;
- stop the frequency direction if D2, D2b and D3 are all negative; if D1 / D4 point to a size prior, move to
  size-aware targets / multi-scale inference and the complementary-model plan instead.

## Risks
| # | Risk | Handling |
|---|---|---|
| R1 | Only 10 small lesions in Validation: low power | pooled over 3 seeds, per-lesion paired tests; Test reported as replication; an inconclusive result is reported as such |
| R2 | Resampler differs from `to_working_space` | `base_rs` vs `prep` per case (max abs difference, share of differing voxels); `base` sanity vs stored predictions |
| R3 | Filtered / zoomed inputs are out of distribution | D3 read one-way; zoom 1.4 is inside the training zoom range 0.7-1.4 |
| R4 | Box mapping errors (flip, half-voxel conventions) | all mappings are affine on box edges; `phase0` = `base` and `z` mapping checked on the GT-box hit rate of `base` |
| R5 | Login-node work / data leaving the cluster | everything in Slurm jobs; no image data is written outside $S or committed |
| R6 | Multiple comparisons (3 sigmas, 3 zooms, 4 diagnostics) | rules fixed above; everything else descriptive |
