# Pre-registration: nnU-Net / nnDetection augmentations that DETR v2 does not have yet (official TDSC split, boxes only)

Written 2026-10-23 before these runs (smoke runs only). User request (10-23): use the nn frameworks' augmentations in
DETR as well. Context: on the official split, nnU-Net trained on cuboid box masks reached Test 0.682 vs DETR v2
0.627 (`_docs/reports/Test对比 官方划分 只用框.md`); its edge is ranking (sensitivity 0.61 at 0.125 FP/scan vs
0.31-0.53), not recall (0.70 vs 0.83-0.86).

## What DETR v2 already has and what is missing
| nnU-Net v2 default (nnDetection similar) | DETR v2 (`configs/detr_v2.yaml`) |
|---|---|
| scaling 0.7-1.4, p 0.2 | three-axis zoom 0.7-1.4 (+-10% per axis), p 0.25 |
| Gaussian noise, blur (p 0.2), simulated low resolution (p 0.25) | noise (std 8 / 255), blur p 0.2, low resolution p 0.25, speckle p 0.2 |
| gamma p 0.3 and inverted gamma p 0.1 (0.7-1.5) | gamma on every patch (exp U(-0.3, 0.3)), inverted gamma p 0.1 |
| multiplicative brightness 0.75-1.25, p 0.15 | brightness 0.85-1.15 and offset +-0.08, every patch |
| mirroring of all axes, p 0.5 each | lateral axes d and w only (depth h carries acquisition physics: shadows point to the deep side) |
| **rotation +-30 deg, p 0.2 per axis** | **missing** |
| **contrast 0.75-1.25, p 0.15** | **missing** |
| **test-time mirroring** (all axes) | **off** (`eval.tta_flips` exists: d / w / dw flips, WBF IoU 0.5) |

## Runs
- **N1, flip TTA (inference only).** swa.pt of `detr_{v2,paste,shadow,patch256}_s{0,1,2}` on Validation with
  `eval.tta_flips=true` (fixed settings: d, w and dw flips, WBF IoU 0.5; no h flip since training never flips h).
  `scripts/slurm/tta_v2_jubail.sh`, one job.
- **N2, `detr_nnaug_s{0,1,2}`.** DETR v2 + rotation about the depth axis h, U(-30, 30) deg, p 0.2 + contrast
  0.75-1.25, p 0.15 (`data.rot_prob=0.2 data.contrast_prob=0.15`), single factor on v2 like the other v2 arms,
  36k iterations, swa.pt. `scripts/slurm/detr_v2_jubail.sh nnaug`.
  Rotation without masks: the patch is rotated (bilinear, zero outside); each box becomes the hull of its rotated
  inscribed ellipsoid (unit-tested to +-2 voxels on ellipsoids). Only the lateral plane, like the flips.

## Read-out (Validation, local official FROC, pandas 2.3.3; Test only in the final comparison)
- N1: per arm, mean over seeds of TTA vs no-TTA (same checkpoint), paired case bootstrap; adopt TTA if +0.03 and
  > 0 in >= 80% of resamples (same rule as the v2 arms); top-1 hit, far-FP share, sensitivity at 0.125 FP.
- N2: the v2 rule (`detr_v2.md`): mean swa.pt Validation FROC >= v2 + 0.03 and paired bootstrap > 0 in >= 80%;
  mechanism metrics: near-miss share and large / small lesion size ratio.

## Risks
| # | Risk | Handling |
|---|---|---|
| N-R1 | Lateral voxels are anisotropic (d is resampled 865 -> 336, w 330 -> 336), so a voxel-space rotation is a rotation plus shear physically; the lateral axis swap (rot90) hurt in E14 | +-30 deg only (not 90), p 0.2; read-out vs v2 decides |
| N-R2 | Boxes after rotation are approximate (lesions fill 24.7% of their box on average; an ellipsoid fills 52%) | Ellipsoid hull instead of the box hull (that one would be up to 1.41x larger); near-miss share and size ratio are read out |
| N-R3 | Box-level flip TTA did not help before: 130-case OOF of the CV runs, +0.002 (jb_base) and -0.053 (jb_oof); nnU-Net averages dense probabilities before thresholding, which DETR's set output cannot do | Inference only, cheap; adopted only by the rule above |
| N-R4 | Jubail shared with the user's own jobs (sopt-*) | 3 training jobs + 1 short inference job; no more than the earlier v2 arms |
| N-R5 | 30-case Validation is noisy | 3 seeds, paired bootstrap, mechanism metrics |
