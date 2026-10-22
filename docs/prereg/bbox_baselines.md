# Pre-registration: bbox-only baselines (official TDSC split)

Written 2026-10-22, before any of these runs. Rule from now on: **every baseline gets only the 3D boxes**, never the
lesion masks (the follow-up dataset has only boxes). Methods that need a mask get a cuboid mask of the box
(`medical_one.data.tdsc.box_mask`: voxels [round(lo), round(hi)) of the working-space box of meta.json, written by
`scripts/preprocess_tdsc.py --box-masks` to `<prep_dir>/box_masks/{Train,Validation}`). Earlier mask-trained numbers
(nnU-Net 250 ep, Test 0.644; the nnDetection run of 10-21, stopped at 10 / 50 epochs) are reference only.

## Runs
- **nnDetection, boxes** (`b_nndet_box`, 3090 GPU 0, official image `nndetection:97a58f3`): exactly the standard run
  (`scripts/baselines/nndet_pipeline.py`, RetinaUNetV001 / D3V001 plan, 50 x 2500 batches, nnDetection's own
  sweep on Validation, default test-time mirroring), only `--masks-dir <prep>/box_masks`. Separate data directory
  and run name; the stopped mask run is not touched.
- **nnDetection, boxes, no segmentation loss** (`b_nndet_box_noseg`, 3090 GPU 0 after `b_nndet_box`; added 2026-10-22
  before either nnDetection box run finished): identical, but module `RetinaUNetV001NoSeg`
  (`scripts/baselines/nndet_plugins/`): the segmentation loss is weighted 0, so only the box classification /
  regression supervise the network (nnDetection still needs the cuboid "mask" as its input format, for box extraction,
  foreground sampling and planning). The untrained segmentation head stays because nnDetection's evaluation and
  test-time mirroring read it. A 1-epoch x 20-batch smoke run goes first. Answers how much of nnDetection's result comes
  from its segmentation branch when only boxes exist.
- **nnU-Net, boxes** (`b_nnunet_box`, Jubail V100): `scripts/baselines/nnunet_pipeline.py`, 3d_fullres, 250 epochs,
  fold 0 = Train / Validation (as the mask run), `--masks-dir <prep>/box_masks`. Box variant (threshold / score) =
  highest Validation FROC of the four variants (ties: listed order).
- **Original DETR, rerun** (`detr_bbox_s{0,1,2}`, Jubail V100, fp32): `configs/abus3d_tdsc.yaml` (two-stage deformable
  DETR, 900 queries) with 6000 iterations (the length of pilot6k, the best original DETR) and Validation every
  1000; best.pt = highest Validation official FROC. It already used boxes only; rerun so that it is trained and
  scored under the same conditions as DenseDet3D / CoDETR3D. Deliverable seed 0 (fixed now); 3-seed mean and range
  reported.

## Risks and what we do about them
| # | Risk | Handling |
|---|---|---|
| B1 | Cuboid labels mark the tissue around the lesion as foreground: the real lesion fills on average 24.7% of its box (median 24.1%, min 10.0%; Train + Validation, measured 2026-10-22) | Inherent to giving boxes to a mask-based method; detection targets (boxes) are unchanged; the no-segmentation-loss run isolates this |
| B2 | nnDetection's +-30 deg rotation (p 0.2) re-derives boxes from the rotated cuboid, which are looser (up to ~1.37x per axis) | Kept: the baseline is the default nnDetection; noted as a known bias |
| B3 | Rounding to voxels moves each face by <= 0.5 voxel | Unit test; boxes >= ~10 voxels per axis, so negligible |
| B4 | Overwriting the stopped mask run or its preprocessed data | New run names and data directories |
| B5 | Host RAM on the 3090 (the 10-21 OOM) | One nnDetection job only, 6 augmentation workers, status checks |
| B6 | nnU-Net is new on Jubail (own env) | Same nnU-Net v2 major version as the mask run; 1-epoch smoke test first |
| B7 | 30-case Validation is noisy (bootstrap SD ~0.09) | No tuning beyond the pre-registered choices; Test once |
| B8 | The DETR rerun differs from pilot6k's code of 10-14 (EMA, official-FROC selection, eval cadence) | Reported as a fresh baseline; pilot6k stays a historical number |

## Test reporting
Once, after all bbox-only baselines and the DenseDet3D / CoDETR3D runs are final; scored locally (pandas 2.3.3) with
the E12 checks. Main comparisons: CoDETR3D and DenseDet3D deliverables vs nnDetection (boxes), paired case bootstrap;
secondary: vs nnU-Net (boxes) and the original DETR rerun.
