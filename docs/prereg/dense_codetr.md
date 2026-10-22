# Pre-registration: DenseDet3D and CoDETR3D (official TDSC split)

Written before any training of these models (2026-10-22). Motivation: E13/E14 (CenterDETR3D v1 negative; the
detectors over-fit the 100 training cases, Validation peaks at 4k-7k iterations, sparse one-to-one supervision,
in-sample mining finds nothing, rot90 breaks the d/w shape prior). nnU-Net / nnDetection do not over-fit; the main
difference is dense supervision with in-batch hard negatives. Here we learn nnDetection's dense detection recipe
(re-implemented, no nnDetection code or planner) and test it alone and as Co-DETR auxiliary heads for a DETR.

## Arms (each: seeds 0, 1, 2; Jubail V100, fp32)
- **DenseDet3D** (`configs/dense_det.yaml`): ResUNet3D + neck (strides 4/8/16), center heat-map head (stride 4,
  as expert 1 of CenterDETR3D) + dense anchor-based box head on the three levels.
  - Box head as nnDetection `RetinaUNetV001` / `v001.yaml` (commit 97a58f3): shared towers (1 conv, 128 ch, GN),
    learned per-level regression scale, prior 0.01; ATSS (4 candidates per level and anchor shape, threshold
    mean + std, no center-in-box); batched hard-negative sampler (32 anchors per image, positive fraction 0.33,
    negatives = 2 x positives drawn from the 20 x highest-scoring negatives, at least 1); BCE (mean over sampled)
    + GIoU (sum over sampled positives / their number).
  - Anchor shapes: our rule, k-means (log space) on the Train box sizes, 3 per level, small to large.
  - No segmentation head (bbox-only constraint).
- **CoDETR3D** (`configs/co_detr.yaml`): two-stage deformable DETR (3 enc / 6 dec, 60 queries) on the same
  backbone, Rank-DETR losses (GIoU-aware classification; high-order matching -p * IoU^4 over the last 25%).
  Co-DETR (ICCV 2023, Sense-X/Co-DETR `co_deformable_detr_r50_1x_coco.py`): the dense box head and the heat-map
  head are auxiliary heads on the encoder memory (reshaped to per-level maps); dense-head loss weight
  lambda_2 x decoder layers = 12; the dense head's ATSS positives (at most 300 per image, padded to >= 9 with random
  negative anchors trained as background) are customised positive queries in a separate decoder pass with fixed
  targets, weight lambda_1 = 1. Deviations: one ATSS-like aux head instead of ATSS + Faster R-CNN; the heat-map head
  is the second aux head (no positive queries from it); the positive queries share the main query projection.

## Shared settings
- Data: official split, train = Train (100), selection on Validation (30), Test (70) predicted at the end only.
- Patch sampling: 50% of patches keep the lesion center inside (nnDetection's foreground share), 50% random.
- Augmentation: lateral flips, lateral zoom 0.8-1.2, gamma / contrast / brightness, Gaussian noise (std 8), blur 0.2,
  speckle 0.2, low-res 0.25, inverted gamma 0.1. **No rot90** (E14).
- 12000 iterations, batch 2, AdamW 2e-4 (warmup 500, cosine), grad clip 0.1, EMA 0.999, Validation every 1000.
- Inference: 3 x 3 sliding windows, top 300 per window, **no score threshold**, window NMS, center keep, cross-window
  NMS 0.1, at most 50 boxes per volume.
- Checkpoint: best.pt = highest Validation official FROC of the first output (DenseDet3D "dense", CoDETR3D "detr")
  among the 12 evaluations. No SWA.

## Selection (Validation only)
- Grid on best.pt (`scripts/finalize_center.py`): output {dense, dense_heat} (DenseDet3D) / {detr, dense_heat}
  (CoDETR3D) x window NMS IoU {0.1, 0.2}. `dense_heat` scores a dense box by sqrt(classifier x heat at its center).
- Cell = highest mean Validation FROC over the 3 seeds (ties: order listed, then smaller NMS).
- **Deliverable seed = seed 0, fixed now** (E14: a 30-case Validation cannot rank seeds; bootstrap SD 0.094).
  The 3-seed mean and range are reported alongside.

## Test reporting (once, after all runs and the nnDetection baseline are final)
- Chosen cell of each arm: FROC per seed, mean, range, AP@0.1/0.3/0.5, top-1 hit, max recall, sensitivities, with
  the E12 checks; scored locally (pandas 2.3.3, as E12/E13).
- Main comparison: each arm's deliverable vs the standard nnDetection single model (paired case bootstrap);
  "better" = higher Test FROC, "significantly better" = 95% CI lower bound > 0. Secondary: vs nnU-Net (0.644),
  DETR pilot6k (0.581), CenterDETR3D baseline (0.565) and v1 (0.456) of E12/E13, and DenseDet3D vs CoDETR3D.

## Amendment 2026-10-22 (before any full training run; only 40-iteration smoke runs existed, Validation FROC 0)
After the risk analysis (`_docs/analysis/3D检测潜在问题与对策调查.md`, outside the repo): positive-patch share
0.7 -> 0.5 (both arms; R3) and CoDETR3D queries 300 -> 60 (R2). The other proposed changes (ellipse-box rotation,
shadow augmentation, Train-subset monitoring, smoothed checkpoint selection, size-stratified reporting) need code
and go to the next round; size-stratified Test results are added to the report as a secondary analysis only.
