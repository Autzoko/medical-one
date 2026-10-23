# Pre-registration: DETR v3 + CPM3D + 3D ROI verifier (cascade; official TDSC split, boxes only)

Written 2026-10-23 before any of these runs (smoke runs only). User decisions (10-23): keep a framework like the
original DETR + YOLO + ResNet pipeline (main detector + complementary detector + second-stage model), not
necessarily those models or 3D + 2D; the complementary detector is a CPM-style detector (not nnDetection /
nnU-Net); negatives and AdaBoost-style weighting are used throughout.

## Evidence that motivates it (Validation, no training)
- E16: 20% of DETR outcomes are near misses (a box touches the lesion, IoU <= 0.3); large lesions get 0.5-0.75x
  boxes (sliding-window truncation: training targets are clipped boxes), tiny lesions 1.5-2.4x; top-1 0.60.
- E17 (oracles on 18 DETR runs, FROC 0.604): repairing the best-overlapping box of each case 0.736 (+0.13);
  moving the hit to rank 1 0.681 (+0.08); both 0.876 (+0.27); in-case rank decay +0.009; self-WBF 0.
- Literature (report `_docs/reports/三维超声肿瘤检测创新方向.md`): the TDSC 2023 winner is CPM-Net based (Test 0.847).

## Components
1. **DETR v3** (`detr_v3_s{0,1,2}`): `configs/detr_v2.yaml` + copy-paste 0.3 + synthetic shadow 0.3 + AdaBoost case
   weights + out-of-fold hard negatives (patches centred on them 20%, `loss.w_neg = 1`). Weights / negatives from
   the inner-fold DETRs of `detr_boost.md` (`scripts/boost/boost_weights.py`, `scripts/hardneg/mine.py`).
2. **CPM3D** (`models/cpm.py`, `configs/cpm.yaml`): ResUNet3D + stride-4 point head (cls / log size / offset);
   CPM-Net matching (7 cells inside the box nearest its center, other inside cells ignored), focal loss
   (alpha 0.75, gamma 2), L1 + GIoU; trained and applied on the **whole** working volume (336 x 96 x 336, one
   window), 12k iterations, batch 2, EMA + swa.pt (8k-12k). Arms: `cpm_s*` (augmentation incl. paste + shadow) and
   `cpm_boost_s*` (+ AdaBoost case weights + out-of-fold hard negatives with `w_neg = 1`).
3. **Verifier / refiner** (`cascade/verifier.py`, `scripts/cascade/train_verifier.py`): ROI = candidate box x 2.5 from
   the whole volume -> 48 x 32 x 48; small 3D ResNet + 7 box scalars -> hit logit, box deltas, malignancy logit;
   2 refinement iterations at inference. Training (Train only): out-of-fold DETR candidates (top 30 per case) +
   16 jittered GT boxes (scale 0.4-2.5x) + 8 random boxes per case per epoch, 40 epochs. AdaBoost: cases sampled
   by their boosting weight; detector candidates weighted clip(exp(|hit - detector score|), 0.5, 3). Stacking
   (boosting step): 4 inner-fold verifiers -> out-of-fold features [logit(detector score), logit(verifier hit)]
   at the refined box -> 2-feature logistic regression = final score (`stack.json`); then one verifier on all Train.
4. **Cascade** (`scripts/cascade/apply_cascade.py`, fixed): DETR (+ CPM) WBF IoU 0.3 equal weights -> top 30 ->
   refine x2 + stacked score -> NMS 0.1 -> <= 50 boxes.

## Read-out (Validation, local official FROC, pandas 2.3.3; Test once at the end)
- CPM vs DETR alone; complementarity (top-1 on DETR-weak cases, case correlation) as in E16.
- Ablation chain per DETR seed: DETR -> + CPM (fuse) -> + verifier rescore -> + refine -> full; for DETR v2 /
  paste / shadow (already trained) and DETR v3.
- Mechanism metrics (pre-registered): near-miss share, large-lesion size ratio, top-1, sensitivity at
  0.125-0.5 FP/scan, far-FP share.
- Adoption of a stage: 3-seed mean FROC +0.03 and paired case bootstrap > 0 in >= 80% of resamples.

## Risks
| # | Risk | Handling |
|---|---|---|
| C1 | CPM on whole volumes exceeds V100 memory / 24 h | smoke run logs peak memory and s/it before the full runs |
| C2 | Our earlier center heads were weak (0.25-0.37) | CPM differs (whole volume, CPM matching, own backbone); read-out reports it alone first |
| C3 | Verifier over-fits ~3000 candidates of 100 cases | small net, dropout, fixed 40 epochs, strong ROI augmentation, jittered / random boxes |
| C4 | Inner-fold (75-case) candidates differ from the final DETR's | jittered GT boxes cover the scale range; read-out per stage on Validation |
| C5 | Stacking fitted on DETR-only candidates, applied to DETR + CPM fused scores | equal-weight WBF keeps scores in [0, 1]; the "+ CPM" stage is reported with and without the verifier |
| C6 | Training-label boundary noise (organisers corrected only Test; Val 116 box 1.47x its main component) | reported per case; Test 175 (box 9.9x its lesion, stray voxels) is unreachable for every method |
| C7 | Validation noise (30 cases) | 3 seeds, paired bootstrap, mechanism metrics |

## Amendment 2026-10-23 (before any full run of this round)
- **CPM3D paused** (user decision): `cpm_s*` / `cpm_boost_s*` are not run in this round (code and smoke results kept;
  measured 9.6 GB, 0.87 s/it). The cascade is DETR v3 -> verifier; the "+ CPM" stage is dropped from the chain.
- **`detr_v3_p256_s{0,1,2}`** (user decision): DETR v3 with 256 x 96 x 256 patches / windows (`data.patch=[256,96,256]`),
  ~14 h per seed. Reason: patch256 passed the v2 rule (Val +0.051, bootstrap 0.85); on Validation its gain was in
  small lesions (top-1 near misses 0.23 -> 0.07, 3 seeds), not in large-lesion box size (0.88 -> 0.86x), so it may
  overlap with copy-paste. Read-out: v3_p256 vs v3, same rule (3-seed mean +0.03, bootstrap > 0 in >= 80%); the
  verifier chain is run on whichever v3 the rule picks (v3 if neither passes).
- Each launcher part writes its own boosting weights / negatives files (same inputs, deterministic), so parallel
  jobs never read a half-written file.
