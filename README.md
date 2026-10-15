# medical-one

3D lesion detection for automated breast ultrasound (ABUS). The first model is a PyTorch
reimplementation of the 3D detector reverse-engineered from the ABUS-2.3.0 release
(see the audit report `ABUS_3D检测器完整技术报告.md`), trained from scratch on TDSC-ABUS.

## Model

| Part | Configuration (from the exported ONNX graph) |
|---|---|
| Input | 1×1×176×96×176 patch, uint8 → `(x/255-0.5)/0.5` |
| Backbone | 3D residual encoder-decoder; features de2 (stride 4), de3 (8), s4 (16) |
| Neck | 1³ conv to 192 ch + GroupNorm(24) per level |
| Transformer | two-stage deformable DETR, 3 encoder / 6 decoder layers, 6 heads, 4 points, 900 queries |
| Output | 900 logits + 900 boxes (d0,h0,w0,d1,h1,w1) in [0,1] |

The architecture is confirmed by the export. The training recipe (Hungarian matching, focal
loss, L1 + 3D GIoU, auxiliary losses) and a few details (backbone norm, position-encoding
constants, proposal priors) are our own choices, since the original training code is unavailable.

## Layout

```
configs/                experiment configs (YAML)
scripts/                entry points: preprocess, train, evaluate, benchmark
  slurm/                DGX Spark job script
src/medical_one/
  models/               backbone, 3D deformable attention, transformer, detector
  data/                 TDSC-ABUS preprocessing, patch dataset
  losses/               set criterion (Hungarian + focal + L1 + GIoU)
  engine/               trainer, sliding-window inference
  evaluation/           FROC / hit-rate metrics
  utils/                3D box ops
tests/                  CPU unit tests
```

## Usage

```bash
pip install -e ".[dev]"
python scripts/preprocess_tdsc.py --config configs/abus3d_tdsc.yaml
python scripts/train.py --config configs/abus3d_tdsc.yaml name=run1 train.iters=20000
python scripts/evaluate.py <run_dir>/best.pt --split Test
pytest
```

Any config value can be overridden on the command line as `section.key=value`
(`evaluate.py` takes them after `--set`). Several checkpoints passed to `evaluate.py` are ensembled;
`eval.tta_flips=true` adds flip test-time augmentation, both fused with weighted box fusion.

### DGX Spark cluster (Slurm)

Compute nodes are aarch64 GB10 and do not mount the login node's `/home`; data, runs and logs live on
the shared mount `/media/cvpr/langtian/abus3d`. Submit from the repo on `dgx-login`:

```bash
COMMIT=$(git rev-parse HEAD) sbatch -J exp1 scripts/slurm/dgx_job.sbatch train name=exp1 train.iters=10000
```

## Data preprocessing

TDSC-ABUS NRRD arrays are `[N0, N1, N2]` with N0 the depth axis (0.073 mm). Following the deployed
pipeline: transpose to `[N1, N0, N2]`, flip depth, trilinear resize to 336×112×336, keep the first
96 depth slices. Lesion boxes are derived from the masks; they agree with the official `bbx_labels.csv`.
Training samples random 176×96×176 patches; evaluation runs 9 overlapping windows per volume.

## Evaluation

Lesion-level metrics at IoU 0.1 / 0.3: top-1 hit rate, recall, and FROC (mean sensitivity at
0.125–8 FPs per volume). This protocol is ours and has not been checked against the official
TDSC challenge script.
