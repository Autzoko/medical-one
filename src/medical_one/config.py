"""Experiment configuration: YAML file -> nested dataclasses, with ``key.sub=value`` CLI overrides."""
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class PathsConfig:
    tdsc_root: str = "~/data-001/langtian/data/medical-image/ABUS_TDSC"
    extracted_train: str = "~/langtian/abus3d_repro/data/raw"  # parent of the unzipped Train/
    prep_dir: str = "~/langtian/abus3d_repro/data/prep"
    runs_dir: str = "~/langtian/abus3d_repro/runs"


@dataclass
class ModelConfig:
    arch: str = "detr"  # "detr": ABUSDet3D; "center_detr": expert 1 (center heat map) + expert 2 (DETR), CenterDETR3D
    norm: str = "gn"
    dim: int = 192
    num_queries: int = 900
    dropout: float = 0.0
    center_hidden: int = 96  # center_detr: channels of the center-head branches
    # center_detr training-only query groups and decoder refinement (all off = the baseline CenterDETR3D)
    num_o2m: int = 0  # one-to-many queries (H-DETR hybrid matching); targets repeated loss.o2m_k times
    dn_number: int = 0  # DINO contrastive denoising: groups = dn_number // (2 * max GT per patch)
    dn_noise: float = 1.0  # DINO box noise scale (positives move < 1 x half-size per corner, negatives 1..2 x)
    num_neg_queries: int = 0  # mined hard negatives fed as background queries (needs data.neg_refresh_every)
    look_forward_twice: bool = False  # DINO: box refinement keeps the gradient to the previous layer's box


@dataclass
class DataConfig:
    pos_prob: float = 0.7
    margin: int = 24
    num_workers: int = 8
    scale_range: float = 0.0  # lateral zoom augmentation, e.g. 0.2 -> crop 0.8x..1.2x of the patch
    noise_std: float = 0.0  # Gaussian noise in uint8 units
    train_splits: List[str] = field(default_factory=lambda: ["Train"])  # e.g. [Train, Validation] for a final fit
    cv_folds: int = 0  # > 0: pool Train + Validation and hold out fold ``cv_fold`` of ``cv_folds`` for validation
    cv_fold: int = 0
    neg_file: str = ""  # hard negatives (``hardneg.save_negatives``); empty = none
    neg_prob: float = 0.0  # share of patches centred on a hard negative (taken from the random patches)
    neg_refresh_every: int = 0  # > 0: every N iterations re-mine negatives with the current (EMA) model
    neg_min_score: float = 0.3  # online mining: score threshold, boxes per case, keep earlier negatives
    neg_top: int = 5
    neg_keep_old: bool = True
    # extra augmentation (0 = off); box-exact only, since training has boxes but no masks
    rot90_prob: float = 0.0  # swap the two lateral axes (d <-> w); with the flips: all 8 lateral symmetries
    blur_prob: float = 0.0  # Gaussian blur, sigma U(0.5, 1) voxels (nnDetection base_more: p 0.2)
    speckle_prob: float = 0.0  # multiplicative noise x * (1 + N(0, s)), s ~ U(0, 0.1) (ultrasound speckle)
    lowres_prob: float = 0.0  # simulate low resolution: zoom U(0.5, 1) nearest down, linear up (nnU-Net: p 0.25)
    inv_gamma_prob: float = 0.0  # gamma on the inverted image (nnDetection base_more: p 0.1)


@dataclass
class LossConfig:
    w_cls: float = 2.0
    w_l1: float = 5.0
    w_giou: float = 2.0
    w_neg: float = 0.0  # unmatched predictions overlapping a hard negative get (1 + w_neg) x classification loss
    neg_iou: float = 0.3  # ... when their IoU with it exceeds this
    # center_detr: expert-1 loss (penalty-reduced focal heat + L1 log size + L1 offset + GIoU at center cells)
    w_center_heat: float = 1.0
    w_center_size: float = 1.0
    w_center_offset: float = 1.0
    w_center_giou: float = 1.0
    center_sigma_ratio: float = 0.1667  # Gaussian sigma per axis = ratio x box size (cells), at least min_sigma
    center_min_sigma: float = 0.5
    # center_detr expert-2 extras
    o2m_k: int = 6  # H-DETR: each target repeated k times for the one-to-many queries
    w_o2m: float = 1.0
    w_dn: float = 1.0
    w_neg_queries: float = 1.0
    iou_aware_cls: bool = False  # Rank-DETR GIoU-aware classification loss
    high_order_alpha: float = 4.0  # Rank-DETR high-order matching cost -p * IoU^alpha ...
    high_order_start: float = 1.0  # ... from this fraction of training on (Rank-DETR: 67500/90000 = 0.75; 1 = never)


@dataclass
class TrainConfig:
    iters: int = 20000
    batch_size: int = 2
    lr: float = 2e-4
    offset_lr_mult: float = 0.1
    weight_decay: float = 1e-4
    warmup_iters: int = 500
    grad_clip: float = 0.1
    amp: bool = True
    log_every: int = 50
    eval_every: int = 2000
    seed: int = 0
    ema_decay: float = 0.0  # 0 disables EMA; evaluation/selection then uses the raw weights
    max_gpu_mem_gb: float = 0.0  # 0 = no cap; otherwise cap this process to protect shared GPUs
    predict_test: bool = True  # after training, save Test predictions of best.pt (scored only at the end)
    stop_iter: int = 0  # stop early (and save) at this iteration while keeping the ``iters`` LR schedule; 0 = iters
    swa_last: int = 0  # > 0: average the evaluated (EMA) weights of the last N evaluations -> swa.pt


@dataclass
class EvalConfig:
    iou_thrs: List[float] = field(default_factory=lambda: [0.1, 0.3])
    select_metric: str = "froc@0.3"
    tta_flips: bool = False  # also predict on d / w / dw flipped volumes and fuse with WBF
    wbf_iou: float = 0.5  # WBF cluster IoU for TTA / ensembles
    # sliding-window post-processing (defaults follow the deployed per-window logic)
    top_k: int = 300
    score_thr: float = 0.01
    window_nms: float = 0.05
    max_per_window: int = 100
    fuse_nms: float = 0.1
    max_dets: int = 50
    deployed_fusion: bool = False  # deployed cross-window step (inclusive NMS, WBF max, top 20) instead of NMS
    head: str = "detr"  # center_detr models: which expert's boxes to predict ("detr" = expert 2, "center" = expert 1)

    def postprocess(self) -> Dict[str, Any]:
        keys = ("top_k", "score_thr", "window_nms", "max_per_window", "fuse_nms", "max_dets", "deployed_fusion",
                "head")
        return {k: getattr(self, k) for k in keys}


@dataclass
class Config:
    name: str = "abus3d_tdsc"
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def expand_paths(self) -> "Config":
        for f in fields(self.paths):
            setattr(self.paths, f.name, os.path.expanduser(getattr(self.paths, f.name)))
        return self

    @property
    def run_dir(self) -> str:
        return os.path.join(self.paths.runs_dir, self.name)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _update(obj: Any, values: Dict[str, Any]) -> None:
    for key, value in values.items():
        if not hasattr(obj, key):
            raise KeyError(f"unknown config key: {key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            _update(current, value)
        else:
            setattr(obj, key, _coerce(value, current))


def _coerce(value: Any, current: Any) -> Any:
    """Match the default's scalar type; YAML parses e.g. ``1e-4`` (no decimal point) as a string."""
    if isinstance(current, bool) or not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return value
    if isinstance(current, float):
        return float(value)
    if isinstance(current, int) and isinstance(value, str):
        return int(value)
    return value


def _parse_override(item: str) -> Dict[str, Any]:
    key, _, raw = item.partition("=")
    node: Dict[str, Any] = {}
    out = node
    parts = key.split(".")
    for p in parts[:-1]:
        node[p] = {}
        node = node[p]
    node[parts[-1]] = yaml.safe_load(raw)
    return out


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> Config:
    cfg = Config()
    if path:
        with open(path) as f:
            _update(cfg, yaml.safe_load(f) or {})
    for item in overrides or []:
        _update(cfg, _parse_override(item))
    return cfg.expand_paths()
