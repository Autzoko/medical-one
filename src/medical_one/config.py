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
    norm: str = "gn"
    dim: int = 192
    num_queries: int = 900
    dropout: float = 0.0


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


@dataclass
class LossConfig:
    w_cls: float = 2.0
    w_l1: float = 5.0
    w_giou: float = 2.0
    w_neg: float = 0.0  # unmatched predictions overlapping a hard negative get (1 + w_neg) x classification loss
    neg_iou: float = 0.3  # ... when their IoU with it exceeds this


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

    def postprocess(self) -> Dict[str, Any]:
        keys = ("top_k", "score_thr", "window_nms", "max_per_window", "fuse_nms", "max_dets", "deployed_fusion")
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
