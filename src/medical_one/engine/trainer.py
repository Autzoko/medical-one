"""Training loop: AdamW, linear warmup + cosine decay, bf16 autocast, periodic validation."""
import copy
import json
import math
import os
import random
import time
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch

from ..config import Config
from ..data.dataset import PatchDataset, collate
from ..data.tdsc import load_meta, load_volume
from ..evaluation.metrics import compute_metrics
from ..losses.set_criterion import SetCriterion
from ..models.detector import ABUSDet3D
from .inference import predict_volume, predict_volume_fused


def build_model(cfg: Config) -> ABUSDet3D:
    m = cfg.model
    return ABUSDet3D(norm=m.norm, dim=m.dim, num_queries=m.num_queries, drop=m.dropout)


@torch.no_grad()
def predict_split(model: Union[ABUSDet3D, Sequence[ABUSDet3D]], cfg: Config, metas: List[dict],
                  device: str) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """A single model without TTA uses plain sliding-window NMS; otherwise predictions are WBF-fused."""
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    modes = [m.training for m in models]
    for m in models:
        m.eval()
    post = cfg.eval.postprocess()
    preds = []
    for meta in metas:
        volume = load_volume(cfg.paths.prep_dir, meta)
        if len(models) == 1 and not cfg.eval.tta_flips:
            preds.append(predict_volume(models[0], volume, device, **post))
        else:
            preds.append(predict_volume_fused(models, volume, device, cfg.eval.tta_flips, cfg.eval.wbf_iou, **post))
    for m, mode in zip(models, modes):
        m.train(mode)
    return preds


def evaluate(model: Union[ABUSDet3D, Sequence[ABUSDet3D]], cfg: Config, metas: List[dict],
             device: str) -> Dict[str, object]:
    return compute_metrics(predict_split(model, cfg, metas, device), metas, cfg.eval.iou_thrs)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Trainer:
    def __init__(self, cfg: Config, device: str = "cuda"):
        self.cfg, self.device = cfg, device
        _seed(cfg.train.seed)
        torch.backends.cudnn.benchmark = True
        if cfg.train.max_gpu_mem_gb > 0 and device.startswith("cuda"):
            index = torch.device(device).index
            index = torch.cuda.current_device() if index is None else index
            total = torch.cuda.get_device_properties(index).total_memory / 2**30
            torch.cuda.set_per_process_memory_fraction(min(1.0, cfg.train.max_gpu_mem_gb / total), index)
        self.splits = load_meta(cfg.paths.prep_dir)
        self.model = build_model(cfg).to(device)
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False) if cfg.train.ema_decay > 0 else None
        self.criterion = SetCriterion(cfg.loss.w_cls, cfg.loss.w_l1, cfg.loss.w_giou)
        t = cfg.train
        offsets = [p for n, p in self.model.named_parameters() if "sampling_offsets" in n]
        others = [p for n, p in self.model.named_parameters() if "sampling_offsets" not in n]
        self.opt = torch.optim.AdamW([{"params": others, "lr": t.lr},
                                      {"params": offsets, "lr": t.lr * t.offset_lr_mult}],
                                     weight_decay=t.weight_decay)
        self.base_lrs = [g["lr"] for g in self.opt.param_groups]
        os.makedirs(cfg.run_dir, exist_ok=True)
        with open(os.path.join(cfg.run_dir, "config.json"), "w") as f:
            json.dump(cfg.to_dict(), f, indent=1)
        self.log_file = open(os.path.join(cfg.run_dir, "log.jsonl"), "a")

    def _log(self, record: Dict[str, object], prefix: str = "") -> None:
        print(prefix + json.dumps(record), flush=True)
        self.log_file.write(json.dumps(record) + "\n")
        self.log_file.flush()

    def _set_lr(self, it: int) -> None:
        t = self.cfg.train
        factor = min(1.0, it / t.warmup_iters) * 0.5 * (1 + math.cos(math.pi * it / t.iters))
        for group, base in zip(self.opt.param_groups, self.base_lrs):
            group["lr"] = base * factor

    @torch.no_grad()
    def _update_ema(self) -> None:
        if self.ema is None:
            return
        d = self.cfg.train.ema_decay
        for e, p in zip(self.ema.state_dict().values(), self.model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(p.detach(), alpha=1 - d)
            else:
                e.copy_(p)

    def _save(self, name: str, it: int, **extra) -> None:
        """``model`` holds the weights used for evaluation (EMA if enabled); ``raw`` the trained weights."""
        state = {"model": (self.ema if self.ema is not None else self.model).state_dict(), "it": it, **extra}
        if self.ema is not None:
            state["raw"] = self.model.state_dict()
        torch.save(state, os.path.join(self.cfg.run_dir, name))

    def fit(self) -> None:
        cfg, t = self.cfg, self.cfg.train
        train_metas = [m for split in cfg.data.train_splits for m in self.splits[split]]
        stop = t.stop_iter or t.iters
        dataset = PatchDataset(cfg.paths.prep_dir, train_metas, stop * t.batch_size,
                               cfg.data.pos_prob, cfg.data.margin, scale_range=cfg.data.scale_range,
                               noise_std=cfg.data.noise_std)
        loader = torch.utils.data.DataLoader(dataset, t.batch_size, num_workers=cfg.data.num_workers,
                                             collate_fn=collate, persistent_workers=cfg.data.num_workers > 0)
        best, start = -1.0, time.time()
        self.model.train()
        for it, (x, targets) in enumerate(loader, 1):
            self._set_lr(it)
            x = x.to(self.device, non_blocking=True)
            targets = [tg.to(self.device) for tg in targets]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=t.amp):
                dec_outputs, enc_output = self.model(x)
            loss, last = self.criterion.total(dec_outputs, enc_output, targets)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), t.grad_clip)
            self.opt.step()
            self._update_ema()
            if it % t.log_every == 0:
                self._log({"it": it, "loss": round(loss.item(), 4), "last": round(last.item(), 4),
                           "gnorm": round(grad_norm.item(), 3), "lr": self.opt.param_groups[0]["lr"],
                           "sec": round(time.time() - start, 1)})
            if it % t.eval_every == 0 or it == stop:
                res = evaluate(self.ema if self.ema is not None else self.model, cfg, self.splits["Validation"], self.device)
                res["it"] = it
                self._log({"eval": res}, prefix="EVAL ")
                self._save("last.pt", it)
                if res[cfg.eval.select_metric] > best:
                    best = res[cfg.eval.select_metric]
                    self._save("best.pt", it, val=res)
