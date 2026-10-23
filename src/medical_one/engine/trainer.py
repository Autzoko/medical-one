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
from ..data.tdsc import inner_split, load_meta, load_volume, train_val_metas
from ..evaluation.metrics import compute_metrics
from ..evaluation.preds import save_predictions
from ..hardneg import load_negatives, mine, save_negatives, union
from ..losses.set_criterion import SetCriterion
from ..models.center_detr import CenterDETR3D
from ..models.center_head import CenterLoss
from ..models.cpm import CPM3D, CPMLoss
from ..models.dense_det import CoDETR3D, DenseDet3D, positive_queries
from ..models.dense_head import DenseLoss, plan_anchors
from ..models.detector import ABUSDet3D
from .inference import predict_volume, predict_volume_fused


def build_model(cfg: Config) -> Union[ABUSDet3D, CenterDETR3D, DenseDet3D, CoDETR3D, CPM3D]:
    """Anchor shapes of the dense models are a buffer: set by the trainer (``plan_anchors``) or a checkpoint."""
    m = cfg.model
    if m.arch == "cpm":
        return CPM3D(norm=m.norm, hidden=m.cpm_hidden, top_k=m.cpm_top_k)
    if m.arch == "dense":
        return DenseDet3D(norm=m.norm, dim=m.dim, center_hidden=m.center_hidden, dense_hidden=m.dense_hidden,
                          dense_convs=m.dense_convs, anchors_per_level=m.anchors_per_level)
    if m.arch == "co_detr":
        return CoDETR3D(norm=m.norm, dim=m.dim, num_queries=m.num_queries, drop=m.dropout,
                        center_hidden=m.center_hidden, dense_hidden=m.dense_hidden, dense_convs=m.dense_convs,
                        anchors_per_level=m.anchors_per_level, max_pos_queries=m.max_pos_queries)
    if m.arch == "center_detr":
        return CenterDETR3D(norm=m.norm, dim=m.dim, num_queries=m.num_queries, drop=m.dropout,
                            center_hidden=m.center_hidden, num_o2m=m.num_o2m, dn_number=m.dn_number,
                            dn_noise=m.dn_noise, num_neg_queries=m.num_neg_queries,
                            look_forward_twice=m.look_forward_twice)
    assert m.arch == "detr", m.arch
    return ABUSDet3D(norm=m.norm, dim=m.dim, num_queries=m.num_queries, drop=m.dropout)


def heads(cfg: Config) -> List[str]:
    """Outputs whose boxes are evaluated and saved; the first drives checkpoint selection."""
    return {"center_detr": ["detr", "center"], "dense": ["dense", "dense_heat", "center"],
            "co_detr": ["detr", "dense", "dense_heat"]}.get(cfg.model.arch, ["detr"])


def _suffix(head: str) -> str:
    return "" if head == "detr" else f"_{head}"


@torch.no_grad()
def predict_split(model, cfg: Config, metas: List[dict], device: str,
                  head: str = "") -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """A single model without TTA uses plain sliding-window NMS; otherwise predictions are WBF-fused.
    ``head`` overrides ``cfg.eval.head`` (which expert of a CenterDETR3D)."""
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    modes = [m.training for m in models]
    for m in models:
        m.eval()
    post = cfg.eval.postprocess()
    post["patch"] = tuple(cfg.data.patch)
    if head:
        post["head"] = head
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
        if cfg.model.arch in ("dense", "co_detr"):  # our planning rule: k-means on the training box sizes
            train_metas = self._metas()[0]
            sizes = np.array([np.subtract(m["box"][3:], m["box"][:3]) for m in train_metas])
            anchors = plan_anchors(sizes, per_level=cfg.model.anchors_per_level, seed=0)
            self.model.dense.set_anchors(anchors)
            os.makedirs(cfg.run_dir, exist_ok=True)
            with open(os.path.join(cfg.run_dir, "anchors.json"), "w") as f:
                json.dump({"anchor_sizes_vox": anchors.round(2).tolist(), "n_boxes": len(sizes)}, f)
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False) if cfg.train.ema_decay > 0 else None
        self.criterion = SetCriterion(cfg.loss.w_cls, cfg.loss.w_l1, cfg.loss.w_giou,
                                      w_neg=cfg.loss.w_neg, neg_iou=cfg.loss.neg_iou,
                                      iou_aware=cfg.loss.iou_aware_cls, high_order_alpha=cfg.loss.high_order_alpha)
        lc = cfg.loss
        self.center_loss = self.dense_loss = self.cpm_loss = None
        if cfg.model.arch == "cpm":
            self.cpm_loss = CPMLoss(lc.cpm_topk, lc.cpm_alpha, lc.cpm_gamma, lc.w_cpm_cls, lc.w_cpm_size,
                                    lc.w_cpm_offset, lc.w_cpm_giou, lc.cpm_ignore_inside, lc.w_neg)
        if cfg.model.arch in ("dense", "co_detr"):
            self.dense_loss = DenseLoss(lc.dense_candidates, lc.dense_batch_per_image, lc.dense_pos_fraction,
                                        lc.dense_pool)
            self.vox_scale = torch.tensor(tuple(cfg.data.patch) * 2, dtype=torch.float32, device=device)
        if cfg.model.arch in ("center_detr", "dense", "co_detr"):
            self.center_loss = CenterLoss(lc.w_center_heat, lc.w_center_size, lc.w_center_offset, lc.w_center_giou,
                                          lc.center_sigma_ratio, lc.center_min_sigma)
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
        self.swa_sum, self.swa_n = None, 0

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

    def _metas(self) -> Tuple[List[dict], List[dict], List[dict]]:
        """(train, validation, held-out training cases of ``data.inner_fold``; empty without inner folds)."""
        d = self.cfg.data
        train, val = train_val_metas(self.splits, d.train_splits, d.cv_folds, d.cv_fold)
        if d.inner_folds > 0:
            train, held_out = inner_split(train, d.inner_folds, d.inner_fold)
            return train, val, held_out
        return train, val, []

    def _case_weights(self, metas: List[dict]):
        if not self.cfg.data.case_weight_file:
            return None
        with open(self.cfg.data.case_weight_file) as f:
            w = {(c["split"], c["case"]): c["weight"] for c in json.load(f)["cases"]}
        return [w[(m["split"], m["case"])] for m in metas]

    def fit(self) -> None:
        cfg, t = self.cfg, self.cfg.train
        train_metas, val_metas, oof_metas = self._metas()
        stop = t.stop_iter or t.iters
        negatives = load_negatives(cfg.data.neg_file) if cfg.data.neg_file else None  # only train cases are read
        every = cfg.data.neg_refresh_every
        if every > 0 and negatives is None:
            negatives = {}
        dataset = PatchDataset(cfg.paths.prep_dir, train_metas, stop * t.batch_size,
                               cfg.data.pos_prob, cfg.data.margin, scale_range=cfg.data.scale_range,
                               noise_std=cfg.data.noise_std, negatives=negatives, neg_prob=cfg.data.neg_prob,
                               extra_aug={"rot90": cfg.data.rot90_prob, "blur": cfg.data.blur_prob,
                                          "speckle": cfg.data.speckle_prob, "lowres": cfg.data.lowres_prob,
                                          "inv_gamma": cfg.data.inv_gamma_prob, "shadow": cfg.data.shadow_prob},
                               patch=cfg.data.patch, scale_prob=cfg.data.scale_prob, scale_iso=cfg.data.scale_iso,
                               scale_axis=cfg.data.scale_axis, paste_prob=cfg.data.paste_prob,
                               paste_context=cfg.data.paste_context, small_oversample=cfg.data.small_oversample,
                               case_weights=self._case_weights(train_metas))
        # Online bootstrapping: the run is cut into segments; after each, the current model re-mines its own
        # false positives on the training cases and the next segment's loader (fresh workers) samples them.
        bounds = [b for b in range(every, stop, every)] + [stop] if every > 0 else [stop]
        best, start, it = -1.0, time.time(), 0
        self.model.train()
        for seg_end in bounds:
            dataset.length = (seg_end - it) * t.batch_size
            loader = torch.utils.data.DataLoader(dataset, t.batch_size, num_workers=cfg.data.num_workers,
                                                 collate_fn=collate, persistent_workers=cfg.data.num_workers > 0)
            for batch in loader:
                it += 1
                best = self._step(it, batch, stop, best, start, val_metas)
            if seg_end < stop:
                negatives = self._refresh_negatives(it, train_metas, negatives)
                dataset.set_negatives(negatives)
        final = self._finish_swa(stop, val_metas) if self.swa_n else (self.ema or self.model)
        if oof_metas:  # out-of-fold predictions of the held-out training cases (deliverable weights)
            save_predictions(os.path.join(cfg.run_dir, "preds_oof.json"),
                             predict_split(final, cfg, oof_metas, self.device), oof_metas,
                             {"checkpoint": "swa.pt" if self.swa_n else "final", "it": stop,
                              "inner_folds": cfg.data.inner_folds, "inner_fold": cfg.data.inner_fold})
        if cfg.train.predict_test:
            self._predict_test()

    def _accumulate_swa(self) -> None:
        """Running sum of the evaluated weights (EMA if enabled) over the last ``swa_last`` evaluations."""
        state = (self.ema if self.ema is not None else self.model).state_dict()
        if self.swa_sum is None:
            self.swa_sum = {k: v.detach().float().clone() if v.dtype.is_floating_point else v.clone()
                            for k, v in state.items()}
        else:
            for k, v in state.items():
                if v.dtype.is_floating_point:
                    self.swa_sum[k] += v.detach().float()
                else:
                    self.swa_sum[k] = v.clone()
        self.swa_n += 1

    def _finish_swa(self, it: int, val_metas: List[dict]) -> None:
        cfg = self.cfg
        avg = {k: (v / self.swa_n).to(self.model.state_dict()[k].dtype) if v.dtype.is_floating_point else v
               for k, v in self.swa_sum.items()}
        model = build_model(cfg).to(self.device)
        model.load_state_dict(avg)
        preds = {h: predict_split(model, cfg, val_metas, self.device, head=h) for h in heads(cfg)}
        res = compute_metrics(preds[heads(cfg)[0]], val_metas, cfg.eval.iou_thrs)
        self._log({"eval_swa": res, "it": it, "n_avg": self.swa_n}, prefix="EVAL ")
        torch.save({"model": avg, "it": it, "n_avg": self.swa_n, "val": res}, os.path.join(cfg.run_dir, "swa.pt"))
        for h, p in preds.items():
            save_predictions(os.path.join(cfg.run_dir, f"preds_val_swa{_suffix(h)}.json"), p, val_metas,
                             {"checkpoint": "swa.pt", "it": it, "n_avg": self.swa_n, "head": h})
        return model.eval()

    def _refresh_negatives(self, it: int, train_metas: List[dict], negatives):
        """In-sample mining with the evaluation model (EMA if enabled), merged with earlier negatives."""
        cfg, t0 = self.cfg, time.time()
        model = self.ema if self.ema is not None else self.model
        preds = predict_split(model, cfg, train_metas, self.device)
        new = mine(preds, train_metas, cfg.data.neg_min_score, cfg.data.neg_top)
        merged = union(negatives, new, top=2 * cfg.data.neg_top) if cfg.data.neg_keep_old and negatives else new
        save_negatives(os.path.join(cfg.run_dir, f"negatives_it{it}.json"), merged, {"it": it})
        n_new = [len(new[(m["split"], m["case"])][1]) for m in train_metas]
        n_all = [len(merged.get((m["split"], m["case"]), (None, []))[1]) for m in train_metas]
        self._log({"it": it, "neg_refresh": {"mined": int(sum(n_new)), "cases_mined": round(float(np.mean(
            [n > 0 for n in n_new])), 3), "total": int(sum(n_all)), "sec": round(time.time() - t0, 1)}})
        return merged

    def _step(self, it: int, batch, stop: int, best: float, start: float, val_metas: List[dict]) -> float:
        cfg, t = self.cfg, self.cfg.train
        self._set_lr(it)
        x = batch[0].to(self.device, non_blocking=True)
        targets = [tg.to(self.device) for tg in batch[1]]
        negs = [n.to(self.device) for n in batch[2]] if len(batch) > 2 else None
        self.criterion.high_order = it > cfg.loss.high_order_start * t.iters
        center_parts = extra_parts = None
        if self.cpm_loss is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=t.amp):
                out = self.model(x)
            loss, center_parts = self.cpm_loss(out, targets, negs)
            last = center_parts["cls"]
        elif self.dense_loss is not None:
            loss, last, center_parts, extra_parts = self._dense_losses(x, targets)
        elif self.center_loss is None:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=t.amp):
                dec_outputs, enc_output = self.model(x)
            loss, last = self.criterion.total(dec_outputs, enc_output, targets, negs)
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=t.amp):
                out = self.model(x, targets, negs)
            loss, last = self.criterion.total(out["o2o"], out["enc"], targets, negs)
            center, center_parts = self.center_loss(out["center"], targets)
            loss, extra_parts = loss + center, {}
            lc = cfg.loss
            if out["o2m"] is not None:
                extra_parts["o2m"] = sum(self.criterion(o, targets, repeat=lc.o2m_k) for o in out["o2m"])
                loss = loss + lc.w_o2m * extra_parts["o2m"]
            if out["dn"] is not None:
                extra_parts["dn"] = sum(self.criterion.dn_loss(o, targets, out["dn_meta"]) for o in out["dn"])
                loss = loss + lc.w_dn * extra_parts["dn"]
            if out["neg"] is not None:
                extra_parts["neg"] = sum(self.criterion.background_loss(o, targets) for o in out["neg"])
                loss = loss + lc.w_neg_queries * extra_parts["neg"]
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), t.grad_clip)
        self.opt.step()
        self._update_ema()
        if it % t.log_every == 0:
            rec = {"it": it, "loss": round(loss.item(), 4), "last": round(last.item(), 4),
                   "gnorm": round(grad_norm.item(), 3), "lr": self.opt.param_groups[0]["lr"],
                   "sec": round(time.time() - start, 1)}
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                rec["mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            if center_parts is not None:
                rec["center"] = {k: round(v.item(), 4) for k, v in center_parts.items()}
            if extra_parts:
                rec["extra"] = {k: round(v.item(), 4) for k, v in extra_parts.items()}
            self._log(rec)
        if it % t.eval_every == 0 or it == stop:
            eval_model = self.ema if self.ema is not None else self.model
            preds = {h: predict_split(eval_model, cfg, val_metas, self.device, head=h) for h in heads(cfg)}
            res = compute_metrics(preds[heads(cfg)[0]], val_metas, cfg.eval.iou_thrs)
            res["it"] = it
            self._log({"eval": res}, prefix="EVAL ")
            for h in heads(cfg)[1:]:
                r = compute_metrics(preds[h], val_metas, cfg.eval.iou_thrs)
                self._log({"eval_head": h, "it": it, **{k: r[k] for k in ("tdsc_froc", "tdsc_sens", "ap@0.3",
                                                                          "tdsc_max_recall", "top1_hit@0.3")}},
                          prefix="EVAL ")
            self._save("last.pt", it)
            if t.swa_last and it > stop - t.swa_last * t.eval_every:
                self._accumulate_swa()
            if res[cfg.eval.select_metric] > best:  # selection: the first output (heads) on Validation only
                best = res[cfg.eval.select_metric]
                self._save("best.pt", it, val=res)
                for h, p in preds.items():
                    save_predictions(os.path.join(cfg.run_dir, f"preds_val_best{_suffix(h)}.json"), p, val_metas,
                                     {"checkpoint": "best.pt", "it": it, "head": h})
        return best

    def _dense_losses(self, x: torch.Tensor, targets: List[torch.Tensor]):
        """DenseDet3D: heat-map loss + dense box loss. CoDETR3D: one-to-one decoder (Hungarian, Rank-DETR losses if
        enabled) + encoder proposals + aux heads on the encoder memory (heat map, dense boxes x ``w_dense``) +
        customised positive queries from the dense head's ATSS positives (x ``w_pos_queries``)."""
        cfg, lc, t = self.cfg, self.cfg.loss, self.cfg.train
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=t.amp):
            if cfg.model.arch == "dense":
                out = self.model(x)
            else:
                state = self.model.encode(x)
                out = self.model.aux(state)
                dec = self.model.decode(state)
        center, center_parts = self.center_loss(out["center"], targets)
        dense, dense_parts, match = self.dense_loss(out["dense"], targets, self.vox_scale)
        extra = {"dense": dense.detach(), "dense_pos": dense_parts["pos"]}
        if cfg.model.arch == "dense":
            return center + lc.w_dense * dense, dense_parts["cls"] + dense_parts["reg"], center_parts, extra
        loss = sum(self.criterion(o, targets) for o in dec) + self.criterion(self.model.transformer.encoder_output(state),
                                                                            targets)
        last = self.criterion(dec[-1], targets).detach()
        loss = loss + center + lc.w_dense * dense
        if lc.w_pos_queries > 0:
            ref, index, pairs = positive_queries(out["dense"], match, targets, self.model.max_pos_queries,
                                                 self.model.min_pos_queries)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=t.amp):
                pos_dec = self.model.decode(state, pos_ref=ref, pos_index=index)
            pos_loss = sum(self.criterion.assigned_loss(o, targets, pairs) for o in pos_dec)
            loss = loss + lc.w_pos_queries * pos_loss
            extra["pos_queries"] = pos_loss.detach()
        return loss, last, center_parts, extra

    @torch.no_grad()
    def _predict_test(self) -> None:
        """Test-set predictions of best.pt and last.pt, saved for later scoring (never used for selection)."""
        model = build_model(self.cfg).to(self.device)
        metas = self.splits["Test"]
        ckpts = [("best.pt", "preds_Test"), ("last.pt", "preds_Test_last")]
        if os.path.exists(os.path.join(self.cfg.run_dir, "swa.pt")):
            ckpts.append(("swa.pt", "preds_Test_swa"))
        for ckpt, out in ckpts:
            state = torch.load(os.path.join(self.cfg.run_dir, ckpt), map_location=self.device, weights_only=False)
            model.load_state_dict(state["model"])
            for h in heads(self.cfg):
                save_predictions(os.path.join(self.cfg.run_dir, f"{out}{_suffix(h)}.json"),
                                 predict_split(model, self.cfg, metas, self.device, head=h),
                                 metas, {"checkpoint": ckpt, "it": state["it"], "head": h})
