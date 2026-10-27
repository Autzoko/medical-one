"""Zero-training diagnostics of the low small-lesion scores of DETR v3_p256 (docs/prereg/small_lesion_diag.md).

  python scripts/diag/small_lesion_diag.py RUN_s0 RUN_s1 RUN_s2 --split Validation --raw-root RAW --out OUT \
      [--variants base phase1 ...] [--part 0 --parts 4] [--set data.patch=[256,96,256] ...]

For every case the input variants are built (from the prepared working volume or, by ``build_working``, from the raw
NRRD), every model's ``swa.pt`` predicts them with sliding windows on all three axes, and the boxes are mapped back to
the base working grid. Writes ``preds_<split>_<variant>_s<i>.part<k>.json`` (``save_predictions`` format) and, when
the raw volume is read, ``d5_<split>.part<k>.json`` (header spacing, speckle autocorrelation, CNR, appearance of the
top boxes). Raw volumes never leave the job: only boxes, scores and summary numbers are written.
"""
import argparse
import json
import os
import time
from typing import Callable, Dict, List, Sequence, Tuple

import nrrd
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from medical_one.config import load_config
from medical_one.data.dataset import normalize
from medical_one.data.tdsc import KEEP_DEPTH, RESIZE, case_paths, load_meta, load_volume
from medical_one.engine import build_model
from medical_one.engine.inference import CENTER_MARGIN, window_starts
from medical_one.evaluation.preds import save_predictions
from medical_one.utils.boxes import nms_3d

RAW_VARIANTS = ("phase1", "phase2", "phase3", "aa0.5", "aa1", "aa1.5", "z1n", "z1")
PREP_VARIANTS = ("base", "shift_d+1", "shift_d-1", "shift_h+1", "shift_h-1", "shift_w+1", "shift_w-1", "z2")
ZOOM = 1.4


# ---------------------------------------------------------------- resampling (raw -> working space)
def _linear(x: np.ndarray, axis: int, u: np.ndarray) -> np.ndarray:
    """Linear interpolation of ``x`` along ``axis`` at source coordinates ``u`` (PyTorch's align_corners=False rule:
    negative coordinates clamp to 0, the upper neighbour clamps to the last sample)."""
    n = x.shape[axis]
    u = np.clip(u, 0.0, n - 1.0)
    i0 = np.floor(u).astype(np.int64)
    i1 = np.minimum(i0 + 1, n - 1)
    shape = [1] * x.ndim
    shape[axis] = len(u)
    w = (u - i0).astype(np.float32).reshape(shape)
    return np.take(x, i0, axis) * (1 - w) + np.take(x, i1, axis) * w


def build_working(raw: np.ndarray, size: Sequence[int] = RESIZE, keep: int = KEEP_DEPTH, shift_h: float = 0.0,
                  sigma_mult: float = 0.0) -> np.ndarray:
    """``to_working_space`` as an explicit separable resampler: transpose to [N1, N0, N2], flip depth, sample the grid
    ``size`` (voxel centres, align_corners=False), keep the first ``keep`` depth slices, round to uint8.
    ``shift_h`` moves the depth sampling grid by that many input voxels (flipped axis); ``sigma_mult`` = c > 0
    applies a Gaussian pre-filter with sigma = c (f - 1) / 2 input voxels on every axis shrunk by f > 1."""
    v = np.ascontiguousarray(raw.transpose(1, 0, 2)[:, ::-1, :]).astype(np.float32)
    shrink = [n / s for n, s in zip(v.shape, size)]
    if sigma_mult > 0:
        for axis, f in enumerate(shrink):
            if f > 1:
                v = gaussian_filter1d(v, sigma_mult * (f - 1) / 2, axis=axis, mode="nearest")
    for axis in (1, 0, 2):  # the depth axis first: it shrinks most
        u = (np.arange(size[axis]) + 0.5) * shrink[axis] - 0.5 + (shift_h if axis == 1 else 0.0)
        v = _linear(v, axis, u)
    return np.clip(np.round(v[:, :keep]), 0, 255).astype(np.uint8)


def shift_volume(vol: np.ndarray, axis: int, s: int) -> np.ndarray:
    """out[o] = vol[o + s] along ``axis`` with edge replication (content moves by -s)."""
    idx = np.clip(np.arange(vol.shape[axis]) + s, 0, vol.shape[axis] - 1)
    return np.take(vol, idx, axis)


def zoom_prep(vol: np.ndarray, z: float = ZOOM) -> np.ndarray:
    size = tuple(int(round(n * z)) for n in vol.shape)
    x = F.interpolate(torch.from_numpy(vol).float()[None, None], size=size, mode="trilinear", align_corners=False)
    return x[0, 0].round().clamp(0, 255).to(torch.uint8).numpy()


# ---------------------------------------------------------------- variants: (volume, box map back to the base grid)
Mapper = Callable[[torch.Tensor], torch.Tensor]


def _affine(scale: Sequence[float], offset: Sequence[float]) -> Mapper:
    s = torch.tensor(list(scale) * 2, dtype=torch.float)
    o = torch.tensor(list(offset) * 2, dtype=torch.float)
    return lambda b: b * s + o


def make_variants(names: Sequence[str], prep: np.ndarray, raw: np.ndarray = None) -> Dict[str, Tuple[np.ndarray, Mapper]]:
    out = {}
    ident = _affine((1, 1, 1), (0, 0, 0))
    for name in names:
        if name == "base":
            out[name] = (prep, ident)
        elif name.startswith("shift_"):
            axis, s = "dhw".index(name[6]), int(name[7:])
            off = [0.0, 0.0, 0.0]
            off[axis] = s
            out[name] = (shift_volume(prep, axis, s), _affine((1, 1, 1), off))
        elif name == "z2":
            v = zoom_prep(prep)
            out[name] = (v, _affine([a / b for a, b in zip(prep.shape, v.shape)], (0, 0, 0)))
        elif name.startswith("phase"):
            q = int(name[5:])
            f_h = raw.shape[0] / RESIZE[1]
            out[name] = (build_working(raw, shift_h=q / 4 * f_h), _affine((1, 1, 1), (0, q / 4, 0)))
        elif name.startswith("aa"):
            out[name] = (build_working(raw, sigma_mult=float(name[2:])), ident)
        elif name in ("z1n", "z1"):
            size = tuple(int(round(n * ZOOM)) for n in RESIZE)
            keep = int(round(KEEP_DEPTH * ZOOM))
            v = build_working(raw, size=size, keep=keep, sigma_mult=1.0 if name == "z1" else 0.0)
            out[name] = (v, _affine([a / b for a, b in zip(RESIZE, size)], (0, 0, 0)))
        else:
            raise ValueError(name)
    return out


# ---------------------------------------------------------------- inference with windows on all three axes
@torch.no_grad()
def predict_any(model, volume: np.ndarray, device: str, patch: Sequence[int], top_k: int = 300,
                score_thr: float = 0.01, window_nms: float = 0.05, max_per_window: int = 100, fuse_nms: float = 0.1,
                max_dets: int = 50, **_) -> Tuple[torch.Tensor, torch.Tensor]:
    """``predict_volume`` generalised to volumes larger than the patch on the depth axis too. With depth = patch depth
    it is the same computation (one depth window, no depth centre check)."""
    patch = tuple(patch)
    starts = [window_starts(n, p) for n, p in zip(volume.shape, patch)]
    x = normalize(torch.from_numpy(volume).float().to(device))
    scale = torch.tensor(patch * 2, device=device, dtype=torch.float)
    all_boxes, all_scores = [], []
    for sd in starts[0]:
        for sh in starts[1]:
            for sw in starts[2]:
                window = x[sd:sd + patch[0], sh:sh + patch[1], sw:sw + patch[2]].contiguous()[None, None]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                    logits, corners = model.predict(window)
                scores, boxes = logits[0, :, 0].float().sigmoid(), corners[0].float() * scale
                top = scores.topk(min(top_k, len(scores))).indices
                scores, boxes = scores[top], boxes[top]
                keep = scores > score_thr
                scores, boxes = scores[keep], boxes[keep]
                keep = nms_3d(boxes, scores, window_nms)[:max_per_window]
                scores, boxes = scores[keep], boxes[keep]
                c = (boxes[:, :3] + boxes[:, 3:]) / 2
                ok = torch.ones(len(c), dtype=torch.bool, device=device)
                for a, (st, sts) in enumerate(zip((sd, sh, sw), starts)):
                    if len(sts) == 1:
                        continue
                    lo = 0 if st == sts[0] else CENTER_MARGIN
                    hi = patch[a] if st == sts[-1] else patch[a] - CENTER_MARGIN
                    ok &= (c[:, a] > lo) & (c[:, a] < hi)
                offset = torch.tensor([sd, sh, sw] * 2, device=device, dtype=torch.float)
                all_boxes.append(boxes[ok] + offset)
                all_scores.append(scores[ok])
    boxes, scores = torch.cat(all_boxes), torch.cat(all_scores)
    keep = nms_3d(boxes, scores, fuse_nms)[:max_dets]
    return boxes[keep].cpu(), scores[keep].cpu()


# ---------------------------------------------------------------- D5 signal measurements
def _fwhm(ac: np.ndarray) -> float:
    """2 x the lag where a normalised autocorrelation (lag 0 = 1) first falls below 0.5 (linear interpolation)."""
    below = np.nonzero(ac < 0.5)[0]
    if not len(below):
        return float("nan")
    k = below[0]
    return float(2 * (k - 1 + (ac[k - 1] - 0.5) / (ac[k - 1] - ac[k])))


def speckle_fwhm(raw: np.ndarray, lo: Sequence[int], hi: Sequence[int], block=(96, 48, 24)) -> Dict[str, object]:
    """Autocorrelation FWHM per raw axis (N0, N1, N2) in a parenchyma block at the lesion's depth (N0) and sweep (N2)
    position, beside the lesion along N1; high-passed (minus a Gaussian sigma 6) before the autocorrelation."""
    c = [(a + b) // 2 for a, b in zip(lo, hi)]
    for side in (+1, -1):
        c1 = c[1] + side * ((hi[1] - lo[1]) // 2 + block[1] // 2 + 20)
        starts = [c[0] - block[0] // 2, c1 - block[1] // 2, c[2] - block[2] // 2]
        if any(s < 0 or s + b > n for s, b, n in zip(starts, block, raw.shape)):
            continue
        x = raw[tuple(slice(s, s + b) for s, b in zip(starts, block))].astype(np.float32)
        if x.mean() < 20 or (x == 0).mean() > 0.05:  # outside the field of view or in a shadow
            continue
        x = x - gaussian_filter(x, 6)
        res = {}
        for axis in range(3):
            spec = np.abs(np.fft.rfft(x, n=2 * x.shape[axis], axis=axis)) ** 2
            ac = np.fft.irfft(spec, axis=axis).take(range(x.shape[axis] // 2), axis=axis)
            ac = ac.mean(axis=tuple(a for a in range(3) if a != axis))
            res[f"fwhm_ax{axis}"] = round(_fwhm(ac / ac[0]), 3)
        res["side"] = side
        return res
    return {"side": 0}


def _ellipsoid_masks(box: Sequence[float], shape: Sequence[int]):
    c = [(box[a] + box[a + 3]) / 2 for a in range(3)]
    h = [max((box[a + 3] - box[a]) / 2, 0.5) for a in range(3)]
    r = np.sqrt(sum(((g + 0.5 - c[a]) / h[a]) ** 2 for a, g in enumerate(np.ogrid[:shape[0], :shape[1], :shape[2]])))
    return r <= 0.8, (r >= 1.3) & (r <= 1.8)


def appearance(vol: np.ndarray, box: Sequence[float]) -> Dict[str, float]:
    """Mean / std inside the box's inscribed ellipsoid (x0.8) and in a shell (1.3-1.8), zero voxels excluded; CNR."""
    inner, shell = _ellipsoid_masks(box, vol.shape)
    nz = vol > 0
    a, b = vol[inner & nz].astype(np.float32), vol[shell & nz].astype(np.float32)
    if len(a) < 8 or len(b) < 8:
        return {}
    return {"mean_in": round(float(a.mean()), 2), "std_in": round(float(a.std()), 2),
            "mean_shell": round(float(b.mean()), 2), "std_shell": round(float(b.std()), 2),
            "cnr": round(float(abs(a.mean() - b.mean()) / max(b.std(), 1e-6)), 4),
            "size": round(float(np.cbrt(np.prod([box[k + 3] - box[k] for k in range(3)]))), 2)}


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="run directories (their swa.pt), one per seed")
    ap.add_argument("--config", default="configs/detr_v2.yaml")
    ap.add_argument("--split", required=True, choices=["Train", "Validation", "Test"])
    ap.add_argument("--raw-root", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="*", default=list(PREP_VARIANTS) + list(RAW_VARIANTS))
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--max-cases", type=int, default=0, help="smoke test: only the first N cases of the part")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--set", dest="overrides", nargs="*", action="extend", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    post = cfg.eval.postprocess()
    patch = tuple(cfg.data.patch)
    models = []
    for run in args.runs:
        m = build_model(cfg).to(args.device).eval()
        m.load_state_dict(torch.load(os.path.join(run, "swa.pt"), map_location=args.device, weights_only=False)["model"])
        models.append(m)
    metas = load_meta(cfg.paths.prep_dir)[args.split][args.part::args.parts]
    if args.max_cases:
        metas = metas[:args.max_cases]
    need_raw = any(v in RAW_VARIANTS for v in args.variants)
    assert not need_raw or args.raw_root, "raw variants need --raw-root"
    os.makedirs(args.out, exist_ok=True)
    preds = {(v, i): [] for v in args.variants for i in range(len(models))}
    d5 = []
    for n, meta in enumerate(metas):
        t0 = time.time()
        prep = load_volume(cfg.paths.prep_dir, meta)
        raw, rec = None, {"split": meta["split"], "case": meta["case"]}
        if need_raw:
            raw, header = nrrd.read(case_paths(args.raw_root, meta["split"], meta["case"])[0])
            assert list(raw.shape) == meta["shape"], (raw.shape, meta["shape"])
            for k in ("spacings", "space directions"):
                if k in header:
                    rec[k] = np.asarray(header[k], dtype=float).round(5).tolist()
            base_rs = build_working(raw)
            diff = np.abs(base_rs.astype(np.int16) - prep.astype(np.int16))
            rec["base_rs_maxdiff"], rec["base_rs_frac_diff"] = int(diff.max()), round(float((diff > 0).mean()), 6)
            rec["speckle"] = speckle_fwhm(raw, meta["mask_lo"], meta["mask_hi"])
        variants = make_variants(args.variants, prep, raw)
        for v, (vol, mapper) in variants.items():
            for i, model in enumerate(models):
                boxes, scores = predict_any(model, vol, args.device, patch, **post)
                preds[(v, i)].append((mapper(boxes), scores))
        if need_raw:
            rec["gt_prep"] = appearance(prep, meta["box"])
            if "aa1" in variants:
                rec["gt_aa1"] = appearance(variants["aa1"][0], meta["box"])
            b0, s0 = preds[("base", 0)][-1] if "base" in variants else (torch.zeros(0, 6), torch.zeros(0))
            rec["top_boxes"] = [{"rank": r, "score": round(float(s0[r]), 5), "box": [round(x, 2) for x in b0[r].tolist()],
                                 **appearance(prep, b0[r].tolist())} for r in range(min(5, len(s0)))]
            d5.append(rec)
        del raw
        print(json.dumps({"case": meta["case"], "n": n + 1, "of": len(metas), "sec": round(time.time() - t0, 1),
                          **{k: rec[k] for k in ("base_rs_maxdiff", "base_rs_frac_diff") if k in rec}}), flush=True)
    tag = f"{args.split}_{{}}_s{{}}.part{args.part}"
    for (v, i), p in preds.items():
        save_predictions(os.path.join(args.out, f"preds_{tag.format(v, i)}.json"), p, metas,
                         {"run": args.runs[i], "variant": v, "patch": list(patch)})
    if d5:
        with open(os.path.join(args.out, f"d5_{args.split}.part{args.part}.json"), "w") as f:
            json.dump(d5, f)


if __name__ == "__main__":
    main()
