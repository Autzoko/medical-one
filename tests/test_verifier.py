"""Cascade verifier: ROI sampling from the whole volume, delta round trip, refinement, stacking fit."""
import importlib.util
import os

import numpy as np
import torch

from medical_one.cascade.verifier import (Verifier3D, apply_deltas, extract_rois, jitter_boxes, labels_for, refine,
                                          target_deltas)

spec = importlib.util.spec_from_file_location(
    "train_verifier", os.path.join(os.path.dirname(__file__), "..", "scripts", "cascade", "train_verifier.py"))
tv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tv)


def test_roi_samples_the_box_region():
    vol = torch.zeros(336, 96, 336)
    vol[100:140, 30:50, 200:260] = 1.0
    box = torch.tensor([[100.0, 30, 200, 140, 50, 260]])
    r = extract_rois(vol, box, enlarge=1.0, min_extent=(1, 1, 1))
    assert r.shape == (1, 1, 48, 32, 48) and r[0, 0, 1:-1, 1:-1, 1:-1].min() > 0.99
    r2 = extract_rois(vol, box, enlarge=2.0, min_extent=(1, 1, 1))  # box = central half per axis
    inner = r2[0, 0, 13:35, 9:23, 13:35]
    assert inner.min() > 0.99 and r2[0, 0, :10].max() < 0.01


def test_deltas_round_trip_and_labels():
    boxes = jitter_boxes(torch.tensor([50.0, 20, 60, 90, 40, 130]), 20, np.random.RandomState(0))
    gt = torch.tensor([50.0, 20, 60, 90, 40, 130])
    fixed = apply_deltas(boxes, target_deltas(boxes, gt))
    ok = (target_deltas(boxes, gt)[:, :3].abs() <= 1).all(1)
    assert torch.allclose(fixed[ok], gt.expand(int(ok.sum()), 6), atol=1e-3)
    hit, iou = labels_for(gt[None], gt)
    assert hit.item() == 1 and abs(iou.item() - 1) < 1e-6


def test_refine_runs_and_fit_logistic():
    torch.manual_seed(0)
    m = Verifier3D(widths=(8, 8, 16, 16))
    vol = torch.randn(336, 96, 336)
    b, p, mal = refine(m, vol, torch.tensor([[10.0, 10, 10, 40, 30, 50], [100.0, 20, 100, 200, 60, 220]]), iters=2)
    assert b.shape == (2, 6) and p.shape == (2,) and ((p >= 0) & (p <= 1)).all()
    rng = np.random.RandomState(0)
    X = rng.randn(500, 2)
    y = (X @ np.array([2.0, -1.0]) + 0.5 + rng.randn(500) * 0.3 > 0).astype(float)
    coef, b0 = tv.fit_logistic(X, y, l2=0.01)
    assert coef[0] > 1 and coef[1] < -0.5 and b0 > 0
