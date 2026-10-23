"""AdaBoost-style reweighting of the DETR (docs/prereg/detr_boost.md): folds inside Train, case-weighted sampling,
boosting weights from out-of-fold errors."""
import importlib.util
import os
import random

import numpy as np

from medical_one.data.tdsc import inner_split

spec = importlib.util.spec_from_file_location(
    "boost_weights", os.path.join(os.path.dirname(__file__), "..", "scripts", "boost", "boost_weights.py"))
bw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bw)


def test_inner_split_covers_train_once_and_is_stratified():
    train = [{"split": "Train", "case": i, "label": "M" if i % 3 else "B"} for i in range(100)]
    held = []
    for k in range(4):
        tr, ho = inner_split(train, 4, k)
        assert not {m["case"] for m in tr} & {m["case"] for m in ho}
        assert len(tr) + len(ho) == 100 and 24 <= len(ho) <= 26
        assert abs(sum(m["label"] == "B" for m in ho) - 34 / 4) <= 1
        held += [m["case"] for m in ho]
    assert sorted(held) == list(range(100))
    assert inner_split(train, 4, 1) == inner_split(list(reversed(train)), 4, 1)  # order independent


def test_boost_weights():
    e = np.array([0.0] * 6 + [0.5, 1.0, 1.0, 1.0])
    w, eps, alpha = bw.boost_weights(e)
    assert abs(w.mean() - 1) < 1e-9 and alpha > 0
    assert w[7] > w[6] > w[0] and w[7] == w[9]
    assert w.min() >= 0.5 * 0.5 / 1 and w.max() <= 3.0  # clipped, then half uniform
    w1, _, _ = bw.boost_weights(np.full(10, 0.3))
    assert np.allclose(w1, 1)


def test_case_weights_bias_sampling(monkeypatch):
    import medical_one.data.dataset as dataset_mod
    from medical_one.data.dataset import PatchDataset

    monkeypatch.setattr(dataset_mod, "load_volume", lambda prep_dir, m: np.zeros((336, 96, 336), np.uint8))
    metas = [{"split": "Train", "case": c, "box": [40, 30, 50, 70, 50, 90]} for c in range(3)]
    ds = PatchDataset("", metas, 10, pos_prob=0.5, case_weights=[1.0, 1.0, 4.0])
    random.seed(0)
    picks = [ds._pick_case() for _ in range(6000)]
    for positive in (True, False):
        counts = np.bincount([i for i, p in picks if p == positive], minlength=3)
        assert counts[2] > 3 * counts[0] and counts[2] > 3 * counts[1]
