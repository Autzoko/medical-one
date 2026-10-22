"""Box-only augmentations of the DETR v2 round (three-axis zoom, lesion copy-paste, synthetic shadow, small-lesion
oversampling) and sliding windows for other patch sizes."""
import random

import numpy as np
import pytest
import torch

import medical_one.data.dataset as dataset_mod
from medical_one.data.dataset import PatchDataset
from medical_one.engine.inference import predict_volume, window_starts

SHAPE = (336, 96, 336)
BOXES = {1: [40, 30, 50, 70, 50, 90], 2: [200, 50, 220, 212, 58, 236], 3: [150, 10, 100, 230, 80, 200]}


def _volume(box):
    v = np.full(SHAPE, 60, np.uint8)
    d0, h0, w0, d1, h1, w1 = box
    v[d0:d1, h0:h1, w0:w1] = 220
    return v


@pytest.fixture
def make_ds(monkeypatch):
    monkeypatch.setattr(dataset_mod, "load_volume", lambda prep_dir, m: _volume(BOXES[m["case"]]))

    def make(cases=(1,), **kw):
        metas = [{"split": "Train", "case": c, "box": BOXES[c]} for c in cases]
        return PatchDataset("", metas, 100, **kw)
    return make


def _bright_extent(x):
    x = x[0]
    hi = x > (x.max() + x.median()) / 2
    nz = torch.nonzero(hi)
    return torch.cat([nz.min(0).values, nz.max(0).values + 1]).float()


def test_three_axis_zoom_is_box_exact(make_ds):
    random.seed(0)
    ds = make_ds(pos_prob=1.0, scale_prob=1.0, scale_iso=(0.7, 1.4), scale_axis=0.1)
    crops = []
    orig = ds._crop_sizes
    ds._crop_sizes = lambda shape: crops.append(orig(shape)) or crops[-1]
    for _ in range(25):
        x, t = ds[0]
        assert len(t) == 1
        box = t[0] * torch.tensor(ds.patch * 2, dtype=torch.float32)
        assert (box - _bright_extent(x)).abs().max() < 2.5, (box, _bright_extent(x))
    assert any(c[1] > SHAPE[1] for c in crops) and any(c[1] < SHAPE[1] for c in crops)  # zero-pad and crop in depth


def test_extract_zero_pads_outside_the_volume():
    v = np.arange(4 * 3 * 5, dtype=np.uint8).reshape(4, 3, 5)
    out = PatchDataset._extract(v, (-1, 0, 2), (3, 4, 4))
    assert out.shape == (3, 4, 4)
    assert (out[0] == 0).all() and (out[:, 3] == 0).all() and (out[:, :, 3] == 0).all()
    assert (out[1:, :3, :3] == v[:2, :, 2:5]).all()


def test_paste_adds_a_second_exact_box(make_ds):
    random.seed(1)
    ds = make_ds(cases=(1, 2), pos_prob=1.0, paste_prob=1.0)
    twos = 0
    for _ in range(20):
        x, t = ds[0]
        if len(t) == 2:
            twos += 1
            b = (t * torch.tensor(ds.patch * 2, dtype=torch.float32)).round().long()
            for box in b:
                c = (box[:3] + box[3:]) // 2
                assert x[0, c[0], c[1], c[2]] > 0  # lesion (~0.7) vs background (~-0.5), after gamma / brightness
            assert ((b[0, 3:] <= b[1, :3]) | (b[1, 3:] <= b[0, :3])).any()  # the two boxes do not overlap
    assert twos >= 5


def test_shadow_only_darkens_and_deepens():
    random.seed(2)
    x = torch.full((40, 30, 50), 100.0)
    y = PatchDataset._shadow(x)
    assert (y <= x).all() and (y < 99).any()
    assert (y[:, :-1] <= y[:, 1:] + 1e-4).all()  # deeper (smaller h) is never brighter


def test_small_lesions_oversampled(make_ds):
    random.seed(3)
    ds = make_ds(cases=(1, 2, 3), pos_prob=1.0, small_oversample=0.5)
    counts = np.bincount([ds._pick_case()[0] for _ in range(3000)], minlength=3)
    assert counts[1] > counts[0] > counts[2]  # case 2 smallest, case 3 largest


def test_window_starts():
    assert window_starts(336, 176) == (0, 80, 160)
    assert window_starts(336, 256) == (0, 80)
    assert window_starts(336, 336) == (0,)


class _Dummy:
    def __init__(self):
        self.shapes = []

    def predict(self, x):
        self.shapes.append(tuple(x.shape))
        boxes = torch.tensor([[[0.4, 0.4, 0.4, 0.6, 0.6, 0.6]]])
        return torch.full((1, 1, 1), 3.0), boxes


def test_predict_volume_with_a_larger_patch():
    m = _Dummy()
    boxes, scores = predict_volume(m, np.zeros(SHAPE, np.uint8), "cpu", score_thr=0.0, patch=(256, 96, 256))
    assert m.shapes == [(1, 1, 256, 96, 256)] * 4
    assert len(boxes) >= 1 and (boxes[:, 3:] > boxes[:, :3]).all()
