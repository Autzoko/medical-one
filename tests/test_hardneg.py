import numpy as np
import torch

from medical_one.data.dataset import PATCH, PatchDataset, collate
from medical_one.hardneg import far_from_gt, load_negatives, mine, save_negatives, union, v_to_working, working_to_v
from medical_one.losses import SetCriterion
from medical_one.utils.boxes import cxcyczwhd_to_xyxy, xyxy_to_cxcyczwhd

GT = [150.0, 30.0, 140.0, 190.0, 50.0, 200.0]
NEG = [20.0, 10.0, 20.0, 50.0, 30.0, 60.0]


def test_mine_keeps_confident_boxes_away_from_the_lesion():
    meta = [{"split": "Train", "case": 0, "box": GT}]
    boxes = torch.tensor([GT, [185.0, 30, 195, 220, 50, 230], NEG, [22.0, 10, 22, 50, 30, 60], [250.0, 10, 250, 280, 30, 290]])
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.1])
    negs = mine([(boxes, scores)], meta, min_score=0.3)
    b, s = negs[("Train", 0)]
    # the lesion, its touching neighbour (inside the 25% margin), the duplicate (NMS) and the weak box go
    assert np.allclose(b, [NEG]) and np.allclose(s, [0.7])
    assert far_from_gt(np.array([NEG, GT]), GT).tolist() == [True, False]


def test_union_and_roundtrip(tmp_path):
    a = {("Train", 0): (np.array([NEG]), np.array([0.5]))}
    b = {("Train", 0): (np.array([[21.0, 10, 20, 50, 30, 60], [250.0, 10, 250, 280, 30, 290]]), np.array([0.9, 0.4])),
         ("Train", 1): (np.zeros((0, 6)), np.zeros(0))}
    u = union(a, b)
    assert np.allclose(u[("Train", 0)][1], [0.9, 0.4]) and len(u[("Train", 1)][1]) == 0
    save_negatives(str(tmp_path / "n.json"), u, {})
    back = load_negatives(str(tmp_path / "n.json"))
    assert np.allclose(back[("Train", 0)][0], u[("Train", 0)][0])


def test_v_and_working_space_are_a_scale():
    v_shape = (608, 330, 865)
    b = np.array([[10.0, 20, 30, 40, 50, 60]])
    assert np.allclose(working_to_v(v_to_working(b, v_shape), v_shape), b)


def test_negative_patches_follow_the_negative(tmp_path):
    vol = np.zeros((336, 96, 336), np.uint8)
    vol[150:190, 30:50, 140:200] = 255
    vol[200:240, 10:30, 20:60] = 128
    np.save(tmp_path / "000.npy", vol)
    meta = [{"split": ".", "case": 0, "box": GT}]
    neg = [200.0, 10.0, 20.0, 240.0, 30.0, 60.0]
    ds = PatchDataset(str(tmp_path), meta, 20, pos_prob=0.0, negatives={(".", 0): (np.array([neg]), np.array([0.5]))},
                      neg_prob=1.0, scale_range=0.2)
    for _ in range(30):
        x, t, n = ds[0]
        assert len(n) == 1  # every patch is centred on the negative
        b = (n[0].numpy() * np.asarray(PATCH * 2)).round().astype(int)
        inner = x[0, b[0] + 2:b[3] - 2, b[1] + 2:b[4] - 2, b[2] + 2:b[5] - 2]
        # the box stays on the uniform mid-grey block (background -> -1, lesion -> +1 before intensity jitter)
        assert inner.numel() == 0 or (inner.std() < 1e-4 and -0.7 < inner.mean() < 0.7)
    x, t, n = collate([ds[0], ds[0]])
    assert x.shape == (2, 1, *PATCH) and len(n) == 2
    plain = PatchDataset(str(tmp_path), meta, 20)
    assert len(plain[0]) == 2  # without negatives the items are unchanged


def test_neg_weight_only_on_unmatched_overlapping_queries():
    gt = torch.tensor([[0.2, 0.2, 0.2, 0.4, 0.4, 0.4]])
    neg = torch.tensor([[0.6, 0.6, 0.6, 0.8, 0.8, 0.8]])
    corners = torch.tensor([[0.2, 0.2, 0.2, 0.4, 0.4, 0.4], [0.6, 0.6, 0.6, 0.8, 0.8, 0.8],
                            [0.61, 0.6, 0.6, 0.8, 0.8, 0.8], [0.0, 0.0, 0.0, 0.1, 0.1, 0.1]])
    boxes = xyxy_to_cxcyczwhd(corners)[None]
    assert torch.allclose(cxcyczwhd_to_xyxy(boxes[0]), corners)
    crit = SetCriterion(w_neg=2.0)
    w = crit.neg_weight(boxes, [neg])
    assert w[0, :, 0].tolist() == [1.0, 3.0, 3.0, 1.0]
    logits = torch.zeros(1, 4, 1)
    out = {"logits": logits, "boxes": boxes}
    base, hard = SetCriterion()(out, [gt]), crit(out, [gt], [neg])
    assert hard > base
    assert torch.allclose(SetCriterion(w_neg=2.0)(out, [gt], [torch.zeros(0, 6)]), base)


def test_yolo_negative_slices():
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "pipeline"))
    from make_yolo_data import neg_slices

    s = neg_slices(np.array([[10.2, 5, 6, 13.0, 9, 12], [-3.0, 1, 1, 1.5, 2, 2]]), 12)
    assert sorted(s) == [0, 1, 11] and s[11] == [(5, 6, 9, 12)]  # slices crossed (clamped to the volume)
