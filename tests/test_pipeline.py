import numpy as np
import torch

from medical_one.data.tdsc import box_to_working_space, boxes_to_original
from medical_one.pipeline.fusion import fuse_case, select_3d, wbf
from medical_one.pipeline.native import letterbox, native_view, raw_box_to_v, roi_image, three_slice, unletterbox
from medical_one.pipeline.yolo2d import aggregate_slices, v_boxes_to_working, yolo_nms


def test_native_view_matches_deployed_index():
    r = np.arange(4 * 3 * 5, dtype=np.int32).reshape(4, 3, 5)
    v = native_view(r)
    assert v.shape == (3, 4, 5)
    assert v[2, 1, 3] == r[4 - 1 - 1, 2, 3]


def test_three_slice_clamps_ends():
    v = np.arange(3)[:, None, None] * np.ones((3, 2, 2), np.uint8)
    assert three_slice(v, 0)[0, 0].tolist() == [0, 0, 1]
    assert three_slice(v, 2)[0, 0].tolist() == [1, 2, 2]


def test_letterbox_shape_and_inverse():
    img = np.zeros((865, 353, 3), np.uint8)
    out, r, (dw, dh) = letterbox(img)
    assert out.shape[0] == 960 and out.shape[1] % 32 == 0 and out.shape[1] < 960
    box = np.array([[100.0, 200.0, 150.0, 300.0]])
    letter = box.copy() * r
    letter[:, [0, 2]] += (out.shape[1] - 353 * r) / 2
    assert np.allclose(unletterbox(letter, out.shape[:2], img.shape[:2]), box)


def test_aggregation_stops_at_slice_gap_and_keeps_far_side():
    plane = [10, 10, 50, 50]
    rows = [plane + [0.9, s] for s in (4, 5, 6, 7)] + [plane + [0.8, s] for s in (12, 13, 14, 15)]
    groups = aggregate_slices(np.array(rows, float))
    assert [(g[1], g[2]) for g in groups] == [(3.0, 8.0), (11.0, 16.0)]


def test_aggregation_is_seed_neighbourhood_not_closure():
    a, b, c = [0, 0, 10, 10], [6, 0, 16, 10], [12, 0, 22, 10]  # a~b and b~c overlap, a and c do not
    rows = np.array([a + [0.9, 0], b + [0.8, 0], c + [0.7, 0]], float)
    groups = aggregate_slices(rows)
    assert len(groups) == 2 and groups[0][0] == 0


def test_yolo_nms_threshold_and_product_score():
    pred = torch.tensor([[[50, 50, 20, 20, 0.9, 0.5], [51, 50, 20, 20, 0.8, 1.0], [200, 200, 10, 10, 0.04, 1.0]]])
    det = yolo_nms(pred)[0]
    assert len(det) == 1 and torch.isclose(det[0, 4], torch.tensor(0.8))


def test_wbf_single_member_halved_and_avg():
    b = np.array([[0.1, 0.1, 0.1, 0.3, 0.3, 0.3]])
    fb, fs = wbf([(b, np.array([0.6])), (np.zeros((0, 6)), np.zeros(0))], 0.05, "avg")
    assert np.isclose(fs[0], 0.3)
    fb, fs = wbf([(b, np.array([0.6])), (b + 0.01, np.array([0.4]))], 0.05, "avg")
    assert len(fs) == 1 and np.isclose(fs[0], 0.5)
    assert np.allclose(fb[0], (0.6 * b[0] + 0.4 * (b[0] + 0.01)) / 1.0)


def test_select_3d_rank_rules_and_boost():
    boxes = np.zeros((7, 6))
    scores = np.array([0.1, 0.5, 0.15, 0.2, 0.7, 0.9, 0.9])
    _, s = select_3d(boxes, scores)
    assert np.allclose(s, [0.15, 0.75, 0.3, 0.95], atol=1e-6)


def test_fuse_case_gate_and_threshold():
    d3 = np.array([[100, 40, 100, 140, 60, 140.0]])
    closed = fuse_case({"gate": False, "boxes": []}, [600, 865, 340], d3, np.array([0.9]))
    assert len(closed[1]) == 0
    fb, fs = fuse_case({"gate": True, "boxes": []}, [600, 865, 340], d3, np.array([0.9]))
    assert np.isclose(fs[0], 0.475) and np.allclose(fb[0], d3[0])


def test_v_and_working_space_roundtrip():
    shape = (865, 600, 340)
    lo, hi = [300, 100, 50], [400, 200, 120]
    vbox = raw_box_to_v(lo, hi, shape)
    work = v_boxes_to_working(vbox, (shape[1], shape[0], shape[2]))
    back = boxes_to_original(torch.from_numpy(work).double(), shape)[0].numpy()
    assert np.allclose(back, lo + hi)
    ref, _ = box_to_working_space(lo, hi, shape)
    assert np.allclose(work[0], ref, atol=1e-2)


def test_roi_image_is_square_224():
    v = np.random.randint(0, 255, (20, 100, 60), np.uint8)
    roi = roi_image(v, [5, 10, 20, 9, 30, 25])
    assert roi.shape == (224, 224, 3)
