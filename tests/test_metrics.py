import numpy as np
import torch

from medical_one.data.tdsc import box_to_working_space, boxes_to_original
from medical_one.evaluation import compute_metrics
from medical_one.evaluation.tdsc_official import average_precision, official_froc

META = {"split": "Test", "case": 0, "shape": [865, 600, 340], "mask_lo": [300, 50, 100], "mask_hi": [420, 260, 160]}
META["box"] = box_to_working_space(META["mask_lo"], META["mask_hi"], META["shape"])[0]


def test_boxes_to_original_inverts_working_space_mapping():
    raw = boxes_to_original(torch.tensor([META["box"]], dtype=torch.float64), META["shape"])[0]
    assert np.allclose(raw.numpy(), META["mask_lo"] + META["mask_hi"], atol=1e-2)


def test_perfect_detector_scores_one_and_duplicates_are_free():
    box = torch.tensor([META["box"]])
    perfect = compute_metrics([(box, torch.tensor([0.9]))], [META])
    assert perfect["tdsc_froc"] > 0.999 and perfect["ap@0.3"] > 0.999
    dup = compute_metrics([(box.repeat(3, 1), torch.tensor([0.9, 0.8, 0.7]))], [META])
    assert dup["tdsc_froc"] > 0.999  # official: duplicate hits are not false positives
    assert dup["froc@0.3"] < perfect["froc@0.3"] or dup["froc@0.3"] == 1.0


def test_official_froc_counts_misses_as_false_positives():
    per_volume = [(np.array([0.9, 0.5]), np.array([0.1, 0.6])), (np.array([0.4]), np.array([0.0]))]
    score, key, _, _ = official_froc(per_volume)
    assert 0 < score < 1 and abs(key[-1] - 0.5) < 1e-6  # 1 of 2 lesions found, at 1 FP/scan (+EPS as official)
    assert average_precision(per_volume, 0.3) < 1
