import numpy as np
import torch

from medical_one.data.dataset import PATCH, PatchDataset
from medical_one.engine.inference import _unflip
from medical_one.utils.boxes import weighted_box_fusion


def test_wbf_merges_overlaps_and_downweights_singletons():
    a = (torch.tensor([[0.0, 0, 0, 10, 10, 10], [50.0, 50, 50, 60, 60, 60]]), torch.tensor([0.8, 0.6]))
    b = (torch.tensor([[1.0, 0, 0, 11, 10, 10]]), torch.tensor([0.4]))
    boxes, scores = weighted_box_fusion([a, b], iou_threshold=0.5)
    assert len(boxes) == 2
    assert torch.allclose(scores, torch.tensor([0.6, 0.3]))
    assert torch.allclose(boxes[0], torch.tensor([1 / 3, 0, 0, 31 / 3, 10, 10]))


def test_unflip_inverts_flip():
    shape = (336, 96, 336)
    box = torch.tensor([[10.0, 5, 300, 50, 20, 330]])
    flipped = _unflip(box, (0, 2), shape)
    assert torch.allclose(flipped, torch.tensor([[286.0, 5, 6, 326, 20, 36]]))
    assert torch.allclose(_unflip(flipped, (0, 2), shape), box)


def test_scale_augmentation_keeps_box_on_lesion(tmp_path):
    vol = np.zeros((336, 96, 336), np.uint8)
    vol[150:190, 30:50, 140:200] = 255
    np.save(tmp_path / "000.npy", vol)
    meta = [{"split": ".", "case": 0, "box": [150.0, 30.0, 140.0, 190.0, 50.0, 200.0]}]
    ds = PatchDataset(str(tmp_path), meta, 20, pos_prob=1.0, scale_range=0.3)
    ds.augment = True
    for _ in range(20):
        x, t = ds[0]
        assert x.shape == (1, *PATCH)
        if len(t):
            b = (t[0].numpy() * np.asarray(PATCH * 2)).round().astype(int)
            inner = x[0, b[0] + 2:b[3] - 2, b[1] + 2:b[4] - 2, b[2] + 2:b[5] - 2]
            assert inner.numel() == 0 or inner.mean() > 0.5  # box still covers the bright lesion
