"""Cuboid masks for the bbox-only baselines."""
import numpy as np

from medical_one.data.tdsc import box_mask


def _extent(mask):
    nz = np.nonzero(mask)
    return [int(a.min()) for a in nz] + [int(a.max()) + 1 for a in nz]


def test_box_mask_is_the_rounded_box():
    box = [10.4, 3.6, 200.5, 40.49, 20.5, 251.2]
    m = box_mask(box)
    assert m.shape == (336, 96, 336) and m.dtype == np.uint8
    assert _extent(m) == [10, 4, 201, 40, 21, 251]
    assert m.sum() == 30 * 17 * 50  # a full cuboid
    assert np.abs(np.array(_extent(m)) - np.array(box)).max() <= 0.5


def test_box_mask_degenerate_and_border():
    assert box_mask([5.2, 95.9, 7.0, 5.3, 96.0, 7.1]).sum() == 1  # thinner than a voxel, at the depth crop
    assert _extent(box_mask([-2, -1, 330, 12, 100, 340])) == [0, 0, 330, 12, 96, 336]
