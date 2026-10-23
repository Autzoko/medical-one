"""CPM3D: center-points matching targets, loss / backward, prediction shapes, hard-negative weighting."""
import torch

from medical_one.models.center_head import decode_boxes
from medical_one.models.cpm import CPM3D, CPMLoss


def test_targets_pick_nearest_cells_inside_the_box():
    loss = CPMLoss(topk=7)
    box = torch.tensor([[0.25, 0.25, 0.25, 0.75, 0.75, 0.75]])
    t = loss.targets([box], (8, 8, 8), "cpu")
    pos = t["pos"]
    assert len(pos["b"]) == 7 and t["cls"].sum() == 7
    # positives are inside the box and the decoded target box equals the GT box
    size = torch.zeros(1, 3, 8, 8, 8)
    off = torch.zeros(1, 3, 8, 8, 8)
    for i, s, o in zip(pos["i"], pos["size"], pos["offset"]):
        size.view(1, 3, -1)[0, :, i], off.view(1, 3, -1)[0, :, i] = s, o
    dec = decode_boxes(size, off)[0, pos["i"]]
    assert torch.allclose(dec, box.expand(7, 6), atol=1e-5)
    # other cells inside the box are ignored, cells outside are negatives
    inside_ignored = ((t["weight"][0] == 0)).sum()
    assert inside_ignored == 4 ** 3 - 7


def test_tiny_box_gets_its_nearest_cell_and_hard_negatives_are_weighted():
    loss = CPMLoss(topk=7, w_neg=2.0)
    box = torch.tensor([[0.51, 0.51, 0.51, 0.53, 0.53, 0.53]])  # no cell center inside
    neg = torch.tensor([[0.0, 0.0, 0.0, 0.3, 0.3, 0.3]])
    t = loss.targets([box], (8, 8, 8), "cpu", negs=[neg])
    assert len(t["pos"]["b"]) == 1
    assert (t["weight"][0] == 3.0).sum() == 8  # 2 x 2 x 2 cells of the hard-negative box


def test_forward_loss_and_predict():
    torch.manual_seed(0)
    model = CPM3D(hidden=24, top_k=10)
    x = torch.randn(2, 1, 32, 16, 32)
    out = model(x)
    assert out["cls"].shape == (2, 1, 8, 4, 8)
    boxes = [torch.tensor([[0.2, 0.2, 0.2, 0.6, 0.7, 0.5]]), torch.zeros(0, 6)]
    loss, parts = CPMLoss()(out, boxes)
    loss.backward()
    assert torch.isfinite(loss) and parts["npos"] > 0
    logits, corners = model.predict(x)
    assert logits.shape == (2, 10, 1) and corners.shape == (2, 10, 6)
    assert (corners >= 0).all() and (corners <= 1).all()
