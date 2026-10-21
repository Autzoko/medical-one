import math

import torch

from medical_one.models.center_detr import CenterDETR3D
from medical_one.models.center_head import STRIDE, CenterHead, CenterLoss, decode_boxes, peaks

GRID = (11, 6, 11)                               # a 44 x 24 x 44 patch
BOX = torch.tensor([[0.20, 0.25, 0.30, 0.50, 0.75, 0.62]])  # normalised corners


def _perfect_output(loss: CenterLoss, boxes):
    """Raw maps that decode exactly to the targets: heat logit +-20 at / off the centers, exact size / offset."""
    tgt = loss.targets(boxes, GRID, "cpu")
    heat = torch.where(tgt["heat"].eq(1), torch.tensor(20.0), torch.tensor(-20.0))
    size, offset = torch.zeros(len(boxes), 3, *GRID), torch.zeros(len(boxes), 3, *GRID)
    pos = tgt["pos"]
    size.flatten(2)[pos["b"], :, pos["i"]] = pos["size"]
    offset.flatten(2)[pos["b"], :, pos["i"]] = pos["offset"]
    return {"heat": heat, "size": size, "offset": offset}, tgt


def test_targets_peak_and_gaussian():
    loss = CenterLoss()
    tgt = loss.targets([BOX], GRID, "cpu")
    heat = tgt["heat"][0, 0]
    patch = torch.tensor([n * STRIDE for n in GRID], dtype=torch.float)
    center = (BOX[0, :3] + BOX[0, 3:]) / 2 * patch / STRIDE
    cell = center.floor().long()
    assert heat.max() == 1 and heat[tuple(cell)] == 1 and int(heat.eq(1).sum()) == 1
    assert int(tgt["pos"]["i"][0]) == (cell[0] * GRID[1] + cell[1]) * GRID[2] + cell[2]
    # sigma per axis = ratio x size in cells: one sigma away along d the target is exp(-1/2)
    size = (BOX[0, 3:] - BOX[0, :3]) * patch / STRIDE
    sigma_d = max(float(size[0]) / 6, 0.5)
    one = cell.clone()
    one[0] += 1
    assert math.isclose(float(heat[tuple(one)]), math.exp(-1 / (2 * sigma_d ** 2)), rel_tol=1e-4)
    assert 0 <= float(heat.min()) and torch.allclose(tgt["pos"]["box"], BOX)


def test_empty_patch_has_no_positives():
    tgt = CenterLoss().targets([torch.zeros(0, 6)], GRID, "cpu")
    assert tgt["heat"].sum() == 0 and len(tgt["pos"]["b"]) == 0


def test_perfect_prediction_decodes_box_and_has_tiny_loss():
    loss = CenterLoss()
    boxes = [BOX, torch.zeros(0, 6), BOX.flip(1).flip(1) * 0.9 + 0.05]
    out, tgt = _perfect_output(loss, boxes)
    dec = decode_boxes(out["size"], out["offset"])
    pos = tgt["pos"]
    assert torch.allclose(dec[pos["b"], pos["i"]], pos["box"], atol=1e-5)
    total, parts = loss(out, boxes)
    assert parts["size"] < 1e-6 and parts["offset"] < 1e-6 and parts["giou"] < 1e-5
    assert parts["heat"] < 1e-3 and torch.isfinite(total)


def test_loss_decreases_heat_towards_target():
    torch.manual_seed(0)
    head = CenterHead(dim=24, hidden=24)
    feat = torch.randn(1, 24, *GRID)
    opt = torch.optim.Adam(head.parameters(), 1e-2)
    loss = CenterLoss()
    first = None
    for _ in range(60):
        total, _ = loss(head(feat), [BOX])
        first = first if first is not None else total.item()
        opt.zero_grad()
        total.backward()
        opt.step()
    assert total.item() < 0.3 * first
    idx, _ = peaks(head(feat)["heat"].detach(), 1)
    assert int(idx[0, 0]) == int(loss.targets([BOX], GRID, "cpu")["pos"]["i"][0])


def test_peaks_are_local_maxima_in_score_order():
    heat = torch.full((1, 1, 5, 5, 5), -5.0)
    heat[0, 0, 1, 1, 1], heat[0, 0, 1, 1, 2] = 3.0, 2.0   # 2.0 is next to 3.0: not a peak
    heat[0, 0, 4, 4, 4] = 1.0
    idx, logits = peaks(heat, 2)
    assert idx[0].tolist() == [(1 * 5 + 1) * 5 + 1, (4 * 5 + 4) * 5 + 4] and logits[0].tolist() == [3.0, 1.0]


def test_center_detr_queries_index_level0_memory():
    torch.manual_seed(0)
    model = CenterDETR3D(num_queries=20).eval()
    x = torch.randn(1, 1, 32, 32, 32)
    feats = model.backbone.neck(model.backbone.backbone(x))
    assert tuple(feats[0].shape[2:]) == (8, 8, 8)            # stride 4
    center = model.center(feats[0])
    idx, ref, logits = model._queries(center)
    assert idx.shape == (1, 20) and ref.shape == (1, 20, 6) and (ref > 0).all() and (ref < 1).all()
    # the flattened level-0 grid is the first block of the transformer memory, in (d, h, w) order
    flat = feats[0].flatten(2).transpose(1, 2)
    d, rest = divmod(int(idx[0, 0]), 64)
    h, w = divmod(rest, 8)
    assert torch.equal(flat[0, idx[0, 0]], feats[0][0, :, d, h, w])
    assert logits[0, 0] == center["heat"][0, 0, d, h, w]


def test_center_detr_forward_backward_and_predict():
    from medical_one.losses import SetCriterion

    torch.manual_seed(0)
    model = CenterDETR3D(num_queries=30)
    x = torch.randn(2, 1, 32, 32, 32)
    dec, enc, center = model(x)
    assert len(dec) == 6 and dec[-1]["logits"].shape == (2, 30, 1) and center["heat"].shape == (2, 1, 8, 8, 8)
    targets = [BOX, torch.zeros(0, 6)]
    loss, _ = SetCriterion().total(dec, enc, targets)
    c, _ = CenterLoss()(center, targets)
    (loss + c).backward()
    assert torch.isfinite(loss + c)
    assert model.center.heat[-1].weight.grad.abs().sum() > 0           # expert 1 is trained by its own loss
    assert model.transformer.decoder.class_embed[0].weight.grad.abs().sum() > 0
    model.eval()
    for head in ("detr", "center"):
        logits, corners = model.predict(x, head=head)
        assert logits.shape == (2, 30, 1) and corners.shape == (2, 30, 6)
        assert (corners[..., 3:] >= corners[..., :3]).all()


def test_trainer_step_center_detr(tmp_path):
    from medical_one.config import load_config
    from medical_one.engine.trainer import build_model, heads

    cfg = load_config(None, ["model.arch=center_detr", "model.num_queries=30"])
    assert isinstance(build_model(cfg), CenterDETR3D) and heads(cfg) == ["detr", "center"]
    assert load_config(None, []).model.arch == "detr"
