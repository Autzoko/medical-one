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
    heat = torch.linspace(-9, -6, 125).view(1, 1, 5, 5, 5).clone()  # a ramp: one peak at the far corner
    heat[0, 0, 1, 1, 1], heat[0, 0, 1, 1, 2] = 3.0, 2.0   # 2.0 is next to 3.0: not a peak
    heat[0, 0, 4, 4, 2] = 1.0
    idx, logits = peaks(heat, 3)
    assert idx[0].tolist()[:2] == [(1 * 5 + 1) * 5 + 1, (4 * 5 + 4) * 5 + 2] and logits[0].tolist()[:2] == [3.0, 1.0]
    # fewer peaks than k: non-peaks follow by score, and the first cells do not depend on k
    more, more_logits = peaks(heat, 10)
    assert torch.equal(more[:, :3], idx) and more_logits[0, 3] == 2.0


def test_center_detr_queries_index_level0_memory():
    torch.manual_seed(0)
    model = CenterDETR3D(num_queries=20).eval()
    x = torch.randn(1, 1, 32, 32, 32)
    feats = model.backbone.neck(model.backbone.backbone(x))
    assert tuple(feats[0].shape[2:]) == (8, 8, 8)            # stride 4
    center = model.center(feats[0])
    idx, ref, logits = model._queries(center, 20)
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
    out = model(x)
    dec, enc, center = out["o2o"], out["enc"], out["center"]
    assert len(dec) == 6 and dec[-1]["logits"].shape == (2, 30, 1) and center["heat"].shape == (2, 1, 8, 8, 8)
    assert out["o2m"] is None and out["dn"] is None and out["neg"] is None
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


# ---------------------------------------------------------------- v1: query groups, Rank-DETR loss, augmentation
from medical_one.losses.set_criterion import SetCriterion, iou_aware_focal_loss, sigmoid_focal_loss  # noqa: E402
from medical_one.models.query_groups import attention_mask, denoising_queries, negative_queries  # noqa: E402
from medical_one.utils.boxes import cxcyczwhd_to_xyxy  # noqa: E402


def test_denoising_noise_ranges_and_layout():
    torch.manual_seed(0)
    t = torch.tensor([[0.40, 0.40, 0.40, 0.60, 0.60, 0.60]])
    refs, meta = denoising_queries([t, torch.zeros(0, 6)], number=10, noise=0.2, device="cpu")
    assert meta == {"groups": 5, "single": 1} and refs.shape == (2, 10, 6)
    half = 0.1
    corners = cxcyczwhd_to_xyxy(refs[0]).view(5, 2, 6)
    rel = ((corners - t) / (half * 0.2)).abs()                   # |offset| in units of noise * half-size
    assert (rel[:, 0] <= 1 + 1e-4).all()                         # positives: < 1
    assert (rel[:, 1] >= 1 - 1e-4).all() and (rel[:, 1] <= 2 + 1e-4).all()   # negatives: 1..2
    assert torch.allclose(refs[1], torch.full_like(refs[1], 0.5))  # no GT: padded slots
    assert denoising_queries([torch.zeros(0, 6)] * 2, 100, 1.0, "cpu")[0].shape == (2, 0, 6)


def test_attention_mask_blocks_groups():
    k1, k2, g, single, n = 3, 2, 2, 1, 2
    m = attention_mask(k1, k2, g, single, n, "cpu")
    L = k1 + k2 + g * 2 * single + n
    assert m.shape == (L, L)
    see = ~m
    assert see[:k1, :k1].all() and not see[:k1, k1:].any()               # one-to-one: itself only
    assert see[k1:5, k1:5].all() and not see[k1:5, :k1].any() and not see[k1:5, 5:].any()
    assert see[5:7, :k1].all() and see[5:7, 5:7].all() and not see[5:7, 7:].any() and not see[5:7, k1:5].any()
    assert see[7:9, :k1].all() and see[7:9, 7:9].all() and not see[7:9, 5:7].any()
    assert see[9:, :k1].all() and see[9:, 9:].all() and not see[9:, 5:9].any()
    assert attention_mask(3, 0, 0, 0, 0, "cpu") is None


def test_training_groups_do_not_leak_into_one_to_one():
    torch.manual_seed(0)
    model = CenterDETR3D(num_queries=20, num_o2m=30, dn_number=10, num_neg_queries=3, look_forward_twice=True)
    model.train()
    x = torch.randn(2, 1, 32, 32, 32)
    targets = [BOX, torch.zeros(0, 6)]
    negs = [torch.tensor([[0.6, 0.1, 0.6, 0.8, 0.3, 0.8]]), torch.zeros(0, 6)]
    full = model(x, targets, negs)
    plain = model(x)
    assert full["o2m"][0]["logits"].shape == (2, 30, 1) and full["dn"][0]["logits"].shape == (2, 10, 1)
    assert full["neg"][0]["logits"].shape == (2, 1, 1)
    for a, b in zip(full["o2o"], plain["o2o"]):
        assert torch.allclose(a["logits"], b["logits"], atol=1e-5) and torch.allclose(a["boxes"], b["boxes"], atol=1e-6)


def test_iou_aware_loss_reduces_to_focal_for_hard_targets():
    torch.manual_seed(0)
    logits = torch.randn(2, 7, 1)
    pos = (torch.rand(2, 7, 1) > 0.7).float()
    assert torch.allclose(iou_aware_focal_loss(logits, pos, pos), sigmoid_focal_loss(logits, pos), atol=1e-6)
    soft = pos * 0.6
    assert not torch.allclose(iou_aware_focal_loss(logits, soft, pos), sigmoid_focal_loss(logits, pos))


def test_high_order_matching_is_argmax_score_times_iou4():
    from medical_one.utils.boxes import box_iou_giou

    crit = SetCriterion(high_order_alpha=4.0)
    crit.high_order = True
    target = torch.tensor([[0.2, 0.2, 0.2, 0.6, 0.6, 0.6]])
    boxes = torch.tensor([[0.4, 0.4, 0.4, 0.4, 0.4, 0.4],       # exact, p 0.27
                          [0.41, 0.41, 0.41, 0.42, 0.42, 0.42],  # IoU ~0.88, p 0.95
                          [0.45, 0.45, 0.45, 0.6, 0.6, 0.6]])    # IoU ~0.3, p 0.99
    logits = torch.tensor([[-1.0], [3.0], [5.0]])
    score = logits[:, 0].sigmoid() * box_iou_giou(cxcyczwhd_to_xyxy(boxes), target)[0][:, 0] ** 4
    assert int(crit.match(logits, boxes, target)[0][0]) == int(score.argmax()) == 1


def test_dn_loss_rewards_exact_positive_slots():
    crit = SetCriterion(iou_aware=True)
    targets = [BOX, torch.zeros(0, 6)]
    meta = {"groups": 2, "single": 1}
    from medical_one.utils.boxes import xyxy_to_cxcyczwhd
    gt = xyxy_to_cxcyczwhd(BOX)[0]
    boxes = torch.full((2, 4, 6), 0.5)
    logits = torch.full((2, 4, 1), -6.0)
    for g in range(2):
        boxes[0, g * 2] = gt
        logits[0, g * 2] = 6.0
    good = crit.dn_loss({"logits": logits, "boxes": boxes}, targets, meta)
    swapped = crit.dn_loss({"logits": logits.roll(1, 1), "boxes": boxes.roll(1, 1)}, targets, meta)
    assert good < 0.05 and swapped > 1.0


def test_negative_queries_padding():
    refs = negative_queries([torch.tensor([[0.1, 0.1, 0.1, 0.3, 0.3, 0.3]] * 3), torch.zeros(0, 6)], 2, 2, "cpu")
    assert refs.shape == (2, 2, 6) and torch.allclose(refs[1], torch.full((2, 6), 0.5))
    assert negative_queries(None, 5, 2, "cpu").shape == (2, 0, 6)


def test_rot90_augmentation_is_box_exact():
    import numpy as np

    from medical_one.data.dataset import PATCH, PatchDataset

    ds = PatchDataset.__new__(PatchDataset)
    ds.extra_aug = {"rot90": 1.0, "blur": 0.0, "speckle": 0.0, "lowres": 0.0, "inv_gamma": 0.0}
    x = torch.zeros(PATCH)
    box = np.array([[10, 20, 100, 30, 40, 150]], np.float32)
    x[10:30, 20:40, 100:150] = 255
    y, b = ds._extra(x, box)
    d0, h0, w0, d1, h1, w1 = b[0].astype(int)
    assert y[d0:d1, h0:h1, w0:w1].min() == 255 and y.sum() == x.sum()
    assert (b[0] == [100, 20, 10, 150, 40, 30]).all()
