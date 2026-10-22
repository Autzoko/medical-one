"""Dense box head (nnDetection recipe), DenseDet3D and CoDETR3D."""
import numpy as np
import pytest
import torch

from medical_one.config import Config
from medical_one.engine.trainer import Trainer, build_model
from medical_one.losses.set_criterion import SetCriterion
from medical_one.models.center_head import CenterLoss
from medical_one.models.dense_det import positive_queries
from medical_one.models.dense_head import STRIDES, DenseLoss, atss_match, decode, plan_anchors
from medical_one.utils.boxes import box_iou_giou

GRID = (32, 32, 32)  # tiny patch: levels 8^3, 4^3, 2^3
SIZES = np.array([[40, 8, 12], [60, 10, 20], [20, 6, 8], [90, 14, 30], [30, 7, 10], [50, 9, 16]] * 5, float)


def _cfg(arch):
    cfg = Config()
    cfg.model.arch, cfg.model.dim, cfg.model.num_queries = arch, 48, 20
    cfg.model.center_hidden, cfg.model.dense_hidden = 24, 32
    cfg.train.amp = False
    return cfg


def _model(arch):
    torch.manual_seed(0)
    m = build_model(_cfg(arch))
    m.dense.set_anchors(plan_anchors(SIZES / 6))  # anchors fitting the tiny patch
    return m


def test_plan_anchors_sorted_by_volume():
    a = plan_anchors(SIZES)
    assert a.shape == (3, 3, 3)
    vol = a.reshape(-1, 3).prod(1)
    assert (np.diff(vol) >= 0).all()


def test_anchor_layout_and_zero_delta_decode():
    m = _model("dense")
    out = m(torch.zeros(1, 1, *GRID))["dense"]
    cells = [np.prod([n // s for n in GRID]) for s in STRIDES]
    assert out["counts"] == [c * 3 for c in cells]
    assert out["logits"].shape == (1, sum(out["counts"]))
    assert torch.allclose(decode(torch.zeros_like(out["deltas"][0]), out["anchors"]), out["anchors"], atol=1e-6)


def test_atss_positive_contains_identical_anchor():
    m = _model("dense")
    out = m(torch.zeros(1, 1, *GRID))["dense"]
    vox = torch.tensor(GRID * 2, dtype=torch.float32)
    anchors = out["anchors"] * vox
    j = out["counts"][0] + 17  # an anchor on level 1
    match = atss_match(anchors[j:j + 1], anchors, out["counts"], 3)
    assert match[j] == 0
    pos = torch.where(match == 0)[0]
    assert (box_iou_giou(anchors[pos], anchors[j:j + 1])[0][:, 0] > 0).all()
    assert (atss_match(torch.zeros(0, 6), anchors, out["counts"], 3) == -1).all()


def test_sampler_counts_and_hard_pool():
    loss = DenseLoss()
    match = torch.full((2, 5000), -1)
    match[0, :7], match[1, :3] = 0, 0
    logits = torch.randn(2, 5000)
    pm, nm = loss.sample(match, logits)
    assert pm.sum() == 10 and nm.sum() == 20 and not (pm & nm).any()
    neg_scores = logits.flatten()[(match.flatten() < 0)]
    pool_floor = neg_scores.topk(20 * 20).values.min()
    assert (logits[nm] >= pool_floor).all()


def test_dense_loss_perfect_regression():
    m = _model("dense")
    out = m(torch.zeros(1, 1, *GRID))["dense"]
    vox = torch.tensor(GRID * 2, dtype=torch.float32)
    target = [torch.tensor([[0.2, 0.25, 0.3, 0.6, 0.55, 0.7]])]
    loss = DenseLoss()
    match = loss.assign(out, target, vox)
    pos = torch.where(match[0] >= 0)[0]
    assert len(pos) >= 1
    a = out["anchors"][pos]
    ac, asz = (a[:, :3] + a[:, 3:]) / 2, a[:, 3:] - a[:, :3]
    t = target[0].expand(len(pos), -1)
    tc, ts = (t[:, :3] + t[:, 3:]) / 2, t[:, 3:] - t[:, :3]
    deltas = out["deltas"].detach().clone()
    deltas[0, pos] = torch.cat([(tc - ac) / asz, (ts / asz).log()], 1)
    out = dict(out, deltas=deltas)
    _, parts, _ = loss(out, target, vox)
    assert parts["reg"] < 1e-4


@pytest.mark.parametrize("head", ["dense", "dense_heat", "center"])
def test_densedet_predict(head):
    m = _model("dense").eval()
    logits, boxes = m.predict(torch.randn(1, 1, *GRID), head=head)
    assert logits.shape[:2] == boxes.shape[:2] and torch.isfinite(logits).all()
    assert (boxes >= 0).all() and (boxes <= 1).all()


def test_positive_queries_cells_contain_anchor_centers():
    m = _model("co_detr")
    state = m.encode(torch.randn(2, 1, *GRID))
    out = m.aux(state)
    vox = torch.tensor(GRID * 2, dtype=torch.float32)
    targets = [torch.tensor([[0.2, 0.25, 0.3, 0.6, 0.55, 0.7]]), torch.zeros(0, 6)]
    match = DenseLoss().assign(out["dense"], targets, vox)
    ref, index, pairs = positive_queries(out["dense"], match, targets, min_pos=9)
    assert ref.shape[:2] == index.shape == (2, max(9, int((match[0] >= 0).sum())))
    assert len(pairs[0][0]) == int((match[0] >= 0).sum()) and len(pairs[1][0]) == 0
    # memory position -> level / cell -> its center must be the anchor center
    shapes = state["shapes"]
    starts = np.cumsum([0] + [np.prod(s) for s in shapes])
    for b in range(2):
        for q in range(ref.shape[1]):
            i = int(index[b, q])
            lvl = int(np.searchsorted(starts, i, side="right") - 1)
            c = np.unravel_index(i - starts[lvl], shapes[lvl])
            center = (np.array(c) + 0.5) * STRIDES[lvl] / np.array(GRID)
            assert np.allclose(center, ref[b, q, :3].numpy(), atol=1e-4)


def test_codetr_losses_backward():
    cfg = _cfg("co_detr")
    cfg.loss.iou_aware_cls, cfg.loss.w_dense = True, 12.0
    tr = Trainer.__new__(Trainer)
    tr.cfg, tr.device = cfg, "cpu"
    tr.model = _model("co_detr")
    tr.criterion = SetCriterion(iou_aware=True)
    tr.center_loss, tr.dense_loss = CenterLoss(), DenseLoss()
    tr.vox_scale = torch.tensor(GRID * 2, dtype=torch.float32)
    targets = [torch.tensor([[0.2, 0.25, 0.3, 0.6, 0.55, 0.7]]), torch.zeros(0, 6)]
    loss, last, center_parts, extra = tr._dense_losses(torch.randn(2, 1, *GRID), targets)
    assert torch.isfinite(loss) and "pos_queries" in extra and extra["dense_pos"] > 0
    loss.backward()
    grads = [p.grad for n, p in tr.model.named_parameters() if n.startswith("transformer.decoder.layers.0")]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    assert tr.model.dense.cls_out.weight.grad.abs().sum() > 0


def test_densedet_losses_backward():
    cfg = _cfg("dense")
    tr = Trainer.__new__(Trainer)
    tr.cfg, tr.device = cfg, "cpu"
    tr.model = _model("dense")
    tr.center_loss, tr.dense_loss = CenterLoss(), DenseLoss()
    tr.vox_scale = torch.tensor(GRID * 2, dtype=torch.float32)
    targets = [torch.tensor([[0.2, 0.25, 0.3, 0.6, 0.55, 0.7]])]
    loss, last, _, extra = tr._dense_losses(torch.randn(1, 1, *GRID), targets)
    loss.backward()
    assert torch.isfinite(loss) and tr.model.dense.reg_out.weight.grad.abs().sum() > 0


def test_anchor_buffer_roundtrip():
    m = _model("dense")
    m2 = build_model(_cfg("dense"))
    m2.load_state_dict(m.state_dict())
    assert torch.equal(m2.dense.anchor_sizes, m.dense.anchor_sizes)
