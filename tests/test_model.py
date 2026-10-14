import torch

from medical_one.losses import SetCriterion
from medical_one.models import ABUSDet3D
from medical_one.models.deform_attn import ms_deform_attn_3d


def test_deform_attn_samples_voxel_centers():
    # One level, constant-per-voxel value; sampling exactly at a voxel center returns that voxel.
    d, h, w = 4, 3, 5
    vol = torch.arange(d * h * w, dtype=torch.float).view(1, d * h * w, 1, 1)
    idx = (2, 1, 3)
    loc = torch.tensor([(i + 0.5) / n for i, n in zip(idx, (d, h, w))]).view(1, 1, 1, 1, 1, 3)
    out = ms_deform_attn_3d(vol, [(d, h, w)], loc, torch.ones(1, 1, 1, 1, 1))
    assert out.item() == idx[0] * h * w + idx[1] * w + idx[2]


def test_small_forward_backward():
    torch.manual_seed(0)
    model = ABUSDet3D(num_queries=50)
    x = torch.randn(1, 1, 32, 32, 32)
    outs, enc = model(x)
    assert len(outs) == 6
    assert outs[-1]["logits"].shape == (1, 50, 1) and outs[-1]["boxes"].shape == (1, 50, 6)
    logits, corners = model.predict(x)
    assert corners.shape == (1, 50, 6)
    target = [torch.tensor([[0.2, 0.2, 0.2, 0.5, 0.6, 0.4]])]
    loss, _ = SetCriterion().total(outs, enc, target)
    loss.backward()
    assert torch.isfinite(loss)


def test_param_count_matches_export():
    # The exported graph holds ~17.56M weights once precomputed constants are excluded.
    n = sum(p.numel() for p in ABUSDet3D().parameters())
    assert abs(n / 1e6 - 17.57) < 0.05


def test_trainer_ema_tracks_weights(tmp_path, monkeypatch):
    from medical_one.config import load_config
    from medical_one.engine import trainer as tr

    monkeypatch.setattr(tr, "load_meta", lambda _: {"Train": [], "Validation": [], "Test": []})
    cfg = load_config(None, [f"paths.runs_dir={tmp_path}", "train.ema_decay=0.5", "model.num_queries=10"])
    t = tr.Trainer(cfg, device="cpu")
    p = next(t.model.parameters())
    with torch.no_grad():
        p.add_(1.0)
    before = next(t.ema.parameters()).clone()
    t._update_ema()
    assert torch.allclose(next(t.ema.parameters()), 0.5 * before + 0.5 * p)
