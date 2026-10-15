from medical_one.config import load_config


def test_overrides_are_coerced_to_default_types():
    cfg = load_config(None, ["train.lr=1e-4", "train.iters=500", "data.scale_range=0", "eval.tta_flips=true"])
    assert cfg.train.lr == 1e-4 and isinstance(cfg.train.lr, float)
    assert cfg.train.iters == 500
    assert cfg.data.scale_range == 0.0 and isinstance(cfg.data.scale_range, float)
    assert cfg.eval.tta_flips is True
