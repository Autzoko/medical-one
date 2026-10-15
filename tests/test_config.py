from medical_one.config import load_config


def test_overrides_are_coerced_to_default_types():
    cfg = load_config(None, ["train.lr=1e-4", "train.iters=500", "data.scale_range=0", "eval.tta_flips=true"])
    assert cfg.train.lr == 1e-4 and isinstance(cfg.train.lr, float)
    assert cfg.train.iters == 500
    assert cfg.data.scale_range == 0.0 and isinstance(cfg.data.scale_range, float)
    assert cfg.eval.tta_flips is True


def test_cv_split_partitions_train_and_validation():
    from medical_one.data.tdsc import cv_split

    splits = {"Train": [{"case": i, "label": "BM"[i % 2]} for i in range(100)],
              "Validation": [{"case": 100 + i, "label": "BM"[i % 3 == 0]} for i in range(30)]}
    held = []
    for k in range(5):
        train, val = cv_split(splits, 5, k)
        assert len(train) + len(val) == 130 and not {m["case"] for m in train} & {m["case"] for m in val}
        held += [m["case"] for m in val]
    assert sorted(held) == list(range(130))
