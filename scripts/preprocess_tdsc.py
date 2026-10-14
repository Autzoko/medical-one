"""Convert TDSC-ABUS NRRD volumes + masks into the model working space (336x96x336 uint8 + boxes)."""
import argparse

from medical_one.config import load_config
from medical_one.data.tdsc import preprocess_tdsc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/abus3d_tdsc.yaml")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("overrides", nargs="*", help="key.sub=value")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    metas = preprocess_tdsc(cfg.paths.tdsc_root, cfg.paths.prep_dir, cfg.paths.extracted_train, args.workers)
    truncated = sum(m.depth_kept_frac < 1 for m in metas)
    lost = sum(m.depth_kept_frac <= 0 for m in metas)
    print(f"cases={len(metas)} truncated_by_depth_crop={truncated} lost={lost} -> {cfg.paths.prep_dir}")


if __name__ == "__main__":
    main()
