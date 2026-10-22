"""Convert TDSC-ABUS NRRD volumes + masks into the model working space (336x96x336 uint8 + boxes)."""
import argparse

from medical_one.config import load_config
from medical_one.data.tdsc import export_box_masks, export_masks, preprocess_tdsc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/abus3d_tdsc.yaml")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--masks-only", action="store_true", help="only write working-space masks to <prep_dir>/masks")
    ap.add_argument("--box-masks", action="store_true",
                    help="only write cuboid masks of the boxes in meta.json to <prep_dir>/box_masks (bbox-only baselines)")
    ap.add_argument("overrides", nargs="*", help="key.sub=value")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.box_masks:
        res = export_box_masks(cfg.paths.prep_dir)
        print(f"box masks={len(res)} min voxels={min(n for _, _, n in res)} -> {cfg.paths.prep_dir}/box_masks")
        return
    if args.masks_only:
        res = export_masks(cfg.paths.tdsc_root, cfg.paths.prep_dir, cfg.paths.extracted_train, args.workers)
        print(f"masks={len(res)} empty={sum(n == 0 for _, _, n in res)} -> {cfg.paths.prep_dir}/masks")
        return
    metas = preprocess_tdsc(cfg.paths.tdsc_root, cfg.paths.prep_dir, cfg.paths.extracted_train, args.workers)
    truncated = sum(m.depth_kept_frac < 1 for m in metas)
    lost = sum(m.depth_kept_frac <= 0 for m in metas)
    print(f"cases={len(metas)} truncated_by_depth_crop={truncated} lost={lost} -> {cfg.paths.prep_dir}")


if __name__ == "__main__":
    main()
