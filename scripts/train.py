"""Train ABUSDet3D on preprocessed TDSC-ABUS."""
import argparse

from medical_one.config import load_config
from medical_one.engine import Trainer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/abus3d_tdsc.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("overrides", nargs="*", help="key.sub=value, e.g. train.iters=6000 name=pilot")
    args = ap.parse_intermixed_args()  # overrides may come before and after --config
    Trainer(load_config(args.config, args.overrides), args.device).fit()


if __name__ == "__main__":
    main()
