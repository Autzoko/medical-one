"""Measure forward+backward time and peak GPU memory on random input."""
import argparse
import time

import torch

from medical_one.config import load_config
from medical_one.engine import build_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/abus3d_tdsc.yaml")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--iters", type=int, default=4)
    args = ap.parse_args()
    model = build_model(load_config(args.config)).cuda()
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), 1e-4)
    for bs in args.batch_sizes:
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(bs, 1, 176, 96, 176, device="cuda")
        for it in range(args.iters):
            if it == 1:
                torch.cuda.synchronize()
                start = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outs, enc = model(x)
                loss = sum(o["logits"].mean() + o["boxes"].mean() for o in outs) + enc["logits"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        sec = (time.time() - start) / (args.iters - 1)
        print(f"bs={bs}: {sec:.2f}s/iter, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
