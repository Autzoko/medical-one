"""Run one nnDetection command on this repo's torch 2.x stack.

nnDetection pins 2021-era dependencies. ``patch_repo`` makes small source edits to the pinned checkout
(idempotent, so every process, including spawned workers, sees them):
  - ``torch._six`` (removed in torch 2.0) provided only ``string_classes``; it becomes a local tuple;
  - the CUDA 3D-NMS extension ``nndet._C`` is not built (the nodes' CUDA toolkit is 12.x while torch is
    cu130), so CUDA boxes use ``nms_torch`` below: the same greedy rule as nnDetection's own CPU fallback;
  - the planner's nvidia-smi memory reading, [N/A] on the unified-memory GB10, falls back to the driver.
At run time MLflow logging is replaced by Lightning's CSVLogger (it only records metrics).

  python scripts/baselines/nndet_launch.py NNDET_ROOT {prep,unpack,train,predict} ARGS...
"""
import importlib
import os
import sys

COMMANDS = {"prep": ("scripts.preprocess", "main"), "unpack": ("scripts.utils", "unpack"),
            "train": ("scripts.train", "train"), "predict": ("scripts.predict", "main")}

NMS_TORCH = '''import numpy as np
import torch

from nndet.core.boxes.ops import box_iou


def nms_torch(boxes, scores, thresh):
    """Greedy NMS: highest score first, a box is dropped when its IoU with a kept box exceeds ``thresh``
    (the rule of ``nms_cpu``). Returns kept indices in decreasing score order."""
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    order = torch.argsort(scores, descending=True)
    over = (box_iou(boxes[order], boxes[order]) > thresh).cpu().numpy()
    removed = np.zeros(len(order), bool)
    keep = []
    for i in range(len(order)):
        if not removed[i]:
            keep.append(i)
            removed |= over[i]
    return order[torch.as_tensor(keep, device=boxes.device)]
'''

SMI_INT = '''def _smi_int(reading: str) -> int:
    """nvidia-smi reports [N/A] memory on unified-memory GPUs (GB10); use the driver's used memory then."""
    try:
        return int(reading)
    except ValueError:
        free, total = torch.cuda.mem_get_info()
        return (total - free) // 2 ** 20


def smi_memory_allocated(gpu_id: int = 0) -> int:'''

EDITS = {
    "nndet/utils/tensor.py": [("from torch._six import string_classes", "string_classes = (str, bytes)")],
    "nndet/core/boxes/nms.py": [("    nms_gpu = None",
                                 "    from nndet.core.boxes.nms_torch import nms_torch as nms_gpu")],
    # only logged (the available-memory branch that used it is commented out upstream)
    "nndet/planning/estimator.py": [("def smi_memory_allocated(gpu_id: int = 0) -> int:", SMI_INT),
                                    ("    reading = int(sp.check_output(", "    reading = _smi_int(sp.check_output(")],
}


def patch_repo(root: str) -> None:
    with open(os.path.join(root, "nndet/core/boxes/nms_torch.py"), "w") as f:
        f.write(NMS_TORCH)
    for rel, edits in EDITS.items():
        path = os.path.join(root, rel)
        with open(path) as f:
            src = f.read()
        for old, new in edits:
            if new not in src:
                assert src.count(old) == 1, (rel, old)
                src = src.replace(old, new)
        with open(path, "w") as f:
            f.write(src)


def csv_logger(experiment_name, tags=None, save_dir="./mlruns"):
    from pytorch_lightning.loggers import CSVLogger

    return CSVLogger(save_dir, name=experiment_name)


def main() -> None:
    root, cmd, args = sys.argv[1], sys.argv[2], sys.argv[3:]
    sys.path.insert(0, root)
    module_name, fn = COMMANDS[cmd]
    module = importlib.import_module(module_name)
    if cmd == "train":
        module.MLFlowLogger = csv_logger
    sys.argv = [f"nndet_{cmd}"] + args
    getattr(module, fn)()


if __name__ == "__main__":
    main()
