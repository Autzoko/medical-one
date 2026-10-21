"""MedNeXt (Roy et al., MICCAI 2023) inside the nnU-Net v2 framework: same preprocessing, augmentation, sampling,
sliding-window inference and segmentation-to-box step as the nnU-Net baseline, only the network and optimiser change.

The network is MONAI's MedNeXt (``create_mednext``: S / B / M / L, kernel 3, residual up/down blocks, GroupNorm).
Following the MedNeXt nnU-Net trainers: AdamW, lr 1e-3, poly schedule, deep supervision on the 4 decoder scales.
MedNeXt downsamples 4 times by 2 isotropically, so it needs a patch size divisible by 16 on every axis; that patch
size lives in a separate plans file (``mednext_plans``), derived from nnUNetPlans so the preprocessed data is shared.
Every MedNeXt block is gradient-checkpointed while training (identical maths, ~3x less activation memory), which
keeps nnU-Net's planned patch size on a 24 GB card (MedNeXt-B, 2 x 96x160x160: 7.4 GB, 0.71 s/iter on a 3090).

nnU-Net finds trainers by class name inside its own package, so ``install()`` copies this module into
``nnunetv2/training/nnUNetTrainer/variants/`` of the active environment.
"""
import json
import os
import shutil

import torch
from torch import nn

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

DS_SCALES = [[1.0, 1.0, 1.0], [0.5] * 3, [0.25] * 3, [0.125] * 3, [0.0625] * 3]


def _mednext(variant, num_input_channels, num_output_channels, enable_deep_supervision):
    from monai.networks.nets.mednext import create_mednext

    from monai.networks.blocks.mednext_block import MedNeXtBlock
    from torch.utils.checkpoint import checkpoint

    net = create_mednext(variant, spatial_dims=3, in_channels=num_input_channels, out_channels=num_output_channels,
                         kernel_size=3, deep_supervision=enable_deep_supervision)
    for m in net.modules():
        if isinstance(m, MedNeXtBlock):
            m.forward = (lambda f: lambda x: checkpoint(f, x, use_reentrant=False) if torch.is_grad_enabled() else f(x))(m.forward)
    return _AlwaysDS(net)


class _AlwaysDS(nn.Module):
    """MONAI's MedNeXt returns the deep-supervision tuple only in train mode, but nnU-Net also computes the
    deep-supervised validation loss in eval mode; this wrapper returns the list whenever ``do_ds`` is on."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    @property
    def do_ds(self):
        return self.net.do_ds

    @do_ds.setter
    def do_ds(self, value):
        self.net.do_ds = value

    def forward(self, x):
        if not self.net.do_ds or self.net.training:
            out = self.net(x)
            return list(out) if isinstance(out, (tuple, list)) else out
        self.net.train(True)  # only the return branch depends on .training (GroupNorm, no dropout)
        try:
            return list(self.net(x))
        finally:
            self.net.train(False)


class nnUNetTrainer_MedNeXtB(nnUNetTrainer):
    variant = "B"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-3
        self.weight_decay = 1e-4
        patch = self.configuration_manager.patch_size
        assert all(p % 16 == 0 for p in patch), f"MedNeXt needs a patch size divisible by 16, got {patch}"

    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager, num_input_channels, num_output_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return _mednext(cls.variant, num_input_channels, num_output_channels, enable_deep_supervision)

    def _get_deep_supervision_scales(self):
        return DS_SCALES if self.enable_deep_supervision else None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

    def set_deep_supervision_enabled(self, enabled: bool):
        mod = self.network.module if self.is_ddp else self.network
        mod = getattr(mod, "_orig_mod", mod)
        mod.do_ds = enabled


class nnUNetTrainer_MedNeXtS(nnUNetTrainer_MedNeXtB):
    variant = "S"


class nnUNetTrainer_MedNeXtB_250epochs(nnUNetTrainer_MedNeXtB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 250


def mednext_plans(preprocessed_dir, src="nnUNetPlans", dst="nnUNetPlans_MedNeXt", multiple=16):
    """Copy of the nnU-Net plans whose 3d_fullres patch size is rounded to a multiple of 16 per axis (nearest,
    at least 16). The data identifier is kept, so the existing preprocessed arrays are reused."""
    path = os.path.join(preprocessed_dir, f"{dst}.json")
    if os.path.exists(path):
        return dst
    with open(os.path.join(preprocessed_dir, f"{src}.json")) as f:
        plans = json.load(f)
    plans["plans_name"] = dst
    cfg = plans["configurations"]["3d_fullres"]
    cfg["patch_size"] = [max(multiple, int(round(p / multiple)) * multiple) for p in cfg["patch_size"]]
    with open(path, "w") as f:
        json.dump(plans, f, indent=1)
    print(f"MedNeXt plans: patch {cfg['patch_size']}, batch {cfg['batch_size']} -> {path}", flush=True)
    return dst


def install():
    """Make the trainers visible to nnUNetv2_train / nnUNetv2_predict (they search nnunetv2's own package)."""
    import nnunetv2

    dst = os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer", "variants", "medical_one_mednext.py")
    shutil.copyfile(os.path.abspath(__file__), dst)
    return dst


if __name__ == "__main__":  # quick shape check of the deep-supervision outputs
    net = _mednext("B", 1, 2, True)
    x = torch.zeros(1, 1, 32, 64, 64)
    print([tuple(o.shape) for o in net.eval()(x)], sum(p.numel() for p in net.parameters()))
