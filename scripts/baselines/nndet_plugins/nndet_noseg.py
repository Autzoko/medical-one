"""nnDetection module variant without segmentation supervision (bbox-only baseline, docs/prereg/bbox_baselines.md).

``RetinaUNetV001NoSeg`` is RetinaUNetV001 with the segmentation loss weighted 0: the detection head (ATSS, hard
negative sampling, BCE + GIoU), architecture, planning, augmentation, inference and sweep are unchanged. The
segmentation head is kept (untrained) only because nnDetection's evaluation and test-time mirroring read its output.
"""
from nndet.arch.heads.segmenter import DiCESegmenterFgBg
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001


class ZeroLossSegmenter(DiCESegmenterFgBg):
    def compute_loss(self, pred_seg, target):
        zero = pred_seg["seg_logits"].sum() * 0.0
        return {"seg_ce": zero, "seg_dice": zero}


@MODULE_REGISTRY.register
class RetinaUNetV001NoSeg(RetinaUNetV001):
    segmenter_cls = ZeroLossSegmenter
