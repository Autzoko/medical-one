"""MONAI 3D RetinaNet baseline (ResNet50-FPN, ATSS matching, hard negative sampling).

Follows the MONAI LUNA16 detection tutorial (monai.apps.detection); only the anchor shapes are adapted to
TDSC lesion sizes in working-space voxels (median d/h/w = 76/14/29). Inputs, patch sampling and
augmentation are the same as for the DETR model, and inference runs on the whole working-space volume.
"""
from typing import Sequence, Tuple

import numpy as np
import torch

from ..data.dataset import normalize

# (d, h, w) at the stride-4 level; levels 1..4 scale them by 1, 2, 4, 8 -> covers ~16..256 x 3..48 x 6..96 voxels
BASE_ANCHORS = ((16, 3, 6), (24, 4, 8), (32, 6, 12))
RETURNED_LAYERS = (1, 2, 3)


def build_retinanet(base_anchors: Sequence[Sequence[int]] = BASE_ANCHORS, score_thresh: float = 0.02,
                    nms_thresh: float = 0.22, detections: int = 50):
    from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
    from monai.apps.detection.networks.retinanet_network import RetinaNet, resnet_fpn_feature_extractor
    from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
    from monai.networks.nets import resnet

    anchors = AnchorGeneratorWithAnchorShape(feature_map_scales=[2 ** i for i in range(len(RETURNED_LAYERS) + 1)],
                                             base_anchor_shapes=[list(a) for a in base_anchors])
    backbone = resnet.ResNet(block=resnet.ResNetBottleneck, layers=[3, 4, 6, 3], block_inplanes=resnet.get_inplanes(),
                             n_input_channels=1, conv1_t_stride=(2, 2, 2), conv1_t_size=(7, 7, 7))
    features = resnet_fpn_feature_extractor(backbone, 3, pretrained_backbone=False, returned_layers=RETURNED_LAYERS)
    size_divisible = [s * 2 * 2 ** max(RETURNED_LAYERS) for s in features.body.conv1.stride]
    net = RetinaNet(spatial_dims=3, num_classes=1, num_anchors=anchors.num_anchors_per_location()[0],
                    feature_extractor=features, size_divisible=size_divisible)
    detector = RetinaNetDetector(network=net, anchor_generator=anchors)
    detector.set_atss_matcher(num_candidates=4, center_in_gt=False)
    detector.set_hard_negative_sampler(batch_size_per_image=64, positive_fraction=0.3, pool_size=20, min_neg=16)
    detector.set_target_keys(box_key="box", label_key="label")
    detector.set_box_selector_parameters(score_thresh=score_thresh, topk_candidates_per_level=1000,
                                         nms_thresh=nms_thresh, detections_per_img=detections)
    return detector


def to_targets(targets: Sequence[torch.Tensor], patch: Sequence[int], device: str):
    """Normalized corner boxes [T,6] -> MONAI target dicts in voxel corners (axes in tensor order)."""
    scale = torch.tensor(list(patch) * 2, dtype=torch.float, device=device)
    return [{"box": t.to(device) * scale, "label": torch.zeros(len(t), dtype=torch.long, device=device)} for t in targets]


@torch.no_grad()
def predict_volume_retinanet(detector, volume: np.ndarray, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    """Whole-volume inference; returns working-space corner boxes [N,6] and scores [N] sorted by score."""
    detector.eval()
    x = normalize(torch.from_numpy(volume).float().to(device))[None, None]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        out = detector(x, use_inferer=False)[0]
    boxes, scores = out["box"].float().cpu(), out["label_scores"].float().cpu()
    order = scores.argsort(descending=True)
    return boxes[order], scores[order]
