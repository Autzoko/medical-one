import torch

from medical_one.utils.boxes import box_iou_giou, cxcyczwhd_to_xyxy, nms_3d, xyxy_to_cxcyczwhd


def test_roundtrip():
    b = torch.tensor([[0.1, 0.2, 0.3, 0.5, 0.6, 0.9]])
    assert torch.allclose(cxcyczwhd_to_xyxy(xyxy_to_cxcyczwhd(b)), b)


def test_iou_giou():
    a = torch.tensor([[0.0, 0, 0, 2, 2, 2]])
    b = torch.tensor([[1.0, 1, 1, 3, 3, 3], [4.0, 4, 4, 5, 5, 5]])
    iou, giou = box_iou_giou(a, b)
    assert torch.allclose(iou[0, 0], torch.tensor(1 / 15))
    assert iou[0, 1] == 0 and giou[0, 1] < 0
    assert torch.allclose(box_iou_giou(a, a)[1], torch.ones(1, 1))


def test_nms_keeps_best_and_disjoint():
    boxes = torch.tensor([[0.0, 0, 0, 2, 2, 2], [0.1, 0, 0, 2, 2, 2], [5.0, 5, 5, 6, 6, 6]])
    scores = torch.tensor([0.5, 0.9, 0.3])
    assert nms_3d(boxes, scores, 0.05) == [1, 2]
