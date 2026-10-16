"""Native-resolution volume V and the 2.5D image construction shared by the YOLO branch and the classifier.

As in the deployed pipeline (audit: ``deep_2d.md`` sections 1-3): the raw array R [N0, N1, N2] becomes
V = R.permute(1, 0, 2).flip(1), i.e. V[u, v, w] = R[N0-1-v, u, w]. Slices are taken along u; a 2.5D image
stacks the slices (u-1, u, u+1) (clamped at the volume ends) as channels, so its rows are v and its
columns are w. The 3D branch works on V resized to 336 x 112 x 336 (``data.tdsc.to_working_space``), so
the two branches share the (u, v, w) axis order.
"""
import math
from typing import Sequence, Tuple

import cv2
import numpy as np

LETTER_SIZE = 960
LETTER_STRIDE = 32
LETTER_COLOR = 114


def native_view(raw: np.ndarray) -> np.ndarray:
    """R [N0, N1, N2] -> V [N1, N0, N2] with the depth axis flipped (contiguous copy)."""
    return np.ascontiguousarray(raw.transpose(1, 0, 2)[:, ::-1, :])


def raw_box_to_v(lo: Sequence[int], hi: Sequence[int], shape: Sequence[int]) -> np.ndarray:
    """Half-open raw-array box (axis 0..2) -> half-open V box (u0, v0, w0, u1, v1, w1)."""
    n0 = shape[0]
    return np.array([lo[1], n0 - hi[0], lo[2], hi[1], n0 - lo[0], hi[2]], np.float64)


def three_slice(v: np.ndarray, i: int) -> np.ndarray:
    """HWC uint8 image with channels (V[i-1], V[i], V[i+1]), indices clamped to the volume."""
    n = v.shape[0]
    return np.ascontiguousarray(np.stack([v[max(i - 1, 0)], v[i], v[min(i + 1, n - 1)]], -1))


def letterbox(img: np.ndarray, new: int = LETTER_SIZE, stride: int = LETTER_STRIDE,
              color: int = LETTER_COLOR) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """YOLOv5 letterbox with auto=True, scaleFill=False, scaleUp=True: the long side becomes ``new`` and
    the short side is padded to a multiple of ``stride`` (so the input is not always square)."""
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw, dh = (new - new_w) % stride / 2, (new - new_h) % stride / 2
    if (w, h) != (new_w, new_h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color,) * 3)
    return img, r, (dw, dh)


def unletterbox(boxes: np.ndarray, letter_shape: Sequence[int], orig_shape: Sequence[int]) -> np.ndarray:
    """xyxy boxes on the letterboxed image -> original slice pixels; no clipping, then rounded (as the DLL)."""
    gain = min(letter_shape[0] / orig_shape[0], letter_shape[1] / orig_shape[1])
    pad_x = (letter_shape[1] - orig_shape[1] * gain) / 2
    pad_y = (letter_shape[0] - orig_shape[0] * gain) / 2
    out = boxes.astype(np.float64).copy()
    out[:, [0, 2]] -= pad_x
    out[:, [1, 3]] -= pad_y
    out[:, :4] /= gain
    return np.round(out)


def roi_image(v: np.ndarray, box: Sequence[float], context: int = 32, size: int = 224) -> np.ndarray:
    """Classifier input for a V-space box (u0, v0, w0, u1, v1, w1), as in the deployed ResNet stage
    (audit ``deep_fusion_cls.md`` part 2): the 2.5D image at the box's center slice, the in-plane box
    widened by ``context`` pixels, zero-padded to a square (extra pixel at bottom/right), resized to
    ``size`` (INTER_LINEAR). Returns HWC uint8."""
    n, h, w = v.shape
    center = min(max(math.trunc(box[0] + box[3]) // 2, 0), n - 1)
    img = three_slice(v, center)
    lv, lw = max(math.trunc(box[1]), 0), max(math.trunc(box[2]), 0)
    uv, uw = min(math.trunc(box[4]), h - 1), min(math.trunc(box[5]), w - 1)
    a_v, b_v = max(lv - context, 0), min(uv + context, h)
    a_w, b_w = max(lw - context, 0), min(uw + context, w)
    crop = img[a_v:b_v, a_w:b_w]
    ch, cw = crop.shape[:2]
    delta = abs(ch - cw)
    lead, trail = delta // 2, delta - delta // 2
    if ch <= cw:
        crop = cv2.copyMakeBorder(crop, lead, trail, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        crop = cv2.copyMakeBorder(crop, 0, 0, lead, trail, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
