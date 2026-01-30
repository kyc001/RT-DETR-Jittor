"""Box operations for RT-DETR
严格按照PyTorch版本: rtdetr_pytorch/src/zoo/rtdetr/box_ops.py
"""

import jittor as jt


__all__ = [
    'box_cxcywh_to_xyxy',
    'box_xyxy_to_cxcywh',
    'box_area',
    'box_iou',
    'generalized_box_iou',
    'masks_to_boxes'
]


def box_cxcywh_to_xyxy(x):
    """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return jt.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h) format"""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return jt.stack(b, dim=-1)


def box_area(boxes):
    """Compute the area of a set of bounding boxes
    boxes should be in (x1, y1, x2, y2) format
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes
    boxes should be in (x1, y1, x2, y2) format

    Returns:
        iou: [N, M] IoU matrix
        union: [N, M] union area matrix
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = jt.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = jt.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min_v=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format (xyxy format)

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all().item()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all().item()

    iou, union = box_iou(boxes1, boxes2)

    lt = jt.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = jt.maximum(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min_v=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks,
    (H, W) are the spatial dimensions.

    Returns a [N, 4] tensor, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return jt.zeros((0, 4), dtype=jt.float32)

    h, w = masks.shape[-2:]

    y = jt.arange(0, h, dtype=jt.float32)
    x = jt.arange(0, w, dtype=jt.float32)
    y, x = jt.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = jt.where(masks.bool(), x_mask, jt.array(1e8)).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = jt.where(masks.bool(), y_mask, jt.array(1e8)).flatten(1).min(-1)[0]

    return jt.stack([x_min, y_min, x_max, y_max], 1)
