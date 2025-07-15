"""Box operations for RT-DETR
Aligned with PyTorch version implementation
"""

import jittor as jt
import numpy as np

def ensure_float32(x):
    """确保张量为float32类型"""
    if isinstance(x, jt.Var):
        return x.float32()
    else:
        return x

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
    """Compute the area of a set of bounding boxes"""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes"""
    boxes1 = ensure_float32(boxes1)
    boxes2 = ensure_float32(boxes2)
    
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = jt.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = jt.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min_v=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return ensure_float32(iou)

def generalized_box_iou(boxes1, boxes2):
    """Generalized IoU from https://giou.stanford.edu/"""
    boxes1 = ensure_float32(boxes1)
    boxes2 = ensure_float32(boxes2)
    
    # Convert to xyxy format if needed
    if boxes1.shape[-1] == 4 and boxes2.shape[-1] == 4:
        # Assume already in xyxy format, but convert from cxcywh if needed
        boxes1 = box_cxcywh_to_xyxy(boxes1)
        boxes2 = box_cxcywh_to_xyxy(boxes2)
    
    # Calculate intersection
    lt = jt.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = jt.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    
    wh = (rb - lt).clamp(min_v=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    # Calculate union
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - inter
    
    # Calculate IoU
    iou = inter / union
    
    # Calculate enclosing box
    lti = jt.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rbi = jt.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    
    whi = (rbi - lti).clamp(min_v=0)  # [N,M,2]
    areai = whi[:, :, 0] * whi[:, :, 1]
    
    return ensure_float32(iou - (areai - union) / areai)

def masks_to_boxes(masks):
    """Compute bounding boxes around the provided masks"""
    if masks.numel() == 0:
        return jt.zeros((0, 4), dtype=jt.float32)

    h, w = masks.shape[-2:]

    y = jt.arange(0, h, dtype=jt.float32)
    x = jt.arange(0, w, dtype=jt.float32)
    y, x = jt.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~masks, 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~masks, 1e8).flatten(1).min(-1)[0]

    return jt.stack([x_min, y_min, x_max, y_max], 1)

__all__ = [
    'box_cxcywh_to_xyxy',
    'box_xyxy_to_cxcywh', 
    'box_area',
    'box_iou',
    'generalized_box_iou',
    'masks_to_boxes'
]
