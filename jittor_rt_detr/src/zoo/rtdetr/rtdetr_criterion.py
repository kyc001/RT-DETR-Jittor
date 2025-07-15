"""RT-DETR Criterion (Loss Function)
Aligned with PyTorch version implementation
"""

import jittor as jt
import jittor.nn as nn
from scipy.optimize import linear_sum_assignment
import numpy as np

def ensure_float32(x):
    """确保张量为float32类型"""
    if isinstance(x, jt.Var):
        return x.float32()
    elif isinstance(x, (list, tuple)):
        return [ensure_float32(item) for item in x]
    else:
        return x

def ensure_int64(x):
    """确保张量为int64类型"""
    if isinstance(x, jt.Var):
        return x.int64()
    else:
        return x

def box_cxcywh_to_xyxy(x):
    """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return jt.stack(b, dim=-1)

def generalized_box_iou(boxes1, boxes2):
    """Generalized IoU from https://giou.stanford.edu/"""
    # Ensure float32
    boxes1 = ensure_float32(boxes1)
    boxes2 = ensure_float32(boxes2)
    
    # Convert to xyxy format
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

class HungarianMatcher(nn.Module):
    """Hungarian Matcher for RT-DETR"""
    
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2, use_focal=True, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal = use_focal
        self.alpha = alpha
        self.gamma = gamma

    def execute(self, outputs, targets):
        with jt.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # 展平以计算成本矩阵 - 确保数据类型一致
            pred_logits = ensure_float32(outputs["pred_logits"])
            pred_boxes = ensure_float32(outputs["pred_boxes"])
            
            if self.use_focal:
                out_prob = ensure_float32(jt.sigmoid(pred_logits.flatten(0, 1)))
            else:
                out_prob = ensure_float32(pred_logits.flatten(0, 1).softmax(-1))

            out_bbox = ensure_float32(pred_boxes.flatten(0, 1))

            # 收集目标标签和边界框 - 强制数据类型一致
            tgt_ids = ensure_int64(jt.concat([ensure_int64(v["labels"]) for v in targets]))
            tgt_bbox = ensure_float32(jt.concat([ensure_float32(v["boxes"]) for v in targets]))

            # 计算分类成本
            if self.use_focal:
                neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            else:
                cost_class = -out_prob[:, tgt_ids]

            # 计算L1成本 - Jittor没有cdist，使用手动实现
            cost_bbox = jt.abs(out_bbox[:, None, :] - tgt_bbox[None, :, :]).sum(dim=-1)

            # 计算giou成本
            cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

            # 最终成本矩阵
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(jt.array(i, dtype=jt.int64), jt.array(j, dtype=jt.int64)) for i, j in indices]

class SetCriterion(nn.Module):
    """RT-DETR Loss Criterion"""
    
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (Focal Loss)"""
        pred_logits = ensure_float32(outputs['pred_logits'])

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = jt.concat([ensure_int64(t["labels"][J]) for t, (_, J) in zip(targets, indices)])
        target_classes = jt.full(pred_logits.shape[:2], self.num_classes, dtype=jt.int64)
        target_classes[idx] = target_classes_o

        # 简化的focal loss实现
        # 将logits reshape为 [batch*queries, num_classes]
        pred_logits_flat = pred_logits.view(-1, pred_logits.shape[-1])
        target_classes_flat = target_classes.view(-1)

        # 计算交叉熵损失
        ce_loss = nn.cross_entropy_loss(pred_logits_flat, target_classes_flat, reduction='none')
        ce_loss = ce_loss.view(pred_logits.shape[:2])

        # Focal loss权重
        p_t = jt.exp(-ce_loss)
        loss_ce = self.focal_alpha * (1 - p_t) ** self.focal_gamma * ce_loss

        return ensure_float32(loss_ce.mean())

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Bounding box losses"""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = ensure_float32(outputs['pred_boxes'][idx])
        target_boxes = ensure_float32(jt.concat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0))

        # Jittor的l1_loss不支持reduction参数，手动计算
        loss_bbox = jt.abs(src_boxes - target_boxes)
        loss_giou = 1 - jt.diag(generalized_box_iou(src_boxes, target_boxes))

        return ensure_float32(loss_bbox.sum() / num_boxes), ensure_float32(loss_giou.sum() / num_boxes)

    def _get_src_permutation_idx(self, indices):
        batch_idx = jt.concat([jt.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = jt.concat([src for (src, _) in indices])
        return batch_idx, src_idx

    def execute(self, outputs, targets):
        """Forward pass"""
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        
        # Compute the average number of target boxes across all nodes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = jt.array([num_boxes], dtype=jt.float32)
        num_boxes = jt.clamp(num_boxes, min_v=1).item()

        # Compute all the requested losses
        losses = {}
        
        # Classification loss
        losses['loss_ce'] = self.loss_labels(outputs, targets, indices, num_boxes)
        
        # Box losses
        loss_bbox, loss_giou = self.loss_boxes(outputs, targets, indices, num_boxes)
        losses['loss_bbox'] = loss_bbox
        losses['loss_giou'] = loss_giou

        # Auxiliary losses
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                losses[f'loss_ce_aux_{i}'] = self.loss_labels(aux_outputs, targets, indices, num_boxes)
                l_bbox, l_giou = self.loss_boxes(aux_outputs, targets, indices, num_boxes)
                losses[f'loss_bbox_aux_{i}'] = l_bbox
                losses[f'loss_giou_aux_{i}'] = l_giou

        return losses

def build_criterion(num_classes):
    """Build RT-DETR criterion"""
    matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2, use_focal=True)
    
    weight_dict = {
        'loss_ce': 1,
        'loss_bbox': 5,
        'loss_giou': 2,
    }
    
    # Add auxiliary losses
    aux_weight_dict = {}
    for i in range(6):  # 6 decoder layers
        aux_weight_dict.update({k + f'_aux_{i}': v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    
    losses = ['labels', 'boxes']
    
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        focal_alpha=0.25,
        focal_gamma=2.0
    )
    
    return criterion

__all__ = ['SetCriterion', 'HungarianMatcher', 'build_criterion']
