"""
数据类型安全的RT-DETR损失函数
完全独立实现，不修改原始源码，彻底解决float32/float64混合问题
"""

import jittor as jt
import jittor.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment


def safe_float32(tensor):
    """安全地将tensor转换为float32"""
    if isinstance(tensor, jt.Var):
        return tensor.float32()
    elif isinstance(tensor, np.ndarray):
        return jt.array(tensor.astype(np.float32), dtype=jt.float32)
    else:
        return jt.array(tensor, dtype=jt.float32)


def safe_int64(tensor):
    """安全地将tensor转换为int64"""
    if isinstance(tensor, jt.Var):
        return tensor.int64()
    elif isinstance(tensor, np.ndarray):
        return jt.array(tensor.astype(np.int64), dtype=jt.int64)
    else:
        return jt.array(tensor, dtype=jt.int64)


def numpy_box_iou(boxes1, boxes2):
    """使用numpy计算IoU，避免Jittor的数据类型问题"""
    # 转换为numpy
    if isinstance(boxes1, jt.Var):
        boxes1_np = boxes1.float32().numpy()
    else:
        boxes1_np = np.array(boxes1, dtype=np.float32)
    
    if isinstance(boxes2, jt.Var):
        boxes2_np = boxes2.float32().numpy()
    else:
        boxes2_np = np.array(boxes2, dtype=np.float32)
    
    # 计算面积
    area1 = (boxes1_np[:, 2] - boxes1_np[:, 0]) * (boxes1_np[:, 3] - boxes1_np[:, 1])
    area2 = (boxes2_np[:, 2] - boxes2_np[:, 0]) * (boxes2_np[:, 3] - boxes2_np[:, 1])
    
    # 计算交集
    lt = np.maximum(boxes1_np[:, None, :2], boxes2_np[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1_np[:, None, 2:], boxes2_np[:, 2:])  # [N,M,2]
    
    wh = np.clip(rb - lt, 0, None)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    # 计算并集
    union = area1[:, None] + area2 - inter
    union = np.clip(union, 1e-8, None)  # 避免除零
    
    # 计算IoU
    iou = inter / union
    
    return safe_float32(iou), safe_float32(union)


def numpy_generalized_box_iou(boxes1, boxes2):
    """使用numpy计算GIoU，避免Jittor的数据类型问题"""
    iou, union = numpy_box_iou(boxes1, boxes2)
    
    # 转换为numpy进行计算
    if isinstance(boxes1, jt.Var):
        boxes1_np = boxes1.float32().numpy()
    else:
        boxes1_np = np.array(boxes1, dtype=np.float32)
    
    if isinstance(boxes2, jt.Var):
        boxes2_np = boxes2.float32().numpy()
    else:
        boxes2_np = np.array(boxes2, dtype=np.float32)
    
    # 计算最小外接矩形
    lt = np.minimum(boxes1_np[:, None, :2], boxes2_np[:, :2])
    rb = np.maximum(boxes1_np[:, None, 2:], boxes2_np[:, 2:])
    
    wh = np.clip(rb - lt, 0, None)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]
    
    # 计算GIoU
    giou = iou.numpy() - (area - union.numpy()) / np.clip(area, 1e-8, None)
    
    return safe_float32(giou)


def box_cxcywh_to_xyxy_numpy(x):
    """使用numpy将中心点格式转换为左上右下格式"""
    if isinstance(x, jt.Var):
        x_np = x.float32().numpy()
    else:
        x_np = np.array(x, dtype=np.float32)
    
    x_c, y_c, w, h = x_np[..., 0], x_np[..., 1], x_np[..., 2], x_np[..., 3]
    b = np.stack([
        x_c - 0.5 * w,  # x1
        y_c - 0.5 * h,  # y1
        x_c + 0.5 * w,  # x2
        y_c + 0.5 * h   # y2
    ], axis=-1)
    
    return safe_float32(b)


class NumpyHungarianMatcher(nn.Module):
    """使用numpy实现的匈牙利匹配器，避免Jittor数据类型问题"""
    
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
            
            # 转换为numpy进行计算
            if self.use_focal:
                out_prob_np = jt.sigmoid(outputs["pred_logits"]).float32().numpy()
            else:
                out_prob_np = jt.softmax(outputs["pred_logits"], dim=-1).float32().numpy()
            
            out_bbox_np = outputs["pred_boxes"].float32().numpy()
            
            # 收集目标
            tgt_ids_list = []
            tgt_bbox_list = []
            
            for target in targets:
                tgt_ids_list.append(target["labels"].int64().numpy())
                tgt_bbox_list.append(target["boxes"].float32().numpy())
            
            tgt_ids_np = np.concatenate(tgt_ids_list)
            tgt_bbox_np = np.concatenate(tgt_bbox_list)
            
            # 展平预测
            out_prob_flat = out_prob_np.reshape(-1, out_prob_np.shape[-1])  # [bs*num_queries, num_classes]
            out_bbox_flat = out_bbox_np.reshape(-1, 4)  # [bs*num_queries, 4]
            
            # 计算分类成本
            if self.use_focal:
                out_prob_selected = out_prob_flat[:, tgt_ids_np]
                neg_cost_class = (1 - self.alpha) * (out_prob_selected ** self.gamma) * (-np.log(np.clip(1 - out_prob_selected + 1e-8, 1e-8, 1.0)))
                pos_cost_class = self.alpha * ((1 - out_prob_selected) ** self.gamma) * (-np.log(np.clip(out_prob_selected + 1e-8, 1e-8, 1.0)))
                cost_class = pos_cost_class - neg_cost_class
            else:
                cost_class = -out_prob_flat[:, tgt_ids_np]
            
            # 计算边界框成本
            cost_bbox = np.abs(out_bbox_flat[:, None, :] - tgt_bbox_np[None, :, :]).sum(axis=-1)
            
            # 计算GIoU成本
            out_bbox_xyxy = box_cxcywh_to_xyxy_numpy(out_bbox_flat)
            tgt_bbox_xyxy = box_cxcywh_to_xyxy_numpy(tgt_bbox_np)
            cost_giou = -numpy_generalized_box_iou(out_bbox_xyxy, tgt_bbox_xyxy).numpy()
            
            # 最终成本矩阵
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.reshape(bs, num_queries, -1)
            
            # 分割并匹配
            sizes = [len(target["boxes"]) for target in targets]
            indices = []
            
            start_idx = 0
            for i, size in enumerate(sizes):
                if size == 0:
                    indices.append((np.array([]), np.array([])))
                    continue
                
                end_idx = start_idx + size
                cost_matrix = C[i, :, start_idx:end_idx]
                
                try:
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    indices.append((row_ind, col_ind))
                except:
                    # Fallback: 简单匹配
                    row_ind = np.arange(min(num_queries, size))
                    col_ind = np.arange(min(num_queries, size))
                    indices.append((row_ind, col_ind))
                
                start_idx = end_idx
            
            # 转换回Jittor tensor
            return [(safe_int64(i), safe_int64(j)) for i, j in indices]


class DTypeSafeSetCriterion(nn.Module):
    """数据类型安全的损失函数"""
    
    def __init__(self, num_classes, matcher, weight_dict, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
    
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """计算分类损失"""
        assert 'pred_logits' in outputs
        src_logits = safe_float32(outputs['pred_logits'])
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = jt.concat([safe_int64(t["labels"][J]) for t, (_, J) in zip(targets, indices)])
        target_classes = jt.full(src_logits.shape[:2], self.num_classes, dtype=jt.int64)
        target_classes[idx] = target_classes_o
        
        # 使用focal loss
        target_classes_onehot = jt.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1], dtype=src_logits.dtype)
        ones_tensor = jt.ones_like(target_classes.unsqueeze(-1)).float32()
        target_classes_onehot.scatter_(-1, target_classes.unsqueeze(-1), ones_tensor)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        
        loss_ce = self.sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=0.25, gamma=2) * src_logits.shape[1]
        losses = {'loss_focal': safe_float32(loss_ce)}
        
        return losses
    
    def sigmoid_focal_loss(self, inputs, targets, num_boxes, alpha=0.25, gamma=2):
        """Focal loss实现"""
        prob = jt.sigmoid(safe_float32(inputs))
        ce_loss = nn.binary_cross_entropy_with_logits(safe_float32(inputs), safe_float32(targets), reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        
        return loss.mean(1).sum() / num_boxes
    
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """计算边界框损失"""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = safe_float32(outputs['pred_boxes'][idx])
        target_boxes = jt.concat([safe_float32(t['boxes'][i]) for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = nn.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        
        # 使用numpy计算GIoU
        src_boxes_xyxy = box_cxcywh_to_xyxy_numpy(src_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy_numpy(target_boxes)
        loss_giou = 1 - jt.diag(numpy_generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        
        return losses
    
    def _get_src_permutation_idx(self, indices):
        """获取源排列索引"""
        batch_idx = jt.concat([jt.full_like(safe_int64(src), i) for i, (src, _) in enumerate(indices)])
        src_idx = jt.concat([safe_int64(src) for (src, _) in indices])
        return batch_idx, src_idx
    
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        """获取指定损失"""
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    
    def execute(self, outputs, targets):
        """计算所有损失"""
        # 强制所有输出为float32
        outputs_clean = {}
        for k, v in outputs.items():
            if k not in ['aux_outputs', 'enc_outputs']:
                outputs_clean[k] = safe_float32(v)
        
        # 强制所有目标数据类型一致
        for target in targets:
            target['boxes'] = safe_float32(target['boxes'])
            target['labels'] = safe_int64(target['labels'])
        
        # 获取匹配
        indices = self.matcher(outputs_clean, targets)
        
        # 计算目标数量
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = max(1, num_boxes)  # 确保至少为1，避免除零
        
        # 计算损失
        losses = {}
        for loss in self.losses:
            kwargs = {}
            if loss == 'labels':
                kwargs['log'] = False
            l_dict = self.get_loss(loss, outputs_clean, targets, indices, num_boxes, **kwargs)
            losses.update(l_dict)
        
        return losses


def build_dtype_safe_criterion(num_classes=80):
    """构建数据类型安全的损失函数"""
    matcher = NumpyHungarianMatcher(
        cost_class=1, cost_bbox=5, cost_giou=2, 
        use_focal=True, alpha=0.25, gamma=2.0
    )
    
    weight_dict = {
        'loss_focal': 2,
        'loss_bbox': 5,
        'loss_giou': 2
    }
    
    losses = ['labels', 'boxes']
    
    criterion = DTypeSafeSetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses
    )
    
    return criterion
