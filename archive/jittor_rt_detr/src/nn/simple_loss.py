"""
简化的RT-DETR损失函数
完全避开数据类型不匹配问题，使用最简单的损失计算
"""

import jittor as jt
import jittor.nn as nn
import numpy as np


class SimplifiedRTDETRLoss(nn.Module):
    """
    简化的RT-DETR损失函数
    避开复杂的匈牙利匹配，使用简单的最近邻匹配
    """
    
    def __init__(self, num_classes=80, weight_dict=None):
        super().__init__()
        self.num_classes = num_classes
        
        # 默认权重
        if weight_dict is None:
            self.weight_dict = {
                'loss_ce': 1.0,
                'loss_bbox': 5.0,
                'loss_giou': 2.0
            }
        else:
            self.weight_dict = weight_dict
    
    def simple_matcher(self, pred_boxes, target_boxes):
        """
        简单的最近邻匹配器
        为每个目标框找到最近的预测框
        """
        # 确保数据类型一致
        pred_boxes = pred_boxes.float32()
        target_boxes = target_boxes.float32()
        
        num_queries = pred_boxes.shape[0]
        num_targets = target_boxes.shape[0]
        
        if num_targets == 0:
            return [], []
        
        # 计算中心点距离
        pred_centers = pred_boxes[:, :2]  # [num_queries, 2]
        target_centers = target_boxes[:, :2]  # [num_targets, 2]
        
        # 手动计算距离矩阵
        # pred_centers: [num_queries, 2], target_centers: [num_targets, 2]
        pred_expanded = pred_centers.unsqueeze(1)  # [num_queries, 1, 2]
        target_expanded = target_centers.unsqueeze(0)  # [1, num_targets, 2]
        distances = ((pred_expanded - target_expanded) ** 2).sum(dim=-1).sqrt()  # [num_queries, num_targets]
        
        # 为每个目标找到最近的预测
        matched_pred_indices = []
        matched_target_indices = []

        for target_idx in range(num_targets):
            # 找到距离最近的预测框
            pred_idx = distances[:, target_idx].argmin(dim=0)[0].item()
            matched_pred_indices.append(pred_idx)
            matched_target_indices.append(target_idx)
        
        return matched_pred_indices, matched_target_indices
    
    def compute_classification_loss(self, pred_logits, target_labels, matched_indices):
        """计算分类损失"""
        matched_pred_indices, matched_target_indices = matched_indices
        
        if not matched_pred_indices:
            return jt.array(0.0, dtype=jt.float32)
        
        # 获取匹配的预测和目标
        matched_pred_logits = pred_logits[matched_pred_indices]  # [num_matched, num_classes]
        matched_target_labels = target_labels[matched_target_indices]  # [num_matched]
        
        # 交叉熵损失
        loss_ce = nn.cross_entropy_loss(matched_pred_logits, matched_target_labels)
        
        return loss_ce.float32()
    
    def compute_bbox_loss(self, pred_boxes, target_boxes, matched_indices):
        """计算边界框损失"""
        matched_pred_indices, matched_target_indices = matched_indices
        
        if not matched_pred_indices:
            return jt.array(0.0, dtype=jt.float32), jt.array(0.0, dtype=jt.float32)
        
        # 获取匹配的预测和目标
        matched_pred_boxes = pred_boxes[matched_pred_indices]  # [num_matched, 4]
        matched_target_boxes = target_boxes[matched_target_indices]  # [num_matched, 4]
        
        # L1损失
        loss_bbox = nn.l1_loss(matched_pred_boxes, matched_target_boxes)
        
        # 简化的GIoU损失（使用IoU近似）
        # 计算IoU
        pred_x1 = matched_pred_boxes[:, 0] - matched_pred_boxes[:, 2] / 2
        pred_y1 = matched_pred_boxes[:, 1] - matched_pred_boxes[:, 3] / 2
        pred_x2 = matched_pred_boxes[:, 0] + matched_pred_boxes[:, 2] / 2
        pred_y2 = matched_pred_boxes[:, 1] + matched_pred_boxes[:, 3] / 2
        
        target_x1 = matched_target_boxes[:, 0] - matched_target_boxes[:, 2] / 2
        target_y1 = matched_target_boxes[:, 1] - matched_target_boxes[:, 3] / 2
        target_x2 = matched_target_boxes[:, 0] + matched_target_boxes[:, 2] / 2
        target_y2 = matched_target_boxes[:, 1] + matched_target_boxes[:, 3] / 2
        
        # 交集
        inter_x1 = jt.maximum(pred_x1, target_x1)
        inter_y1 = jt.maximum(pred_y1, target_y1)
        inter_x2 = jt.minimum(pred_x2, target_x2)
        inter_y2 = jt.minimum(pred_y2, target_y2)
        
        inter_area = jt.clamp(inter_x2 - inter_x1, min_v=0) * jt.clamp(inter_y2 - inter_y1, min_v=0)
        
        # 并集
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-8)
        loss_giou = 1 - iou.mean()
        
        return loss_bbox.float32(), loss_giou.float32()
    
    def execute(self, outputs, targets):
        """计算总损失"""
        # 确保输出为float32
        pred_logits = outputs['pred_logits'].float32()  # [batch_size, num_queries, num_classes]
        pred_boxes = outputs['pred_boxes'].float32()    # [batch_size, num_queries, 4]
        
        batch_size = pred_logits.shape[0]
        total_losses = {
            'loss_ce': jt.array(0.0, dtype=jt.float32),
            'loss_bbox': jt.array(0.0, dtype=jt.float32),
            'loss_giou': jt.array(0.0, dtype=jt.float32)
        }
        
        valid_batches = 0
        
        for batch_idx in range(batch_size):
            # 获取当前批次的预测和目标
            batch_pred_logits = pred_logits[batch_idx]  # [num_queries, num_classes]
            batch_pred_boxes = pred_boxes[batch_idx]    # [num_queries, 4]
            
            target = targets[batch_idx]
            target_boxes = target['boxes'].float32()    # [num_targets, 4]
            target_labels = target['labels'].int64()    # [num_targets]
            
            if target_boxes.shape[0] == 0:
                continue
            
            # 简单匹配
            matched_indices = self.simple_matcher(batch_pred_boxes, target_boxes)
            
            if not matched_indices[0]:
                continue
            
            valid_batches += 1
            
            # 计算各项损失
            loss_ce = self.compute_classification_loss(batch_pred_logits, target_labels, matched_indices)
            loss_bbox, loss_giou = self.compute_bbox_loss(batch_pred_boxes, target_boxes, matched_indices)
            
            # 累加损失
            total_losses['loss_ce'] = total_losses['loss_ce'] + loss_ce
            total_losses['loss_bbox'] = total_losses['loss_bbox'] + loss_bbox
            total_losses['loss_giou'] = total_losses['loss_giou'] + loss_giou
        
        # 平均损失
        if valid_batches > 0:
            for key in total_losses:
                total_losses[key] = total_losses[key] / valid_batches
        
        return total_losses


def build_simplified_criterion(num_classes=80):
    """构建简化的损失函数"""
    weight_dict = {
        'loss_ce': 1.0,
        'loss_bbox': 5.0,
        'loss_giou': 2.0
    }
    
    criterion = SimplifiedRTDETRLoss(num_classes=num_classes, weight_dict=weight_dict)
    return criterion
