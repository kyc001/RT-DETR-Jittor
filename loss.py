# loss.py

import jittor as jt
import jittor.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return jt.stack(b, dim=-1)


def generalized_box_iou(boxes1, boxes2):
    # ... (这部分函数无需修改)
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    inter_xmin = jt.maximum(
        boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
    inter_ymin = jt.maximum(
        boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
    inter_xmax = jt.minimum(
        boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
    inter_ymax = jt.minimum(
        boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))
    inter_area = (inter_xmax - inter_xmin).clamp(min_v=0) * \
        (inter_ymax - inter_ymin).clamp(min_v=0)
    boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = boxes1_area.unsqueeze(
        1) + boxes2_area.unsqueeze(0) - inter_area
    iou = inter_area / union_area.clamp(min_v=1e-6)
    enclose_xmin = jt.minimum(
        boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
    enclose_ymin = jt.minimum(
        boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
    enclose_xmax = jt.maximum(
        boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
    enclose_ymax = jt.maximum(
        boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))
    enclose_area = (enclose_xmax - enclose_xmin) * \
        (enclose_ymax - enclose_ymin)
    giou = iou - (enclose_area - union_area) / enclose_area.clamp(min_v=1e-6)
    return giou


class DETRLoss(nn.Module):
    def __init__(self, num_classes, lambda_cls=1.0, lambda_bbox=5.0, lambda_giou=2.0, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cls = lambda_cls
        self.lambda_bbox = lambda_bbox
        self.lambda_giou = lambda_giou

        empty_weight = jt.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef  # 背景类别的权重较低
        self.register_buffer('empty_weight', empty_weight)

        self.cls_loss = nn.CrossEntropyLoss(
            weight=self.empty_weight, reduction='mean')

    @jt.no_grad()
    def _get_src_permutation_idx(self, indices):
        # 构造用于 gather 的 batch-wise 源索引
        batch_idx = jt.concat([jt.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        src_idx = jt.concat([src for (src, _) in indices])
        return (batch_idx, src_idx)

    @jt.no_grad()
    def _get_tgt_permutation_idx(self, indices):
        # 构造用于 gather 的 batch-wise 目标索引
        batch_idx = jt.concat([jt.full_like(tgt, i)
                              for i, (_, tgt) in enumerate(indices)])
        tgt_idx = jt.concat([tgt for (_, tgt) in indices])
        return (batch_idx, tgt_idx)

    @jt.no_grad()
    def hungarian_match(self, pred_logits, pred_boxes, targets):
        # pred_logits: (B, num_queries, C)
        # pred_boxes: (B, num_queries, 4)
        B, N, C = pred_logits.shape

        # 将 batch 内所有预测和目标展平，以便计算一个大的代价矩阵
        out_prob = nn.softmax(pred_logits.flatten(0, 1), dim=-1)  # (B*N, C)
        out_bbox = pred_boxes.flatten(0, 1)  # (B*N, 4)

        tgt_ids = jt.concat([v["labels"] for v in targets])
        tgt_bbox = jt.concat([v["boxes"] for v in targets])

        # 分类代价
        cost_class = -out_prob[:, tgt_ids]

        # 回归代价 (L1)
        cost_bbox = jt.abs(out_bbox.unsqueeze(
            1) - tgt_bbox.unsqueeze(0)).sum(dim=-1)

        # GIoU 代价 (需要将 cxcywh 转为 xyxy)
        cost_giou = - \
            generalized_box_iou(box_cxcywh_to_xyxy(
                out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # 最终代价矩阵
        C = self.lambda_bbox * cost_bbox + self.lambda_cls * \
            cost_class + self.lambda_giou * cost_giou
        C = C.view(B, N, -1)

        # 按每张图片的目标数量切分代价矩阵，并分别进行匹配
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i].numpy())
                   for i, c in enumerate(C.split(sizes, -1))]

        return [(jt.array(i, dtype='int64'), jt.array(j, dtype='int64')) for i, j in indices]

    def _get_loss(self, loss_name, outputs, targets, indices, num_boxes):
        if loss_name == 'labels':
            # 分类损失
            src_logits = outputs['pred_logits']
            target_classes_o = jt.concat(
                [t["labels"][J] for t, (_, J) in zip(targets, indices)])
            target_classes = jt.full(
                src_logits.shape[:2], self.num_classes, dtype='int64')
            src_idx = self._get_src_permutation_idx(indices)
            target_classes[src_idx] = target_classes_o
            return self.cls_loss(src_logits.transpose(1, 2), target_classes)

        if loss_name in ['boxes', 'giou']:
            # 回归损失
            idx = self._get_src_permutation_idx(indices)
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = jt.concat([t['boxes'][i]
                                     for t, (_, i) in zip(targets, indices)])

            if num_boxes == 0:
                return jt.zeros(1)

            if loss_name == 'boxes':
                return (jt.abs(src_boxes - target_boxes)).sum() / num_boxes
            else:  # giou
                giou = generalized_box_iou(box_cxcywh_to_xyxy(
                    src_boxes), box_cxcywh_to_xyxy(target_boxes)).diag()
                return (1 - giou).sum() / num_boxes

    def execute(self, pred_logits, pred_boxes, targets):
        # 注意：这里的输入是单层的预测，为了简化，我们暂时不计算辅助损失
        # 完整的实现会在这里加一个 for 循环，遍历所有解码器层的输出

        outputs = {'pred_logits': pred_logits, 'pred_boxes': pred_boxes}

        # 1. 匈牙利匹配
        indices = self.hungarian_match(pred_logits, pred_boxes, targets)

        # 2. 计算匹配到的物体总数，用于归一化
        num_boxes = sum(len(t["labels"]) for t in targets)

        # 3. 计算各项损失
        loss_labels = self._get_loss(
            'labels', outputs, targets, indices, num_boxes)
        loss_boxes = self._get_loss(
            'boxes', outputs, targets, indices, num_boxes)
        loss_giou = self._get_loss(
            'giou', outputs, targets, indices, num_boxes)

        # 4. 加权求和
        total_loss = self.lambda_cls * loss_labels + \
            self.lambda_bbox * loss_boxes + self.lambda_giou * loss_giou

        # 返回一个字典，便于记录详细日志
        return {
            'total_loss': total_loss,
            'loss_cls': loss_labels,
            'loss_bbox': loss_boxes,
            'loss_giou': loss_giou
        }
