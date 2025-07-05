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
    def __init__(self, num_classes=80, lambda_cls=1.0, lambda_bbox=5.0, lambda_giou=2.0, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cls = lambda_cls
        self.lambda_bbox = lambda_bbox
        self.lambda_giou = lambda_giou

        empty_weight = jt.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.cls_loss = nn.CrossEntropyLoss(weight=self.empty_weight)

    @jt.no_grad()
    def hungarian_match(self, pred_logits, pred_boxes, tgt_labels, tgt_boxes):
        num_queries = pred_logits.shape[0]
        num_targets = tgt_labels.shape[0]

        out_prob = nn.softmax(pred_logits, dim=-1)
        cost_class = -out_prob[:, tgt_labels]

        # ## <<< 关键修正：手动计算 Pairwise L1 Distance >>>
        # pred_boxes: (N, 4) -> (N, 1, 4)
        # tgt_boxes: (M, 4) -> (1, M, 4)
        # 广播计算后得到 (N, M, 4)，然后在最后一个维度求和得到 (N, M) 的L1距离矩阵
        cost_bbox = jt.abs(pred_boxes.unsqueeze(
            1) - tgt_boxes.unsqueeze(0)).sum(dim=-1)

        cost_giou = - \
            generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes), tgt_boxes)

        C = self.lambda_cls * cost_class + self.lambda_bbox * \
            cost_bbox + self.lambda_giou * cost_giou

        indices = linear_sum_assignment(C.numpy())
        return [(jt.array(i, dtype='int64'), jt.array(j, dtype='int64')) for i, j in zip(*indices)]

    def execute(self, pred_logits, pred_boxes, targets):
        # 简化假设：此处的实现是针对 batch_size=1 的。
        # 真实的多batch处理需要在此处添加一个 for 循环，但这会使逻辑变得复杂。
        # 当前我们先确保单batch能跑通。
        pred_logits_b = pred_logits[0]
        pred_boxes_b = pred_boxes[0]
        tgt_labels_b = targets[0]['labels']
        tgt_boxes_b = targets[0]['boxes']

        if tgt_labels_b.shape[0] == 0:
            loss_cls = self.cls_loss(pred_logits_b, jt.full(
                (pred_logits_b.shape[0],), self.num_classes, dtype='int64'))
            return self.lambda_cls * loss_cls

        indices = self.hungarian_match(
            pred_logits_b, pred_boxes_b, tgt_labels_b, tgt_boxes_b)

        target_classes = jt.full(
            (pred_logits_b.shape[0],), self.num_classes, dtype='int64')
        if indices:
            idx_pred, idx_tgt = indices[0]
            target_classes[idx_pred] = tgt_labels_b[idx_tgt]

        loss_cls = self.cls_loss(pred_logits_b, target_classes)

        loss_bbox = jt.zeros(1)
        loss_giou = jt.zeros(1)
        if indices:
            idx_pred, idx_tgt = indices[0]

            matched_pred_boxes = pred_boxes_b[idx_pred]
            matched_tgt_boxes = tgt_boxes_b[idx_tgt]
            num_boxes = matched_tgt_boxes.shape[0]

            loss_bbox = jt.abs(matched_pred_boxes -
                               matched_tgt_boxes).sum() / num_boxes

            pred_boxes_xyxy = box_cxcywh_to_xyxy(matched_pred_boxes)
            giou = generalized_box_iou(
                pred_boxes_xyxy, matched_tgt_boxes).diag()
            loss_giou = (1 - giou).sum() / num_boxes

        total_loss = self.lambda_cls * loss_cls + self.lambda_bbox * \
            loss_bbox + self.lambda_giou * loss_giou
        return total_loss
