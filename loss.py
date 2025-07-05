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
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.cls_loss = nn.CrossEntropyLoss(weight=self.empty_weight)

    @jt.no_grad()
    def _get_src_permutation_idx(self, indices):
        batch_idx = jt.concat([jt.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        src_idx = jt.concat([src for (src, _) in indices])
        return (batch_idx, src_idx)

    @jt.no_grad()
    def _get_tgt_permutation_idx(self, indices):
        batch_idx = jt.concat([jt.full_like(tgt, i)
                              for i, (_, tgt) in enumerate(indices)])
        tgt_idx = jt.concat([tgt for (_, tgt) in indices])
        return (batch_idx, tgt_idx)

    @jt.no_grad()
    def hungarian_match(self, pred_logits, pred_boxes, targets):
        B, N, C = pred_logits.shape
        out_prob = nn.softmax(pred_logits.flatten(0, 1), dim=-1)
        out_bbox = pred_boxes.flatten(0, 1)

        tgt_ids = jt.concat([v["labels"] for v in targets])
        tgt_bbox = jt.concat([v["boxes"] for v in targets])

        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = jt.abs(out_bbox.unsqueeze(
            1) - tgt_bbox.unsqueeze(0)).sum(dim=-1)
        cost_giou = - \
            generalized_box_iou(box_cxcywh_to_xyxy(
                out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        C = self.lambda_bbox * cost_bbox + self.lambda_cls * \
            cost_class + self.lambda_giou * cost_giou
        C = C.view(B, N, -1)

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i].numpy())
                   for i, c in enumerate(C.split(sizes, -1))]

        return [(jt.array(i, dtype='int64'), jt.array(j, dtype='int64')) for i, j in indices]

    def _get_loss(self, loss_name, outputs, targets, indices, num_boxes):
        if loss_name == 'labels':
            src_logits = outputs['pred_logits']  # shape: (B, N, C)
            target_classes_o = jt.concat(
                [t["labels"][J] for t, (_, J) in zip(targets, indices)])

            target_classes = jt.full(
                src_logits.shape[:2], self.num_classes, dtype='int64')  # shape: (B, N)
            src_idx = self._get_src_permutation_idx(indices)
            target_classes[src_idx] = target_classes_o

            # ## <<< 关键修正：将 logits 和 target 展平为 2D 和 1D >>>
            # 将 (B, N, C) -> (B*N, C)
            src_logits_flat = src_logits.reshape(-1, src_logits.shape[-1])
            # 将 (B, N) -> (B*N,)
            target_classes_flat = target_classes.reshape(-1)
            return self.cls_loss(src_logits_flat, target_classes_flat)

        if loss_name in ['boxes', 'giou']:
            idx = self._get_src_permutation_idx(indices)
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = jt.concat([t['boxes'][i]
                                     for t, (_, i) in zip(targets, indices)])

            if num_boxes == 0:
                return jt.zeros(1)

            if loss_name == 'boxes':
                return (jt.abs(src_boxes - target_boxes)).sum() / num_boxes
            else:
                giou = generalized_box_iou(box_cxcywh_to_xyxy(
                    src_boxes), box_cxcywh_to_xyxy(target_boxes)).diag()
                return (1 - giou).sum() / num_boxes

    def execute(self, pred_logits, pred_boxes, targets):
        outputs = {'pred_logits': pred_logits, 'pred_boxes': pred_boxes}

        indices = self.hungarian_match(pred_logits, pred_boxes, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)

        loss_labels = self._get_loss(
            'labels', outputs, targets, indices, num_boxes)
        loss_boxes = self._get_loss(
            'boxes', outputs, targets, indices, num_boxes)
        loss_giou = self._get_loss(
            'giou', outputs, targets, indices, num_boxes)

        total_loss = self.lambda_cls * loss_labels + \
            self.lambda_bbox * loss_boxes + self.lambda_giou * loss_giou

        return {
            'total_loss': total_loss,
            'loss_cls': loss_labels,
            'loss_bbox': loss_boxes,
            'loss_giou': loss_giou
        }
