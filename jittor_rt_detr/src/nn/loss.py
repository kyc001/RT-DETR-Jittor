# loss.py (最终修复版)

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
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # 检查边界框格式是否正确，如果不正确则修复
    # 确保 x1 >= x0, y1 >= y0
    boxes1 = jt.stack([
        jt.minimum(boxes1[:, 0], boxes1[:, 2]),  # x0
        jt.minimum(boxes1[:, 1], boxes1[:, 3]),  # y0
        jt.maximum(boxes1[:, 0], boxes1[:, 2]),  # x1
        jt.maximum(boxes1[:, 1], boxes1[:, 3])   # y1
    ], dim=-1)

    boxes2 = jt.stack([
        jt.minimum(boxes2[:, 0], boxes2[:, 2]),  # x0
        jt.minimum(boxes2[:, 1], boxes2[:, 3]),  # y0
        jt.maximum(boxes2[:, 0], boxes2[:, 2]),  # x1
        jt.maximum(boxes2[:, 1], boxes2[:, 3])   # y1
    ], dim=-1)

    # 计算交集
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

    # 计算各自的面积
    boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # 计算并集
    union_area = boxes1_area.unsqueeze(1) + boxes2_area.unsqueeze(0) - inter_area

    # 计算IoU
    iou = inter_area / union_area.clamp(min_v=1e-6)

    # 计算最小外接矩形
    enclose_xmin = jt.minimum(
        boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
    enclose_ymin = jt.minimum(
        boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
    enclose_xmax = jt.maximum(
        boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
    enclose_ymax = jt.maximum(
        boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))
    enclose_area = (enclose_xmax - enclose_xmin) * (enclose_ymax - enclose_ymin)

    # 计算GIoU
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

        self.cross_entropy = nn.CrossEntropyLoss(weight=self.empty_weight)

    @jt.no_grad()
    def _get_src_permutation_idx(self, indices):
        batch_idx = jt.concat([jt.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        src_idx = jt.concat([src for (src, _) in indices])
        return (batch_idx, src_idx)

    @jt.no_grad()
    def hungarian_match(self, pred_logits, pred_boxes, targets):
        B, N, C = pred_logits.shape
        out_prob = nn.softmax(pred_logits.flatten(0, 1), dim=-1)
        out_bbox = pred_boxes.flatten(0, 1)  # (B*N, 4)

        # 收集所有目标的标签和边界框
        tgt_ids = jt.concat([v["labels"] for v in targets])  # (total_targets,)
        tgt_bbox = jt.concat([v["boxes"] for v in targets])  # (total_targets, 4)

        # 确保tgt_bbox是2D的
        if tgt_bbox.ndim == 1:
            tgt_bbox = tgt_bbox.unsqueeze(0)

        # 计算分类成本
        cost_class = -out_prob[:, tgt_ids]  # (B*N, total_targets)

        # 计算L1边界框成本 - 参考PyTorch版本
        # out_bbox: (B*N, 4), tgt_bbox: (total_targets, 4)
        # 使用广播计算L1距离
        cost_bbox = jt.abs(out_bbox.unsqueeze(1) - tgt_bbox.unsqueeze(0)).sum(dim=-1)  # (B*N, total_targets)

        # 计算GIoU成本
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox)
        )  # (B*N, total_targets)

        # 组合所有成本
        C = self.lambda_bbox * cost_bbox + self.lambda_cls * cost_class + self.lambda_giou * cost_giou
        C = C.view(B, N, -1)  # (B, N, total_targets)

        sizes = [len(v["boxes"]) for v in targets]

        # 安全地转换tensor到numpy进行匈牙利算法
        C_splits = C.split(sizes, -1)
        indices = []
        for i, c in enumerate(C_splits):
            try:
                # 使用stop_grad()确保tensor可以转换为numpy
                c_numpy = c[i].stop_grad().numpy()
                row_ind, col_ind = linear_sum_assignment(c_numpy)
                indices.append((row_ind, col_ind))
            except Exception as e:
                # 如果转换失败，使用简单的贪心匹配作为fallback
                print(f"Warning: Hungarian algorithm failed, using greedy matching: {e}")
                c_tensor = c[i]
                _, min_indices = c_tensor.min(dim=0)
                row_ind = np.arange(len(min_indices))
                col_ind = min_indices.stop_grad().numpy()
                indices.append((row_ind, col_ind))

        return [(jt.array(i, dtype='int64'), jt.array(j, dtype='int64')) for i, j in indices]

    def loss_labels(self, pred_logits, targets, indices):
        src_idx = self._get_src_permutation_idx(indices)
        target_classes_o = jt.concat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)])

        # 修复：使用实际的类别数作为背景类别索引
        actual_num_classes = pred_logits.shape[-1]
        background_class_idx = actual_num_classes - 1  # 最后一个类别作为背景
        target_classes = jt.full(
            pred_logits.shape[:2], background_class_idx, dtype='int64')

        target_classes[src_idx] = target_classes_o

        # 修复：使用实际的类别数进行reshape
        loss_ce = self.cross_entropy(pred_logits.reshape(-1, actual_num_classes),
                                     target_classes.reshape(-1))
        return loss_ce

    def loss_boxes(self, pred_boxes, targets, indices, num_boxes):
        if num_boxes == 0:
            return jt.zeros(1), jt.zeros(1)
        src_idx = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[src_idx]
        target_boxes = jt.concat(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices)])
        loss_bbox = (jt.abs(src_boxes - target_boxes)).sum() / num_boxes

        # ## <<< 关键修复：修正函数名的拼写错误 >>>
        giou = generalized_box_iou(box_cxcywh_to_xyxy(
            src_boxes), box_cxcywh_to_xyxy(target_boxes)).diag()
        loss_giou = (1 - giou).sum() / num_boxes

        return loss_bbox, loss_giou

    def execute(self, all_pred_logits, all_pred_boxes, targets, enc_logits=None, enc_boxes=None):
        final_pred_logits = all_pred_logits[-1]
        final_pred_boxes = all_pred_boxes[-1]
        indices = self.hungarian_match(
            final_pred_logits, final_pred_boxes, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        if jt.world_size > 1:
            jt.sync_all(num_boxes)
        num_boxes = max(num_boxes, 1)

        loss_dict = {}
        total_loss = jt.zeros(1)

        # 解码器各层损失
        for i in range(all_pred_logits.shape[0]):
            pred_logits = all_pred_logits[i]
            pred_boxes = all_pred_boxes[i]
            l_cls = self.loss_labels(pred_logits, targets, indices)
            l_bbox, l_giou = self.loss_boxes(
                pred_boxes, targets, indices, num_boxes)
            total_loss += self.lambda_cls * l_cls + \
                self.lambda_bbox * l_bbox + self.lambda_giou * l_giou
            suffix = '' if i == all_pred_logits.shape[0] - 1 else f'_aux_{i}'
            loss_dict[f'loss_cls{suffix}'] = l_cls
            loss_dict[f'loss_bbox{suffix}'] = l_bbox
            loss_dict[f'loss_giou{suffix}'] = l_giou

        # 编码器辅助损失
        if enc_logits is not None and enc_boxes is not None:
            # 对编码器输出进行top-k选择，与解码器查询数量匹配
            B, N, C = enc_logits.shape
            scores = enc_logits.max(dim=-1)
            _, topk_indices = jt.topk(scores, all_pred_logits.shape[1], dim=1)

            # 选择top-k的编码器预测
            batch_indices = jt.arange(B).unsqueeze(1)
            enc_topk_logits = enc_logits[batch_indices, topk_indices]
            enc_topk_boxes = enc_boxes[batch_indices, topk_indices]

            # 计算编码器辅助损失
            enc_indices = self.hungarian_match(enc_topk_logits, enc_topk_boxes, targets)
            enc_l_cls = self.loss_labels(enc_topk_logits, targets, enc_indices)
            enc_l_bbox, enc_l_giou = self.loss_boxes(
                enc_topk_boxes, targets, enc_indices, num_boxes)

            # 编码器损失权重通常较小
            enc_loss_weight = 0.1
            total_loss += enc_loss_weight * (self.lambda_cls * enc_l_cls +
                                           self.lambda_bbox * enc_l_bbox +
                                           self.lambda_giou * enc_l_giou)

            loss_dict['loss_cls_enc'] = enc_l_cls
            loss_dict['loss_bbox_enc'] = enc_l_bbox
            loss_dict['loss_giou_enc'] = enc_l_giou

        loss_dict['total_loss'] = total_loss
        return loss_dict
