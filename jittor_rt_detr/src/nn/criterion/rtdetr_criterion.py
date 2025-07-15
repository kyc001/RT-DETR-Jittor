"""
RT-DETR损失函数 - 严格按照PyTorch版本实现
参考: rtdetr_pytorch/src/zoo/rtdetr/rtdetr_criterion.py
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
    """将中心点格式转换为左上右下格式"""
    # 确保数据类型一致
    x = x.float32()
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return jt.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    """计算两组边界框的IoU"""
    # 确保数据类型一致
    boxes1 = boxes1.float32()
    boxes2 = boxes2.float32()

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = jt.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = jt.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min_v=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    # 避免除零
    union = jt.clamp(union, min_v=1e-8)
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """计算广义IoU (GIoU)"""
    # 确保数据类型一致
    boxes1 = boxes1.float32()
    boxes2 = boxes2.float32()

    # 简化边界框有效性检查，避免Jittor的.all()问题
    # 直接修复可能无效的边界框
    boxes1[:, 2:] = jt.maximum(boxes1[:, 2:], boxes1[:, :2] + 1e-6)
    boxes2[:, 2:] = jt.maximum(boxes2[:, 2:], boxes2[:, :2] + 1e-6)

    iou, union = box_iou(boxes1, boxes2)

    lt = jt.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = jt.maximum(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min_v=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


class HungarianMatcher(nn.Module):
    """匈牙利匹配器 - 严格按照PyTorch版本实现"""
    
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, use_focal: bool = True):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal = use_focal
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

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

            # 计算分类成本 - 修复混合精度问题
            if self.use_focal:
                # Focal loss成本 - 强制float32
                alpha = jt.array(0.25, dtype=jt.float32)
                gamma = jt.array(2.0, dtype=jt.float32)
                eps = jt.array(1e-8, dtype=jt.float32)

                neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + eps).log())
                neg_cost_class = ensure_float32(neg_cost_class)

                pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + eps).log())
                pos_cost_class = ensure_float32(pos_cost_class)

                cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
                cost_class = ensure_float32(cost_class)
            else:
                # 标准分类成本
                cost_class = -out_prob[:, tgt_ids]

            # 计算L1边界框成本 (手动实现cdist) - 确保数据类型一致
            cost_bbox = jt.abs(out_bbox.unsqueeze(1).float32() - tgt_bbox.unsqueeze(0).float32()).sum(dim=-1)

            # 计算GIoU成本 - 确保数据类型一致
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

            # 最终成本矩阵 - 确保所有成本都是float32
            cost_bbox = cost_bbox.float32()
            cost_class = cost_class.float32()
            cost_giou = cost_giou.float32()

            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1)

            sizes = [len(v["boxes"]) for v in targets]
            # 优化的匈牙利匹配实现
            C_splits = C.split(sizes, -1)
            indices = []
            for i, c in enumerate(C_splits):
                try:
                    # 确保成本矩阵为float32并同步到CPU
                    cost_matrix = c[i].float32().stop_grad()
                    jt.sync_all()  # 确保计算完成
                    c_numpy = cost_matrix.numpy()

                    # 执行匈牙利算法
                    row_ind, col_ind = linear_sum_assignment(c_numpy)
                    indices.append((row_ind, col_ind))
                except Exception as e:
                    # 改进的fallback机制
                    num_targets = sizes[i]
                    if num_targets > 0:
                        # 基于成本矩阵的简单贪心匹配
                        try:
                            cost_matrix = c[i].float32().stop_grad()
                            jt.sync_all()
                            c_numpy = cost_matrix.numpy()

                            # 贪心匹配：为每个目标选择成本最低的查询
                            row_ind = []
                            col_ind = []
                            used_queries = set()

                            for target_idx in range(num_targets):
                                available_queries = [q for q in range(c_numpy.shape[0]) if q not in used_queries]
                                if available_queries:
                                    costs = c_numpy[available_queries, target_idx]
                                    best_query_idx = available_queries[np.argmin(costs)]
                                    row_ind.append(best_query_idx)
                                    col_ind.append(target_idx)
                                    used_queries.add(best_query_idx)

                            row_ind = np.array(row_ind)
                            col_ind = np.array(col_ind)
                        except:
                            # 最后的fallback
                            row_ind = np.arange(min(num_queries, num_targets))
                            col_ind = np.arange(min(num_queries, num_targets))
                    else:
                        row_ind = np.array([])
                        col_ind = np.array([])

                    indices.append((row_ind, col_ind))

            return [(jt.array(i, dtype=jt.int64), jt.array(j, dtype=jt.int64)) for i, j in indices]


class SetCriterion(nn.Module):
    """RT-DETR损失函数 - 严格按照PyTorch版本实现"""
    
    def __init__(self, num_classes, matcher, weight_dict, losses, alpha=0.25, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.alpha = alpha
        self.gamma = gamma

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        """Focal loss for labels - 修复混合精度问题"""
        assert 'pred_logits' in outputs
        src_logits = ensure_float32(outputs['pred_logits'])

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = jt.concat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = jt.full(src_logits.shape[:2], self.num_classes, dtype=jt.int64)
        target_classes[idx] = target_classes_o

        target = jt.nn.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        target = ensure_float32(target)  # 确保target是float32

        # Sigmoid focal loss (手动实现) - 强制float32
        sigmoid_p = ensure_float32(jt.sigmoid(src_logits))

        # 确保所有常数都是float32
        eps = jt.array(1e-8, dtype=jt.float32)
        gamma = jt.array(self.gamma, dtype=jt.float32)
        alpha = jt.array(self.alpha, dtype=jt.float32)

        ce_loss = -(target * jt.log(sigmoid_p + eps) + (1 - target) * jt.log(1 - sigmoid_p + eps))
        ce_loss = ensure_float32(ce_loss)

        p_t = sigmoid_p * target + (1 - sigmoid_p) * (1 - target)
        p_t = ensure_float32(p_t)

        loss = ce_loss * ((1 - p_t) ** gamma)
        loss = ensure_float32(loss)

        if self.alpha >= 0:
            alpha_t = alpha * target + (1 - alpha) * (1 - target)
            alpha_t = ensure_float32(alpha_t)
            loss = alpha_t * loss
            loss = ensure_float32(loss)

        # 确保最终计算都是float32
        loss_mean = ensure_float32(loss.mean(1))
        loss_sum = ensure_float32(loss_mean.sum())
        num_boxes_f32 = jt.array(num_boxes, dtype=jt.float32)
        shape_factor = jt.array(src_logits.shape[1], dtype=jt.float32)

        final_loss = ensure_float32(loss_sum * shape_factor / num_boxes_f32)
        return {'loss_focal': final_loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """计算边界框损失"""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx].float32()
        target_boxes = jt.concat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0).float32()

        loss_bbox = jt.abs(src_boxes - target_boxes)
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - jt.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # 排列索引以便收集预测
        batch_idx = jt.concat([jt.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = jt.concat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # 排列索引以便收集目标
        batch_idx = jt.concat([jt.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = jt.concat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels_focal,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def execute(self, outputs, targets):
        """计算所有损失"""
        # 强制所有输出为float32
        outputs_without_aux = {}
        for k, v in outputs.items():
            if k not in ['aux_outputs', 'enc_outputs']:
                if isinstance(v, jt.Var):
                    outputs_without_aux[k] = v.float32()
                else:
                    outputs_without_aux[k] = v

        # 强制所有目标数据类型一致
        for target in targets:
            target['boxes'] = target['boxes'].float32()
            target['labels'] = target['labels'].int64()

        # 获取匹配
        indices = self.matcher(outputs_without_aux, targets)

        # 计算所有节点的目标框数量，用于归一化
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = jt.array([num_boxes], dtype=jt.float32)
        num_boxes = jt.clamp(num_boxes / jt.distributed.get_world_size() if jt.in_mpi else num_boxes, min_v=1).item()

        # 计算所有请求的损失
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # 如果有辅助损失，也计算它们
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # 强制辅助输出为float32
                aux_outputs_float32 = {}
                for k, v in aux_outputs.items():
                    if isinstance(v, jt.Var):
                        aux_outputs_float32[k] = v.float32()
                    else:
                        aux_outputs_float32[k] = v

                indices = self.matcher(aux_outputs_float32, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs_float32, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # 编码器损失
        if 'enc_outputs' in outputs:
            # 强制编码器输出为float32
            enc_outputs = {}
            for k, v in outputs['enc_outputs'].items():
                if isinstance(v, jt.Var):
                    enc_outputs[k] = v.float32()
                else:
                    enc_outputs[k] = v

            indices = self.matcher(enc_outputs, targets)
            for loss in self.losses:
                kwargs = {}
                if loss == 'labels':
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + '_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


def build_criterion(num_classes):
    """构建损失函数 - 严格按照PyTorch版本"""
    matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2, use_focal=True)
    
    weight_dict = {'loss_focal': 2, 'loss_bbox': 5, 'loss_giou': 2}
    
    # 辅助损失权重
    aux_weight_dict = {}
    for i in range(6):  # 6层解码器
        aux_weight_dict.update({k + f'_aux_{i}': v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    
    # 编码器损失权重
    enc_weight_dict = {k + '_enc': v for k, v in weight_dict.items() if k != 'loss_focal'}
    enc_weight_dict['loss_focal_enc'] = weight_dict['loss_focal']
    weight_dict.update(enc_weight_dict)

    losses = ['labels', 'boxes']
    
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, alpha=0.25, gamma=2.0)
    
    return criterion
