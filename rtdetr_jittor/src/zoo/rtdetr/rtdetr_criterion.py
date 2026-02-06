"""RT-DETR Criterion (Loss Function)
严格按照PyTorch版本: rtdetr_pytorch/src/zoo/rtdetr/rtdetr_criterion.py
"""

import jittor as jt
import jittor.nn as nn
from scipy.optimize import linear_sum_assignment
import numpy as np

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou


__all__ = ['SetCriterion', 'HungarianMatcher', 'build_criterion']


def binary_cross_entropy_with_logits(input, target, weight=None, reduction='mean'):
    """Jittor兼容的BCE with logits"""
    max_val = jt.clamp(-input, min_v=0)
    loss = input - input * target + max_val + jt.log(jt.exp(-max_val) + jt.exp(-input - max_val))

    if weight is not None:
        loss = loss * weight

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        return loss


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='none'):
    """Sigmoid focal loss - 用于不平衡分类"""
    p = jt.sigmoid(inputs)
    ce_loss = binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * ce_loss * ((1 - p_t) ** gamma)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


class HungarianMatcher(nn.Module):
    """Hungarian Matcher for RT-DETR - 严格按照PyTorch版本"""

    def __init__(self, weight_dict=None, use_focal_loss=True, alpha=0.25, gamma=2.0):
        super().__init__()
        # 支持两种初始化方式
        if weight_dict is not None:
            self.cost_class = weight_dict.get('cost_class', 2)
            self.cost_bbox = weight_dict.get('cost_bbox', 5)
            self.cost_giou = weight_dict.get('cost_giou', 2)
        else:
            self.cost_class = 2
            self.cost_bbox = 5
            self.cost_giou = 2

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

    def execute(self, outputs, targets):
        with jt.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # 展平
            out_prob = outputs["pred_logits"].flatten(0, 1)
            out_bbox = outputs["pred_boxes"].flatten(0, 1)

            # 收集目标
            tgt_ids = jt.concat([v["labels"].int64() for v in targets])
            tgt_bbox = jt.concat([v["boxes"].float32() for v in targets])

            # 计算分类成本
            if self.use_focal_loss:
                out_prob = jt.sigmoid(out_prob)
                neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            else:
                out_prob = jt.nn.softmax(out_prob, dim=-1)
                cost_class = -out_prob[:, tgt_ids]

            # 计算L1成本
            cost_bbox = jt.abs(out_bbox[:, None, :] - tgt_bbox[None, :, :]).sum(dim=-1)

            # 计算giou成本
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox),
                box_cxcywh_to_xyxy(tgt_bbox))

            # 最终成本矩阵
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1)

            # 转换为numpy进行匈牙利匹配
            sizes = [len(v["boxes"]) for v in targets]
            C_numpy = C.numpy()

            # 分割并计算匹配
            indices = []
            offset = 0
            for i, size in enumerate(sizes):
                if size > 0:
                    c_i = C_numpy[i, :, offset:offset+size]
                    row_ind, col_ind = linear_sum_assignment(c_i)
                    indices.append((jt.array(row_ind, dtype=jt.int64), jt.array(col_ind, dtype=jt.int64)))
                else:
                    indices.append((jt.zeros(0, dtype=jt.int64), jt.zeros(0, dtype=jt.int64)))
                offset += size

            return indices


class SetCriterion(nn.Module):
    """RT-DETR Loss Criterion - 严格按照PyTorch版本"""

    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, eos_coef=1e-4, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

        # 空白类别权重
        empty_weight = jt.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.empty_weight = empty_weight

        self.alpha = alpha
        self.gamma = gamma

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """分类损失 (NLL)"""
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = jt.concat([t["labels"][J].int64() for t, (_, J) in zip(targets, indices)])
        target_classes = jt.full(src_logits.shape[:2], self.num_classes, dtype=jt.int64)
        target_classes[idx] = target_classes_o

        loss_ce = nn.cross_entropy_loss(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_labels_bce(self, outputs, targets, indices, num_boxes, log=True):
        """BCE分类损失"""
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = jt.concat([t["labels"][J].int64() for t, (_, J) in zip(targets, indices)])
        target_classes = jt.full(src_logits.shape[:2], self.num_classes, dtype=jt.int64)
        target_classes[idx] = target_classes_o

        target = jt.nn.one_hot(target_classes, self.num_classes + 1)[..., :-1].float32()
        loss = binary_cross_entropy_with_logits(src_logits, target, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_bce': loss}

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        """Focal分类损失"""
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = jt.concat([t["labels"][J].int64() for t, (_, J) in zip(targets, indices)])
        target_classes = jt.full(src_logits.shape[:2], self.num_classes, dtype=jt.int64)
        target_classes[idx] = target_classes_o

        target = jt.nn.one_hot(target_classes, self.num_classes + 1)[..., :-1].float32()
        loss = sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        """Varifocal分类损失"""
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = jt.concat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = jt.diag(ious).stop_grad()

        src_logits = outputs['pred_logits']
        target_classes_o = jt.concat([t["labels"][J].int64() for t, (_, J) in zip(targets, indices)])
        target_classes = jt.full(src_logits.shape[:2], self.num_classes, dtype=jt.int64)
        target_classes[idx] = target_classes_o
        target = jt.nn.one_hot(target_classes, self.num_classes + 1)[..., :-1].float32()

        target_score_o = jt.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.float32()
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = jt.sigmoid(src_logits).stop_grad()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """基数误差（仅用于记录）"""
        with jt.no_grad():
            pred_logits = outputs['pred_logits']
            tgt_lengths = jt.array([len(v["labels"]) for v in targets], dtype=jt.float32)
            card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1).float32()
            card_err = jt.abs(card_pred - tgt_lengths).mean()
            return {'cardinality_error': card_err}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """边界框损失"""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = jt.concat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}

        loss_bbox = jt.abs(src_boxes - target_boxes)
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - jt.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = jt.concat([jt.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = jt.concat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = jt.concat([jt.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = jt.concat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'bce': self.loss_labels_bce,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def execute(self, outputs, targets):
        """前向传播"""
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # 匹配
        indices = self.matcher(outputs_without_aux, targets)

        # 计算目标框数量
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = jt.array([num_boxes], dtype=jt.float32)
        num_boxes = jt.clamp(num_boxes, min_v=1).item()

        # 计算所有请求的损失
        losses = {}
        for loss in self.losses:
            kwargs = {}
            if loss == 'labels':
                kwargs = {'log': True}
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # 辅助损失
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # CDN辅助损失 (for rtdetr denoising)
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, 'dn_meta is required for dn_aux_outputs'
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            num_boxes_dn = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes_dn, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """获取CDN匹配索引"""
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = jt.arange(num_gt, dtype=jt.int64)
                gt_idx = jt.concat([gt_idx] * dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((jt.zeros(0, dtype=jt.int64), jt.zeros(0, dtype=jt.int64)))

        return dn_match_indices


def build_criterion(
    num_classes,
    matcher=None,
    weight_dict=None,
    losses=None,
    alpha=0.2,
    gamma=2.0,
    eos_coef=1e-4,
):
    """构建RT-DETR criterion"""
    if matcher is None:
        matcher = HungarianMatcher(
            weight_dict={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
            use_focal_loss=True,
            alpha=0.25,
            gamma=2.0
        )

    if weight_dict is None:
        weight_dict = {
            'loss_vfl': 1,
            'loss_bbox': 5,
            'loss_giou': 2,
        }
        # 添加辅助损失权重
        for i in range(6):
            weight_dict.update({k + f'_aux_{i}': v for k, v in {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2}.items()})

    if losses is None:
        losses = ['vfl', 'boxes']

    criterion = SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        alpha=float(alpha),
        gamma=float(gamma),
        eos_coef=float(eos_coef),
        num_classes=num_classes
    )

    return criterion
