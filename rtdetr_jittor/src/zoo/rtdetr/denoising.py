"""Denoising for RT-DETR
严格按照PyTorch版本: rtdetr_pytorch/src/zoo/rtdetr/denoising.py
"""

import jittor as jt

from .utils import inverse_sigmoid, tile
from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


__all__ = ['get_contrastive_denoising_training_group']


def get_contrastive_denoising_training_group(targets,
                                             num_classes,
                                             num_queries,
                                             class_embed,
                                             num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0,):
    """Contrastive denoising training group - 严格按照PyTorch版本"""
    if num_denoising <= 0:
        return None, None, None, None

    num_gts = [len(t['labels']) for t in targets]

    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group
    # pad gt to max_num of a batch
    bs = len(num_gts)

    input_query_class = jt.full([bs, max_gt_num], num_classes, dtype=jt.int32)
    input_query_bbox = jt.zeros([bs, max_gt_num, 4])
    pad_gt_mask = jt.zeros([bs, max_gt_num], dtype=jt.bool)

    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]['labels'].int32()
            input_query_bbox[i, :num_gt] = targets[i]['boxes'].float32()
            pad_gt_mask[i, :num_gt] = True

    # each group has positive and negative queries.
    input_query_class = tile(input_query_class, [1, 2 * num_group])
    input_query_bbox = tile(input_query_bbox, [1, 2 * num_group, 1])
    pad_gt_mask = tile(pad_gt_mask, [1, 2 * num_group])

    # positive and negative mask
    negative_gt_mask = jt.zeros([bs, max_gt_num * 2, 1])
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = tile(negative_gt_mask, [1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask

    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask.float32()
    dn_positive_idx = jt.nonzero(positive_gt_mask)[:, 1]

    # split dn_positive_idx by num_gts
    split_sizes = [n * num_group for n in num_gts]
    dn_positive_idx_list = []
    start = 0
    for size in split_sizes:
        if size > 0:
            dn_positive_idx_list.append(dn_positive_idx[start:start+size])
        else:
            dn_positive_idx_list.append(jt.zeros(0, dtype=jt.int64))
        start += size

    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)

    if label_noise_ratio > 0:
        mask = jt.rand_like(input_query_class.float32()) < (label_noise_ratio * 0.5)
        # randomly put a new one here
        new_label = jt.randint(0, num_classes, input_query_class.shape, dtype=input_query_class.dtype)
        input_query_class = jt.where(mask & pad_gt_mask, new_label, input_query_class)

    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        diff = tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = jt.randint(0, 2, input_query_bbox.shape).float32() * 2.0 - 1.0
        rand_part = jt.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        rand_part *= rand_sign
        known_bbox = known_bbox + rand_part * diff
        known_bbox = jt.clamp(known_bbox, min_v=0.0, max_v=1.0)
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)

    # Apply class embedding
    input_query_class = class_embed(input_query_class)

    tgt_size = num_denoising + num_queries
    # attn_mask
    attn_mask = jt.full([tgt_size, tgt_size], False, dtype=jt.bool)
    # match query cannot see the reconstruction
    attn_mask[num_denoising:, :num_denoising] = True

    # reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True

    dn_meta = {
        "dn_positive_idx": dn_positive_idx_list,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries]
    }

    return input_query_class, input_query_bbox, attn_mask, dn_meta
