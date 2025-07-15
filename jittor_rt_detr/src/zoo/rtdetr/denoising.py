"""Denoising for RT-DETR
Jittor version aligned with PyTorch implementation
"""

import jittor as jt

from .utils import inverse_sigmoid
from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


def get_contrastive_denoising_training_group(targets,
                                             num_classes,
                                             num_queries,
                                             class_embed,
                                             num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0,):
    """Contrastive denoising training group"""
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
            input_query_class[i, :num_gt] = targets[i]['labels']
            input_query_bbox[i, :num_gt] = targets[i]['boxes']
            pad_gt_mask[i, :num_gt] = 1
    
    # each group has positive and negative queries.
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    
    # positive and negative mask
    negative_gt_mask = jt.zeros([bs, max_gt_num * 2, 1])
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask
    
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = jt.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = dn_positive_idx.long()
    
    if len(dn_positive_idx) > 0:
        gt_class = jt.gather(input_query_class, 1, dn_positive_idx.unsqueeze(0).expand(bs, -1))
        gt_bbox = jt.gather(input_query_bbox, 1, dn_positive_idx.unsqueeze(0).unsqueeze(-1).expand(bs, -1, 4))
        
        # add noise
        gt_class_target = gt_class.clone()
        gt_bbox_target = gt_bbox.clone()
        
        # class noise
        if label_noise_ratio > 0:
            p = jt.rand_like(gt_class_target.float())
            chosen_idx = p < (label_noise_ratio * 0.5)
            # randomly put a new one here
            new_label = jt.randint_like(gt_class_target, 0, num_classes)
            gt_class_target = jt.where(chosen_idx, new_label, gt_class_target)
        
        # box noise
        if box_noise_scale > 0:
            diff = jt.zeros_like(gt_bbox_target)
            diff[:, :, :2] = gt_bbox_target[:, :, 2:] / 2
            diff[:, :, 2:] = gt_bbox_target[:, :, 2:]
            
            gt_bbox_target += jt.multiply(jt.rand_like(gt_bbox_target), diff) * box_noise_scale
            gt_bbox_target = jt.clamp(gt_bbox_target, min_v=0., max_v=1.)
    
    return gt_class_target, gt_bbox_target, dn_positive_idx, num_group


def get_dn_meta(targets, num_denoising, num_classes, device):
    """Get denoising meta information"""
    if num_denoising <= 0:
        return None
    
    num_gts = [len(t['labels']) for t in targets]
    max_gt_num = max(num_gts) if num_gts else 0
    
    if max_gt_num == 0:
        return None
    
    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group
    
    return {
        'num_denoising': num_denoising,
        'num_group': num_group,
        'max_gt_num': max_gt_num,
    }


def dn_post_process(outputs_class, outputs_coord, dn_meta, aux_loss, _set_aux_loss):
    """Post process denoising outputs"""
    if dn_meta and dn_meta['num_group'] > 0:
        num_denoising = dn_meta['num_denoising']
        outputs_class, outputs_class_dn = outputs_class[:, :-num_denoising], outputs_class[:, -num_denoising:]
        outputs_coord, outputs_coord_dn = outputs_coord[:, :-num_denoising], outputs_coord[:, -num_denoising:]
        
        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        if aux_loss:
            out['aux_outputs'] = _set_aux_loss(outputs_class, outputs_coord)
        
        out['dn_aux_outputs'] = [{'pred_logits': outputs_class_dn, 'pred_boxes': outputs_coord_dn}]
        return out
    else:
        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        if aux_loss:
            out['aux_outputs'] = _set_aux_loss(outputs_class, outputs_coord)
        return out


__all__ = [
    'get_contrastive_denoising_training_group',
    'get_dn_meta', 
    'dn_post_process'
]
