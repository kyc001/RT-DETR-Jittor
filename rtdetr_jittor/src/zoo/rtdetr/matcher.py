"""Hungarian Matcher for RT-DETR
Aligned with PyTorch version implementation
"""

import jittor as jt
import jittor.nn as nn
from scipy.optimize import linear_sum_assignment
import numpy as np

from .box_ops import generalized_box_iou

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

class HungarianMatcher(nn.Module):
    """Hungarian Matcher for RT-DETR
    
    This class computes an assignment between the targets and the predictions of the network.
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2, use_focal=True, alpha=0.25, gamma=2.0):
        """Creates the matcher
        
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
            use_focal: Whether to use focal loss for classification cost
            alpha: Alpha parameter for focal loss
            gamma: Gamma parameter for focal loss
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal = use_focal
        self.alpha = alpha
        self.gamma = gamma
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def execute(self, outputs, targets):
        """Performs the matching
        
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
                 
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                 
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with jt.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            pred_logits = ensure_float32(outputs["pred_logits"])
            pred_boxes = ensure_float32(outputs["pred_boxes"])
            
            if self.use_focal:
                out_prob = ensure_float32(jt.sigmoid(pred_logits.flatten(0, 1)))  # [batch_size * num_queries, num_classes]
            else:
                out_prob = ensure_float32(pred_logits.flatten(0, 1).softmax(-1))  # [batch_size * num_queries, num_classes]

            out_bbox = ensure_float32(pred_boxes.flatten(0, 1))  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = ensure_int64(jt.concat([ensure_int64(v["labels"]) for v in targets]))
            tgt_bbox = ensure_float32(jt.concat([ensure_float32(v["boxes"]) for v in targets]))

            # Compute the classification cost
            if self.use_focal:
                # Focal loss classification cost
                neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            else:
                # Standard classification cost
                cost_class = -out_prob[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = jt.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost between boxes
            cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(jt.array(i, dtype=jt.int64), jt.array(j, dtype=jt.int64)) for i, j in indices]

def build_matcher(cost_class=1, cost_bbox=5, cost_giou=2, use_focal=True, alpha=0.25, gamma=2.0):
    """Build Hungarian matcher"""
    return HungarianMatcher(
        cost_class=cost_class,
        cost_bbox=cost_bbox, 
        cost_giou=cost_giou,
        use_focal=use_focal,
        alpha=alpha,
        gamma=gamma
    )

__all__ = ['HungarianMatcher', 'build_matcher']
