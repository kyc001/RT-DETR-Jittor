"""RT-DETR Post-processor
Aligned with PyTorch version implementation
"""

import jittor as jt
import jittor.nn as nn
import numpy as np

from .box_ops import box_cxcywh_to_xyxy

def ensure_float32(x):
    """确保张量为float32类型"""
    if isinstance(x, jt.Var):
        return x.float32()
    else:
        return x

class RTDETRPostProcessor(nn.Module):
    """Post-processor for RT-DETR"""
    
    def __init__(self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False):
        """
        Args:
            num_classes: Number of object classes
            use_focal_loss: Whether focal loss was used during training
            num_top_queries: Number of top queries to keep
            remap_mscoco_category: Whether to remap MSCOCO category IDs
        """
        super().__init__()
        self.num_classes = num_classes
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.remap_mscoco_category = remap_mscoco_category
        
        # MSCOCO category remapping if needed
        if remap_mscoco_category:
            self.coco_category_mapping = self._get_coco_category_mapping()

    def _get_coco_category_mapping(self):
        """Get MSCOCO category ID mapping"""
        # MSCOCO has 80 classes but category IDs are not continuous
        coco_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        return {i: coco_id for i, coco_id in enumerate(coco_ids)}

    def execute(self, outputs, target_sizes=None):
        """Post-process the outputs
        
        Args:
            outputs: Dict containing 'pred_logits' and 'pred_boxes'
            target_sizes: Tensor of shape [batch_size, 2] containing target image sizes (height, width)
            
        Returns:
            List of dicts, each containing 'scores', 'labels', and 'boxes' for one image
        """
        pred_logits = ensure_float32(outputs['pred_logits'])
        pred_boxes = ensure_float32(outputs['pred_boxes'])
        
        batch_size = pred_logits.shape[0]
        
        # Get class probabilities
        if self.use_focal_loss:
            # For focal loss, use sigmoid
            prob = jt.sigmoid(pred_logits)
            # Get max probability and corresponding class for each query
            scores = prob.max(dim=-1)[0]
            labels = prob.argmax(dim=-1)
        else:
            # For standard cross-entropy, use softmax
            prob = jt.nn.softmax(pred_logits, dim=-1)
            # Exclude background class (last class)
            scores = prob[..., :-1].max(dim=-1)[0]
            labels = prob[..., :-1].argmax(dim=-1)
        
        # Convert boxes from cxcywh to xyxy format
        boxes = box_cxcywh_to_xyxy(pred_boxes)
        
        # Scale boxes to target image size if provided
        if target_sizes is not None:
            if len(target_sizes) != batch_size:
                raise ValueError(f"Expected target_sizes to have length {batch_size}, got {len(target_sizes)}")
            
            # target_sizes: [batch_size, 2] -> [height, width]
            img_h, img_w = target_sizes.unbind(1)
            scale_fct = jt.stack([img_w, img_h, img_w, img_h], dim=1)
            boxes = boxes * scale_fct[:, None, :]
        
        results = []
        for i in range(batch_size):
            # Get top-k queries based on scores
            if self.num_top_queries < pred_logits.shape[1]:
                top_scores, top_indices = scores[i].topk(self.num_top_queries)
                top_labels = labels[i][top_indices]
                top_boxes = boxes[i][top_indices]
            else:
                top_scores = scores[i]
                top_labels = labels[i]
                top_boxes = boxes[i]
            
            # Remap category IDs if needed
            if self.remap_mscoco_category:
                remapped_labels = []
                for label in top_labels:
                    label_id = int(label.item())
                    if label_id in self.coco_category_mapping:
                        remapped_labels.append(self.coco_category_mapping[label_id])
                    else:
                        remapped_labels.append(label_id)
                top_labels = jt.array(remapped_labels, dtype=jt.int64)
            
            results.append({
                'scores': ensure_float32(top_scores),
                'labels': top_labels,
                'boxes': ensure_float32(top_boxes)
            })
        
        return results

    def execute_single(self, outputs, target_size=None, score_threshold=0.3):
        """Post-process outputs for a single image with score filtering
        
        Args:
            outputs: Dict containing 'pred_logits' and 'pred_boxes' for single image
            target_size: Tuple (height, width) for target image size
            score_threshold: Minimum score threshold for detections
            
        Returns:
            Dict containing 'scores', 'labels', and 'boxes'
        """
        pred_logits = ensure_float32(outputs['pred_logits'])
        pred_boxes = ensure_float32(outputs['pred_boxes'])
        
        # Remove batch dimension if present
        if pred_logits.dim() == 3:
            pred_logits = pred_logits[0]
            pred_boxes = pred_boxes[0]
        
        # Get class probabilities
        if self.use_focal_loss:
            prob = jt.sigmoid(pred_logits)
            scores = prob.max(dim=-1)[0]
            labels = prob.argmax(dim=-1)
        else:
            prob = jt.nn.softmax(pred_logits, dim=-1)
            scores = prob[..., :-1].max(dim=-1)[0]
            labels = prob[..., :-1].argmax(dim=-1)
        
        # Filter by score threshold
        keep = scores > score_threshold
        scores = scores[keep]
        labels = labels[keep]
        boxes = pred_boxes[keep]
        
        # Convert boxes to xyxy format
        boxes = box_cxcywh_to_xyxy(boxes)
        
        # Scale boxes if target size provided
        if target_size is not None:
            img_h, img_w = target_size
            scale_fct = jt.array([img_w, img_h, img_w, img_h], dtype=jt.float32)
            boxes = boxes * scale_fct
        
        # Remap category IDs if needed
        if self.remap_mscoco_category:
            remapped_labels = []
            for label in labels:
                label_id = int(label.item())
                if label_id in self.coco_category_mapping:
                    remapped_labels.append(self.coco_category_mapping[label_id])
                else:
                    remapped_labels.append(label_id)
            labels = jt.array(remapped_labels, dtype=jt.int64)
        
        return {
            'scores': ensure_float32(scores),
            'labels': labels,
            'boxes': ensure_float32(boxes)
        }

def build_postprocessor(num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False):
    """Build RT-DETR post-processor"""
    return RTDETRPostProcessor(
        num_classes=num_classes,
        use_focal_loss=use_focal_loss,
        num_top_queries=num_top_queries,
        remap_mscoco_category=remap_mscoco_category
    )

__all__ = ['RTDETRPostProcessor', 'build_postprocessor']
