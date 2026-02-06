"""RT-DETR postprocessor aligned with PyTorch behavior."""

from __future__ import annotations

from typing import Any, Dict, List

import jittor as jt
import jittor.nn as nn

from .box_ops import box_cxcywh_to_xyxy

__all__ = ["RTDETRPostProcessor", "build_postprocessor"]


def _to_var(x: Any) -> jt.Var:
    if isinstance(x, jt.Var):
        return x
    return jt.array(x)


class RTDETRPostProcessor(nn.Module):
    def __init__(
        self,
        num_classes: int = 80,
        use_focal_loss: bool = True,
        num_top_queries: int = 300,
        remap_mscoco_category: bool = False,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = int(num_top_queries)
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return (
            f"use_focal_loss={self.use_focal_loss}, "
            f"num_classes={self.num_classes}, "
            f"num_top_queries={self.num_top_queries}"
        )

    def execute(
        self,
        outputs: Dict[str, jt.Var],
        orig_target_sizes: Any = None,
        target_sizes: Any = None,
    ):
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]

        if orig_target_sizes is None:
            orig_target_sizes = target_sizes
        if orig_target_sizes is None:
            raise ValueError("orig_target_sizes/target_sizes must be provided")
        orig_target_sizes = _to_var(orig_target_sizes).float32()
        bbox_pred = box_cxcywh_to_xyxy(boxes)
        bbox_pred = bbox_pred * orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = jt.sigmoid(logits)
            scores, index = jt.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            gather_index = index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1])
            boxes = jt.gather(bbox_pred, 1, gather_index)
        else:
            scores = nn.softmax(logits, dim=-1)[:, :, :-1]
            labels = scores.argmax(dim=-1)
            scores = scores.max(dim=-1)
            boxes = bbox_pred
            if scores.shape[1] > self.num_top_queries:
                scores, index = jt.topk(scores, self.num_top_queries, dim=-1)
                labels = jt.gather(labels, 1, index)
                gather_index = index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1])
                boxes = jt.gather(boxes, 1, gather_index)

        if self.deploy_mode:
            return labels, boxes, scores

        if self.remap_mscoco_category:
            from ...data.coco import mscoco_label2category

            mapped = [mscoco_label2category[int(x)] for x in labels.reshape((-1,)).numpy().tolist()]
            labels = jt.array(mapped, dtype=labels.dtype).reshape(labels.shape)

        results: List[Dict[str, jt.Var]] = []
        for lab, box, sco in zip(labels, boxes, scores):
            results.append({"labels": lab, "boxes": box, "scores": sco})
        return results

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self

    @property
    def iou_types(self):
        return ("bbox",)


def build_postprocessor(
    num_classes: int = 80,
    use_focal_loss: bool = True,
    num_top_queries: int = 300,
    remap_mscoco_category: bool = False,
):
    return RTDETRPostProcessor(
        num_classes=num_classes,
        use_focal_loss=use_focal_loss,
        num_top_queries=num_top_queries,
        remap_mscoco_category=remap_mscoco_category,
    )
