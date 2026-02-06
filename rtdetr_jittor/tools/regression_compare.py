#!/usr/bin/env python3
"""Cross-framework regression comparison between PyTorch and Jittor RT-DETR."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


THIS_FILE = Path(__file__).resolve()
JITTOR_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = JITTOR_ROOT.parent


def _serialize_targets(targets: List[Dict[str, np.ndarray]]) -> List[Dict[str, Any]]:
    out = []
    for t in targets:
        out.append(
            {
                "boxes": t["boxes"].tolist(),
                "labels": t["labels"].tolist(),
                "orig_size": t["orig_size"].tolist(),
                "image_id": int(t["image_id"]),
            }
        )
    return out


def _deserialize_targets(payload: List[Dict[str, Any]]) -> List[Dict[str, np.ndarray]]:
    targets: List[Dict[str, np.ndarray]] = []
    for t in payload:
        targets.append(
            {
                "boxes": np.asarray(t["boxes"], dtype=np.float32),
                "labels": np.asarray(t["labels"], dtype=np.int64),
                "orig_size": np.asarray(t["orig_size"], dtype=np.float32),
                "image_id": np.asarray([int(t["image_id"])], dtype=np.int64),
            }
        )
    return targets


def _make_random_targets(
    batch_size: int,
    num_boxes: int,
    num_classes: int,
    input_size: int,
    seed: int,
) -> List[Dict[str, np.ndarray]]:
    rng = np.random.RandomState(seed)
    targets: List[Dict[str, np.ndarray]] = []
    for idx in range(batch_size):
        centers = rng.uniform(0.1, 0.9, (num_boxes, 2)).astype(np.float32)
        sizes = rng.uniform(0.05, 0.4, (num_boxes, 2)).astype(np.float32)
        boxes = np.concatenate([centers, sizes], axis=1).astype(np.float32)
        boxes = np.clip(boxes, 0.0, 1.0)
        labels = rng.randint(0, num_classes, size=(num_boxes,), dtype=np.int64)
        targets.append(
            {
                "boxes": boxes,
                "labels": labels,
                "orig_size": np.asarray([input_size, input_size], dtype=np.float32),
                "image_id": np.asarray([idx], dtype=np.int64),
            }
        )
    return targets


def _load_runtime_cfg(
    config_path: str,
    input_size: int,
    num_classes: int,
    seed: int,
    num_denoising: int,
) -> Dict[str, Any]:
    sys.path.insert(0, str(JITTOR_ROOT))
    from src.core.engine import load_runtime_config

    overrides = {
        "input_size": int(input_size),
        "num_classes": int(num_classes),
        "seed": int(seed),
        "model.decoder.num_denoising": int(num_denoising),
    }
    cfg = load_runtime_config(config_path, overrides=overrides)
    cfg["model"]["multi_scale"] = None
    cfg["model"]["decoder"]["num_denoising"] = int(num_denoising)
    return cfg


def _collect_matcher_cost_numpy(
    pred_logits: np.ndarray,
    pred_boxes: np.ndarray,
    targets: List[Dict[str, np.ndarray]],
    cost_class_weight: float,
    cost_bbox_weight: float,
    cost_giou_weight: float,
    use_focal_loss: bool = True,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Dict[str, np.ndarray]:
    """Compute matcher cost components in numpy for cross-framework diagnostics."""
    # Local helpers to avoid framework dependency in orchestrator compare.
    def cxcywh_to_xyxy(x: np.ndarray) -> np.ndarray:
        x_c, y_c, w, h = np.split(x, 4, axis=-1)
        return np.concatenate([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], axis=-1)

    def box_area(boxes: np.ndarray) -> np.ndarray:
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    def box_iou(boxes1: np.ndarray, boxes2: np.ndarray):
        area1 = box_area(boxes1)
        area2 = box_area(boxes2)
        lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
        rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = np.clip(rb - lt, a_min=0.0, a_max=None)
        inter = wh[..., 0] * wh[..., 1]
        union = area1[:, None] + area2 - inter
        iou = inter / np.clip(union, 1e-12, None)
        return iou, union

    def generalized_box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        iou, union = box_iou(boxes1, boxes2)
        lt = np.minimum(boxes1[:, None, :2], boxes2[:, :2])
        rb = np.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = np.clip(rb - lt, a_min=0.0, a_max=None)
        area = wh[..., 0] * wh[..., 1]
        return iou - (area - union) / np.clip(area, 1e-12, None)

    bs, num_queries, num_classes = pred_logits.shape
    out_prob = pred_logits.reshape(bs * num_queries, num_classes).astype(np.float32)
    out_bbox = pred_boxes.reshape(bs * num_queries, 4).astype(np.float32)

    tgt_ids = np.concatenate([t["labels"].astype(np.int64) for t in targets], axis=0)
    tgt_bbox = np.concatenate([t["boxes"].astype(np.float32) for t in targets], axis=0)

    if use_focal_loss:
        out_prob = 1.0 / (1.0 + np.exp(-out_prob))
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-np.log(np.clip(1 - out_prob + 1e-8, 1e-8, None)))
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-np.log(np.clip(out_prob + 1e-8, 1e-8, None)))
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
    else:
        expv = np.exp(out_prob - out_prob.max(axis=-1, keepdims=True))
        out_prob = expv / np.clip(expv.sum(axis=-1, keepdims=True), 1e-12, None)
        cost_class = -out_prob[:, tgt_ids]

    cost_bbox = np.abs(out_bbox[:, None, :] - tgt_bbox[None, :, :]).sum(axis=-1)
    cost_giou = -generalized_box_iou(cxcywh_to_xyxy(out_bbox), cxcywh_to_xyxy(tgt_bbox))
    cost_total = (
        cost_bbox_weight * cost_bbox
        + cost_class_weight * cost_class
        + cost_giou_weight * cost_giou
    )
    sizes = [len(v["boxes"]) for v in targets]
    return {
        "cost_class": cost_class.reshape(bs, num_queries, -1),
        "cost_bbox": cost_bbox.reshape(bs, num_queries, -1),
        "cost_giou": cost_giou.reshape(bs, num_queries, -1),
        "cost_total": cost_total.reshape(bs, num_queries, -1),
        "sizes": np.asarray(sizes, dtype=np.int64),
    }


def _safe_to_float_array(x: np.ndarray) -> np.ndarray:
    if np.issubdtype(x.dtype, np.floating):
        return x.astype(np.float64, copy=False)
    return x.astype(np.float64)


def _abs_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    diff = np.abs(_safe_to_float_array(a) - _safe_to_float_array(b))
    return {
        "shape": list(a.shape),
        "max_abs": float(diff.max()) if diff.size > 0 else 0.0,
        "mean_abs": float(diff.mean()) if diff.size > 0 else 0.0,
    }


def _exact_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    same = a == b
    return {
        "shape": list(a.shape),
        "exact_match": bool(np.array_equal(a, b)),
        "equal_ratio": float(same.mean()) if same.size > 0 else 1.0,
    }


def _matcher_assignment_from_cost(cost_total: np.ndarray, sizes: Sequence[int]) -> List[Dict[str, List[int]]]:
    assignments: List[Dict[str, List[int]]] = []
    offset = 0
    for batch_idx, size in enumerate(sizes):
        size = int(size)
        if size <= 0:
            assignments.append({"src": [], "tgt": []})
            continue
        c = cost_total[batch_idx, :, offset : offset + size]
        src, tgt = linear_sum_assignment(c)
        assignments.append(
            {
                "src": src.astype(np.int64).tolist(),
                "tgt": tgt.astype(np.int64).tolist(),
            }
        )
        offset += size
    return assignments


def _permute_jt_to_pt_by_query_cost(
    pt_logits: np.ndarray,
    pt_boxes: np.ndarray,
    jt_logits: np.ndarray,
    jt_boxes: np.ndarray,
    logits_weight: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align JT query order to PT via Hungarian assignment over query-query cost."""
    bs, num_queries, _ = pt_boxes.shape
    aligned_logits = np.empty_like(jt_logits)
    aligned_boxes = np.empty_like(jt_boxes)
    for b in range(bs):
        pt_b = pt_boxes[b]
        jt_b = jt_boxes[b]
        box_cost = np.abs(pt_b[:, None, :] - jt_b[None, :, :]).mean(axis=-1)
        logit_cost = np.abs(pt_logits[b][:, None, :] - jt_logits[b][None, :, :]).mean(axis=-1)
        query_cost = box_cost + logits_weight * logit_cost
        src, tgt = linear_sum_assignment(query_cost)
        # linear_sum_assignment returns sorted src indices; for square matrix this is 0..Q-1.
        reorder = np.zeros((num_queries,), dtype=np.int64)
        reorder[src] = tgt
        aligned_logits[b] = jt_logits[b][reorder]
        aligned_boxes[b] = jt_boxes[b][reorder]
    return aligned_logits, aligned_boxes


def _state_digest(keys: Sequence[str]) -> str:
    payload = "\n".join(keys).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_buffer_like_key(key: str) -> bool:
    return (
        key.endswith("running_mean")
        or key.endswith("running_var")
        or key.endswith("num_batches_tracked")
        or "pos_embed" in key
        or key.endswith("anchors")
        or key.endswith("valid_mask")
    )


def _worker_pytorch(args: argparse.Namespace) -> None:
    import torch

    try:
        import torchvision  # noqa: F401
    except Exception:
        tv_mod = types.ModuleType("torchvision")
        ops_mod = types.ModuleType("torchvision.ops")
        boxes_mod = types.ModuleType("torchvision.ops.boxes")

        def _box_area(boxes):
            return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        def _sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="none"):
            p = torch.sigmoid(inputs)
            ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
            p_t = p * targets + (1.0 - p) * (1.0 - targets)
            alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
            loss = alpha_t * ce_loss * ((1.0 - p_t) ** gamma)
            if reduction == "mean":
                return loss.mean()
            if reduction == "sum":
                return loss.sum()
            return loss

        boxes_mod.box_area = _box_area
        ops_mod.boxes = boxes_mod
        ops_mod.sigmoid_focal_loss = _sigmoid_focal_loss
        tv_mod.disable_beta_transforms_warning = lambda: None
        tv_mod.ops = ops_mod
        sys.modules["torchvision"] = tv_mod
        sys.modules["torchvision.ops"] = ops_mod
        sys.modules["torchvision.ops.boxes"] = boxes_mod

    src_path = PROJECT_ROOT / "rtdetr_pytorch" / "src"
    sys.modules.pop("src", None)
    for key in list(sys.modules.keys()):
        if key.startswith("src."):
            del sys.modules[key]

    src_mod = types.ModuleType("src")
    src_mod.__path__ = [str(src_path)]
    sys.modules["src"] = src_mod

    # Pre-inject subpackages to avoid executing heavy __init__.py imports (e.g. regnet/transformers).
    for pkg, rel in [
        ("src.nn", "nn"),
        ("src.nn.backbone", "nn/backbone"),
        ("src.misc", "misc"),
        ("src.zoo", "zoo"),
        ("src.zoo.rtdetr", "zoo/rtdetr"),
    ]:
        mod = types.ModuleType(pkg)
        mod.__path__ = [str(src_path / rel)]
        sys.modules[pkg] = mod

    with open(args.cfg_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(args.targets_json, "r", encoding="utf-8") as f:
        targets = _deserialize_targets(json.load(f))
    input_np = np.load(args.input_npy).astype(np.float32)

    from src.nn.backbone.presnet import PResNet
    from src.zoo.rtdetr.hybrid_encoder import HybridEncoder
    from src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
    from src.zoo.rtdetr.rtdetr import RTDETR
    from src.zoo.rtdetr.matcher import HungarianMatcher
    from src.zoo.rtdetr.rtdetr_criterion import SetCriterion

    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    backbone_cfg = cfg["model"]["backbone"]
    encoder_cfg = cfg["model"]["encoder"]
    decoder_cfg = cfg["model"]["decoder"]
    num_classes = int(cfg["num_classes"])

    backbone = PResNet(
        depth=int(backbone_cfg.get("depth", 50)),
        variant=backbone_cfg.get("variant", "d"),
        num_stages=int(backbone_cfg.get("num_stages", 4)),
        return_idx=list(backbone_cfg.get("return_idx", [1, 2, 3])),
        freeze_at=int(backbone_cfg.get("freeze_at", 0)),
        freeze_norm=bool(backbone_cfg.get("freeze_norm", True)),
        pretrained=False,
    )
    encoder = HybridEncoder(
        in_channels=backbone.out_channels,
        feat_strides=backbone.out_strides,
        hidden_dim=int(encoder_cfg.get("hidden_dim", 256)),
        nhead=int(encoder_cfg.get("nhead", 8)),
        dim_feedforward=int(encoder_cfg.get("dim_feedforward", 1024)),
        dropout=float(encoder_cfg.get("dropout", 0.0)),
        enc_act=encoder_cfg.get("enc_act", "gelu"),
        use_encoder_idx=list(encoder_cfg.get("use_encoder_idx", [2])),
        num_encoder_layers=int(encoder_cfg.get("num_encoder_layers", 1)),
        pe_temperature=float(encoder_cfg.get("pe_temperature", 10000)),
        expansion=float(encoder_cfg.get("expansion", 1.0)),
        depth_mult=float(encoder_cfg.get("depth_mult", 1.0)),
        act=encoder_cfg.get("act", "silu"),
        eval_spatial_size=encoder_cfg.get("eval_spatial_size"),
    )
    decoder = RTDETRTransformer(
        num_classes=num_classes,
        hidden_dim=int(decoder_cfg.get("hidden_dim", 256)),
        num_queries=int(decoder_cfg.get("num_queries", 300)),
        feat_channels=list(encoder.out_channels),
        feat_strides=list(encoder.out_strides),
        num_levels=int(decoder_cfg.get("num_levels", 3)),
        num_decoder_points=int(decoder_cfg.get("num_decoder_points", 4)),
        nhead=int(decoder_cfg.get("nhead", 8)),
        num_decoder_layers=int(decoder_cfg.get("num_decoder_layers", 6)),
        dim_feedforward=int(decoder_cfg.get("dim_feedforward", 1024)),
        dropout=float(decoder_cfg.get("dropout", 0.0)),
        activation=decoder_cfg.get("activation", "relu"),
        num_denoising=int(decoder_cfg.get("num_denoising", 100)),
        label_noise_ratio=float(decoder_cfg.get("label_noise_ratio", 0.5)),
        box_noise_scale=float(decoder_cfg.get("box_noise_scale", 1.0)),
        learnt_init_query=bool(decoder_cfg.get("learnt_init_query", False)),
        eval_idx=int(decoder_cfg.get("eval_idx", -1)),
        eval_spatial_size=encoder_cfg.get("eval_spatial_size"),
        aux_loss=bool(decoder_cfg.get("aux_loss", True)),
    )
    model = RTDETR(backbone=backbone, encoder=encoder, decoder=decoder, multi_scale=None)
    criterion_cfg = cfg.get("criterion", {})
    matcher_cfg = criterion_cfg.get("matcher", {})
    matcher_weight_cfg = matcher_cfg.get("weight_dict", {}) if isinstance(matcher_cfg.get("weight_dict"), dict) else {}
    matcher = HungarianMatcher(
        weight_dict={
            "cost_class": float(matcher_cfg.get("cost_class", matcher_weight_cfg.get("cost_class", 2))),
            "cost_bbox": float(matcher_cfg.get("cost_bbox", matcher_weight_cfg.get("cost_bbox", 5))),
            "cost_giou": float(matcher_cfg.get("cost_giou", matcher_weight_cfg.get("cost_giou", 2))),
        },
        use_focal_loss=bool(matcher_cfg.get("use_focal_loss", matcher_cfg.get("use_focal", True))),
        alpha=float(matcher_cfg.get("alpha", 0.25)),
        gamma=float(matcher_cfg.get("gamma", 2.0)),
    )
    weight_dict = criterion_cfg.get("weight_dict")
    if isinstance(weight_dict, dict):
        weight_dict = dict(weight_dict)
        if "loss_focal" in weight_dict and "loss_vfl" not in weight_dict:
            weight_dict["loss_vfl"] = weight_dict["loss_focal"]
        if "loss_ce" in weight_dict and "loss_vfl" not in weight_dict:
            weight_dict["loss_vfl"] = weight_dict["loss_ce"]
    else:
        weight_dict = {"loss_vfl": 1, "loss_bbox": 5, "loss_giou": 2}
    losses_cfg = criterion_cfg.get("losses", ["vfl", "boxes"])
    losses = ["vfl" if loss == "labels" else str(loss) for loss in losses_cfg]
    criterion = SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        alpha=float(criterion_cfg.get("alpha", 0.2)),
        gamma=float(criterion_cfg.get("gamma", 2.0)),
        eos_coef=float(criterion_cfg.get("eos_coef", 1e-4)),
        num_classes=num_classes,
    )

    def _to_np(t: torch.Tensor) -> np.ndarray:
        arr = t.detach().cpu().numpy()
        if np.issubdtype(arr.dtype, np.floating):
            return arr.astype(np.float32)
        if np.issubdtype(arr.dtype, np.integer):
            return arr.astype(np.int64)
        return arr

    x = torch.from_numpy(input_np)
    targets_torch = []
    for t in targets:
        targets_torch.append(
            {
                "boxes": torch.from_numpy(t["boxes"]).float(),
                "labels": torch.from_numpy(t["labels"]).long(),
                "orig_size": torch.from_numpy(t["orig_size"]).float(),
                "image_id": torch.from_numpy(t["image_id"]).long(),
            }
        )

    # Keep a frozen initial state for fair PT/JT comparison.
    initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    trace_np: Dict[str, np.ndarray] = {}
    model.load_state_dict(initial_state, strict=True)
    model.eval()
    with torch.no_grad():
        feats = model.backbone(x)
        for idx, feat in enumerate(feats):
            trace_np[f"trace_backbone_{idx}"] = _to_np(feat)

        enc_feats = model.encoder(feats)
        for idx, feat in enumerate(enc_feats):
            trace_np[f"trace_encoder_{idx}"] = _to_np(feat)

        dec = model.decoder
        memory, spatial_shapes, level_start_index = dec._get_encoder_input(enc_feats)
        trace_np["trace_decoder_memory"] = _to_np(memory)
        trace_np["trace_decoder_spatial_shapes"] = np.asarray(spatial_shapes, dtype=np.int64)
        trace_np["trace_decoder_level_start_index"] = np.asarray(level_start_index, dtype=np.int64)

        if dec.training or dec.eval_spatial_size is None:
            anchors, valid_mask = dec._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = dec.anchors.to(memory.device), dec.valid_mask.to(memory.device)
        memory_masked = valid_mask.to(memory.dtype) * memory
        output_memory = dec.enc_output(memory_masked)
        enc_outputs_class = dec.enc_score_head(output_memory)
        enc_outputs_coord_unact = dec.enc_bbox_head(output_memory) + anchors
        class_max = enc_outputs_class.max(-1).values
        _, topk_ind = torch.topk(class_max, dec.num_queries, dim=1)

        topk_ind_expand = topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1])
        reference_points_unact = enc_outputs_coord_unact.gather(dim=1, index=topk_ind_expand)
        enc_topk_bboxes = torch.sigmoid(reference_points_unact)
        enc_topk_logits = enc_outputs_class.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1])
        )

        bs = memory.shape[0]
        if dec.learnt_init_query:
            target = dec.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = output_memory.gather(
                dim=1,
                index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]),
            ).detach()

        dec_was_training = dec.decoder.training
        dec.decoder.train()
        out_bboxes_all, out_logits_all = dec.decoder(
            target,
            reference_points_unact.detach(),
            memory,
            spatial_shapes,
            level_start_index,
            dec.dec_bbox_head,
            dec.dec_score_head,
            dec.query_pos_head,
            attn_mask=None,
        )
        dec.decoder.train(dec_was_training)

        out_eval = {"pred_logits": out_logits_all[-1], "pred_boxes": out_bboxes_all[-1]}

        trace_np["trace_decoder_output_memory"] = _to_np(output_memory)
        trace_np["trace_decoder_enc_outputs_class"] = _to_np(enc_outputs_class)
        trace_np["trace_decoder_enc_outputs_coord_unact"] = _to_np(enc_outputs_coord_unact)
        trace_np["trace_decoder_class_max"] = _to_np(class_max)
        trace_np["trace_decoder_topk_ind"] = _to_np(topk_ind)
        trace_np["trace_decoder_target"] = _to_np(target)
        trace_np["trace_decoder_reference_points_unact"] = _to_np(reference_points_unact)
        trace_np["trace_decoder_enc_topk_bboxes"] = _to_np(enc_topk_bboxes)
        trace_np["trace_decoder_enc_topk_logits"] = _to_np(enc_topk_logits)
        trace_np["trace_decoder_all_bboxes"] = _to_np(out_bboxes_all)
        trace_np["trace_decoder_all_logits"] = _to_np(out_logits_all)

        eval_pred_logits_np = _to_np(out_eval["pred_logits"])
        eval_pred_boxes_np = _to_np(out_eval["pred_boxes"])

    model.load_state_dict(initial_state, strict=True)
    model.train()
    # Keep BN in eval for deterministic cross-framework train-path comparison.
    model.backbone.eval()
    model.encoder.eval()
    for proj in model.decoder.input_proj:
        proj.eval()
    out_train = model(x, targets_torch)
    loss_dict = criterion(out_train, targets_torch)

    with torch.no_grad():
        matcher_indices = criterion.matcher(
            {"pred_logits": out_train["pred_logits"], "pred_boxes": out_train["pred_boxes"]},
            targets_torch,
        )

    train_pred_logits_np = _to_np(out_train["pred_logits"])
    train_pred_boxes_np = _to_np(out_train["pred_boxes"])
    matcher_cost = _collect_matcher_cost_numpy(
        pred_logits=train_pred_logits_np,
        pred_boxes=train_pred_boxes_np,
        targets=targets,
        cost_class_weight=float(matcher.cost_class),
        cost_bbox_weight=float(matcher.cost_bbox),
        cost_giou_weight=float(matcher.cost_giou),
        use_focal_loss=bool(matcher.use_focal_loss),
        alpha=float(matcher.alpha),
        gamma=float(matcher.gamma),
    )

    npz_payload = {
        "pred_logits": eval_pred_logits_np,
        "pred_boxes": eval_pred_boxes_np,
        "train_pred_logits": train_pred_logits_np,
        "train_pred_boxes": train_pred_boxes_np,
        "matcher_cost_class": matcher_cost["cost_class"].astype(np.float32),
        "matcher_cost_bbox": matcher_cost["cost_bbox"].astype(np.float32),
        "matcher_cost_giou": matcher_cost["cost_giou"].astype(np.float32),
        "matcher_cost_total": matcher_cost["cost_total"].astype(np.float32),
        "matcher_sizes": matcher_cost["sizes"].astype(np.int64),
    }
    npz_payload.update(trace_np)
    np.savez(args.output_npz, **npz_payload)

    losses = {k: float(v.detach().cpu().item()) for k, v in loss_dict.items()}
    matcher_dump = []
    for src_idx, tgt_idx in matcher_indices:
        matcher_dump.append(
            {
                "src": src_idx.detach().cpu().numpy().astype(np.int64).tolist(),
                "tgt": tgt_idx.detach().cpu().numpy().astype(np.int64).tolist(),
            }
        )

    state_keys = sorted(initial_state.keys())
    state_numel = int(sum(int(v.numel()) for v in initial_state.values()))
    state_param_like_keys = [k for k in state_keys if not _is_buffer_like_key(k)]
    state_param_like_numel = int(sum(int(initial_state[k].numel()) for k in state_param_like_keys))
    param_count = int(sum(int(p.numel()) for p in model.parameters()))
    trainable_param_count = int(sum(int(p.numel()) for p in model.parameters() if p.requires_grad))

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "loss": losses,
                "matcher": matcher_dump,
                "matcher_from_cost": _matcher_assignment_from_cost(
                    matcher_cost["cost_total"], matcher_cost["sizes"].tolist()
                ),
                "model_stats": {
                    "param_count": param_count,
                    "trainable_param_count": trainable_param_count,
                    "state_key_count": len(state_keys),
                    "state_numel": state_numel,
                    "state_param_like_key_count": len(state_param_like_keys),
                    "state_param_like_numel": state_param_like_numel,
                    "state_digest": _state_digest(state_keys),
                    "state_keys_head": state_keys[:16],
                    "state_keys_tail": state_keys[-16:],
                },
            },
            f,
            indent=2,
        )

    torch.save({"model": initial_state}, args.pt_state_path)


def _worker_jittor(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(JITTOR_ROOT))

    import jittor as jt

    from src.core.engine import (
        build_model_components,
        load_model_state,
        set_runtime,
    )
    from src.zoo.rtdetr.utils import tile

    with open(args.cfg_json, "r", encoding="utf-8") as f:
        cfg_payload = json.load(f)
    with open(args.targets_json, "r", encoding="utf-8") as f:
        targets = _deserialize_targets(json.load(f))
    input_np = np.load(args.input_npy).astype(np.float32)

    cfg = dict(cfg_payload)
    set_runtime(device="cpu", seed=int(cfg["seed"]))

    model, criterion, _ = build_model_components(cfg, num_classes=int(cfg["num_classes"]))
    state = jt.load(args.jt_state_path)
    load_info = load_model_state(model, state)

    def _to_np(v: jt.Var) -> np.ndarray:
        arr = v.numpy()
        if np.issubdtype(arr.dtype, np.floating):
            return arr.astype(np.float32)
        if np.issubdtype(arr.dtype, np.integer):
            return arr.astype(np.int64)
        return arr

    def _numel_by_shape(shape: Sequence[int]) -> int:
        n = 1
        for d in shape:
            n *= int(d)
        return int(n)

    x = jt.array(input_np).float32()
    targets_jt = []
    for t in targets:
        targets_jt.append(
            {
                "boxes": jt.array(t["boxes"]).float32(),
                "labels": jt.array(t["labels"]).int64(),
                "orig_size": jt.array(t["orig_size"]).float32(),
                "image_id": jt.array(t["image_id"]).int64(),
            }
        )

    trace_np: Dict[str, np.ndarray] = {}
    model.eval()
    with jt.no_grad():
        feats = model.backbone(x)
        for idx, feat in enumerate(feats):
            trace_np[f"trace_backbone_{idx}"] = _to_np(feat)

        enc_feats = model.encoder(feats)
        for idx, feat in enumerate(enc_feats):
            trace_np[f"trace_encoder_{idx}"] = _to_np(feat)

        dec = model.decoder
        memory, spatial_shapes, level_start_index = dec._get_encoder_input(enc_feats)
        trace_np["trace_decoder_memory"] = _to_np(memory)
        trace_np["trace_decoder_spatial_shapes"] = np.asarray(spatial_shapes, dtype=np.int64)
        trace_np["trace_decoder_level_start_index"] = np.asarray(level_start_index, dtype=np.int64)

        if dec.is_training() or dec.eval_spatial_size is None:
            anchors, valid_mask = dec._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = dec.anchors, dec.valid_mask
        memory_masked = valid_mask.float32() * memory
        output_memory = dec.enc_output(memory_masked)
        enc_outputs_class = dec.enc_score_head(output_memory)
        enc_outputs_coord_unact = dec.enc_bbox_head(output_memory) + anchors
        class_max = enc_outputs_class.max(-1)
        _, topk_ind = jt.topk(class_max, dec.num_queries, dim=1)

        bs = int(memory.shape[0])
        topk_ind_expand = topk_ind.unsqueeze(-1).expand([bs, dec.num_queries, enc_outputs_coord_unact.shape[-1]])
        reference_points_unact = jt.gather(enc_outputs_coord_unact, 1, topk_ind_expand)
        enc_topk_bboxes = jt.sigmoid(reference_points_unact)
        topk_ind_expand_logits = topk_ind.unsqueeze(-1).expand([bs, dec.num_queries, enc_outputs_class.shape[-1]])
        enc_topk_logits = jt.gather(enc_outputs_class, 1, topk_ind_expand_logits)

        if dec.learnt_init_query:
            target = tile(dec.tgt_embed.weight.unsqueeze(0), [bs, 1, 1])
        else:
            topk_ind_expand_mem = topk_ind.unsqueeze(-1).expand([bs, dec.num_queries, output_memory.shape[-1]])
            target = jt.gather(output_memory, 1, topk_ind_expand_mem).stop_grad()

        dec_was_training = dec.decoder.is_training()
        dec.decoder.train()
        out_bboxes_all, out_logits_all = dec.decoder(
            target,
            reference_points_unact.stop_grad(),
            memory,
            spatial_shapes,
            level_start_index,
            dec.dec_bbox_head,
            dec.dec_score_head,
            dec.query_pos_head,
            attn_mask=None,
        )
        if dec_was_training:
            dec.decoder.train()
        else:
            dec.decoder.eval()

        out_eval = {"pred_logits": out_logits_all[-1], "pred_boxes": out_bboxes_all[-1]}

        trace_np["trace_decoder_output_memory"] = _to_np(output_memory)
        trace_np["trace_decoder_enc_outputs_class"] = _to_np(enc_outputs_class)
        trace_np["trace_decoder_enc_outputs_coord_unact"] = _to_np(enc_outputs_coord_unact)
        trace_np["trace_decoder_class_max"] = _to_np(class_max)
        trace_np["trace_decoder_topk_ind"] = _to_np(topk_ind)
        trace_np["trace_decoder_target"] = _to_np(target)
        trace_np["trace_decoder_reference_points_unact"] = _to_np(reference_points_unact)
        trace_np["trace_decoder_enc_topk_bboxes"] = _to_np(enc_topk_bboxes)
        trace_np["trace_decoder_enc_topk_logits"] = _to_np(enc_topk_logits)
        trace_np["trace_decoder_all_bboxes"] = _to_np(out_bboxes_all)
        trace_np["trace_decoder_all_logits"] = _to_np(out_logits_all)

        # Materialize eval outputs before switching training mode to avoid lazy-eval contamination.
        eval_pred_logits_np = _to_np(out_eval["pred_logits"])
        eval_pred_boxes_np = _to_np(out_eval["pred_boxes"])

    model.train()
    # Keep BN in eval for deterministic cross-framework train-path comparison.
    model.backbone.eval()
    model.encoder.eval()
    for proj in model.decoder.input_proj:
        proj.eval()
    out_train = model(x, targets_jt)
    loss_dict = criterion(out_train, targets_jt)

    with jt.no_grad():
        matcher_indices = criterion.matcher(
            {"pred_logits": out_train["pred_logits"], "pred_boxes": out_train["pred_boxes"]},
            targets_jt,
        )

    train_pred_logits_np = _to_np(out_train["pred_logits"])
    train_pred_boxes_np = _to_np(out_train["pred_boxes"])
    matcher = criterion.matcher
    matcher_cost = _collect_matcher_cost_numpy(
        pred_logits=train_pred_logits_np,
        pred_boxes=train_pred_boxes_np,
        targets=targets,
        cost_class_weight=float(matcher.cost_class),
        cost_bbox_weight=float(matcher.cost_bbox),
        cost_giou_weight=float(matcher.cost_giou),
        use_focal_loss=bool(matcher.use_focal_loss),
        alpha=float(matcher.alpha),
        gamma=float(matcher.gamma),
    )

    npz_payload = {
        "pred_logits": eval_pred_logits_np,
        "pred_boxes": eval_pred_boxes_np,
        "train_pred_logits": train_pred_logits_np,
        "train_pred_boxes": train_pred_boxes_np,
        "matcher_cost_class": matcher_cost["cost_class"].astype(np.float32),
        "matcher_cost_bbox": matcher_cost["cost_bbox"].astype(np.float32),
        "matcher_cost_giou": matcher_cost["cost_giou"].astype(np.float32),
        "matcher_cost_total": matcher_cost["cost_total"].astype(np.float32),
        "matcher_sizes": matcher_cost["sizes"].astype(np.int64),
    }
    npz_payload.update(trace_np)
    np.savez(args.output_npz, **npz_payload)

    losses = {k: float(v.item()) for k, v in loss_dict.items()}
    matcher_dump = []
    for src_idx, tgt_idx in matcher_indices:
        matcher_dump.append(
            {
                "src": src_idx.numpy().astype(np.int64).tolist(),
                "tgt": tgt_idx.numpy().astype(np.int64).tolist(),
            }
        )

    state_dict = model.state_dict()
    state_keys = sorted(state_dict.keys())
    state_numel = int(sum(_numel_by_shape(v.shape) for v in state_dict.values()))
    state_param_like_keys = [k for k in state_keys if not _is_buffer_like_key(k)]
    state_param_like_numel = int(sum(_numel_by_shape(state_dict[k].shape) for k in state_param_like_keys))
    param_count = int(sum(_numel_by_shape(p.shape) for p in model.parameters()))
    trainable_param_count = param_count

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "loss": losses,
                "matcher": matcher_dump,
                "matcher_from_cost": _matcher_assignment_from_cost(
                    matcher_cost["cost_total"], matcher_cost["sizes"].tolist()
                ),
                "load_info": load_info,
                "model_stats": {
                    "param_count": param_count,
                    "trainable_param_count": trainable_param_count,
                    "state_key_count": len(state_keys),
                    "state_numel": state_numel,
                    "state_param_like_key_count": len(state_param_like_keys),
                    "state_param_like_numel": state_param_like_numel,
                    "state_digest": _state_digest(state_keys),
                    "state_keys_head": state_keys[:16],
                    "state_keys_tail": state_keys[-16:],
                },
            },
            f,
            indent=2,
        )


def _compare_outputs(pt_npz: Path, jt_npz: Path, pt_json: Path, jt_json: Path) -> Dict[str, Any]:
    pt_data = np.load(pt_npz)
    jt_data = np.load(jt_npz)

    pt_logits = pt_data["pred_logits"]
    jt_logits = jt_data["pred_logits"]
    pt_boxes = pt_data["pred_boxes"]
    jt_boxes = jt_data["pred_boxes"]

    with open(pt_json, "r", encoding="utf-8") as f:
        pt_meta = json.load(f)
    with open(jt_json, "r", encoding="utf-8") as f:
        jt_meta = json.load(f)

    pt_loss = pt_meta.get("loss", {})
    jt_loss = jt_meta.get("loss", {})
    loss_keys = sorted(set(pt_loss.keys()) & set(jt_loss.keys()))
    loss_diff = {}
    for key in loss_keys:
        loss_diff[key] = {
            "pt": float(pt_loss[key]),
            "jt": float(jt_loss[key]),
            "abs": abs(float(pt_loss[key]) - float(jt_loss[key])),
        }

    # Query permutation-aware comparison to separate semantic mismatch vs ordering mismatch.
    jt_logits_aligned, jt_boxes_aligned = _permute_jt_to_pt_by_query_cost(
        pt_logits=pt_logits,
        pt_boxes=pt_boxes,
        jt_logits=jt_logits,
        jt_boxes=jt_boxes,
    )

    matcher_same = pt_meta.get("matcher", []) == jt_meta.get("matcher", [])
    matcher_cost_assignment_same = pt_meta.get("matcher_from_cost", []) == jt_meta.get("matcher_from_cost", [])

    matcher_cost_report: Dict[str, Any] = {}
    for key in ["matcher_cost_class", "matcher_cost_bbox", "matcher_cost_giou", "matcher_cost_total"]:
        if key in pt_data.files and key in jt_data.files and pt_data[key].shape == jt_data[key].shape:
            matcher_cost_report[key] = _abs_stats(pt_data[key], jt_data[key])
    if "matcher_sizes" in pt_data.files and "matcher_sizes" in jt_data.files:
        if pt_data["matcher_sizes"].shape == jt_data["matcher_sizes"].shape:
            matcher_cost_report["matcher_sizes"] = _exact_stats(
                pt_data["matcher_sizes"], jt_data["matcher_sizes"]
            )
            if "matcher_cost_total" in pt_data.files and "matcher_cost_total" in jt_data.files:
                pt_assign = _matcher_assignment_from_cost(
                    pt_data["matcher_cost_total"], pt_data["matcher_sizes"].tolist()
                )
                jt_assign = _matcher_assignment_from_cost(
                    jt_data["matcher_cost_total"], jt_data["matcher_sizes"].tolist()
                )
                matcher_cost_report["assignment_from_cost_exact_match"] = bool(pt_assign == jt_assign)

    # Layer-by-layer trace summaries.
    trace_report: Dict[str, Any] = {}
    trace_keys = sorted(
        set(k for k in pt_data.files if k.startswith("trace_"))
        & set(k for k in jt_data.files if k.startswith("trace_"))
    )
    for key in trace_keys:
        pt_arr = pt_data[key]
        jt_arr = jt_data[key]
        if pt_arr.shape != jt_arr.shape:
            trace_report[key] = {"shape_pt": list(pt_arr.shape), "shape_jt": list(jt_arr.shape)}
            continue
        if np.issubdtype(pt_arr.dtype, np.integer) and np.issubdtype(jt_arr.dtype, np.integer):
            trace_report[key] = _exact_stats(pt_arr, jt_arr)
        else:
            trace_report[key] = _abs_stats(pt_arr, jt_arr)

    pt_model_stats = pt_meta.get("model_stats", {})
    jt_model_stats = jt_meta.get("model_stats", {})
    param_align = {
        "pt_param_count": int(pt_model_stats.get("param_count", -1)),
        "jt_param_count": int(jt_model_stats.get("param_count", -1)),
        "param_count_exact_match": bool(
            int(pt_model_stats.get("param_count", -1)) == int(jt_model_stats.get("param_count", -2))
        ),
        "pt_trainable_param_count": int(pt_model_stats.get("trainable_param_count", -1)),
        "jt_trainable_param_count": int(jt_model_stats.get("trainable_param_count", -1)),
        "state_key_count_pt": int(pt_model_stats.get("state_key_count", -1)),
        "state_key_count_jt": int(jt_model_stats.get("state_key_count", -1)),
        "state_numel_pt": int(pt_model_stats.get("state_numel", -1)),
        "state_numel_jt": int(jt_model_stats.get("state_numel", -1)),
        "state_param_like_key_count_pt": int(pt_model_stats.get("state_param_like_key_count", -1)),
        "state_param_like_key_count_jt": int(jt_model_stats.get("state_param_like_key_count", -1)),
        "state_param_like_numel_pt": int(pt_model_stats.get("state_param_like_numel", -1)),
        "state_param_like_numel_jt": int(jt_model_stats.get("state_param_like_numel", -1)),
        "state_param_like_numel_exact_match": bool(
            int(pt_model_stats.get("state_param_like_numel", -1))
            == int(jt_model_stats.get("state_param_like_numel", -2))
        ),
        "state_digest_pt": pt_model_stats.get("state_digest", ""),
        "state_digest_jt": jt_model_stats.get("state_digest", ""),
        "state_digest_exact_match": bool(
            pt_model_stats.get("state_digest", "") == jt_model_stats.get("state_digest", "")
        ),
    }

    report = {
        "logits": _abs_stats(pt_logits, jt_logits),
        "boxes": _abs_stats(pt_boxes, jt_boxes),
        "logits_permuted": _abs_stats(pt_logits, jt_logits_aligned),
        "boxes_permuted": _abs_stats(pt_boxes, jt_boxes_aligned),
        "loss": loss_diff,
        "matcher_exact_match": bool(matcher_same),
        "matcher_from_cost_exact_match": bool(matcher_cost_assignment_same),
        "matcher_cost": matcher_cost_report,
        "trace": trace_report,
        "parameter_alignment": param_align,
        "jittor_load_info": jt_meta.get("load_info", {}),
    }
    return report


def _run_orchestrator(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_runtime_cfg(
        config_path=args.config,
        input_size=int(args.input_size),
        num_classes=int(args.num_classes),
        seed=int(args.seed),
        num_denoising=int(args.num_denoising),
    )

    rng = np.random.RandomState(int(args.seed))
    input_np = rng.uniform(
        0.0,
        1.0,
        size=(int(args.batch_size), 3, int(args.input_size), int(args.input_size)),
    ).astype(np.float32)
    targets = _make_random_targets(
        batch_size=int(args.batch_size),
        num_boxes=int(args.num_boxes),
        num_classes=int(args.num_classes),
        input_size=int(args.input_size),
        seed=int(args.seed) + 11,
    )

    input_npy = out_dir / "input.npy"
    targets_json = out_dir / "targets.json"
    cfg_json = out_dir / "runtime_cfg.json"
    pt_npz = out_dir / "pt_outputs.npz"
    pt_json = out_dir / "pt_meta.json"
    jt_npz = out_dir / "jt_outputs.npz"
    jt_json = out_dir / "jt_meta.json"
    pt_state = out_dir / "pt_state.pth"
    jt_state = out_dir / "jt_state.pkl"
    report_json = out_dir / "compare_report.json"

    np.save(input_npy, input_np)
    with open(targets_json, "w", encoding="utf-8") as f:
        json.dump(_serialize_targets(targets), f, indent=2)
    with open(cfg_json, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    cmd_pt = [
        sys.executable,
        str(THIS_FILE),
        "--worker",
        "pytorch",
        "--cfg-json",
        str(cfg_json),
        "--input-npy",
        str(input_npy),
        "--targets-json",
        str(targets_json),
        "--output-npz",
        str(pt_npz),
        "--output-json",
        str(pt_json),
        "--pt-state-path",
        str(pt_state),
    ]
    subprocess.run(cmd_pt, check=True)

    convert_cmd = [
        sys.executable,
        str(JITTOR_ROOT / "tools" / "convert_weights.py"),
        "--pt2jt",
        "-i",
        str(pt_state),
        "-o",
        str(jt_state),
    ]
    subprocess.run(convert_cmd, check=True)

    cmd_jt = [
        sys.executable,
        str(THIS_FILE),
        "--worker",
        "jittor",
        "--cfg-json",
        str(cfg_json),
        "--input-npy",
        str(input_npy),
        "--targets-json",
        str(targets_json),
        "--output-npz",
        str(jt_npz),
        "--output-json",
        str(jt_json),
        "--jt-state-path",
        str(jt_state),
    ]
    subprocess.run(cmd_jt, check=True)

    report = _compare_outputs(pt_npz=pt_npz, jt_npz=jt_npz, pt_json=pt_json, jt_json=jt_json)
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Regression comparison finished:")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {report_json}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyTorch vs Jittor RT-DETR regression compare")
    parser.add_argument("--config", "-c", type=str, default=str(JITTOR_ROOT / "configs" / "rtdetr" / "rtdetr_r18vd_6x_coco.yml"))
    parser.add_argument("--output-dir", type=str, default=str(JITTOR_ROOT / "migration_artifacts" / "regression"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-boxes", type=int, default=5)
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument(
        "--num-denoising",
        type=int,
        default=0,
        help="Denoising query count for train-path comparison (default 0 for deterministic alignment).",
    )

    parser.add_argument("--worker", type=str, default="", choices=["", "pytorch", "jittor"])
    parser.add_argument("--cfg-json", type=str, default="")
    parser.add_argument("--input-npy", type=str, default="")
    parser.add_argument("--targets-json", type=str, default="")
    parser.add_argument("--output-npz", type=str, default="")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--pt-state-path", type=str, default="")
    parser.add_argument("--jt-state-path", type=str, default="")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.worker == "pytorch":
        _worker_pytorch(args)
        return
    if args.worker == "jittor":
        _worker_jittor(args)
        return

    _run_orchestrator(args)


if __name__ == "__main__":
    main()
