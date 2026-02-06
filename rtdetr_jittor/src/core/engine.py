"""Reusable migration engine for RT-DETR-style Jittor projects."""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

import jittor as jt

from ..data import CocoDetectionDataset, DummyDetectionDataset, SimpleDetectionDataLoader
from ..nn.backbone.resnet import PResNet
from ..optim.ema import ModelEMA
from ..zoo.rtdetr.rtdetr import RTDETR
from ..zoo.rtdetr.hybrid_encoder import HybridEncoder
from ..zoo.rtdetr.rtdetr_criterion import HungarianMatcher, build_criterion
from ..zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
from ..zoo.rtdetr.rtdetr_postprocessor import build_postprocessor

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "device": "cuda",
    "output_dir": "./outputs/rtdetr_jittor",
    "input_size": 640,
    "num_classes": 80,
    "train": {
        "epochs": 72,
        "batch_size": 2,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 0,
        "min_lr": 1e-6,
        "grad_accum_steps": 1,
        "ema": True,
        "ema_decay": 0.9999,
        "ema_warmups": 2000,
        "print_freq": 10,
        "save_every": 1,
        "eval_every": 1,
    },
    "eval": {
        "score_threshold": 0.3,
        "topk": 300,
    },
    "data": {
        "root": "./data/coco2017",
        "train_images": "train2017",
        "val_images": "val2017",
        "train_ann": "annotations/instances_train2017.json",
        "val_ann": "annotations/instances_val2017.json",
        "hflip_prob": 0.5,
    },
    "model": {
        "multi_scale": None,
        "backbone": {
            "depth": 50,
            "variant": "d",
            "num_stages": 4,
            "return_idx": [1, 2, 3],
            "freeze_at": 0,
            "freeze_norm": True,
            "pretrained": False,
        },
        "encoder": {
            "hidden_dim": 256,
            "nhead": 8,
            "dim_feedforward": 1024,
            "dropout": 0.0,
            "enc_act": "gelu",
            "use_encoder_idx": [2],
            "num_encoder_layers": 1,
            "pe_temperature": 10000,
            "expansion": 1.0,
            "depth_mult": 1.0,
            "act": "silu",
            "eval_spatial_size": [640, 640],
        },
        "decoder": {
            "hidden_dim": 256,
            "num_queries": 300,
            "num_levels": 3,
            "num_decoder_points": 4,
            "nhead": 8,
            "num_decoder_layers": 6,
            "dim_feedforward": 1024,
            "dropout": 0.0,
            "activation": "relu",
            "num_denoising": 100,
            "label_noise_ratio": 0.5,
            "box_noise_scale": 1.0,
            "learnt_init_query": False,
            "eval_idx": -1,
            "aux_loss": True,
        },
    },
    "criterion": {
        "matcher": {
            "cost_class": 2,
            "cost_bbox": 5,
            "cost_giou": 2,
            "use_focal_loss": True,
            "alpha": 0.25,
            "gamma": 2.0,
        },
        "alpha": 0.2,
        "gamma": 2.0,
        "eos_coef": 1e-4,
        "weight_dict": None,
        "losses": ["vfl", "boxes"],
    },
}


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _set_nested(cfg: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required for config parsing. Please install PyYAML.")
    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return content


def _load_yaml_with_includes(path: Path, visited: Optional[set] = None) -> Dict[str, Any]:
    """Load YAML config and resolve PyTorch-style __include__ recursively."""
    if visited is None:
        visited = set()
    real = str(path.resolve())
    if real in visited:
        raise ValueError(f"Recursive include detected: {path}")
    visited.add(real)

    current = _load_yaml_file(path)
    includes = current.pop("__include__", []) if isinstance(current, dict) else []
    if isinstance(includes, str):
        includes = [includes]

    merged: Dict[str, Any] = {}
    for include in includes:
        include_path = Path(include)
        if not include_path.is_absolute():
            include_path = (path.parent / include_path).resolve()
        base_cfg = _load_yaml_with_includes(include_path, visited=visited)
        _deep_merge(merged, base_cfg)

    if isinstance(current, dict):
        _deep_merge(merged, current)
    return merged


def _apply_legacy_config(raw_cfg: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    for key in ["seed", "output_dir", "input_size", "num_classes"]:
        if key in raw_cfg:
            cfg[key] = raw_cfg[key]

    if "epochs" in raw_cfg:
        cfg["train"]["epochs"] = int(raw_cfg["epochs"])
    if "batch_size" in raw_cfg:
        cfg["train"]["batch_size"] = int(raw_cfg["batch_size"])
    if "lr" in raw_cfg:
        cfg["train"]["lr"] = float(raw_cfg["lr"])
    if "weight_decay" in raw_cfg:
        cfg["train"]["weight_decay"] = float(raw_cfg["weight_decay"])

    if "data_root" in raw_cfg:
        cfg["data"]["root"] = raw_cfg["data_root"]
    if "train_ann" in raw_cfg:
        cfg["data"]["train_ann"] = raw_cfg["train_ann"]
    if "val_ann" in raw_cfg:
        cfg["data"]["val_ann"] = raw_cfg["val_ann"]

    if isinstance(raw_cfg.get("train"), dict):
        _deep_merge(cfg["train"], raw_cfg["train"])
    if isinstance(raw_cfg.get("eval"), dict):
        _deep_merge(cfg["eval"], raw_cfg["eval"])
    if isinstance(raw_cfg.get("data"), dict):
        _deep_merge(cfg["data"], raw_cfg["data"])
    if isinstance(raw_cfg.get("model"), dict):
        _deep_merge(cfg["model"], raw_cfg["model"])
    if isinstance(raw_cfg.get("criterion"), dict):
        _deep_merge(cfg["criterion"], raw_cfg["criterion"])

    if isinstance(raw_cfg.get("RTDETR"), dict):
        if "multi_scale" in raw_cfg["RTDETR"]:
            cfg["model"]["multi_scale"] = raw_cfg["RTDETR"]["multi_scale"]

    if isinstance(raw_cfg.get("PResNet"), dict):
        _deep_merge(cfg["model"]["backbone"], raw_cfg["PResNet"])
    if isinstance(raw_cfg.get("HybridEncoder"), dict):
        _deep_merge(cfg["model"]["encoder"], raw_cfg["HybridEncoder"])
    if isinstance(raw_cfg.get("RTDETRTransformer"), dict):
        _deep_merge(cfg["model"]["decoder"], raw_cfg["RTDETRTransformer"])

    if isinstance(raw_cfg.get("SetCriterion"), dict):
        set_criterion_cfg = raw_cfg["SetCriterion"]
        if "num_classes" in set_criterion_cfg:
            cfg["num_classes"] = int(set_criterion_cfg["num_classes"])
        if isinstance(set_criterion_cfg.get("matcher"), dict):
            matcher_cfg = copy.deepcopy(set_criterion_cfg["matcher"])
            nested_weight = matcher_cfg.pop("weight_dict", None)
            if isinstance(nested_weight, dict):
                _deep_merge(cfg["criterion"]["matcher"], nested_weight)
            _deep_merge(cfg["criterion"]["matcher"], matcher_cfg)
            if "use_focal" in matcher_cfg:
                cfg["criterion"]["matcher"]["use_focal_loss"] = bool(matcher_cfg["use_focal"])
        for key in ["alpha", "gamma", "eos_coef"]:
            if key in set_criterion_cfg:
                cfg["criterion"][key] = float(set_criterion_cfg[key])
        if isinstance(set_criterion_cfg.get("weight_dict"), dict):
            cfg["criterion"]["weight_dict"] = copy.deepcopy(set_criterion_cfg["weight_dict"])
        if isinstance(set_criterion_cfg.get("losses"), list):
            losses = []
            for item in set_criterion_cfg["losses"]:
                if item == "labels":
                    losses.append("vfl")
                else:
                    losses.append(item)
            cfg["criterion"]["losses"] = losses


def load_runtime_config(
    config_path: Optional[str],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    config_dir = Path.cwd()
    resolved_config_path = ""
    if config_path:
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = (Path.cwd() / config_file).resolve()
        if not config_file.is_file():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        raw_cfg = _load_yaml_with_includes(config_file)
        _apply_legacy_config(raw_cfg, cfg)
        config_dir = config_file.parent
        resolved_config_path = str(config_file)

    if overrides:
        for key, value in overrides.items():
            if value is None:
                continue
            _set_nested(cfg, key, value)

    cfg["_config_dir"] = str(config_dir)
    cfg["_config_path"] = resolved_config_path

    enc_eval_size = cfg["model"]["encoder"].get("eval_spatial_size")
    override_input_size = False
    if overrides:
        override_input_size = "input_size" in overrides
    if enc_eval_size is None or override_input_size:
        cfg["model"]["encoder"]["eval_spatial_size"] = [int(cfg["input_size"]), int(cfg["input_size"])]

    return cfg


def set_runtime(device: str = "cuda", seed: int = 42) -> None:
    device = (device or "cuda").lower()
    jt.flags.use_cuda = 1 if device == "cuda" else 0
    jt.flags.auto_mixed_precision_level = 0

    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, jt.Var):
        return x.numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _to_jt_var(x: Any) -> jt.Var:
    if isinstance(x, jt.Var):
        return x
    if isinstance(x, np.ndarray):
        return jt.array(x)
    return jt.array(np.asarray(x))


def _split_qkv_tensor(x: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = _to_numpy(x)
    if arr.ndim == 1:
        axis = 0
    elif arr.ndim == 2:
        axis = 0
    else:
        axis = 0
    if arr.shape[axis] % 3 != 0:
        raise ValueError(f"Cannot split qkv tensor with shape {arr.shape}")
    q, k, v = np.split(arr, 3, axis=axis)
    return q, k, v


def _adapt_state_dict_for_jittor(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt PyTorch-style attention keys for Jittor custom attention blocks."""
    if not isinstance(state_dict, dict):
        return state_dict
    adapted = dict(state_dict)
    for key, value in list(state_dict.items()):
        if key.endswith(".self_attn.in_proj_weight"):
            base = key[: -len("in_proj_weight")]
            try:
                q, k, v = _split_qkv_tensor(value)
            except Exception:
                continue
            adapted[f"{base}q_proj.weight"] = q
            adapted[f"{base}k_proj.weight"] = k
            adapted[f"{base}v_proj.weight"] = v
        elif key.endswith(".self_attn.in_proj_bias"):
            base = key[: -len("in_proj_bias")]
            try:
                q, k, v = _split_qkv_tensor(value)
            except Exception:
                continue
            adapted[f"{base}q_proj.bias"] = q
            adapted[f"{base}k_proj.bias"] = k
            adapted[f"{base}v_proj.bias"] = v
    return adapted


def _var_to_int(x: Any) -> int:
    arr = _to_numpy(x).reshape(-1)
    if arr.size == 0:
        return 0
    return int(arr[0])


def _cxcywh_to_xyxy_abs(cxcywh: np.ndarray, orig_h: float, orig_w: float) -> np.ndarray:
    if cxcywh.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cx = cxcywh[:, 0]
    cy = cxcywh[:, 1]
    bw = cxcywh[:, 2]
    bh = cxcywh[:, 3]
    x1 = (cx - bw * 0.5) * orig_w
    y1 = (cy - bh * 0.5) * orig_h
    x2 = (cx + bw * 0.5) * orig_w
    y2 = (cy + bh * 0.5) * orig_h
    out = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    out[:, 0::2] = np.clip(out[:, 0::2], 0.0, orig_w)
    out[:, 1::2] = np.clip(out[:, 1::2], 0.0, orig_h)
    return out


def _xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
    if xyxy.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = xyxy.copy().astype(np.float32)
    out[:, 2] = out[:, 2] - out[:, 0]
    out[:, 3] = out[:, 3] - out[:, 1]
    return out


def _normalize_prediction_arrays(
    scores: np.ndarray,
    labels: np.ndarray,
    boxes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    boxes = np.asarray(boxes, dtype=np.float32)

    if boxes.ndim == 1:
        boxes = boxes.reshape(-1, 4)
    elif boxes.ndim > 2:
        boxes = boxes.reshape(-1, boxes.shape[-1])

    if boxes.shape[-1] != 4:
        boxes = boxes[:, :4]

    count = min(len(scores), len(labels), len(boxes))
    if count <= 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 4), dtype=np.float32),
        )

    return scores[:count], labels[:count], boxes[:count]


class WarmupCosineScheduler:
    """Per-update warmup + cosine decay scheduler."""

    def __init__(
        self,
        optimizer,
        base_lr: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: int,
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))
        self.step_id = 0
        self.optimizer.lr = self.base_lr if self.warmup_steps <= 0 else 0.0

    def _compute_lr(self, step_id: int) -> float:
        if self.warmup_steps > 0 and step_id < self.warmup_steps:
            return self.base_lr * float(step_id + 1) / float(self.warmup_steps)

        if self.total_steps <= self.warmup_steps:
            return self.base_lr

        progress = float(step_id - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def step(self) -> float:
        lr = self._compute_lr(self.step_id)
        self.optimizer.lr = float(lr)
        self.step_id += 1
        return float(lr)

    def state_dict(self) -> Dict[str, Any]:
        return {"step_id": self.step_id}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self.step_id = int(state.get("step_id", self.step_id))
        self.optimizer.lr = float(self._compute_lr(self.step_id))


def _resolve_data_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    config_dir = Path(cfg["_config_dir"])
    data_root = _resolve_path(str(cfg["data"]["root"]), config_dir)
    return {
        "root": data_root,
        "train_images": _resolve_path(str(cfg["data"]["train_images"]), data_root),
        "val_images": _resolve_path(str(cfg["data"]["val_images"]), data_root),
        "train_ann": _resolve_path(str(cfg["data"]["train_ann"]), data_root),
        "val_ann": _resolve_path(str(cfg["data"]["val_ann"]), data_root),
    }


def build_datasets(
    cfg: Dict[str, Any],
    use_dummy_data: bool = False,
    max_samples: int = -1,
) -> Tuple[Any, Any]:
    if use_dummy_data:
        num_classes = int(cfg["num_classes"])
        train_samples = max(2, int(max_samples) if max_samples and max_samples > 0 else 8)
        val_samples = max(2, min(16, train_samples))
        train_dataset = DummyDetectionDataset(
            num_samples=train_samples,
            image_size=int(cfg["input_size"]),
            num_classes=num_classes,
            seed=int(cfg["seed"]),
        )
        val_dataset = DummyDetectionDataset(
            num_samples=val_samples,
            image_size=int(cfg["input_size"]),
            num_classes=num_classes,
            seed=int(cfg["seed"]) + 1000,
        )
        return train_dataset, val_dataset

    paths = _resolve_data_paths(cfg)
    train_dataset = CocoDetectionDataset(
        image_dir=str(paths["train_images"]),
        ann_file=str(paths["train_ann"]),
        image_size=int(cfg["input_size"]),
        is_train=True,
        hflip_prob=float(cfg["data"].get("hflip_prob", 0.5)),
        max_samples=int(max_samples),
    )
    val_dataset = CocoDetectionDataset(
        image_dir=str(paths["val_images"]),
        ann_file=str(paths["val_ann"]),
        image_size=int(cfg["input_size"]),
        is_train=False,
        hflip_prob=0.0,
        max_samples=int(max_samples),
    )
    return train_dataset, val_dataset


def build_model_components(
    cfg: Dict[str, Any],
    num_classes: Optional[int] = None,
):
    model_cfg = cfg["model"]
    backbone_cfg = model_cfg["backbone"]
    encoder_cfg = copy.deepcopy(model_cfg["encoder"])
    decoder_cfg = copy.deepcopy(model_cfg["decoder"])

    num_classes = int(num_classes if num_classes is not None else cfg["num_classes"])

    backbone = PResNet(
        depth=int(backbone_cfg.get("depth", 50)),
        variant=backbone_cfg.get("variant", "d"),
        num_stages=int(backbone_cfg.get("num_stages", 4)),
        return_idx=list(backbone_cfg.get("return_idx", [1, 2, 3])),
        freeze_at=int(backbone_cfg.get("freeze_at", 0)),
        freeze_norm=bool(backbone_cfg.get("freeze_norm", True)),
        pretrained=bool(backbone_cfg.get("pretrained", False)),
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

    model = RTDETR(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        multi_scale=model_cfg.get("multi_scale"),
    )

    matcher_cfg = cfg["criterion"]["matcher"]
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

    weight_dict = cfg["criterion"].get("weight_dict")
    if isinstance(weight_dict, dict):
        weight_dict = copy.deepcopy(weight_dict)
        if "loss_focal" in weight_dict and "loss_vfl" not in weight_dict:
            weight_dict["loss_vfl"] = weight_dict["loss_focal"]
        if "loss_ce" in weight_dict and "loss_vfl" not in weight_dict:
            weight_dict["loss_vfl"] = weight_dict["loss_ce"]
    else:
        weight_dict = None

    losses = cfg["criterion"].get("losses", ["vfl", "boxes"])
    losses = ["vfl" if loss == "labels" else loss for loss in losses]

    criterion = build_criterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        alpha=float(cfg["criterion"].get("alpha", 0.2)),
        gamma=float(cfg["criterion"].get("gamma", 2.0)),
        eos_coef=float(cfg["criterion"].get("eos_coef", 1e-4)),
    )

    postprocessor = build_postprocessor(
        num_classes=num_classes,
        use_focal_loss=True,
        num_top_queries=int(cfg["eval"].get("topk", 300)),
        remap_mscoco_category=False,
    )

    return model, criterion, postprocessor


def build_optimizer(model, cfg: Dict[str, Any]):
    params = [p for p in model.parameters() if getattr(p, "requires_grad", True)]
    if not params:
        params = list(model.parameters())
    optimizer = jt.optim.Adam(
        params,
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    return optimizer


def _extract_model_state(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        if "model" in payload and isinstance(payload["model"], dict):
            return payload["model"]
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        if "ema" in payload and isinstance(payload["ema"], dict):
            module_state = payload["ema"].get("module")
            if isinstance(module_state, dict):
                return module_state
        looks_like_state_dict = all(
            isinstance(v, (jt.Var, np.ndarray)) for v in payload.values()
        ) if payload else False
        if looks_like_state_dict:
            return payload
    return {}


def load_model_state(model, state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(state_dict, dict):
        return {"loaded": 0, "missing": len(model.state_dict()), "unexpected": 0, "mismatched": 0}

    state_dict = _adapt_state_dict_for_jittor(state_dict)
    current_state = model.state_dict()
    to_load = {}
    unexpected = 0
    mismatched = 0

    for key, value in state_dict.items():
        if key not in current_state:
            unexpected += 1
            continue

        target_shape = tuple(current_state[key].shape)
        value_var = _to_jt_var(value)
        if tuple(value_var.shape) != target_shape:
            mismatched += 1
            continue
        to_load[key] = value_var

    model.load_state_dict(to_load)
    missing = len(current_state) - len(to_load)
    return {
        "loaded": len(to_load),
        "missing": missing,
        "unexpected": unexpected,
        "mismatched": mismatched,
    }


def load_checkpoint(
    model,
    checkpoint_path: str,
    optimizer=None,
    scheduler: Optional[WarmupCosineScheduler] = None,
    ema: Optional[ModelEMA] = None,
    load_optimizer_state: bool = True,
) -> Dict[str, Any]:
    payload = jt.load(checkpoint_path)
    state_dict = _extract_model_state(payload)
    model_info = load_model_state(model, state_dict)

    start_epoch = 0
    global_step = 0
    best_metric = -1.0
    history = []

    if isinstance(payload, dict):
        if load_optimizer_state and optimizer is not None and isinstance(payload.get("optimizer"), dict):
            try:
                optimizer.load_state_dict(payload["optimizer"])
            except Exception as e:
                print(f"[resume] optimizer state load skipped: {e}")

        if scheduler is not None and isinstance(payload.get("scheduler"), dict):
            scheduler.load_state_dict(payload["scheduler"])

        if ema is not None and isinstance(payload.get("ema"), dict):
            try:
                ema.load_state_dict(payload["ema"])
            except Exception as e:
                print(f"[resume] ema state load skipped: {e}")

        start_epoch = int(payload.get("epoch", -1)) + 1
        global_step = int(payload.get("global_step", 0))
        best_metric = float(payload.get("best_metric", -1.0))
        history = payload.get("history", [])

    return {
        "model_info": model_info,
        "start_epoch": max(0, start_epoch),
        "global_step": max(0, global_step),
        "best_metric": best_metric,
        "history": history if isinstance(history, list) else [],
    }


def _build_coco_predictions(dataset, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    coco_preds: List[Dict[str, Any]] = []
    for pred in predictions:
        image_id = int(pred["image_id"])
        boxes_xyxy = pred["boxes"]
        labels = pred["labels"]
        scores = pred["scores"]
        boxes_xywh = _xyxy_to_xywh(boxes_xyxy)
        for box, label, score in zip(boxes_xywh, labels, scores):
            coco_preds.append(
                {
                    "image_id": image_id,
                    "category_id": int(dataset.label_to_category(int(label))),
                    "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "score": float(score),
                }
            )
    return coco_preds


def _bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    denom = area_a + area_b - inter + 1e-8
    return inter / denom


def compute_fallback_metrics(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    target_map = {int(t["image_id"]): t for t in targets}
    tp, fp, fn = 0, 0, 0
    total_scores: List[float] = []

    for pred in predictions:
        image_id = int(pred["image_id"])
        gt = target_map.get(
            image_id,
            {"boxes": np.zeros((0, 4), dtype=np.float32), "labels": np.zeros((0,), dtype=np.int64)},
        )
        pboxes = pred["boxes"]
        plabels = pred["labels"]
        pscores = pred["scores"]
        total_scores.extend(pscores.tolist())

        gboxes = gt["boxes"]
        glabels = gt["labels"]
        matched = np.zeros((len(gboxes),), dtype=bool)

        if len(pscores) > 0:
            order = np.argsort(-pscores)
            pboxes = pboxes[order]
            plabels = plabels[order]

        for pbox, plabel in zip(pboxes, plabels):
            best_iou = -1.0
            best_idx = -1
            for idx, (gbox, glabel) in enumerate(zip(gboxes, glabels)):
                if matched[idx] or int(glabel) != int(plabel):
                    continue
                iou = _bbox_iou_xyxy(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0 and best_iou >= iou_threshold:
                tp += 1
                matched[best_idx] = True
            else:
                fp += 1

        fn += int((~matched).sum())

    precision = float(tp) / float(tp + fp + 1e-8)
    recall = float(tp) / float(tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / float(precision + recall + 1e-8)
    mean_score = float(np.mean(total_scores)) if total_scores else 0.0

    return {
        "precision@0.5": precision,
        "recall@0.5": recall,
        "f1@0.5": f1,
        "mean_score": mean_score,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def evaluate_predictions(
    dataset,
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    output_dir: Path,
    prefix: str = "eval",
) -> Tuple[Dict[str, Any], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_json = output_dir / f"{prefix}_predictions.json"

    coco_preds = _build_coco_predictions(dataset, predictions)
    with open(pred_json, "w", encoding="utf-8") as f:
        json.dump(coco_preds, f)

    metrics: Dict[str, Any] = {}
    ann_file = getattr(dataset, "ann_file", "")

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        if not ann_file or not os.path.isfile(ann_file):
            raise FileNotFoundError(f"annotation file unavailable: {ann_file}")

        if len(coco_preds) == 0:
            metrics.update(
                {
                    "metric_source": "coco",
                    "AP": 0.0,
                    "AP50": 0.0,
                    "AP75": 0.0,
                    "APs": 0.0,
                    "APm": 0.0,
                    "APl": 0.0,
                    "note": "no predictions generated",
                }
            )
            return metrics, pred_json

        coco_gt = COCO(ann_file)
        coco_dt = coco_gt.loadRes(coco_preds)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats
        metrics.update(
            {
                "metric_source": "coco",
                "AP": float(stats[0]),
                "AP50": float(stats[1]),
                "AP75": float(stats[2]),
                "APs": float(stats[3]),
                "APm": float(stats[4]),
                "APl": float(stats[5]),
            }
        )
        return metrics, pred_json
    except Exception as e:
        fallback = compute_fallback_metrics(predictions, targets, iou_threshold=0.5)
        fallback["metric_source"] = "fallback"
        fallback["fallback_reason"] = str(e)
        return fallback, pred_json


def evaluate_model(
    model,
    criterion,
    postprocessor,
    data_loader: SimpleDetectionDataLoader,
    dataset,
    output_dir: Path,
    score_threshold: float = 0.3,
    prefix: str = "eval",
) -> Dict[str, Any]:
    model.eval()
    criterion.eval()

    loss_sums: Dict[str, float] = {}
    batch_count = 0

    predictions: List[Dict[str, Any]] = []
    targets_for_metrics: List[Dict[str, Any]] = []

    with jt.no_grad():
        for images, targets in data_loader:
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            for key, value in loss_dict.items():
                loss_sums[key] = loss_sums.get(key, 0.0) + float(value.item())
            batch_count += 1

            target_sizes = jt.stack([t["orig_size"].float32() for t in targets], dim=0)
            results = postprocessor(outputs, target_sizes=target_sizes)

            for target, result in zip(targets, results):
                image_id = _var_to_int(target["image_id"])
                scores, labels, boxes = _normalize_prediction_arrays(
                    _to_numpy(result["scores"]),
                    _to_numpy(result["labels"]),
                    _to_numpy(result["boxes"]),
                )

                keep = scores >= float(score_threshold)
                scores = scores[keep]
                labels = labels[keep]
                boxes = boxes[keep]

                predictions.append(
                    {
                        "image_id": image_id,
                        "scores": scores,
                        "labels": labels,
                        "boxes": boxes,
                    }
                )

                orig_size = _to_numpy(target["orig_size"]).astype(np.float32).reshape(-1)
                gt_boxes_cxcywh = _to_numpy(target["boxes"]).astype(np.float32)
                gt_boxes_xyxy = _cxcywh_to_xyxy_abs(
                    gt_boxes_cxcywh,
                    orig_h=float(orig_size[0]),
                    orig_w=float(orig_size[1]),
                )
                gt_labels = _to_numpy(target["labels"]).astype(np.int64)

                targets_for_metrics.append(
                    {
                        "image_id": image_id,
                        "boxes": gt_boxes_xyxy,
                        "labels": gt_labels,
                    }
                )

    avg_losses = {
        f"val_{k}": v / float(max(1, batch_count))
        for k, v in loss_sums.items()
    }
    avg_losses["val_loss"] = float(sum(avg_losses.values()))

    metric_stats, pred_json = evaluate_predictions(
        dataset=dataset,
        predictions=predictions,
        targets=targets_for_metrics,
        output_dir=output_dir,
        prefix=prefix,
    )

    merged = {}
    merged.update(avg_losses)
    merged.update(metric_stats)
    merged["prediction_file"] = str(pred_json)
    merged["num_eval_images"] = len(predictions)
    return merged


def _prepare_image_tensor(image_path: str, input_size: int) -> Tuple[Image.Image, jt.Var, jt.Var]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    resized = image.resize((int(input_size), int(input_size)), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    image_tensor = jt.array(arr).float32()
    target_sizes = jt.array(np.asarray([[height, width]], dtype=np.float32))
    return image, image_tensor, target_sizes


def _draw_detections(
    image: Image.Image,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    score_threshold: float,
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(boxes, labels, scores):
        if float(score) < float(score_threshold):
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1 + 1, y1 + 1), f"{int(label)}:{float(score):.2f}", fill="yellow")
    return image


def infer_single_image(
    model,
    postprocessor,
    image_path: str,
    output_dir: Path,
    input_size: int,
    score_threshold: float = 0.3,
    export_regression: bool = False,
) -> Dict[str, Any]:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    image, image_tensor, target_sizes = _prepare_image_tensor(image_path, input_size=input_size)
    with jt.no_grad():
        raw_outputs = model(image_tensor)
        processed = postprocessor(raw_outputs, target_sizes=target_sizes)[0]

    scores, labels, boxes = _normalize_prediction_arrays(
        _to_numpy(processed["scores"]),
        _to_numpy(processed["labels"]),
        _to_numpy(processed["boxes"]),
    )
    keep = scores >= float(score_threshold)
    scores = scores[keep]
    labels = labels[keep]
    boxes = boxes[keep]

    image_name = Path(image_path).stem
    json_path = output_dir / f"{image_name}_detections.json"
    vis_path = output_dir / f"{image_name}_vis.jpg"
    raw_path = output_dir / f"{image_name}_raw_outputs.npz"

    serializable = []
    for box, label, score in zip(boxes, labels, scores):
        serializable.append(
            {
                "label": int(label),
                "score": float(score),
                "bbox_xyxy": [float(v) for v in box.tolist()],
            }
        )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image_path": image_path,
                "num_detections": len(serializable),
                "detections": serializable,
            },
            f,
            indent=2,
        )

    vis_image = _draw_detections(image=image.copy(), boxes=boxes, labels=labels, scores=scores, score_threshold=score_threshold)
    vis_image.save(vis_path)

    if export_regression:
        np.savez(
            raw_path,
            pred_logits=_to_numpy(raw_outputs["pred_logits"]).astype(np.float32),
            pred_boxes=_to_numpy(raw_outputs["pred_boxes"]).astype(np.float32),
        )

    return {
        "num_detections": int(len(serializable)),
        "json_path": str(json_path),
        "vis_path": str(vis_path),
        "raw_output_path": str(raw_path) if export_regression else "",
    }


def _append_jsonl(log_file: Path, record: Dict[str, Any]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def train_one_epoch(
    model,
    criterion,
    data_loader: SimpleDetectionDataLoader,
    optimizer,
    scheduler: Optional[WarmupCosineScheduler],
    ema: Optional[ModelEMA],
    epoch: int,
    global_step: int,
    grad_accum_steps: int = 1,
    print_freq: int = 10,
) -> Tuple[Dict[str, float], int]:
    model.train()
    criterion.train()
    optimizer.zero_grad()

    start_time = time.time()
    num_batches = max(1, len(data_loader))

    loss_sums: Dict[str, float] = {}
    total_loss_sum = 0.0
    update_count = 0
    current_lr = float(optimizer.lr)

    for step, (images, targets) in enumerate(data_loader):
        outputs = model(images, targets)
        loss_dict = criterion(outputs, targets)
        total_loss = sum(loss_dict.values())
        total_loss_value = float(total_loss.item())

        if not math.isfinite(total_loss_value):
            raise RuntimeError(f"Non-finite loss detected: {total_loss_value}, loss_dict={loss_dict}")

        scaled_loss = total_loss / float(max(1, grad_accum_steps))
        optimizer.backward(scaled_loss)

        should_update = ((step + 1) % max(1, grad_accum_steps) == 0) or ((step + 1) == num_batches)
        if should_update:
            optimizer.step()
            optimizer.zero_grad()
            jt.sync_all()
            if ema is not None:
                ema.update(model)
            if scheduler is not None:
                current_lr = scheduler.step()
            else:
                current_lr = float(optimizer.lr)
            global_step += 1
            update_count += 1

        total_loss_sum += total_loss_value
        for key, value in loss_dict.items():
            loss_sums[key] = loss_sums.get(key, 0.0) + float(value.item())

        if (step + 1) % max(1, print_freq) == 0 or step == 0 or (step + 1) == num_batches:
            print(
                f"[train] epoch={epoch} step={step + 1}/{num_batches} "
                f"loss={total_loss_value:.4f} lr={current_lr:.6e}"
            )

    elapsed = time.time() - start_time
    stats = {
        "epoch": float(epoch),
        "train_loss": total_loss_sum / float(num_batches),
        "lr": float(current_lr),
        "updates": float(update_count),
        "epoch_time_sec": float(elapsed),
    }
    for key, value in loss_sums.items():
        stats[f"train_{key}"] = value / float(num_batches)
    return stats, global_step


def _make_last_checkpoint(
    model,
    optimizer,
    scheduler: Optional[WarmupCosineScheduler],
    ema: Optional[ModelEMA],
    epoch: int,
    global_step: int,
    best_metric: float,
    history: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    ckpt = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else {},
        "history": history,
        "config": config,
    }
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    return ckpt


def _make_best_checkpoint(
    eval_model,
    epoch: int,
    best_metric: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "model": eval_model.state_dict(),
        "config": config,
    }


def run_training(
    cfg: Dict[str, Any],
    output_dir: str,
    resume_path: str = "",
    checkpoint_path: str = "",
    use_dummy_data: bool = False,
    max_samples: int = -1,
) -> Dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_file = output / "train_log.jsonl"

    train_dataset, val_dataset = build_datasets(cfg, use_dummy_data=use_dummy_data, max_samples=max_samples)
    num_classes = int(getattr(train_dataset, "num_classes", cfg["num_classes"]))
    if int(cfg["num_classes"]) != num_classes:
        print(f"[config] override num_classes {cfg['num_classes']} -> {num_classes} (from dataset)")
        cfg["num_classes"] = num_classes

    model, criterion, postprocessor = build_model_components(cfg, num_classes=num_classes)
    optimizer = build_optimizer(model, cfg)

    train_loader = SimpleDetectionDataLoader(
        train_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        drop_last=False,
    )
    val_loader = SimpleDetectionDataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    grad_accum_steps = max(1, int(cfg["train"].get("grad_accum_steps", 1)))
    total_updates = int(math.ceil(len(train_loader) / float(grad_accum_steps))) * int(cfg["train"]["epochs"])
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        base_lr=float(cfg["train"]["lr"]),
        min_lr=float(cfg["train"].get("min_lr", cfg["train"]["lr"] * 0.01)),
        warmup_steps=int(cfg["train"].get("warmup_steps", 0)),
        total_steps=max(1, total_updates),
    )

    ema = None
    if bool(cfg["train"].get("ema", True)):
        ema = ModelEMA(
            model=model,
            decay=float(cfg["train"].get("ema_decay", 0.9999)),
            warmups=int(cfg["train"].get("ema_warmups", 2000)),
        )

    if checkpoint_path:
        warm_start = load_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            optimizer=None,
            scheduler=None,
            ema=None,
            load_optimizer_state=False,
        )
        print(f"[checkpoint] warm start loaded: {warm_start['model_info']}")

    start_epoch = 0
    global_step = 0
    best_metric = -1.0
    history: List[Dict[str, Any]] = []

    if resume_path:
        resume_state = load_checkpoint(
            model=model,
            checkpoint_path=resume_path,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            load_optimizer_state=True,
        )
        print(f"[resume] model load summary: {resume_state['model_info']}")
        start_epoch = resume_state["start_epoch"]
        global_step = resume_state["global_step"]
        best_metric = resume_state["best_metric"]
        history = resume_state["history"]

    last_ckpt_path = output / "last_ckpt.pkl"
    best_ckpt_path = output / "best_ckpt.pkl"

    epochs = int(cfg["train"]["epochs"])
    eval_every = max(1, int(cfg["train"].get("eval_every", 1)))
    save_every = max(1, int(cfg["train"].get("save_every", 1)))
    print_freq = max(1, int(cfg["train"].get("print_freq", 10)))

    print(
        f"[train] samples(train/val)=({len(train_dataset)}/{len(val_dataset)}) "
        f"epochs={epochs} batch_size={cfg['train']['batch_size']} "
        f"grad_accum={grad_accum_steps}"
    )

    for epoch in range(start_epoch, epochs):
        train_stats, global_step = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            epoch=epoch,
            global_step=global_step,
            grad_accum_steps=grad_accum_steps,
            print_freq=print_freq,
        )

        eval_stats: Dict[str, Any] = {}
        eval_model = ema.module if ema is not None else model
        if (epoch + 1) % eval_every == 0:
            eval_stats = evaluate_model(
                model=eval_model,
                criterion=criterion,
                postprocessor=postprocessor,
                data_loader=val_loader,
                dataset=val_dataset,
                output_dir=output,
                score_threshold=float(cfg["eval"].get("score_threshold", 0.3)),
                prefix=f"epoch_{epoch + 1}",
            )

        if "AP" in eval_stats:
            metric = float(eval_stats["AP"])
        elif "f1@0.5" in eval_stats:
            metric = float(eval_stats["f1@0.5"])
        else:
            metric = -float(eval_stats.get("val_loss", train_stats["train_loss"]))

        if metric > best_metric:
            best_metric = metric
            best_payload = _make_best_checkpoint(
                eval_model=eval_model,
                epoch=epoch,
                best_metric=best_metric,
                config=cfg,
            )
            jt.save(best_payload, str(best_ckpt_path))

        record = {}
        record.update(train_stats)
        record.update(eval_stats)
        record["best_metric"] = float(best_metric)
        record["epoch_index"] = int(epoch)
        history.append(record)
        _append_jsonl(log_file, record)

        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            last_payload = _make_last_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                history=history,
                config=cfg,
            )
            jt.save(last_payload, str(last_ckpt_path))

        print(
            f"[epoch {epoch + 1}/{epochs}] train_loss={train_stats['train_loss']:.4f} "
            f"best_metric={best_metric:.4f}"
        )

    return {
        "last_ckpt": str(last_ckpt_path),
        "best_ckpt": str(best_ckpt_path),
        "history_len": len(history),
        "best_metric": float(best_metric),
    }


def run_evaluation(
    cfg: Dict[str, Any],
    checkpoint_path: str,
    output_dir: str,
    use_dummy_data: bool = False,
    max_samples: int = -1,
) -> Dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    _, val_dataset = build_datasets(cfg, use_dummy_data=use_dummy_data, max_samples=max_samples)
    num_classes = int(getattr(val_dataset, "num_classes", cfg["num_classes"]))
    model, criterion, postprocessor = build_model_components(cfg, num_classes=num_classes)

    if checkpoint_path:
        load_state = load_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            optimizer=None,
            scheduler=None,
            ema=None,
            load_optimizer_state=False,
        )
        print(f"[eval] checkpoint loaded: {load_state['model_info']}")

    val_loader = SimpleDetectionDataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    stats = evaluate_model(
        model=model,
        criterion=criterion,
        postprocessor=postprocessor,
        data_loader=val_loader,
        dataset=val_dataset,
        output_dir=output,
        score_threshold=float(cfg["eval"].get("score_threshold", 0.3)),
        prefix="eval",
    )
    _append_jsonl(output / "eval_log.jsonl", stats)
    return stats


def run_inference(
    cfg: Dict[str, Any],
    checkpoint_path: str,
    image_path: str,
    output_dir: str,
    export_regression: bool = False,
) -> Dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    model, _, postprocessor = build_model_components(cfg, num_classes=int(cfg["num_classes"]))

    if checkpoint_path:
        load_state = load_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            optimizer=None,
            scheduler=None,
            ema=None,
            load_optimizer_state=False,
        )
        print(f"[infer] checkpoint loaded: {load_state['model_info']}")

    result = infer_single_image(
        model=model,
        postprocessor=postprocessor,
        image_path=image_path,
        output_dir=output,
        input_size=int(cfg["input_size"]),
        score_threshold=float(cfg["eval"].get("score_threshold", 0.3)),
        export_regression=bool(export_regression),
    )
    _append_jsonl(output / "infer_log.jsonl", result)
    return result
