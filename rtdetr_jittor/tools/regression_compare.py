#!/usr/bin/env python3
"""Cross-framework regression comparison between PyTorch and Jittor RT-DETR."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


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


def _load_runtime_cfg(config_path: str, input_size: int, num_classes: int, seed: int) -> Dict[str, Any]:
    sys.path.insert(0, str(JITTOR_ROOT))
    from src.core.engine import load_runtime_config

    overrides = {
        "input_size": int(input_size),
        "num_classes": int(num_classes),
        "seed": int(seed),
        "model.multi_scale": None,
    }
    cfg = load_runtime_config(config_path, overrides=overrides)
    return cfg


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
    matcher = HungarianMatcher(
        weight_dict={"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
        use_focal_loss=True,
        alpha=0.25,
        gamma=2.0,
    )
    weight_dict = {"loss_vfl": 1, "loss_bbox": 5, "loss_giou": 2}
    for i in range(int(decoder_cfg.get("num_decoder_layers", 6))):
        weight_dict.update({f"loss_vfl_aux_{i}": 1, f"loss_bbox_aux_{i}": 5, f"loss_giou_aux_{i}": 2})
    criterion = SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        losses=["vfl", "boxes"],
        alpha=0.2,
        gamma=2.0,
        eos_coef=1e-4,
        num_classes=num_classes,
    )

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

    model.eval()
    with torch.no_grad():
        out_eval = model(x)

    model.train()
    out_train = model(x, targets_torch)
    loss_dict = criterion(out_train, targets_torch)

    with torch.no_grad():
        matcher_indices = criterion.matcher(
            {"pred_logits": out_train["pred_logits"], "pred_boxes": out_train["pred_boxes"]},
            targets_torch,
        )

    np.savez(
        args.output_npz,
        pred_logits=out_eval["pred_logits"].detach().cpu().numpy().astype(np.float32),
        pred_boxes=out_eval["pred_boxes"].detach().cpu().numpy().astype(np.float32),
    )

    losses = {k: float(v.detach().cpu().item()) for k, v in loss_dict.items()}
    matcher_dump = []
    for src_idx, tgt_idx in matcher_indices:
        matcher_dump.append(
            {
                "src": src_idx.detach().cpu().numpy().astype(np.int64).tolist(),
                "tgt": tgt_idx.detach().cpu().numpy().astype(np.int64).tolist(),
            }
        )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"loss": losses, "matcher": matcher_dump}, f, indent=2)

    torch.save({"model": model.state_dict()}, args.pt_state_path)


def _worker_jittor(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(JITTOR_ROOT))

    import jittor as jt

    from src.core.engine import (
        build_model_components,
        load_model_state,
        load_runtime_config,
        set_runtime,
    )

    with open(args.cfg_json, "r", encoding="utf-8") as f:
        cfg_payload = json.load(f)
    with open(args.targets_json, "r", encoding="utf-8") as f:
        targets = _deserialize_targets(json.load(f))
    input_np = np.load(args.input_npy).astype(np.float32)

    cfg = load_runtime_config(
        cfg_payload["_config_path"] if cfg_payload.get("_config_path") else None,
        overrides={
            "seed": int(cfg_payload["seed"]),
            "input_size": int(cfg_payload["input_size"]),
            "num_classes": int(cfg_payload["num_classes"]),
            "model.multi_scale": None,
            "model.backbone.depth": int(cfg_payload["model"]["backbone"]["depth"]),
            "model.decoder.num_decoder_layers": int(cfg_payload["model"]["decoder"]["num_decoder_layers"]),
            "model.encoder.eval_spatial_size": cfg_payload["model"]["encoder"]["eval_spatial_size"],
        },
    )
    set_runtime(device="cpu", seed=int(cfg["seed"]))

    model, criterion, _ = build_model_components(cfg, num_classes=int(cfg["num_classes"]))
    state = jt.load(args.jt_state_path)
    load_info = load_model_state(model, state)

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

    model.eval()
    with jt.no_grad():
        out_eval = model(x)

    model.train()
    out_train = model(x, targets_jt)
    loss_dict = criterion(out_train, targets_jt)

    with jt.no_grad():
        matcher_indices = criterion.matcher(
            {"pred_logits": out_train["pred_logits"], "pred_boxes": out_train["pred_boxes"]},
            targets_jt,
        )

    np.savez(
        args.output_npz,
        pred_logits=out_eval["pred_logits"].numpy().astype(np.float32),
        pred_boxes=out_eval["pred_boxes"].numpy().astype(np.float32),
    )

    losses = {k: float(v.item()) for k, v in loss_dict.items()}
    matcher_dump = []
    for src_idx, tgt_idx in matcher_indices:
        matcher_dump.append(
            {
                "src": src_idx.numpy().astype(np.int64).tolist(),
                "tgt": tgt_idx.numpy().astype(np.int64).tolist(),
            }
        )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"loss": losses, "matcher": matcher_dump, "load_info": load_info}, f, indent=2)


def _compare_outputs(pt_npz: Path, jt_npz: Path, pt_json: Path, jt_json: Path) -> Dict[str, Any]:
    pt_data = np.load(pt_npz)
    jt_data = np.load(jt_npz)

    pt_logits = pt_data["pred_logits"]
    jt_logits = jt_data["pred_logits"]
    pt_boxes = pt_data["pred_boxes"]
    jt_boxes = jt_data["pred_boxes"]

    logits_abs = np.abs(pt_logits - jt_logits)
    boxes_abs = np.abs(pt_boxes - jt_boxes)

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

    matcher_same = pt_meta.get("matcher", []) == jt_meta.get("matcher", [])
    report = {
        "logits": {
            "max_abs": float(logits_abs.max()),
            "mean_abs": float(logits_abs.mean()),
            "shape": list(pt_logits.shape),
        },
        "boxes": {
            "max_abs": float(boxes_abs.max()),
            "mean_abs": float(boxes_abs.mean()),
            "shape": list(pt_boxes.shape),
        },
        "loss": loss_diff,
        "matcher_exact_match": bool(matcher_same),
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
