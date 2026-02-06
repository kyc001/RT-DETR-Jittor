#!/usr/bin/env python3
"""Single-image inference for RT-DETR Jittor migration validation."""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.core.engine import load_runtime_config, run_inference, set_runtime


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RT-DETR Jittor inference")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/rtdetr/rtdetr_r50vd_6x_coco.yml",
        help="Path to YAML config",
    )
    parser.add_argument("--checkpoint", "-r", type=str, default="", help="Checkpoint path")
    parser.add_argument("--image", "-i", type=str, required=True, help="Input image path")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument(
        "--export-regression",
        action="store_true",
        help="Export raw logits/boxes for regression comparison",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    overrides = {}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.seed is not None:
        overrides["seed"] = int(args.seed)
    if args.score_threshold is not None:
        overrides["eval.score_threshold"] = float(args.score_threshold)
    if args.input_size is not None:
        overrides["input_size"] = int(args.input_size)
    if args.num_classes is not None:
        overrides["num_classes"] = int(args.num_classes)

    cfg = load_runtime_config(args.config, overrides=overrides)
    set_runtime(device=args.device, seed=int(cfg["seed"]))

    result = run_inference(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        output_dir=str(cfg["output_dir"]),
        export_regression=bool(args.export_regression),
    )

    print("Inference finished:")
    print(f"  detections: {result['num_detections']}")
    print(f"  json: {result['json_path']}")
    print(f"  vis: {result['vis_path']}")
    if result.get("raw_output_path"):
        print(f"  raw: {result['raw_output_path']}")


if __name__ == "__main__":
    parser = build_argparser()
    main(parser.parse_args())
