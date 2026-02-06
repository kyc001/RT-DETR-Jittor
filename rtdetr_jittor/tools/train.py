#!/usr/bin/env python3
"""Train RT-DETR in Jittor with reusable migration workflow."""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.core.engine import load_runtime_config, run_training, set_runtime


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RT-DETR Jittor training")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/rtdetr/rtdetr_r50vd_6x_coco.yml",
        help="Path to YAML config",
    )
    parser.add_argument("--output-dir", type=str, default="", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--print-freq", type=int, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA")

    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, default="", help="Warm start checkpoint")

    parser.add_argument("--dummy-data", action="store_true", help="Use synthetic dataset")
    parser.add_argument("--max-samples", type=int, default=-1, help="Limit dataset samples")
    return parser


def main(args: argparse.Namespace) -> None:
    overrides = {}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.seed is not None:
        overrides["seed"] = int(args.seed)
    if args.epochs is not None:
        overrides["train.epochs"] = int(args.epochs)
    if args.batch_size is not None:
        overrides["train.batch_size"] = int(args.batch_size)
    if args.lr is not None:
        overrides["train.lr"] = float(args.lr)
    if args.weight_decay is not None:
        overrides["train.weight_decay"] = float(args.weight_decay)
    if args.warmup_steps is not None:
        overrides["train.warmup_steps"] = int(args.warmup_steps)
    if args.grad_accum_steps is not None:
        overrides["train.grad_accum_steps"] = int(args.grad_accum_steps)
    if args.save_every is not None:
        overrides["train.save_every"] = int(args.save_every)
    if args.eval_every is not None:
        overrides["train.eval_every"] = int(args.eval_every)
    if args.print_freq is not None:
        overrides["train.print_freq"] = int(args.print_freq)
    if args.input_size is not None:
        overrides["input_size"] = int(args.input_size)
    if args.num_classes is not None:
        overrides["num_classes"] = int(args.num_classes)
    if args.score_threshold is not None:
        overrides["eval.score_threshold"] = float(args.score_threshold)
    if args.no_ema:
        overrides["train.ema"] = False

    cfg = load_runtime_config(args.config, overrides=overrides)
    set_runtime(device=args.device, seed=int(cfg["seed"]))

    result = run_training(
        cfg=cfg,
        output_dir=str(cfg["output_dir"]),
        resume_path=args.resume,
        checkpoint_path=args.checkpoint,
        use_dummy_data=bool(args.dummy_data),
        max_samples=int(args.max_samples),
    )

    print("Training finished:")
    print(f"  last_ckpt: {result['last_ckpt']}")
    print(f"  best_ckpt: {result['best_ckpt']}")
    print(f"  best_metric: {result['best_metric']:.6f}")
    print(f"  history_len: {result['history_len']}")


if __name__ == "__main__":
    parser = build_argparser()
    main(parser.parse_args())
