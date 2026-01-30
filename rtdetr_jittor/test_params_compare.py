#!/usr/bin/env python3
"""精确参数量对比 - 排除BatchNorm running stats"""

import sys
sys.path.insert(0, '/wanyuhao/keyunchao/project/RT-DETR-Jittor-main/RT-DETR-Jittor-main/rtdetr_jittor')

import jittor as jt
import jittor.nn as nn
jt.flags.use_cuda = 0

def count_params_jittor(model):
    """Jittor默认方式（包含running stats）"""
    return sum(p.numel() for p in model.parameters())

def count_params_pytorch_style(model):
    """模拟PyTorch的参数统计方式（排除running stats）"""
    total = 0
    for name, param in model.named_parameters():
        # 排除running_mean和running_var
        if 'running_mean' in name or 'running_var' in name:
            continue
        total += param.numel()
    return total

def count_bn_running_stats(model):
    """统计BatchNorm running stats的参数量"""
    total = 0
    for name, param in model.named_parameters():
        if 'running_mean' in name or 'running_var' in name:
            total += param.numel()
    return total

print("="*60)
print("RT-DETR 参数量对比 (Jittor vs PyTorch统计方式)")
print("="*60)

# 1. Backbone
from src.nn.backbone.resnet import PResNet
backbone = PResNet(depth=50, variant='d', return_idx=[1, 2, 3])

bb_jt = count_params_jittor(backbone)
bb_pt = count_params_pytorch_style(backbone)
bb_bn = count_bn_running_stats(backbone)

print(f"\nBackbone (PResNet50-D):")
print(f"  Jittor计数 (含running stats): {bb_jt:>12,} ({bb_jt/1e6:.2f}M)")
print(f"  PyTorch计数 (不含running stats): {bb_pt:>12,} ({bb_pt/1e6:.2f}M)")
print(f"  running stats差异:            {bb_bn:>12,} ({bb_bn/1e6:.2f}M)")

# 2. Encoder
from src.zoo.rtdetr.hybrid_encoder import HybridEncoder
encoder = HybridEncoder(
    in_channels=[512, 1024, 2048],
    feat_strides=[8, 16, 32],
    hidden_dim=256,
    nhead=8,
    dim_feedforward=1024,
    use_encoder_idx=[2],
    num_encoder_layers=1,
)

enc_jt = count_params_jittor(encoder)
enc_pt = count_params_pytorch_style(encoder)
enc_bn = count_bn_running_stats(encoder)

print(f"\nHybridEncoder:")
print(f"  Jittor计数 (含running stats): {enc_jt:>12,} ({enc_jt/1e6:.2f}M)")
print(f"  PyTorch计数 (不含running stats): {enc_pt:>12,} ({enc_pt/1e6:.2f}M)")
print(f"  running stats差异:            {enc_bn:>12,} ({enc_bn/1e6:.2f}M)")

# 3. Decoder
from src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
decoder = RTDETRTransformer(
    num_classes=80,
    hidden_dim=256,
    num_queries=300,
    feat_channels=[256, 256, 256],
    feat_strides=[8, 16, 32],
    num_levels=3,
    num_decoder_points=4,
    nhead=8,
    num_decoder_layers=6,
    dim_feedforward=1024,
)

dec_jt = count_params_jittor(decoder)
dec_pt = count_params_pytorch_style(decoder)
dec_bn = count_bn_running_stats(decoder)

print(f"\nRTDETRTransformer:")
print(f"  Jittor计数 (含running stats): {dec_jt:>12,} ({dec_jt/1e6:.2f}M)")
print(f"  PyTorch计数 (不含running stats): {dec_pt:>12,} ({dec_pt/1e6:.2f}M)")
print(f"  running stats差异:            {dec_bn:>12,} ({dec_bn/1e6:.2f}M)")

# 总结
total_jt = bb_jt + enc_jt + dec_jt
total_pt = bb_pt + enc_pt + dec_pt
total_bn = bb_bn + enc_bn + dec_bn

print(f"\n{'='*60}")
print(f"总结")
print(f"{'='*60}")
print(f"  Jittor总参数 (含running stats): {total_jt:>12,} ({total_jt/1e6:.2f}M)")
print(f"  PyTorch方式 (不含running stats): {total_pt:>12,} ({total_pt/1e6:.2f}M)")
print(f"  running stats总差异:           {total_bn:>12,} ({total_bn/1e6:.2f}M)")
print(f"\n  官方RT-DETR R50: 42M")
print(f"  我们(PyTorch方式): {total_pt/1e6:.2f}M")
print(f"  差异: {(total_pt - 42_000_000):,} ({(total_pt - 42_000_000)/42_000_000*100:.2f}%)")
