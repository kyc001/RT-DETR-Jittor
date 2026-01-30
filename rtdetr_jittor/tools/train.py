#!/usr/bin/env python3
"""Training script for Jittor RT-DETR"""

import os
import sys
import argparse

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import jittor as jt
from src.core.yaml_config import YAMLConfig
from src.nn.backbone.resnet import ResNet50
from src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
from src.nn.criterion.rtdetr_criterion import build_criterion

def main(args):
    """Main training function"""
    # 设置Jittor
    jt.flags.use_cuda = 1
    
    # 加载配置
    if args.config:
        cfg = YAMLConfig(args.config)
    else:
        # 默认配置
        cfg = YAMLConfig.__new__(YAMLConfig)
        cfg.yaml_cfg = {
            'num_classes': 80,
            'hidden_dim': 256,
            'num_queries': 300,
            'lr': 1e-4,
            'epochs': 50
        }
    
    # 创建模型
    backbone = ResNet50(pretrained=False)
    transformer = RTDETRTransformer(
        num_classes=cfg.yaml_cfg.get('num_classes', 80),
        hidden_dim=cfg.yaml_cfg.get('hidden_dim', 256),
        num_queries=cfg.yaml_cfg.get('num_queries', 300),
        feat_channels=[256, 512, 1024, 2048]
    )
    criterion = build_criterion(cfg.yaml_cfg.get('num_classes', 80))
    
    print("✅ 模型创建成功")
    print(f"   参数数量: {sum(p.numel() for p in list(backbone.parameters()) + list(transformer.parameters())):,}")
    
    # 创建优化器
    all_params = list(backbone.parameters()) + list(transformer.parameters())
    optimizer = jt.optim.SGD(all_params, lr=cfg.yaml_cfg.get('lr', 1e-4))
    
    print("🚀 开始训练...")
    
    # 简单的训练循环示例
    for epoch in range(cfg.yaml_cfg.get('epochs', 50)):
        # 这里应该是实际的数据加载和训练逻辑
        print(f"Epoch {epoch + 1}/{cfg.yaml_cfg.get('epochs', 50)}")
        
        # 示例前向传播
        x = jt.randn(1, 3, 640, 640, dtype=jt.float32)
        feats = backbone(x)
        outputs = transformer(feats)
        
        # 示例目标
        targets = [{
            'boxes': jt.rand(3, 4, dtype=jt.float32),
            'labels': jt.array([1, 2, 3], dtype=jt.int64)
        }]
        
        # 损失计算
        loss_dict = criterion(outputs, targets)
        total_loss = sum(loss_dict.values())
        
        # 反向传播
        optimizer.backward(total_loss)
        
        if epoch % 10 == 0:
            print(f"  损失: {total_loss.item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RT-DETR with Jittor')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点')
    parser.add_argument('--test-only', action='store_true', help='仅测试模式')
    
    args = parser.parse_args()
    main(args)
