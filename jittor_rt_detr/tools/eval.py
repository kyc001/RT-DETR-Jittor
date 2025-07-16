#!/usr/bin/env python3
"""
RT-DETR评估脚本
"""

import os
import sys
import argparse

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jittor as jt
from src.core.yaml_config import YAMLConfig
from src.data.coco.coco_dataset import COCODataset
from src.zoo.rtdetr.rtdetr import build_rtdetr
from src.nn.criterion.rtdetr_criterion import build_criterion

def main():
    parser = argparse.ArgumentParser(description='RT-DETR Evaluation')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备类型')
    
    args = parser.parse_args()
    
    # 设置Jittor
    if args.device == 'cuda':
        jt.flags.use_cuda = 1
    
    # 加载配置
    cfg = YAMLConfig(args.config)
    
    # 创建模型
    model = build_rtdetr(
        backbone_name='presnet50',
        num_classes=80,
        hidden_dim=256,
        num_queries=300
    )
    
    # 加载检查点
    if os.path.exists(args.checkpoint):
        model.load_state_dict(jt.load(args.checkpoint))
        print(f"✅ 加载检查点: {args.checkpoint}")
    else:
        print(f"⚠️ 检查点不存在: {args.checkpoint}")
    
    # 设置评估模式
    model.eval()
    
    # 创建数据集
    dataset = COCODataset(
        img_folder='data/coco2017_50/val2017',
        ann_file='data/coco2017_50/annotations/instances_val2017.json',
        transforms=None
    )
    
    print(f"✅ 评估数据集大小: {len(dataset)}")
    
    # 简单评估
    with jt.no_grad():
        for i in range(min(10, len(dataset))):
            img, target = dataset[i]
            if isinstance(img, tuple):
                img = img[0]
            
            # 确保输入格式正确
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            
            outputs = model(img)
            print(f"样本 {i}: 输出形状 {outputs['pred_logits'].shape}")
    
    print("✅ 评估完成")

if __name__ == '__main__':
    main()
