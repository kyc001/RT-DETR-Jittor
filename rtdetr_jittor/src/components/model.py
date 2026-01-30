#!/usr/bin/env python3
"""
RT-DETR模型组件
提供封装的RT-DETR模型，参考ultimate_sanity_check.py的验证实现
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt
import jittor.nn as nn

# 导入验证过的组件
from ..nn.backbone.resnet import ResNet50
from ..zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
from ..nn.criterion.rtdetr_criterion import build_criterion

class RTDETRModel(nn.Module):
    """
    封装的RT-DETR模型
    基于ultimate_sanity_check.py的验证实现
    """
    def __init__(self, num_classes=80, pretrained=True):
        super().__init__()
        
        # 使用验证过的backbone
        self.backbone = ResNet50(pretrained=pretrained)
        
        # 使用验证过的transformer
        self.transformer = RTDETRTransformer(
            num_classes=num_classes,
            hidden_dim=256,
            num_queries=300,
            feat_channels=[256, 512, 1024, 2048]
        )
        
        self.num_classes = num_classes
        
        if pretrained:
            print("✅ 使用Jittor内置预训练权重")
        else:
            print("⚠️ 使用随机初始化权重")
    
    def execute(self, x, targets=None):
        """
        前向传播
        确保返回完整的输出包括编码器输出
        """
        # 使用验证过的前向传播方法
        features = self.backbone(x)
        outputs = self.transformer(features, targets)
        
        # RTDETRTransformer已经在其execute方法中包含了完整的输出
        # 包括pred_logits, pred_boxes, enc_outputs等
        return outputs
    
    def get_criterion(self):
        """获取验证过的损失函数"""
        return build_criterion(num_classes=self.num_classes)
    
    def fix_batchnorm(self):
        """
        修复BatchNorm问题
        参考ultimate_sanity_check.py的实现
        """
        def _fix_batchnorm(module):
            for m in module.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.train()
                    # 确保BatchNorm参数可训练
                    if hasattr(m, 'weight') and m.weight is not None:
                        m.weight.requires_grad = True
                    if hasattr(m, 'bias') and m.bias is not None:
                        m.bias.requires_grad = True
        
        _fix_batchnorm(self)
    
    def get_trainable_params(self):
        """
        获取所有可训练参数
        参考ultimate_sanity_check.py的实现
        """
        all_params = []
        for param in self.parameters():
            if param.requires_grad:
                all_params.append(param)
        return all_params
    
    def get_param_stats(self):
        """获取参数统计信息"""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        total_params = backbone_params + transformer_params
        trainable_params = len(self.get_trainable_params())
        
        return {
            'backbone_params': backbone_params,
            'transformer_params': transformer_params,
            'total_params': total_params,
            'trainable_params': trainable_params
        }
    
    def print_model_info(self):
        """打印模型信息"""
        stats = self.get_param_stats()
        print("📊 模型参数统计:")
        print(f"   Backbone参数: {stats['backbone_params']:,}")
        print(f"   Transformer参数: {stats['transformer_params']:,}")
        print(f"   总参数: {stats['total_params']:,}")
        print(f"   可训练参数数量: {stats['trainable_params']}")

def create_rtdetr_model(num_classes=80, pretrained=True):
    """
    创建RT-DETR模型的工厂函数
    
    Args:
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
    
    Returns:
        model: RT-DETR模型
        criterion: 损失函数
    """
    # 设置Jittor
    jt.flags.use_cuda = 1
    jt.set_global_seed(42)
    jt.flags.auto_mixed_precision_level = 0
    
    # 创建模型
    model = RTDETRModel(num_classes=num_classes, pretrained=pretrained)
    
    # 修复BatchNorm
    model.fix_batchnorm()
    
    # 创建损失函数
    criterion = model.get_criterion()
    
    # 打印模型信息
    model.print_model_info()
    
    return model, criterion

def create_optimizer(model, lr=1e-4, weight_decay=0):
    """
    创建优化器
    参考ultimate_sanity_check.py的设置
    
    Args:
        model: RT-DETR模型
        lr: 学习率
        weight_decay: 权重衰减
    
    Returns:
        optimizer: Jittor优化器
    """
    trainable_params = model.get_trainable_params()
    optimizer = jt.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    
    print(f"📊 优化器配置:")
    print(f"   学习率: {lr}")
    print(f"   权重衰减: {weight_decay}")
    print(f"   可训练参数: {len(trainable_params)}")
    
    return optimizer

if __name__ == "__main__":
    # 测试模型创建
    print("🧪 测试RT-DETR模型组件")
    print("=" * 50)
    
    model, criterion = create_rtdetr_model(num_classes=80, pretrained=True)
    optimizer = create_optimizer(model, lr=1e-4)
    
    # 测试前向传播
    test_input = jt.randn(1, 3, 640, 640)
    with jt.no_grad():
        outputs = model(test_input)
    
    print(f"\n✅ 模型测试成功!")
    print(f"   输出键: {list(outputs.keys())}")
    for key, value in outputs.items():
        print(f"   {key}: {value.shape}")
    
    print(f"\n🎉 RT-DETR模型组件验证完成!")
