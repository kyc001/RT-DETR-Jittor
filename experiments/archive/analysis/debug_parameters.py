#!/usr/bin/env python3
"""
调试模型参数情况
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt
from jittor import nn

# 设置Jittor
jt.flags.use_cuda = 1

def analyze_model_parameters():
    """分析模型参数情况"""
    print("🔍 分析模型参数情况")
    print("=" * 60)
    
    try:
        from jittor_rt_detr.src.nn.backbone.resnet import ResNet50
        from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
        
        # 创建模型
        backbone = ResNet50(pretrained=False)
        transformer = RTDETRTransformer(
            num_classes=80,
            hidden_dim=256,
            num_queries=300,
            feat_channels=[256, 512, 1024, 2048]
        )
        
        print("📊 模型创建成功")
        
        # 分析backbone参数
        print(f"\n🔧 Backbone参数分析:")
        backbone_total_params = 0
        backbone_trainable_params = 0
        backbone_param_count = 0
        
        for name, param in backbone.named_parameters():
            param_elements = param.numel()
            backbone_total_params += param_elements
            backbone_param_count += 1
            
            if param.requires_grad:
                backbone_trainable_params += param_elements
            
            # 显示前10个参数的详细信息
            if backbone_param_count <= 10:
                print(f"   {backbone_param_count:2d}: {name:40} | 形状: {str(param.shape):20} | 元素数: {param_elements:8,} | 可训练: {param.requires_grad}")
        
        if backbone_param_count > 10:
            print(f"   ... 还有 {backbone_param_count - 10} 个参数")
        
        print(f"   总计: {backbone_param_count} 个参数张量, {backbone_total_params:,} 个参数元素")
        print(f"   可训练: {backbone_trainable_params:,} 个参数元素")
        
        # 分析transformer参数
        print(f"\n🔧 Transformer参数分析:")
        transformer_total_params = 0
        transformer_trainable_params = 0
        transformer_param_count = 0
        
        for name, param in transformer.named_parameters():
            param_elements = param.numel()
            transformer_total_params += param_elements
            transformer_param_count += 1
            
            if param.requires_grad:
                transformer_trainable_params += param_elements
            
            # 显示前10个参数的详细信息
            if transformer_param_count <= 10:
                print(f"   {transformer_param_count:2d}: {name:40} | 形状: {str(param.shape):20} | 元素数: {param_elements:8,} | 可训练: {param.requires_grad}")
        
        if transformer_param_count > 10:
            print(f"   ... 还有 {transformer_param_count - 10} 个参数")
        
        print(f"   总计: {transformer_param_count} 个参数张量, {transformer_total_params:,} 个参数元素")
        print(f"   可训练: {transformer_trainable_params:,} 个参数元素")
        
        # 总结
        total_param_count = backbone_param_count + transformer_param_count
        total_param_elements = backbone_total_params + transformer_total_params
        total_trainable_elements = backbone_trainable_params + transformer_trainable_params
        
        print(f"\n📊 总结:")
        print(f"   参数张量总数: {total_param_count}")
        print(f"   参数元素总数: {total_param_elements:,}")
        print(f"   可训练参数元素: {total_trainable_elements:,}")
        print(f"   可训练比例: {total_trainable_elements/total_param_elements*100:.1f}%")
        
        # 检查为什么可训练参数这么少
        print(f"\n🔍 检查参数可训练性:")
        
        # 检查backbone中不可训练的参数
        backbone_frozen_params = 0
        for name, param in backbone.named_parameters():
            if not param.requires_grad:
                backbone_frozen_params += param.numel()
        
        # 检查transformer中不可训练的参数
        transformer_frozen_params = 0
        for name, param in transformer.named_parameters():
            if not param.requires_grad:
                transformer_frozen_params += param.numel()
        
        print(f"   Backbone冻结参数: {backbone_frozen_params:,}")
        print(f"   Transformer冻结参数: {transformer_frozen_params:,}")
        
        if backbone_frozen_params > 0 or transformer_frozen_params > 0:
            print(f"   ⚠️ 发现冻结参数！这可能是问题所在")
        
        # 检查具体哪些参数被冻结了
        print(f"\n🧊 冻结参数详情:")
        frozen_count = 0
        for module_name, module in [("backbone", backbone), ("transformer", transformer)]:
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    frozen_count += 1
                    if frozen_count <= 20:  # 只显示前20个
                        print(f"   {module_name}.{name}: {param.shape} ({param.numel():,} 元素)")
        
        if frozen_count > 20:
            print(f"   ... 还有 {frozen_count - 20} 个冻结参数")
        
        return backbone, transformer
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def fix_parameter_training(backbone, transformer):
    """修复参数训练问题"""
    print(f"\n🔧 修复参数训练问题...")
    
    # 确保所有参数都可训练
    for module_name, module in [("backbone", backbone), ("transformer", transformer)]:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                param.requires_grad = True
                print(f"   启用训练: {module_name}.{name}")
    
    # 重新统计
    total_trainable = 0
    for module in [backbone, transformer]:
        for param in module.parameters():
            if param.requires_grad:
                total_trainable += param.numel()
    
    print(f"   修复后可训练参数: {total_trainable:,}")

def main():
    print("🔍 RT-DETR模型参数调试")
    print("=" * 60)
    
    backbone, transformer = analyze_model_parameters()
    
    if backbone is not None:
        fix_parameter_training(backbone, transformer)

if __name__ == "__main__":
    main()
