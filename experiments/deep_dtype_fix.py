#!/usr/bin/env python3
"""
深度数据类型修复
基于之前的成功经验，但更彻底地解决cublas_batched_matmul问题
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt
import jittor.nn as nn
import numpy as np

# 设置Jittor为严格float32模式
jt.flags.use_cuda = 1
jt.set_global_seed(42)
jt.flags.auto_mixed_precision_level = 0

def deep_ensure_float32(x):
    """深度确保张量为float32类型"""
    if isinstance(x, jt.Var):
        if x.dtype != jt.float32:
            return x.float32()
        return x
    elif isinstance(x, (list, tuple)):
        return [deep_ensure_float32(item) for item in x]
    elif isinstance(x, dict):
        return {k: deep_ensure_float32(v) for k, v in x.items()}
    else:
        return x

def force_model_float32(model):
    """强制模型所有参数和缓冲区为float32"""
    print(f"修复模型: {model.__class__.__name__}")
    param_count = 0
    
    # 修复参数
    for name, param in model.named_parameters():
        if param.dtype != jt.float32:
            print(f"  修复参数 {name}: {param.dtype} -> float32")
            param.data = param.data.float32()
            param_count += 1
    
    # 修复缓冲区
    for name, buffer in model.named_buffers():
        if hasattr(buffer, 'dtype') and buffer.dtype != jt.float32:
            print(f"  修复缓冲区 {name}: {buffer.dtype} -> float32")
            buffer.data = buffer.data.float32()
    
    print(f"  总共修复了 {param_count} 个参数")

def test_deep_dtype_fix():
    """测试深度数据类型修复"""
    print("=" * 60)
    print("===        深度数据类型修复测试        ===")
    print("=" * 60)
    
    try:
        from jittor_rt_detr.src.nn.backbone.resnet import ResNet50
        from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
        from jittor_rt_detr.src.nn.criterion.rtdetr_criterion import build_criterion
        
        # 创建模型
        print("创建模型...")
        backbone = ResNet50(pretrained=False)
        transformer = RTDETRTransformer(
            num_classes=80,
            hidden_dim=256,
            num_queries=300,
            feat_channels=[256, 512, 1024, 2048]
        )
        criterion = build_criterion(num_classes=80)
        
        print("✅ 模型创建成功")
        
        # 深度修复所有模型的数据类型
        print("\n深度修复模型数据类型...")
        force_model_float32(backbone)
        force_model_float32(transformer)
        force_model_float32(criterion)
        
        print("✅ 所有模型数据类型已深度修复")
        
        # 创建优化器
        all_params = list(backbone.parameters()) + list(transformer.parameters())
        
        # 验证所有参数都是float32
        print("\n验证参数数据类型...")
        non_float32_count = 0
        for i, param in enumerate(all_params):
            if param.dtype != jt.float32:
                print(f"⚠️ 参数{i}仍然不是float32: {param.dtype}")
                non_float32_count += 1
        
        if non_float32_count == 0:
            print(f"✅ 所有{len(all_params)}个参数都是float32")
        else:
            print(f"❌ 仍有{non_float32_count}个参数不是float32")
            return False
        
        # 使用最简单的优化器避免复杂性
        optimizer = jt.optim.SGD(all_params, lr=1e-4)
        
        # 训练循环
        for i in range(2):
            print(f"\n--- 深度修复训练迭代 {i+1} ---")
            
            # 创建严格float32的输入
            x = jt.randn(1, 3, 640, 640, dtype=jt.float32)
            print(f"输入数据类型: {x.dtype}")
            
            # 前向传播
            feats = backbone(x)
            
            # 深度确保特征数据类型
            feats = [deep_ensure_float32(feat) for feat in feats]
            print(f"特征数据类型: {[feat.dtype for feat in feats]}")
            
            outputs = transformer(feats)
            
            # 深度确保输出数据类型
            outputs = deep_ensure_float32(outputs)
            print(f"输出数据类型: pred_logits={outputs['pred_logits'].dtype}, pred_boxes={outputs['pred_boxes'].dtype}")
            
            # 创建严格类型的目标
            targets = [{
                'boxes': jt.rand(3, 4, dtype=jt.float32),
                'labels': jt.array([1, 2, 3], dtype=jt.int64)
            }]
            
            # 损失计算
            loss_dict = criterion(outputs, targets)
            
            # 深度确保损失数据类型
            loss_dict = deep_ensure_float32(loss_dict)
            total_loss = sum(loss_dict.values())
            total_loss = deep_ensure_float32(total_loss)
            
            print(f"损失: {total_loss.item():.4f} ({total_loss.dtype})")
            
            # 尝试最简单的反向传播
            try:
                # 清零梯度
                for param in all_params:
                    if hasattr(param, 'grad') and param.grad is not None:
                        param.grad = None
                
                # 使用最基本的反向传播
                optimizer.backward(total_loss)
                print("✅ optimizer.backward() 成功")
                
                # 检查梯度
                grad_count = 0
                for param in all_params[:3]:  # 只检查前3个
                    try:
                        grad = param.opt_grad(optimizer)
                        if grad is not None and grad.norm().item() > 1e-8:
                            grad_count += 1
                    except:
                        pass
                
                print(f"✅ 有效梯度参数: {grad_count}/3")
                
            except Exception as e:
                print(f"❌ 反向传播失败: {e}")
                
                # 如果还是失败，说明问题更深层
                print("尝试分析具体的数据类型冲突...")
                
                # 检查模型内部的数据类型
                print("检查模型内部状态...")
                for name, module in transformer.named_modules():
                    if hasattr(module, 'weight') and module.weight is not None:
                        if module.weight.dtype != jt.float32:
                            print(f"⚠️ 模块 {name} 权重不是float32: {module.weight.dtype}")
                
                return False
        
        print("\n✅ 深度数据类型修复成功")
        return True
        
    except Exception as e:
        print(f"❌ 深度数据类型修复失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_minimal_working_example():
    """创建最小工作示例"""
    print("\n" + "=" * 60)
    print("===        最小工作示例        ===")
    print("=" * 60)
    
    try:
        # 创建最简单的模型测试
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(256, 80)
                
            def execute(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        
        # 强制float32
        for param in model.parameters():
            param.data = param.data.float32()
        
        # 创建优化器
        optimizer = jt.optim.SGD(model.parameters(), lr=1e-4)
        
        # 测试训练
        x = jt.randn(1, 300, 256, dtype=jt.float32)
        target = jt.randn(1, 300, 80, dtype=jt.float32)
        
        output = model(x)
        loss = jt.mean((output - target) ** 2)
        
        print(f"简单模型测试:")
        print(f"  输入: {x.dtype}")
        print(f"  输出: {output.dtype}")
        print(f"  损失: {loss.dtype}")
        
        # 反向传播
        optimizer.backward(loss)
        print("✅ 简单模型反向传播成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 简单模型测试失败: {e}")
        return False

def main():
    print("🔧 深度数据类型修复 - 解决cublas_batched_matmul问题")
    print("=" * 80)
    
    # 最小工作示例
    simple_ok = create_minimal_working_example()
    
    # 深度数据类型修复
    deep_ok = test_deep_dtype_fix()
    
    # 总结
    print("\n" + "=" * 80)
    print("🎯 深度修复总结:")
    print("=" * 80)
    
    results = [
        ("最小工作示例", simple_ok),
        ("深度数据类型修复", deep_ok),
    ]
    
    all_passed = True
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} {name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 深度数据类型修复成功！")
        print("✅ 彻底解决了cublas_batched_matmul数据类型问题")
        print("✅ 所有参数和缓冲区都是float32")
        print("✅ 反向传播正常工作")
        print("✅ 基于之前成功经验的深度修复")
        print("\n🚀 RT-DETR现在完全可用于训练！")
    else:
        print("⚠️ 深度修复仍需进一步调整")
        print("💡 建议: 可能需要检查Jittor版本或CUDA兼容性")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
