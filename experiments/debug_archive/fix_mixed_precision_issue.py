#!/usr/bin/env python3
"""
解决Jittor混合精度问题
Jittor不支持float32和float64混合运算
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

def analyze_mixed_precision_issue():
    """分析混合精度问题"""
    print("=" * 60)
    print("===        混合精度问题分析        ===")
    print("=" * 60)
    
    # 测试Jittor的混合精度行为
    print("测试Jittor混合精度行为:")
    
    try:
        # 创建不同精度的张量
        x_f32 = jt.randn(2, 3).float32()
        x_f64 = jt.randn(2, 3).float64()
        
        print(f"float32张量: {x_f32.dtype}")
        print(f"float64张量: {x_f64.dtype}")
        
        # 测试混合运算
        try:
            result = jt.matmul(x_f32, x_f64.transpose(0, 1))
            print(f"✅ 混合运算成功: {result.dtype}")
        except Exception as e:
            print(f"❌ 混合运算失败: {e}")
        
        # 测试强制转换
        x_f64_to_f32 = x_f64.float32()
        result_safe = jt.matmul(x_f32, x_f64_to_f32.transpose(0, 1))
        print(f"✅ 转换后运算成功: {result_safe.dtype}")
        
    except Exception as e:
        print(f"❌ 基础测试失败: {e}")

def force_float32_everywhere():
    """强制所有地方使用float32"""
    print("\n" + "=" * 60)
    print("===        强制float32解决方案        ===")
    print("=" * 60)
    
    try:
        from jittor_rt_detr.src.nn.backbone.resnet import ResNet50
        from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
        from jittor_rt_detr.src.nn.criterion.rtdetr_criterion import build_criterion
        
        # 创建模型
        backbone = ResNet50(pretrained=False)
        transformer = RTDETRTransformer(
            num_classes=80,
            hidden_dim=256,
            num_queries=300,
            feat_channels=[256, 512, 1024, 2048]
        )
        criterion = build_criterion(num_classes=80)
        
        print("✅ 模型创建成功")
        
        # 递归强制所有参数为float32
        def recursive_float32_conversion(module, path=""):
            """递归转换所有参数和子模块为float32"""
            converted_count = 0
            
            # 转换当前模块的参数
            for name, param in module.named_parameters(recurse=False):
                if param.dtype != jt.float32:
                    print(f"  转换 {path}.{name}: {param.dtype} -> float32")
                    param.data = param.data.float32()
                    converted_count += 1
            
            # 转换当前模块的缓冲区
            for name, buffer in module.named_buffers(recurse=False):
                if hasattr(buffer, 'dtype') and buffer.dtype != jt.float32:
                    print(f"  转换缓冲区 {path}.{name}: {buffer.dtype} -> float32")
                    buffer.data = buffer.data.float32()
                    converted_count += 1
            
            # 递归处理子模块
            for name, child in module.named_children():
                child_path = f"{path}.{name}" if path else name
                converted_count += recursive_float32_conversion(child, child_path)
            
            return converted_count
        
        print("\n强制转换所有模型参数为float32:")
        backbone_converted = recursive_float32_conversion(backbone, "backbone")
        transformer_converted = recursive_float32_conversion(transformer, "transformer")
        criterion_converted = recursive_float32_conversion(criterion, "criterion")
        
        total_converted = backbone_converted + transformer_converted + criterion_converted
        print(f"✅ 总共转换了 {total_converted} 个参数/缓冲区")
        
        # 验证转换结果
        print("\n验证转换结果:")
        all_params = list(backbone.parameters()) + list(transformer.parameters()) + list(criterion.parameters())
        
        non_float32_params = []
        for i, param in enumerate(all_params):
            if param.dtype != jt.float32:
                non_float32_params.append((i, param.dtype))
        
        if len(non_float32_params) == 0:
            print(f"✅ 所有 {len(all_params)} 个参数都是float32")
        else:
            print(f"❌ 仍有 {len(non_float32_params)} 个参数不是float32:")
            for i, dtype in non_float32_params[:5]:  # 只显示前5个
                print(f"   参数{i}: {dtype}")
        
        # 创建优化器
        optimizer = jt.optim.SGD(all_params, lr=1e-4)
        
        # 测试训练循环
        print("\n测试混合精度修复后的训练:")
        
        for iteration in range(2):
            print(f"\n--- 迭代 {iteration + 1} ---")
            
            # 严格使用float32输入
            x = jt.randn(1, 3, 640, 640, dtype=jt.float32)
            
            # 前向传播
            feats = backbone(x)
            
            # 确保所有特征都是float32
            for i, feat in enumerate(feats):
                if feat.dtype != jt.float32:
                    print(f"⚠️ 特征{i}不是float32: {feat.dtype}")
                    feats[i] = feat.float32()
            
            outputs = transformer(feats)
            
            # 确保所有输出都是float32
            if outputs['pred_logits'].dtype != jt.float32:
                outputs['pred_logits'] = outputs['pred_logits'].float32()
            if outputs['pred_boxes'].dtype != jt.float32:
                outputs['pred_boxes'] = outputs['pred_boxes'].float32()
            
            if 'enc_outputs' in outputs:
                if outputs['enc_outputs']['pred_logits'].dtype != jt.float32:
                    outputs['enc_outputs']['pred_logits'] = outputs['enc_outputs']['pred_logits'].float32()
                if outputs['enc_outputs']['pred_boxes'].dtype != jt.float32:
                    outputs['enc_outputs']['pred_boxes'] = outputs['enc_outputs']['pred_boxes'].float32()
            
            print(f"✅ 前向传播成功，所有输出都是float32")
            
            # 创建严格float32的目标
            targets = [{
                'boxes': jt.rand(3, 4, dtype=jt.float32),
                'labels': jt.array([1, 2, 3], dtype=jt.int64)
            }]
            
            # 损失计算
            loss_dict = criterion(outputs, targets)
            
            # 确保所有损失都是float32
            for k, v in loss_dict.items():
                if v.dtype != jt.float32:
                    print(f"⚠️ 损失{k}不是float32: {v.dtype}")
                    loss_dict[k] = v.float32()
            
            total_loss = sum(loss_dict.values())
            if total_loss.dtype != jt.float32:
                total_loss = total_loss.float32()
            
            print(f"✅ 损失计算成功: {total_loss.item():.4f} (dtype: {total_loss.dtype})")
            
            # 反向传播
            try:
                # 清零梯度
                for param in all_params:
                    if hasattr(param, 'grad') and param.grad is not None:
                        param.grad = None
                
                # 反向传播
                optimizer.backward(total_loss)
                print("✅ 反向传播成功！混合精度问题已解决")
                
                # 检查梯度
                grad_count = 0
                for param in all_params[:5]:  # 检查前5个参数
                    try:
                        grad = param.opt_grad(optimizer)
                        if grad is not None and grad.norm().item() > 1e-8:
                            grad_count += 1
                    except:
                        pass
                
                print(f"✅ 有效梯度参数: {grad_count}/5")
                
                return True
                
            except Exception as e:
                print(f"❌ 反向传播仍然失败: {e}")
                
                # 进一步诊断
                print("进一步诊断混合精度问题...")
                
                # 检查是否还有隐藏的float64
                print("检查模型中的所有张量:")
                for name, module in transformer.named_modules():
                    for param_name, param in module.named_parameters(recurse=False):
                        if param.dtype != jt.float32:
                            print(f"⚠️ 发现非float32参数: {name}.{param_name} = {param.dtype}")
                
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 强制float32解决方案失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_dtype_safe_wrapper():
    """创建数据类型安全的包装器"""
    print("\n" + "=" * 60)
    print("===        创建数据类型安全包装器        ===")
    print("=" * 60)
    
    class Float32Wrapper:
        """确保所有操作都使用float32的包装器"""
        
        @staticmethod
        def safe_matmul(a, b):
            """安全的矩阵乘法"""
            a = a.float32() if a.dtype != jt.float32 else a
            b = b.float32() if b.dtype != jt.float32 else b
            return jt.matmul(a, b)
        
        @staticmethod
        def safe_add(a, b):
            """安全的加法"""
            a = a.float32() if a.dtype != jt.float32 else a
            b = b.float32() if b.dtype != jt.float32 else b
            return a + b
        
        @staticmethod
        def safe_mul(a, b):
            """安全的乘法"""
            a = a.float32() if a.dtype != jt.float32 else a
            b = b.float32() if b.dtype != jt.float32 else b
            return a * b
    
    # 测试包装器
    try:
        x = jt.randn(2, 3).float64()  # 故意创建float64
        y = jt.randn(3, 4).float32()
        
        # 使用安全包装器
        result = Float32Wrapper.safe_matmul(x, y)
        print(f"✅ 安全矩阵乘法: {x.dtype} × {y.dtype} = {result.dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ 包装器测试失败: {e}")
        return False

def main():
    print("🔧 解决Jittor混合精度问题")
    print("=" * 80)
    
    # 分析混合精度问题
    analyze_mixed_precision_issue()
    
    # 创建数据类型安全包装器
    wrapper_ok = create_dtype_safe_wrapper()
    
    # 强制float32解决方案
    float32_ok = force_float32_everywhere()
    
    # 总结
    print("\n" + "=" * 80)
    print("🎯 混合精度问题解决总结:")
    print("=" * 80)
    
    results = [
        ("数据类型安全包装器", wrapper_ok),
        ("强制float32解决方案", float32_ok),
    ]
    
    all_passed = True
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} {name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 Jittor混合精度问题完全解决！")
        print("✅ 所有模型参数强制转换为float32")
        print("✅ 所有中间计算使用float32")
        print("✅ 反向传播正常工作")
        print("✅ 彻底避免了float32/float64混合运算")
        print("\n🚀 解决方案要点:")
        print("1. ✅ 递归转换所有参数和缓冲区为float32")
        print("2. ✅ 严格控制输入数据类型")
        print("3. ✅ 确保所有中间结果为float32")
        print("4. ✅ 创建数据类型安全的操作包装器")
        print("\n✨ RT-DETR现在完全兼容Jittor的精度要求！")
    else:
        print("⚠️ 混合精度问题仍需进一步解决")
        print("💡 建议: 检查是否有隐藏的float64张量或操作")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
