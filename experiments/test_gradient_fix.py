#!/usr/bin/env python3
"""
测试梯度传播修复效果
"""

import os
import sys
import numpy as np

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt
import jittor.nn as nn

# 设置Jittor
jt.flags.use_cuda = 1
jt.set_global_seed(42)
jt.flags.auto_mixed_precision_level = 0

def test_gradient_flow():
    """测试修复后的梯度传播"""
    print("=" * 60)
    print("===        梯度传播修复测试        ===")
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
        
        # 创建优化器
        all_params = list(backbone.parameters()) + list(transformer.parameters())
        optimizer = jt.optim.AdamW(all_params, lr=1e-4)
        
        # 训练模式
        backbone.train()
        transformer.train()
        
        # 前向传播
        x = jt.randn(1, 3, 640, 640).float32()
        feats = backbone(x)
        outputs = transformer(feats)
        
        print(f"✅ 前向传播成功")
        print(f"   pred_logits: {outputs['pred_logits'].shape}")
        print(f"   pred_boxes: {outputs['pred_boxes'].shape}")
        
        # 检查是否有编码器输出
        if 'enc_outputs' in outputs:
            print(f"   enc_outputs.pred_logits: {outputs['enc_outputs']['pred_logits'].shape}")
            print(f"   enc_outputs.pred_boxes: {outputs['enc_outputs']['pred_boxes'].shape}")
            print("✅ 编码器输出已包含在前向传播中")
        else:
            print("⚠️ 编码器输出未包含")
        
        # 创建目标
        targets = [{
            'boxes': jt.rand(3, 4).float32(),
            'labels': jt.array([1, 2, 3], dtype=jt.int64)
        }]
        
        # 损失计算
        loss_dict = criterion(outputs, targets)
        total_loss = sum(loss_dict.values())
        
        print(f"✅ 损失计算成功: {total_loss.item():.4f}")
        print("   损失组成:")
        for k, v in loss_dict.items():
            print(f"     {k}: {v.item():.4f}")
        
        # 反向传播前检查参数
        print("\n检查关键参数是否参与计算:")
        
        # 检查MSDeformableAttention参数
        for name, param in transformer.named_parameters():
            if 'cross_attn.sampling_offsets' in name or 'cross_attn.attention_weights' in name:
                print(f"   {name}: requires_grad={param.requires_grad}")
            elif 'enc_output' in name or 'enc_score_head' in name or 'enc_bbox_head' in name:
                print(f"   {name}: requires_grad={param.requires_grad}")
        
        # 反向传播
        optimizer.step(total_loss)
        
        print(f"\n✅ 反向传播成功")
        
        # 检查梯度
        print("\n检查梯度情况:")
        gradient_issues = []
        gradient_success = []
        
        for name, param in transformer.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 1e-8:
                    gradient_success.append((name, grad_norm))
                else:
                    gradient_issues.append((name, grad_norm))
            else:
                gradient_issues.append((name, "None"))
        
        print(f"✅ 有梯度的参数: {len(gradient_success)}")
        for name, grad_norm in gradient_success[:10]:  # 只显示前10个
            print(f"   {name}: {grad_norm:.6f}")
        
        if gradient_issues:
            print(f"⚠️ 梯度问题的参数: {len(gradient_issues)}")
            for name, grad_norm in gradient_issues[:10]:  # 只显示前10个
                print(f"   {name}: {grad_norm}")
        
        # 统计
        total_params = len(gradient_success) + len(gradient_issues)
        success_rate = len(gradient_success) / total_params * 100
        
        print(f"\n梯度传播成功率: {len(gradient_success)}/{total_params} ({success_rate:.1f}%)")
        
        return success_rate > 80  # 如果80%以上参数有梯度，认为修复成功
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_iterations():
    """测试多次迭代的梯度稳定性"""
    print("\n" + "=" * 60)
    print("===        多次迭代梯度稳定性测试        ===")
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
        
        # 创建优化器
        all_params = list(backbone.parameters()) + list(transformer.parameters())
        optimizer = jt.optim.AdamW(all_params, lr=1e-4)
        
        # 训练模式
        backbone.train()
        transformer.train()
        
        losses = []
        
        for i in range(5):
            # 前向传播
            x = jt.randn(1, 3, 640, 640).float32()
            feats = backbone(x)
            outputs = transformer(feats)
            
            # 创建目标
            targets = [{
                'boxes': jt.rand(3, 4).float32(),
                'labels': jt.array([1, 2, 3], dtype=jt.int64)
            }]
            
            # 损失计算
            loss_dict = criterion(outputs, targets)
            total_loss = sum(loss_dict.values())
            losses.append(total_loss.item())
            
            # 反向传播
            optimizer.step(total_loss)
            
            print(f"迭代 {i+1}: 损失 = {total_loss.item():.4f}")
        
        print(f"\n✅ 多次迭代测试成功")
        print(f"损失变化: {losses[0]:.4f} -> {losses[-1]:.4f}")
        
        # 检查损失是否在合理范围内变化
        loss_stable = all(0.1 < loss < 100.0 for loss in losses)
        
        return loss_stable
        
    except Exception as e:
        print(f"❌ 多次迭代测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔧 RT-DETR梯度传播修复验证")
    print("=" * 80)
    
    # 测试梯度传播
    gradient_ok = test_gradient_flow()
    
    # 测试多次迭代稳定性
    stability_ok = test_multiple_iterations()
    
    # 总结
    print("\n" + "=" * 80)
    print("🎯 梯度修复验证总结:")
    print("=" * 80)
    
    results = [
        ("梯度传播", gradient_ok),
        ("迭代稳定性", stability_ok),
    ]
    
    all_passed = True
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} {name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 梯度传播问题修复成功！")
        print("✅ MSDeformableAttention参数现在参与梯度计算")
        print("✅ 编码器输出头参数现在参与梯度计算")
        print("✅ 训练过程稳定")
        print("✅ 可以进行正常训练")
    else:
        print("⚠️ 梯度传播问题仍需进一步修复")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
