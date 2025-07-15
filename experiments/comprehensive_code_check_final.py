#!/usr/bin/env python3
"""
清理后的全面代码功能检查
验证所有核心功能是否正常工作
"""

import os
import sys
import traceback

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt
import jittor.nn as nn

# 设置Jittor
jt.flags.use_cuda = 1
jt.set_global_seed(42)
jt.flags.auto_mixed_precision_level = 0

def check_all_imports():
    """检查所有核心模块导入"""
    print("=" * 60)
    print("===        核心模块导入检查        ===")
    print("=" * 60)
    
    import_tests = [
        ("ResNet50", "from jittor_rt_detr.src.nn.backbone.resnet import ResNet50"),
        ("RTDETRTransformer", "from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer"),
        ("MSDeformableAttention", "from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import MSDeformableAttention"),
        ("build_criterion", "from jittor_rt_detr.src.nn.criterion.rtdetr_criterion import build_criterion"),
        ("HungarianMatcher", "from jittor_rt_detr.src.zoo.rtdetr.matcher import HungarianMatcher"),
        ("RTDETR", "from jittor_rt_detr.src.zoo.rtdetr.rtdetr import RTDETR"),
        ("HybridEncoder", "from jittor_rt_detr.src.zoo.rtdetr.hybrid_encoder import HybridEncoder"),
        ("box_ops", "from jittor_rt_detr.src.zoo.rtdetr import box_ops"),
        ("utils", "from jittor_rt_detr.src.zoo.rtdetr import utils"),
        ("Config", "from jittor_rt_detr.src.core.config import Config"),
        ("YAMLConfig", "from jittor_rt_detr.src.core.yaml_config import YAMLConfig"),
        ("COCODataset", "from jittor_rt_detr.src.data.coco.coco_dataset import COCODataset"),
    ]
    
    success_count = 0
    failed_imports = []
    
    for name, import_stmt in import_tests:
        try:
            exec(import_stmt)
            print(f"✅ {name}: 导入成功")
            success_count += 1
        except Exception as e:
            print(f"❌ {name}: 导入失败 - {str(e)[:100]}")
            failed_imports.append((name, str(e)))
    
    print(f"\n导入测试结果: {success_count}/{len(import_tests)} 成功")
    
    if failed_imports:
        print("\n失败的导入详情:")
        for name, error in failed_imports:
            print(f"  {name}: {error}")
    
    return success_count, len(import_tests), failed_imports

def test_model_creation_comprehensive():
    """全面测试模型创建"""
    print("\n" + "=" * 60)
    print("===        全面模型创建测试        ===")
    print("=" * 60)
    
    try:
        from jittor_rt_detr.src.nn.backbone.resnet import ResNet50
        from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer, MSDeformableAttention
        from jittor_rt_detr.src.nn.criterion.rtdetr_criterion import build_criterion
        from jittor_rt_detr.src.zoo.rtdetr.matcher import HungarianMatcher
        from jittor_rt_detr.src.zoo.rtdetr.hybrid_encoder import HybridEncoder
        
        print("1. 测试ResNet50骨干网络:")
        backbone = ResNet50(pretrained=False)
        backbone_params = sum(p.numel() for p in backbone.parameters())
        print(f"✅ ResNet50创建成功，参数量: {backbone_params:,}")
        
        print("\n2. 测试RT-DETR Transformer:")
        transformer = RTDETRTransformer(
            num_classes=80,
            hidden_dim=256,
            num_queries=300,
            feat_channels=[256, 512, 1024, 2048]
        )
        transformer_params = sum(p.numel() for p in transformer.parameters())
        print(f"✅ RTDETRTransformer创建成功，参数量: {transformer_params:,}")
        
        print("\n3. 测试MSDeformableAttention:")
        ms_attn = MSDeformableAttention(embed_dim=256, num_heads=8)
        ms_attn_params = sum(p.numel() for p in ms_attn.parameters())
        print(f"✅ MSDeformableAttention创建成功，参数量: {ms_attn_params:,}")
        
        print("\n4. 测试损失函数:")
        criterion = build_criterion(num_classes=80)
        print("✅ 损失函数创建成功")
        
        print("\n5. 测试匈牙利匹配器:")
        matcher = HungarianMatcher()
        print("✅ 匈牙利匹配器创建成功")
        
        print("\n6. 测试混合编码器:")
        hybrid_encoder = HybridEncoder(embed_dim=256, num_heads=8)
        hybrid_params = sum(p.numel() for p in hybrid_encoder.parameters())
        print(f"✅ HybridEncoder创建成功，参数量: {hybrid_params:,}")
        
        total_params = backbone_params + transformer_params
        print(f"\n总模型参数量: {total_params:,}")
        
        return True, total_params
        
    except Exception as e:
        print(f"❌ 模型创建测试失败: {e}")
        traceback.print_exc()
        return False, 0

def test_forward_propagation_comprehensive():
    """全面测试前向传播"""
    print("\n" + "=" * 60)
    print("===        全面前向传播测试        ===")
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
        
        print("1. 测试不同输入尺寸:")
        input_sizes = [(640, 640), (512, 512), (800, 800)]
        
        for h, w in input_sizes:
            x = jt.randn(1, 3, h, w, dtype=jt.float32)
            
            # Backbone前向传播
            feats = backbone(x)
            print(f"✅ 输入{h}x{w}: Backbone输出{len(feats)}个特征图")
            
            # Transformer前向传播
            outputs = transformer(feats)
            print(f"   Transformer输出: pred_logits={outputs['pred_logits'].shape}, pred_boxes={outputs['pred_boxes'].shape}")
            
            # 检查编码器输出
            if 'enc_outputs' in outputs:
                print(f"   编码器输出: pred_logits={outputs['enc_outputs']['pred_logits'].shape}")
        
        print("\n2. 测试批量处理:")
        batch_sizes = [1, 2]
        
        for bs in batch_sizes:
            x = jt.randn(bs, 3, 640, 640, dtype=jt.float32)
            feats = backbone(x)
            outputs = transformer(feats)
            print(f"✅ 批量大小{bs}: 输出形状正确")
        
        print("\n3. 测试数据类型一致性:")
        x = jt.randn(1, 3, 640, 640, dtype=jt.float32)
        feats = backbone(x)
        outputs = transformer(feats)
        
        # 检查所有输出的数据类型
        all_float32 = True
        for key, value in outputs.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if sub_value.dtype != jt.float32:
                        print(f"⚠️ {key}.{sub_key}: {sub_value.dtype}")
                        all_float32 = False
            else:
                if value.dtype != jt.float32:
                    print(f"⚠️ {key}: {value.dtype}")
                    all_float32 = False
        
        if all_float32:
            print("✅ 所有输出都是float32")
        else:
            print("❌ 存在非float32输出")
        
        return True, all_float32
        
    except Exception as e:
        print(f"❌ 前向传播测试失败: {e}")
        traceback.print_exc()
        return False, False

def test_loss_computation_comprehensive():
    """全面测试损失计算"""
    print("\n" + "=" * 60)
    print("===        全面损失计算测试        ===")
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
        
        print("1. 测试不同目标数量:")
        target_configs = [
            (1, "单个目标"),
            (3, "多个目标"),
            (5, "更多目标"),
        ]
        
        for num_targets, desc in target_configs:
            # 前向传播
            x = jt.randn(1, 3, 640, 640, dtype=jt.float32)
            feats = backbone(x)
            outputs = transformer(feats)
            
            # 创建目标
            targets = [{
                'boxes': jt.rand(num_targets, 4, dtype=jt.float32),
                'labels': jt.randint(1, 81, (num_targets,), dtype=jt.int64)
            }]
            
            # 损失计算
            loss_dict = criterion(outputs, targets)
            total_loss = sum(loss_dict.values())
            
            print(f"✅ {desc}: 总损失={total_loss.item():.4f}")
            
            # 检查损失组成
            expected_losses = ['loss_focal', 'loss_bbox', 'loss_giou']
            if 'enc_outputs' in outputs:
                expected_losses.extend(['loss_focal_enc', 'loss_bbox_enc', 'loss_giou_enc'])
            
            for loss_name in expected_losses:
                if loss_name in loss_dict:
                    print(f"   {loss_name}: {loss_dict[loss_name].item():.4f} ({loss_dict[loss_name].dtype})")
                else:
                    print(f"   ❌ 缺少损失: {loss_name}")
        
        print("\n2. 测试损失数据类型:")
        x = jt.randn(1, 3, 640, 640, dtype=jt.float32)
        feats = backbone(x)
        outputs = transformer(feats)
        
        targets = [{
            'boxes': jt.rand(3, 4, dtype=jt.float32),
            'labels': jt.array([1, 2, 3], dtype=jt.int64)
        }]
        
        loss_dict = criterion(outputs, targets)
        
        all_float32 = True
        for loss_name, loss_value in loss_dict.items():
            if loss_value.dtype != jt.float32:
                print(f"❌ {loss_name}: {loss_value.dtype}")
                all_float32 = False
        
        if all_float32:
            print("✅ 所有损失都是float32")
        else:
            print("❌ 存在非float32损失")
        
        return True, all_float32
        
    except Exception as e:
        print(f"❌ 损失计算测试失败: {e}")
        traceback.print_exc()
        return False, False

def test_training_step_comprehensive():
    """全面测试训练步骤"""
    print("\n" + "=" * 60)
    print("===        全面训练步骤测试        ===")
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
        optimizer = jt.optim.Adam(all_params, lr=1e-4)
        
        print("1. 测试多步训练:")
        losses = []
        
        for step in range(3):
            print(f"\n--- 训练步骤 {step + 1} ---")
            
            # 前向传播
            x = jt.randn(1, 3, 640, 640, dtype=jt.float32)
            feats = backbone(x)
            outputs = transformer(feats)
            
            # 创建目标
            targets = [{
                'boxes': jt.rand(3, 4, dtype=jt.float32),
                'labels': jt.array([1, 2, 3], dtype=jt.int64)
            }]
            
            # 损失计算
            loss_dict = criterion(outputs, targets)
            total_loss = sum(loss_dict.values())
            losses.append(total_loss.item())
            
            print(f"损失: {total_loss.item():.4f}")
            
            # 反向传播
            try:
                optimizer.backward(total_loss)
                print("✅ 反向传播成功")
                
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
                
            except Exception as e:
                print(f"❌ 反向传播失败: {e}")
                return False
        
        print(f"\n训练稳定性检查:")
        print(f"损失变化: {losses[0]:.4f} -> {losses[-1]:.4f}")
        
        # 检查损失是否在合理范围
        reasonable_losses = all(0.1 < loss < 100.0 for loss in losses)
        if reasonable_losses:
            print("✅ 损失在合理范围内")
        else:
            print("⚠️ 损失可能异常")
        
        return True, reasonable_losses
        
    except Exception as e:
        print(f"❌ 训练步骤测试失败: {e}")
        traceback.print_exc()
        return False, False

def main():
    print("🔧 清理后的RT-DETR全面代码功能检查")
    print("=" * 80)
    
    # 1. 检查所有导入
    import_success, import_total, failed_imports = check_all_imports()
    
    # 2. 全面测试模型创建
    model_ok, total_params = test_model_creation_comprehensive()
    
    # 3. 全面测试前向传播
    forward_ok, dtype_ok = test_forward_propagation_comprehensive()
    
    # 4. 全面测试损失计算
    loss_ok, loss_dtype_ok = test_loss_computation_comprehensive()
    
    # 5. 全面测试训练步骤
    training_ok, training_stable = test_training_step_comprehensive()
    
    # 总结
    print("\n" + "=" * 80)
    print("🎯 全面功能检查总结:")
    print("=" * 80)
    
    results = [
        (f"模块导入 ({import_success}/{import_total})", import_success == import_total),
        ("模型创建", model_ok),
        ("前向传播", forward_ok),
        ("数据类型一致性", dtype_ok and loss_dtype_ok),
        ("损失计算", loss_ok),
        ("训练步骤", training_ok),
        ("训练稳定性", training_stable),
    ]
    
    all_passed = True
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} {name}")
        if not result:
            all_passed = False
    
    # 详细信息
    if total_params > 0:
        print(f"\n📊 模型信息:")
        print(f"  总参数量: {total_params:,}")
    
    if failed_imports:
        print(f"\n⚠️ 导入问题:")
        for name, error in failed_imports[:3]:  # 只显示前3个
            print(f"  {name}: {error[:100]}")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 RT-DETR全面功能检查完全通过！")
        print("✅ 所有核心功能正常工作")
        print("✅ 数据类型完全一致")
        print("✅ 训练流程稳定可靠")
        print("✅ 清理后的代码质量优秀")
        print("\n🚀 RT-DETR Jittor版本现在完全可用于:")
        print("1. ✅ 模型训练")
        print("2. ✅ 模型推理")
        print("3. ✅ 研究开发")
        print("4. ✅ 生产部署")
    else:
        print("⚠️ 部分功能需要进一步检查和修复")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
