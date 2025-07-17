#!/usr/bin/env python3
"""
调试模型权重加载问题
"""

import os
import sys
import numpy as np

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt

# 设置Jittor
jt.flags.use_cuda = 1

def check_saved_model():
    """检查保存的模型权重"""
    print("🔍 检查保存的模型权重...")
    
    model_path = '/home/kyc/project/RT-DETR/results/full_training/rtdetr_trained.pkl'
    
    try:
        checkpoint = jt.load(model_path)
        
        print(f"✅ 模型文件加载成功")
        print(f"📊 检查点内容:")
        for key in checkpoint.keys():
            print(f"   {key}: {type(checkpoint[key])}")
        
        # 检查backbone权重
        if 'backbone_state_dict' in checkpoint:
            backbone_state = checkpoint['backbone_state_dict']
            print(f"\n🔧 Backbone权重:")
            print(f"   参数数量: {len(backbone_state)}")
            
            # 显示前几个参数
            for i, (name, param) in enumerate(list(backbone_state.items())[:5]):
                if hasattr(param, 'shape'):
                    print(f"   {name}: {param.shape}")
                else:
                    print(f"   {name}: {type(param)}")
        
        # 检查transformer权重
        if 'transformer_state_dict' in checkpoint:
            transformer_state = checkpoint['transformer_state_dict']
            print(f"\n🔧 Transformer权重:")
            print(f"   参数数量: {len(transformer_state)}")
            
            # 显示前几个参数
            for i, (name, param) in enumerate(list(transformer_state.items())[:5]):
                if hasattr(param, 'shape'):
                    print(f"   {name}: {param.shape}")
                else:
                    print(f"   {name}: {type(param)}")
        
        return checkpoint
        
    except Exception as e:
        print(f"❌ 模型文件检查失败: {e}")
        return None

def create_fresh_model():
    """创建新的模型（参考ultimate_sanity_check.py）"""
    print("\n🔄 创建新模型...")
    
    try:
        from jittor_rt_detr.src.nn.backbone.resnet import ResNet50
        from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
        
        # 创建模型（完全按照ultimate_sanity_check.py的方式）
        backbone = ResNet50(pretrained=False)
        transformer = RTDETRTransformer(
            num_classes=80,
            hidden_dim=256,
            num_queries=300,
            feat_channels=[256, 512, 1024, 2048]
        )
        
        print(f"✅ 新模型创建成功")
        
        # 检查参数数量
        backbone_params = sum(p.numel() for p in backbone.parameters())
        transformer_params = sum(p.numel() for p in transformer.parameters())
        
        print(f"📊 模型参数:")
        print(f"   Backbone参数: {backbone_params:,}")
        print(f"   Transformer参数: {transformer_params:,}")
        print(f"   总参数: {backbone_params + transformer_params:,}")
        
        return backbone, transformer
        
    except Exception as e:
        print(f"❌ 新模型创建失败: {e}")
        return None, None

def compare_models(checkpoint, fresh_backbone, fresh_transformer):
    """比较保存的权重和新模型"""
    print("\n🔍 比较模型权重...")
    
    try:
        # 加载权重到新模型
        if 'backbone_state_dict' in checkpoint and 'transformer_state_dict' in checkpoint:
            print("🔄 加载保存的权重...")
            
            fresh_backbone.load_state_dict(checkpoint['backbone_state_dict'])
            fresh_transformer.load_state_dict(checkpoint['transformer_state_dict'])
            
            print("✅ 权重加载成功")
            
            # 测试推理
            print("\n🧪 测试推理...")
            
            # 创建虚拟输入
            dummy_input = jt.randn(1, 3, 640, 640)
            
            # 设置评估模式
            fresh_backbone.eval()
            fresh_transformer.eval()
            
            with jt.no_grad():
                # 前向传播
                features = fresh_backbone(dummy_input)
                outputs = fresh_transformer(features)
                
                # 获取预测结果
                pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
                pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4]
                
                # 后处理
                pred_scores = jt.nn.softmax(pred_logits, dim=-1)
                pred_scores_no_bg = pred_scores[:, :-1]  # 排除背景类
                
                # 获取最高分数的类别
                max_result = jt.max(pred_scores_no_bg, dim=-1)
                if isinstance(max_result, tuple):
                    max_scores = max_result[0]
                else:
                    max_scores = max_result

                argmax_result = jt.argmax(pred_scores_no_bg, dim=-1)
                if isinstance(argmax_result, tuple):
                    pred_classes = argmax_result[0]
                else:
                    pred_classes = argmax_result
                
                # 转换为numpy
                scores_np = max_scores.numpy()
                classes_np = pred_classes.numpy()
                
                print(f"📊 推理结果:")
                print(f"   分数范围: {scores_np.min():.4f} - {scores_np.max():.4f}")
                print(f"   类别索引范围: {classes_np.min()} - {classes_np.max()}")
                
                # 统计类别分布
                unique_classes, counts = np.unique(classes_np, return_counts=True)
                print(f"   类别分布 (前5个):")
                sorted_indices = np.argsort(counts)[::-1][:5]
                for i, idx in enumerate(sorted_indices):
                    class_idx = unique_classes[idx]
                    count = counts[idx]
                    print(f"     类别{class_idx}: {count}次")
                
                # 检查是否所有预测都是同一个类别
                if len(unique_classes) == 1:
                    print(f"⚠️ 警告: 所有预测都是类别{unique_classes[0]}")
                    return False
                else:
                    print(f"✅ 预测类别多样化: {len(unique_classes)}个不同类别")
                    return True
                
        else:
            print("❌ 检查点中缺少必要的权重")
            return False
            
    except Exception as e:
        print(f"❌ 模型比较失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_image(backbone, transformer):
    """使用真实图像测试"""
    print("\n🖼️ 使用真实图像测试...")
    
    try:
        import json
        from PIL import Image
        
        # 加载真实图像
        data_dir = '/home/kyc/project/RT-DETR/data/coco2017_50'
        images_dir = os.path.join(data_dir, "train2017")
        annotations_file = os.path.join(data_dir, "annotations", "instances_train2017.json")
        
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # 获取第一张图像
        first_image = coco_data['images'][0]
        image_path = os.path.join(images_dir, first_image['file_name'])
        
        print(f"📷 测试图像: {first_image['file_name']}")
        
        # 加载并预处理图像
        image = Image.open(image_path).convert('RGB')
        image_resized = image.resize((640, 640))
        image_array = np.array(image_resized).astype(np.float32) / 255.0
        image_tensor = jt.array(image_array.transpose(2, 0, 1)).unsqueeze(0)
        
        # 推理
        backbone.eval()
        transformer.eval()
        
        with jt.no_grad():
            features = backbone(image_tensor)
            outputs = transformer(features)
            
            # 后处理
            pred_logits = outputs['pred_logits'][0]
            pred_scores = jt.nn.softmax(pred_logits, dim=-1)
            pred_scores_no_bg = pred_scores[:, :-1]
            
            max_result = jt.max(pred_scores_no_bg, dim=-1)
            if isinstance(max_result, tuple):
                max_scores = max_result[0]
            else:
                max_scores = max_result

            argmax_result = jt.argmax(pred_scores_no_bg, dim=-1)
            if isinstance(argmax_result, tuple):
                pred_classes = argmax_result[0]
            else:
                pred_classes = argmax_result
            
            scores_np = max_scores.numpy()
            classes_np = pred_classes.numpy()
            
            print(f"📊 真实图像推理结果:")
            print(f"   分数范围: {scores_np.min():.4f} - {scores_np.max():.4f}")
            print(f"   类别索引范围: {classes_np.min()} - {classes_np.max()}")
            
            # 统计类别分布
            unique_classes, counts = np.unique(classes_np, return_counts=True)
            print(f"   类别分布:")
            sorted_indices = np.argsort(counts)[::-1][:10]
            for i, idx in enumerate(sorted_indices):
                class_idx = unique_classes[idx]
                count = counts[idx]
                print(f"     类别{class_idx}: {count}次")
            
            # 显示最高分数的预测
            top_indices = np.argsort(scores_np)[::-1][:5]
            print(f"   前5个最高分数预测:")
            for i, idx in enumerate(top_indices):
                class_idx = classes_np[idx]
                score = scores_np[idx]
                print(f"     {i+1}: 类别{class_idx}, 分数{score:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 真实图像测试失败: {e}")
        return False

def main():
    print("🔍 RT-DETR模型权重调试")
    print("=" * 60)
    
    # 1. 检查保存的模型
    checkpoint = check_saved_model()
    if checkpoint is None:
        return
    
    # 2. 创建新模型
    fresh_backbone, fresh_transformer = create_fresh_model()
    if fresh_backbone is None:
        return
    
    # 3. 比较模型
    comparison_success = compare_models(checkpoint, fresh_backbone, fresh_transformer)
    
    # 4. 使用真实图像测试
    real_image_success = test_with_real_image(fresh_backbone, fresh_transformer)
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 调试总结")
    print("=" * 60)
    
    print(f"✅ 模型文件加载: 成功")
    print(f"✅ 新模型创建: 成功")
    print(f"{'✅' if comparison_success else '❌'} 权重加载和推理: {'成功' if comparison_success else '失败'}")
    print(f"{'✅' if real_image_success else '❌'} 真实图像测试: {'成功' if real_image_success else '失败'}")
    
    if not comparison_success:
        print("\n💡 建议:")
        print("   1. 检查模型权重是否正确保存")
        print("   2. 检查模型结构是否匹配")
        print("   3. 重新训练模型")

if __name__ == "__main__":
    main()
