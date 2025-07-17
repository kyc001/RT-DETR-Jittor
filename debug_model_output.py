#!/usr/bin/env python3
"""
调试模型输出格式
检查训练好的模型实际输出什么，找出问题所在
"""

import os
import sys
import numpy as np
from PIL import Image

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

def debug_jittor_model():
    """调试Jittor模型输出"""
    try:
        import jittor as jt
        from jittor_rt_detr.src.nn.backbone.resnet import ResNet50
        from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
        import jittor.nn as nn
        
        jt.flags.use_cuda = 1
        
        class RTDETRModel(nn.Module):
            def __init__(self, num_classes=80):
                super().__init__()
                self.backbone = ResNet50(pretrained=False)
                self.transformer = RTDETRTransformer(
                    num_classes=num_classes,
                    hidden_dim=256,
                    num_queries=300,
                    feat_channels=[256, 512, 1024, 2048]
                )
            
            def execute(self, x, targets=None):
                features = self.backbone(x)
                return self.transformer(features, targets)
        
        model = RTDETRModel(num_classes=80)
        
        # 加载训练好的权重
        model_path = "/home/kyc/project/RT-DETR/results/jittor_finetune/rtdetr_jittor_finetune_50img_50epoch.pkl"
        if os.path.exists(model_path):
            model.load(model_path)
            print(f"✅ 成功加载模型: {model_path}")
        else:
            print(f"❌ 模型文件不存在: {model_path}")
            return
        
        model.eval()
        
        # 创建测试输入
        test_input = jt.randn(1, 3, 640, 640)
        
        print(f"\n🔍 调试模型输出...")
        print(f"输入形状: {test_input.shape}")
        
        # 前向传播
        with jt.no_grad():
            outputs = model(test_input)
        
        print(f"\n📊 模型输出分析:")
        print(f"输出类型: {type(outputs)}")
        
        if isinstance(outputs, dict):
            print(f"输出键: {list(outputs.keys())}")
            for key, value in outputs.items():
                print(f"  {key}: 形状={value.shape}, 类型={type(value)}")
                print(f"    数据范围: [{value.min():.4f}, {value.max():.4f}]")
                print(f"    数据类型: {value.dtype}")
                
                # 检查是否有NaN或Inf
                if hasattr(value, 'numpy'):
                    val_np = value.numpy()
                    nan_count = np.isnan(val_np).sum()
                    inf_count = np.isinf(val_np).sum()
                    print(f"    NaN数量: {nan_count}, Inf数量: {inf_count}")
        else:
            print(f"输出不是字典类型: {outputs}")
        
        # 测试后处理
        print(f"\n🔧 测试后处理...")
        try:
            if 'pred_logits' in outputs and 'pred_boxes' in outputs:
                pred_logits = outputs['pred_logits'][0]
                pred_boxes = outputs['pred_boxes'][0]
                
                print(f"pred_logits形状: {pred_logits.shape}")
                print(f"pred_boxes形状: {pred_boxes.shape}")
                
                # 计算softmax
                scores = jt.nn.softmax(pred_logits, dim=-1)
                print(f"scores形状: {scores.shape}")
                
                # 尝试获取最大值
                try:
                    max_scores, predicted_labels = jt.max(scores[:, :-1], dim=-1)
                    print(f"max_scores形状: {max_scores.shape}")
                    print(f"predicted_labels形状: {predicted_labels.shape}")
                    print(f"最大置信度: {max_scores.max():.4f}")
                    print(f"最小置信度: {max_scores.min():.4f}")
                    
                    # 检查有多少高置信度预测
                    high_conf_mask = max_scores > 0.3
                    high_conf_count = high_conf_mask.sum()
                    print(f"置信度>0.3的预测数: {high_conf_count}")
                    
                    if high_conf_count > 0:
                        print(f"✅ 模型有高置信度预测")
                    else:
                        print(f"❌ 模型没有高置信度预测")
                        
                except Exception as e:
                    print(f"❌ jt.max失败: {e}")
                    
            else:
                print(f"❌ 输出缺少必要的键")
                
        except Exception as e:
            print(f"❌ 后处理测试失败: {e}")
        
        # 测试真实图像
        print(f"\n🖼️ 测试真实图像...")
        test_img_path = "/home/kyc/project/RT-DETR/data/coco2017_50/val2017/000000369771.jpg"
        if os.path.exists(test_img_path):
            image = Image.open(test_img_path).convert('RGB')
            image_resized = image.resize((640, 640), Image.LANCZOS)
            img_array = np.array(image_resized).astype(np.float32) / 255.0
            img_tensor = jt.array(img_array.transpose(2, 0, 1)).float32().unsqueeze(0)
            
            with jt.no_grad():
                real_outputs = model(img_tensor)
            
            print(f"真实图像输出:")
            for key, value in real_outputs.items():
                print(f"  {key}: 形状={value.shape}")
                if key == 'pred_logits':
                    scores = jt.nn.softmax(value[0], dim=-1)
                    max_scores = jt.max(scores[:, :-1], dim=-1)[0]
                    print(f"    最大置信度: {max_scores.max():.4f}")
                    high_conf = (max_scores > 0.3).sum()
                    print(f"    高置信度预测数: {high_conf}")
        
        return True
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_training_data():
    """调试训练数据"""
    print(f"\n📊 调试训练数据...")
    
    train_ann_file = "/home/kyc/project/RT-DETR/data/coco2017_50/annotations/instances_train2017.json"
    if not os.path.exists(train_ann_file):
        print(f"❌ 训练标注文件不存在")
        return
    
    import json
    with open(train_ann_file, 'r') as f:
        coco_data = json.load(f)
    
    print(f"训练图像数: {len(coco_data['images'])}")
    print(f"训练标注数: {len(coco_data['annotations'])}")
    print(f"类别数: {len(coco_data['categories'])}")
    
    # 检查类别分布
    category_counts = {}
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
    
    print(f"\n类别分布 (前10个):")
    sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for cat_id, count in sorted_cats:
        print(f"  类别{cat_id}: {count}个标注")
    
    # 检查边界框
    bbox_areas = []
    for ann in coco_data['annotations']:
        x, y, w, h = ann['bbox']
        area = w * h
        bbox_areas.append(area)
    
    bbox_areas = np.array(bbox_areas)
    print(f"\n边界框统计:")
    print(f"  平均面积: {bbox_areas.mean():.2f}")
    print(f"  最小面积: {bbox_areas.min():.2f}")
    print(f"  最大面积: {bbox_areas.max():.2f}")
    print(f"  面积标准差: {bbox_areas.std():.2f}")

def debug_loss_function():
    """调试损失函数"""
    print(f"\n🔧 调试损失函数...")
    
    # 检查训练脚本中的损失函数
    train_script = "/home/kyc/project/RT-DETR/jittor_finetune_train.py"
    if os.path.exists(train_script):
        with open(train_script, 'r') as f:
            content = f.read()
        
        # 查找损失相关代码
        if 'loss' in content.lower():
            print(f"✅ 训练脚本包含损失计算")
        else:
            print(f"❌ 训练脚本可能缺少损失计算")
        
        # 检查关键函数
        key_functions = ['focal_loss', 'bbox_loss', 'giou_loss', 'criterion']
        for func in key_functions:
            if func in content:
                print(f"  ✅ 找到函数: {func}")
            else:
                print(f"  ❌ 缺少函数: {func}")

def main():
    print("🚨 RT-DETR模型失败调试")
    print("=" * 60)
    
    # 调试模型输出
    model_ok = debug_jittor_model()
    
    # 调试训练数据
    debug_training_data()
    
    # 调试损失函数
    debug_loss_function()
    
    print(f"\n📝 调试总结:")
    if model_ok:
        print(f"  ✅ 模型加载成功")
    else:
        print(f"  ❌ 模型加载失败")
    
    print(f"\n🔧 建议的修复步骤:")
    print(f"  1. 检查模型输出格式是否正确")
    print(f"  2. 验证后处理逻辑")
    print(f"  3. 检查损失函数计算")
    print(f"  4. 验证数据加载和预处理")
    print(f"  5. 重新训练模型")

if __name__ == "__main__":
    main()
