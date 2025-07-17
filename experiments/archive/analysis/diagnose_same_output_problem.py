#!/usr/bin/env python3
"""
诊断模型对每张图像输出相同结果的问题
"""

import os
import sys
import json
import numpy as np
from PIL import Image
from collections import Counter

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt

# 设置Jittor
jt.flags.use_cuda = 1

def load_model():
    """加载最新训练的模型"""
    print("🔄 加载最新训练的模型...")
    
    try:
        from jittor_rt_detr.src.nn.backbone.resnet import ResNet50
        from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
        
        # 创建模型架构
        backbone = ResNet50(pretrained=False)
        transformer = RTDETRTransformer(
            num_classes=80,
            hidden_dim=256,
            num_queries=300,
            feat_channels=[256, 512, 1024, 2048]
        )
        
        # 尝试加载最新的模型权重
        model_paths = [
            '/home/kyc/project/RT-DETR/results/sanity_check_training/rtdetr_10img_fixed_200epoch.pkl',
            '/home/kyc/project/RT-DETR/results/sanity_check_training/rtdetr_10img_150epoch.pkl',
            '/home/kyc/project/RT-DETR/results/sanity_check_training/rtdetr_5img_100epoch.pkl'
        ]
        
        loaded_model = None
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    checkpoint = jt.load(model_path)
                    backbone.load_state_dict(checkpoint['backbone_state_dict'])
                    transformer.load_state_dict(checkpoint['transformer_state_dict'])
                    
                    backbone.eval()
                    transformer.eval()
                    
                    print(f"✅ 成功加载模型: {model_path}")
                    print(f"   训练轮数: {checkpoint.get('epoch', 'Unknown')}")
                    print(f"   训练损失: {checkpoint.get('loss', 'Unknown'):.4f}")
                    
                    loaded_model = model_path
                    break
                except Exception as e:
                    print(f"⚠️ 无法加载 {model_path}: {e}")
                    continue
        
        if loaded_model is None:
            print("❌ 无法加载任何模型，使用未训练的模型进行测试")
        
        return backbone, transformer, loaded_model
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return None, None, None

def convert_to_coco_class(class_idx):
    """将模型类别索引转换为COCO类别ID和名称"""
    COCO_CLASSES = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
        21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
        27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
        34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
        39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
        43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup',
        48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
        53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
        58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
        63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
        70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
        76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
        80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
        85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
        89: 'hair drier', 90: 'toothbrush'
    }
    
    if class_idx == 0:
        coco_id = 1  # person
    elif class_idx == 2:
        coco_id = 3  # car
    elif class_idx == 26:
        coco_id = 27  # backpack
    elif class_idx == 32:
        coco_id = 33  # suitcase
    elif class_idx == 83:
        coco_id = 84  # book
    else:
        coco_id = class_idx + 1
    
    class_name = COCO_CLASSES.get(coco_id, f'class_{coco_id}')
    return coco_id, class_name

def detailed_inference_analysis(backbone, transformer):
    """详细分析模型对不同图像的推理结果"""
    print("\n🔍 详细推理分析")
    print("=" * 60)
    print("分析: 模型对不同图像是否产生不同的内部表示和输出")
    
    try:
        # 加载测试图像
        data_dir = '/home/kyc/project/RT-DETR/data/coco2017_50'
        images_dir = os.path.join(data_dir, "train2017")
        annotations_file = os.path.join(data_dir, "annotations", "instances_train2017.json")
        
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # 选择5张不同的图像进行测试
        test_images = coco_data['images'][:5]
        
        all_results = []
        
        for i, img_info in enumerate(test_images):
            print(f"\n📊 图像 {i+1}: {img_info['file_name']}")
            
            try:
                # 加载图像
                image_path = os.path.join(images_dir, img_info['file_name'])
                image = Image.open(image_path).convert('RGB')
                original_size = image.size
                
                # 预处理图像
                image_resized = image.resize((640, 640))
                image_array = np.array(image_resized).astype(np.float32) / 255.0
                image_tensor = jt.array(image_array.transpose(2, 0, 1)).unsqueeze(0)
                
                print(f"   原始尺寸: {original_size}")
                print(f"   输入张量形状: {image_tensor.shape}")
                
                with jt.no_grad():
                    # 前向传播
                    features = backbone(image_tensor)
                    outputs = transformer(features)
                    
                    # 分析backbone特征
                    print(f"   Backbone特征:")
                    for j, feat in enumerate(features):
                        feat_mean = feat.mean().item()
                        feat_std = feat.std().item()
                        print(f"     层{j}: 均值={feat_mean:.6f}, 标准差={feat_std:.6f}")
                    
                    # 分析transformer输出
                    pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
                    pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4]
                    
                    print(f"   Transformer输出:")
                    print(f"     pred_logits形状: {pred_logits.shape}")
                    print(f"     pred_boxes形状: {pred_boxes.shape}")
                    
                    # 分析logits分布
                    logits_mean = pred_logits.mean().item()
                    logits_std = pred_logits.std().item()
                    logits_max = pred_logits.max().item()
                    logits_min = pred_logits.min().item()
                    
                    print(f"     logits统计: 均值={logits_mean:.6f}, 标准差={logits_std:.6f}")
                    print(f"     logits范围: [{logits_min:.6f}, {logits_max:.6f}]")
                    
                    # 分析boxes分布
                    boxes_mean = pred_boxes.mean().item()
                    boxes_std = pred_boxes.std().item()
                    boxes_max = pred_boxes.max().item()
                    boxes_min = pred_boxes.min().item()
                    
                    print(f"     boxes统计: 均值={boxes_mean:.6f}, 标准差={boxes_std:.6f}")
                    print(f"     boxes范围: [{boxes_min:.6f}, {boxes_max:.6f}]")
                    
                    # 后处理
                    pred_scores = jt.nn.softmax(pred_logits, dim=-1)
                    pred_scores_no_bg = pred_scores[:, :-1]  # 排除背景类
                    
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
                    boxes_np = pred_boxes.numpy()
                    
                    print(f"   后处理结果:")
                    print(f"     分数范围: [{scores_np.min():.6f}, {scores_np.max():.6f}]")
                    print(f"     类别范围: [{classes_np.min()}, {classes_np.max()}]")
                    
                    # 获取前5个最高分数的预测
                    top_indices = np.argsort(scores_np)[::-1][:5]
                    print(f"     前5个预测:")
                    
                    top_predictions = []
                    for j, idx in enumerate(top_indices):
                        class_idx = classes_np[idx]
                        score = scores_np[idx]
                        box = boxes_np[idx]
                        coco_id, class_name = convert_to_coco_class(class_idx)
                        
                        print(f"       {j+1}: {class_name} (分数: {score:.6f}, 框: [{box[0]:.3f},{box[1]:.3f},{box[2]:.3f},{box[3]:.3f}])")
                        top_predictions.append((class_name, score, box.tolist()))
                    
                    # 记录结果
                    all_results.append({
                        'file_name': img_info['file_name'],
                        'logits_stats': (logits_mean, logits_std, logits_min, logits_max),
                        'boxes_stats': (boxes_mean, boxes_std, boxes_min, boxes_max),
                        'score_range': (scores_np.min(), scores_np.max()),
                        'top_predictions': top_predictions
                    })
                    
            except Exception as e:
                print(f"   ❌ 处理失败: {e}")
                continue
        
        # 分析结果相似性
        print(f"\n📈 相似性分析:")
        print("=" * 60)
        
        if len(all_results) >= 2:
            # 比较logits统计
            logits_means = [r['logits_stats'][0] for r in all_results]
            logits_stds = [r['logits_stats'][1] for r in all_results]
            
            logits_mean_diff = max(logits_means) - min(logits_means)
            logits_std_diff = max(logits_stds) - min(logits_stds)
            
            print(f"Logits统计差异:")
            print(f"   均值差异: {logits_mean_diff:.8f}")
            print(f"   标准差差异: {logits_std_diff:.8f}")
            
            if logits_mean_diff < 1e-6 and logits_std_diff < 1e-6:
                print(f"   ❌ Logits几乎完全相同 - 模型输出固定模式")
            else:
                print(f"   ✅ Logits有差异 - 模型对不同图像有不同响应")
            
            # 比较预测结果
            print(f"\n预测结果比较:")
            first_predictions = [pred[0] for pred in all_results[0]['top_predictions']]
            
            all_same = True
            for i, result in enumerate(all_results[1:], 1):
                current_predictions = [pred[0] for pred in result['top_predictions']]
                if current_predictions != first_predictions:
                    all_same = False
                    print(f"   图像{i+1}与图像1的预测不同")
                    break
            
            if all_same:
                print(f"   ❌ 所有图像的预测完全相同")
                print(f"   固定预测模式: {first_predictions}")
            else:
                print(f"   ✅ 不同图像产生不同预测")
        
        return all_results
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def suggest_solutions():
    """建议解决方案"""
    print(f"\n🔧 解决方案建议:")
    print("=" * 60)
    
    print(f"1. **检查训练数据**:")
    print(f"   - 确保训练图像真的不同")
    print(f"   - 检查标注是否正确")
    print(f"   - 验证数据预处理是否有问题")
    
    print(f"\n2. **调整训练策略**:")
    print(f"   - 进一步降低学习率 (5e-5 或更低)")
    print(f"   - 增加数据增强 (随机裁剪、颜色变换)")
    print(f"   - 使用更强的正则化")
    
    print(f"\n3. **模型架构调整**:")
    print(f"   - 检查BatchNorm是否正确设置")
    print(f"   - 验证损失函数是否合适")
    print(f"   - 考虑使用不同的优化器")
    
    print(f"\n4. **训练过程监控**:")
    print(f"   - 监控每个epoch的预测变化")
    print(f"   - 检查梯度是否正常流动")
    print(f"   - 验证损失是否真的在下降")

def main():
    print("🔍 RT-DETR 相同输出问题诊断")
    print("分析为什么模型对每张图像输出相同结果")
    print("=" * 60)
    
    # 1. 加载模型
    backbone, transformer, loaded_model = load_model()
    if backbone is None:
        print("❌ 无法加载模型")
        return
    
    # 2. 详细推理分析
    results = detailed_inference_analysis(backbone, transformer)
    
    # 3. 建议解决方案
    suggest_solutions()
    
    print(f"\n🎯 结论:")
    if results and len(results) > 1:
        print(f"   已完成{len(results)}张图像的详细分析")
        print(f"   请查看上述分析结果判断问题根源")
    else:
        print(f"   分析失败，请检查模型和数据")

if __name__ == "__main__":
    main()
