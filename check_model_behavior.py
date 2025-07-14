#!/usr/bin/env python3
"""
检查模型行为脚本
分析模型是否真的学到了有用的特征
"""

import os
import sys
import json
import numpy as np
from PIL import Image

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'jittor_rt_detr'))

import jittor as jt
from src.nn.model import RTDETR

# 设置Jittor
jt.flags.use_cuda = 1

def analyze_model_outputs():
    """分析模型在不同图片上的输出"""
    print("=" * 60)
    print("===      模型行为分析      ===")
    print("=" * 60)
    
    # 加载模型
    model_path = "checkpoints/small_scale_final_model.pkl"
    model = RTDETR(num_classes=80)
    model = model.float32()
    
    if os.path.exists(model_path):
        state_dict = jt.load(model_path)
        model.load_state_dict(state_dict)
        print("✅ 模型加载成功")
    else:
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    model.eval()
    
    # 加载COCO类别信息
    ann_file = "data/coco2017_50/annotations/instances_train2017.json"
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    coco_categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    coco_ids = sorted(coco_categories.keys())
    model_idx_to_coco_id = {i: coco_id for i, coco_id in enumerate(coco_ids)}
    
    # 测试几张不同的图片
    test_images = [
        "000000225405.jpg",  # 之前测试过的图片
        "000000293858.jpg",  # 刚才测试的图片
        "000000404249.jpg"   # 另一张测试图片
    ]
    
    for img_name in test_images:
        print(f"\n>>> 分析图片: {img_name}")
        
        img_path = f"data/coco2017_50/train2017/{img_name}"
        if not os.path.exists(img_path):
            print(f"  图片不存在: {img_path}")
            continue
        
        # 预处理图片
        image = Image.open(img_path).convert('RGB')
        original_size = image.size
        
        resized_image = image.resize((640, 640), Image.LANCZOS)
        img_array = np.array(resized_image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        img_tensor = jt.array(img_array.transpose(2, 0, 1), dtype='float32').unsqueeze(0)
        
        # 推理
        with jt.no_grad():
            outputs = model(img_tensor)
        
        logits, boxes, _, _ = outputs
        pred_logits = logits[-1][0]
        pred_boxes = boxes[-1][0]
        
        # 转换为numpy
        pred_logits = pred_logits.float32()
        pred_boxes = pred_boxes.float32()
        
        logits_np = pred_logits.stop_grad().numpy()
        boxes_np = pred_boxes.stop_grad().numpy()
        
        # 分析logits分布
        print(f"  Logits统计:")
        print(f"    - 形状: {logits_np.shape}")
        print(f"    - 最大值: {np.max(logits_np):.3f}")
        print(f"    - 最小值: {np.min(logits_np):.3f}")
        print(f"    - 平均值: {np.mean(logits_np):.3f}")
        print(f"    - 标准差: {np.std(logits_np):.3f}")
        
        # 使用Softmax
        exp_logits = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
        scores = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        max_scores = np.max(scores, axis=1)
        max_classes = np.argmax(scores, axis=1)
        
        print(f"  预测分析:")
        print(f"    - 最高置信度: {np.max(max_scores):.3f}")
        print(f"    - 最低置信度: {np.min(max_scores):.3f}")
        print(f"    - 平均置信度: {np.mean(max_scores):.3f}")
        
        # 分析类别分布
        unique_classes, counts = np.unique(max_classes, return_counts=True)
        print(f"    - 预测的不同类别数: {len(unique_classes)}")
        
        # 显示主要预测的类别
        class_counts = list(zip(unique_classes, counts))
        class_counts.sort(key=lambda x: x[1], reverse=True)
        
        print(f"    - 主要预测类别:")
        for class_idx, count in class_counts[:5]:
            if class_idx < len(model_idx_to_coco_id):
                coco_id = model_idx_to_coco_id[class_idx]
                class_name = coco_categories[coco_id]
                percentage = count / len(max_classes) * 100
                print(f"      {class_name}: {count} 个 ({percentage:.1f}%)")
        
        # 检查是否所有查询都预测同一个类别
        if len(unique_classes) == 1:
            print(f"    ⚠️ 所有查询都预测为同一个类别！")
        elif len(unique_classes) < 5:
            print(f"    ⚠️ 预测类别过少，可能过拟合")
        else:
            print(f"    ✅ 预测有一定多样性")

def analyze_training_data_distribution():
    """分析训练数据的类别分布"""
    print(f"\n" + "=" * 60)
    print("===      训练数据类别分布分析      ===")
    print("=" * 60)
    
    ann_file = "data/coco2017_50/annotations/instances_train2017.json"
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # 统计每个类别的出现次数
    category_counts = {}
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        cat_name = categories[cat_id]
        category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
    
    # 按数量排序
    sorted_counts = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"训练数据中的类别分布 (总共 {len(sorted_counts)} 个类别):")
    
    total_annotations = sum(category_counts.values())
    
    for i, (cat_name, count) in enumerate(sorted_counts):
        percentage = count / total_annotations * 100
        print(f"  {i+1:2d}. {cat_name:15s}: {count:3d} 个 ({percentage:5.1f}%)")
        
        # 只显示前20个
        if i >= 19:
            remaining = len(sorted_counts) - 20
            if remaining > 0:
                print(f"  ... 还有 {remaining} 个类别")
            break
    
    # 分析数据不平衡程度
    max_count = max(category_counts.values())
    min_count = min(category_counts.values())
    imbalance_ratio = max_count / min_count
    
    print(f"\n数据不平衡分析:")
    print(f"  - 最多的类别: {max_count} 个")
    print(f"  - 最少的类别: {min_count} 个")
    print(f"  - 不平衡比例: {imbalance_ratio:.1f}:1")
    
    if imbalance_ratio > 100:
        print(f"  ⚠️ 严重的数据不平衡！")
    elif imbalance_ratio > 10:
        print(f"  ⚠️ 中等程度的数据不平衡")
    else:
        print(f"  ✅ 数据相对平衡")

def main():
    # 1. 分析模型输出行为
    analyze_model_outputs()
    
    # 2. 分析训练数据分布
    analyze_training_data_distribution()
    
    # 3. 总结
    print(f"\n" + "=" * 60)
    print("🔍 模型行为分析总结:")
    print("1. 检查模型是否对所有图片都输出相同的预测")
    print("2. 分析训练数据的类别分布是否平衡")
    print("3. 判断模型是否真的学到了有用的特征")
    print("=" * 60)

if __name__ == "__main__":
    main()
