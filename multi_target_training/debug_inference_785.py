#!/usr/bin/env python3
"""
调试推理脚本 - 分析为什么skis没有被检测到
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import json

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR/jittor_rt_detr')

import jittor as jt
from src.nn.model import RTDETR

# 简单的数据变换
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

class Resize:
    def __init__(self, size):
        self.size = size
    
    def __call__(self, image):
        return image.resize(self.size, Image.LANCZOS)

class ToTensor:
    def __call__(self, image):
        image_array = np.array(image, dtype=np.float32) / 255.0
        return jt.array(image_array.transpose(2, 0, 1))

class Normalize:
    def __init__(self, mean, std):
        self.mean = jt.array(mean).reshape(3, 1, 1)
        self.std = jt.array(std).reshape(3, 1, 1)
    
    def __call__(self, tensor):
        return (tensor - self.mean) / self.std

def load_class_mapping(ann_file):
    """加载类别映射"""
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # 找到图片785的标注
    target_image = None
    for img in coco_data['images']:
        if img['file_name'] == "000000000785.jpg":
            target_image = img
            break
    
    if target_image is None:
        raise ValueError("找不到图片: 000000000785.jpg")
    
    # 找到该图片的所有标注
    image_annotations = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == target_image['id']:
            image_annotations.append(ann)
    
    # 构建类别映射
    unique_cat_ids = list(set(ann['category_id'] for ann in image_annotations))
    unique_cat_ids.sort()
    
    # 创建类别ID到名称的映射
    cat_id_to_name = {}
    for cat in coco_data['categories']:
        cat_id_to_name[cat['id']] = cat['name']
    
    # 创建索引到类别名称的映射
    idx_to_name = {}
    for idx, cat_id in enumerate(unique_cat_ids):
        idx_to_name[idx] = cat_id_to_name[cat_id]
    
    return idx_to_name

def preprocess_image(image_path):
    """预处理图片"""
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # 数据变换
    transform = Compose([
        Resize((640, 640)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 应用变换并添加batch维度
    image_tensor = transform(image).unsqueeze(0)  # (1, 3, 640, 640)
    
    return image_tensor, original_size

def debug_predictions(logits, boxes, class_mapping, original_size):
    """调试预测结果，查看所有query的情况"""
    # 使用最后一层的预测
    pred_logits = logits[-1][0]  # (num_queries, num_classes)
    pred_boxes = boxes[-1][0]    # (num_queries, 4)
    
    # 计算置信度分数
    pred_scores = jt.sigmoid(pred_logits)  # (num_queries, num_classes)
    
    print(f"\n🔍 调试分析 - 查看所有query的预测情况:")
    print(f"预测logits形状: {pred_logits.shape}")
    print(f"预测boxes形状: {pred_boxes.shape}")
    print(f"类别映射: {class_mapping}")
    
    # 分析每个类别的最高分数
    for class_idx, class_name in class_mapping.items():
        class_scores = pred_scores[:, class_idx].numpy()
        max_score = np.max(class_scores)
        max_idx = np.argmax(class_scores)
        
        print(f"\n📊 类别 {class_name} (索引{class_idx}):")
        print(f"  最高分数: {max_score:.4f} (query {max_idx})")
        
        # 找到分数>0.05的所有query
        high_score_indices = np.where(class_scores > 0.05)[0]
        print(f"  分数>0.05的query数量: {len(high_score_indices)}")
        
        if len(high_score_indices) > 0:
            print(f"  前5个高分query:")
            sorted_indices = np.argsort(class_scores)[::-1][:5]
            for i, idx in enumerate(sorted_indices):
                score = class_scores[idx]
                if score > 0.01:  # 只显示有意义的分数
                    box = pred_boxes[idx].numpy()
                    cx, cy, w, h = box
                    x1 = (cx - w/2) * original_size[0]
                    y1 = (cy - h/2) * original_size[1]
                    x2 = (cx + w/2) * original_size[0]
                    y2 = (cy + h/2) * original_size[1]
                    print(f"    Query {idx}: 分数={score:.4f}, 边界框=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    
    # 查看每个query的最高分类
    print(f"\n🎯 查看所有query的最高分类:")
    max_classes = np.argmax(pred_scores.numpy(), axis=1)
    max_scores = np.max(pred_scores.numpy(), axis=1)
    
    # 统计每个类别被预测为最高分类的次数
    class_counts = {}
    for class_idx in class_mapping.keys():
        class_counts[class_idx] = np.sum(max_classes == class_idx)
    
    for class_idx, class_name in class_mapping.items():
        count = class_counts[class_idx]
        print(f"  {class_name}: {count} 个query将其作为最高分类")
    
    # 查看分数最高的10个query
    print(f"\n🏆 分数最高的10个query:")
    top_indices = np.argsort(max_scores)[::-1][:10]
    for i, idx in enumerate(top_indices):
        score = max_scores[idx]
        class_idx = max_classes[idx]
        class_name = class_mapping[class_idx]
        box = pred_boxes[idx].numpy()
        cx, cy, w, h = box
        x1 = (cx - w/2) * original_size[0]
        y1 = (cy - h/2) * original_size[1]
        x2 = (cx + w/2) * original_size[0]
        y2 = (cy + h/2) * original_size[1]
        print(f"  {i+1}. Query {idx}: {class_name} 分数={score:.4f}, 边界框=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

def main():
    parser = argparse.ArgumentParser(description="调试推理 - 000000000785.jpg")
    parser.add_argument('--model_path', type=str, default='multi_target_training/checkpoints/multi_target_model_final.pkl', help='模型路径')
    parser.add_argument('--image_path', type=str, default='data/coco/val2017/000000000785.jpg', help='图片路径')
    parser.add_argument('--ann_file', type=str, default='data/coco/annotations/instances_val2017.json', help='标注文件')
    
    args = parser.parse_args()
    
    print("=== 🔍 调试推理：000000000785.jpg ===")
    print(f"模型路径: {args.model_path}")
    print(f"图片路径: {args.image_path}")
    
    # 检查文件
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"图片文件不存在: {args.image_path}")
    
    try:
        # 1. 加载类别映射
        print("加载类别映射...")
        class_mapping = load_class_mapping(args.ann_file)
        print(f"✅ 类别映射: {class_mapping}")
        
        # 2. 创建模型
        print("加载模型...")
        num_classes = len(class_mapping)
        model = RTDETR(num_classes=num_classes)
        
        # 3. 加载权重
        state_dict = jt.load(args.model_path)
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ 模型加载成功!")
        
        # 4. 预处理图片
        image_tensor, original_size = preprocess_image(args.image_path)
        print(f"图片原始尺寸: {original_size}")
        
        # 5. 推理
        print("执行推理...")
        with jt.no_grad():
            logits, boxes, enc_logits, enc_boxes = model(image_tensor)
            print("✅ 推理完成!")
        
        # 6. 调试分析
        debug_predictions(logits, boxes, class_mapping, original_size)
        
        print("\n🎉 调试分析完成!")
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
