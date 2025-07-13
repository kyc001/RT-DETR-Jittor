#!/usr/bin/env python3
"""
多目标推理脚本 - 000000000785.jpg
测试训练好的多目标RT-DETR模型，检测人和滑雪板
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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

def postprocess_predictions(logits, boxes, class_mapping, original_size, conf_threshold=0.5):
    """后处理预测结果"""
    # 使用最后一层的预测
    pred_logits = logits[-1][0]  # (num_queries, num_classes)
    pred_boxes = boxes[-1][0]    # (num_queries, 4)
    
    # 计算置信度分数
    pred_scores = jt.sigmoid(pred_logits)  # (num_queries, num_classes)
    
    detections = []
    
    for query_idx in range(pred_logits.shape[0]):
        scores = pred_scores[query_idx].numpy()
        box = pred_boxes[query_idx].numpy()
        
        # 找到最高分数的类别
        max_class_idx = np.argmax(scores)
        max_score = scores[max_class_idx]
        
        if max_score > conf_threshold:
            # 转换边界框坐标
            cx, cy, w, h = box
            x1 = (cx - w/2) * original_size[0]
            y1 = (cy - h/2) * original_size[1]
            x2 = (cx + w/2) * original_size[0]
            y2 = (cy + h/2) * original_size[1]
            
            # 确保坐标在图片范围内
            x1 = max(0, min(x1, original_size[0]))
            y1 = max(0, min(y1, original_size[1]))
            x2 = max(0, min(x2, original_size[0]))
            y2 = max(0, min(y2, original_size[1]))
            
            class_name = class_mapping.get(max_class_idx, f"class_{max_class_idx}")
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'class_name': class_name,
                'confidence': max_score,
                'query_id': query_idx
            })
    
    return detections

def apply_nms(detections, iou_threshold=0.5):
    """应用非极大值抑制"""
    if not detections:
        return []
    
    # 按置信度排序
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    while detections:
        # 保留置信度最高的检测
        current = detections.pop(0)
        keep.append(current)
        
        # 移除与当前检测IoU过高的其他检测
        remaining = []
        for det in detections:
            if calculate_iou(current['bbox'], det['bbox']) < iou_threshold:
                remaining.append(det)
        detections = remaining
    
    return keep

def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 计算交集
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # 计算并集
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def visualize_results(image_path, detections, output_path):
    """可视化检测结果"""
    # 加载原图
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # 定义颜色
    colors = {
        'person': (255, 0, 0),    # 红色
        'skis': (0, 255, 0),      # 绿色
    }
    
    print(f"📊 检测结果:")
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class_name']
        confidence = det['confidence']
        
        color = colors.get(class_name, (0, 0, 255))  # 默认蓝色
        
        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # 绘制标签
        label = f"{class_name}: {confidence:.3f}"
        draw.text((x1, y1-25), label, fill=color, font=font)
        
        print(f"  {i+1}. {class_name}: 置信度={confidence:.3f}, 边界框=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    
    # 保存结果
    image.save(output_path)
    print(f"✅ 检测结果已保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="多目标推理 - 000000000785.jpg")
    parser.add_argument('--model_path', type=str, default='multi_target_training/checkpoints/multi_target_model_final.pkl', help='模型路径')
    parser.add_argument('--image_path', type=str, default='data/coco/val2017/000000000785.jpg', help='图片路径')
    parser.add_argument('--ann_file', type=str, default='data/coco/annotations/instances_val2017.json', help='标注文件')
    parser.add_argument('--output_path', type=str, default='multi_target_training/detection_result_785.jpg', help='输出路径')
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--nms_threshold', type=float, default=0.3, help='NMS阈值')
    
    args = parser.parse_args()
    
    print("=== 🎯 多目标推理测试 ===")
    print(f"模型路径: {args.model_path}")
    print(f"图片路径: {args.image_path}")
    print(f"置信度阈值: {args.conf_threshold}")
    print(f"NMS阈值: {args.nms_threshold}")
    
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
        print("加载多目标模型...")
        num_classes = len(class_mapping)
        model = RTDETR(num_classes=num_classes)
        
        # 3. 加载权重
        state_dict = jt.load(args.model_path)
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ 多目标模型加载成功!")
        
        # 4. 预处理图片
        image_tensor, original_size = preprocess_image(args.image_path)
        
        # 5. 推理
        print("执行多目标推理...")
        with jt.no_grad():
            logits, boxes, enc_logits, enc_boxes = model(image_tensor)
            print("✅ 推理完成!")
        
        # 6. 后处理
        detections = postprocess_predictions(logits, boxes, class_mapping, original_size, args.conf_threshold)
        print(f"📊 初步检测到 {len(detections)} 个目标")
        
        # 7. 应用NMS
        final_detections = apply_nms(detections, args.nms_threshold)
        print(f"📊 NMS后保留 {len(final_detections)} 个目标")
        
        # 8. 可视化结果
        visualize_results(args.image_path, final_detections, args.output_path)
        
        print("🎉 多目标推理完成!")
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
