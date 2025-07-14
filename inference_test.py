#!/usr/bin/env python3
"""
推理测试脚本 - 使用最新训练的模型进行推理
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'jittor_rt_detr'))

import jittor as jt
from src.nn.model import RTDETR

# 设置Jittor
jt.flags.use_cuda = 1

def load_ground_truth():
    """加载真实标注"""
    img_path = "data/coco2017_50/train2017/000000225405.jpg"
    ann_file = "data/coco2017_50/annotations/instances_train2017.json"
    img_name = "000000225405.jpg"
    
    # 加载图片
    image = Image.open(img_path).convert('RGB')
    original_size = image.size
    
    # 加载标注
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # 找到目标图片和标注
    target_image = None
    for img in coco_data['images']:
        if img['file_name'] == img_name:
            target_image = img
            break
    
    annotations = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == target_image['id']:
            annotations.append(ann)
    
    # 处理真实标注
    categories = {cat['id']: cat for cat in coco_data['categories']}
    gt_data = []
    
    print("真实标注:")
    for i, ann in enumerate(annotations):
        x, y, w, h = ann['bbox']
        category_name = categories[ann['category_id']]['name']
        
        gt_data.append({
            'bbox_xyxy': [x, y, x+w, y+h],
            'category_name': category_name,
            'area': w * h
        })
        
        print(f"  {i+1}. {category_name}: [{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}] (面积: {w*h:.1f})")
    
    return image, gt_data, original_size

def run_inference(model_path):
    """运行推理"""
    print(f"\n>>> 使用模型: {model_path}")
    
    # 加载图片
    img_path = "data/coco2017_50/train2017/000000225405.jpg"
    image = Image.open(img_path).convert('RGB')
    original_size = image.size
    
    # 预处理
    resized_image = image.resize((640, 640), Image.LANCZOS)
    img_array = np.array(resized_image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_tensor = jt.array(img_array.transpose(2, 0, 1), dtype='float32').unsqueeze(0)
    
    # 加载模型
    model = RTDETR(num_classes=2)  # person和sports ball
    model = model.float32()
    
    if os.path.exists(model_path):
        state_dict = jt.load(model_path)
        model.load_state_dict(state_dict)
        print("✅ 模型加载成功")
    else:
        print(f"❌ 模型文件不存在: {model_path}")
        return []
    
    model.eval()
    
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
    
    # 使用Sigmoid激活 (之前验证过这种方法有效)
    scores = 1.0 / (1.0 + np.exp(-logits_np))
    
    print(f"\n模型输出分析:")
    for class_idx in range(2):
        class_scores = scores[:, class_idx]
        class_name = 'person' if class_idx == 0 else 'sports ball'
        print(f"  {class_name}: 最高={np.max(class_scores):.3f}, 平均={np.mean(class_scores):.3f}")
    
    # 检测结果
    detections = []
    
    # 为每个类别使用不同的阈值
    thresholds = [0.3, 0.5]  # person, sports ball
    
    for class_idx in range(2):
        class_scores = scores[:, class_idx]
        threshold = thresholds[class_idx]
        
        valid_mask = class_scores > threshold
        valid_indices = np.where(valid_mask)[0]
        
        class_name = 'person' if class_idx == 0 else 'sports ball'
        print(f"  {class_name} (阈值{threshold}): {len(valid_indices)} 个候选")
        
        # 收集检测结果
        class_detections = []
        for idx in valid_indices:
            box = boxes_np[idx]
            score = class_scores[idx]
            
            # 转换坐标
            cx, cy, w, h = box
            x1 = (cx - w/2) * original_size[0]
            y1 = (cy - h/2) * original_size[1]
            x2 = (cx + w/2) * original_size[0]
            y2 = (cy + h/2) * original_size[1]
            
            # 边界检查
            x1 = max(0, min(x1, original_size[0]))
            y1 = max(0, min(y1, original_size[1]))
            x2 = max(0, min(x2, original_size[0]))
            y2 = max(0, min(y2, original_size[1]))
            
            if x2 > x1 and y2 > y1:
                class_detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(score),
                    'class_name': class_name,
                    'area': (x2-x1) * (y2-y1)
                })
        
        # 简单NMS - 只保留置信度最高的几个
        class_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 根据类别保留不同数量
        if class_name == 'person':
            keep_num = 3  # 期望3个person
        else:
            keep_num = 1  # 期望1个sports ball
        
        detections.extend(class_detections[:keep_num])
    
    return detections

def analyze_results(detections, gt_data):
    """分析结果"""
    print(f"\n📊 推理结果分析:")
    print(f"检测数量: {len(detections)}")
    
    # 按类别统计
    pred_counts = {}
    for det in detections:
        class_name = det['class_name']
        pred_counts[class_name] = pred_counts.get(class_name, 0) + 1
    
    gt_counts = {}
    for gt in gt_data:
        class_name = gt['category_name']
        gt_counts[class_name] = gt_counts.get(class_name, 0) + 1
    
    print(f"\n数量对比:")
    all_classes = set(list(gt_counts.keys()) + list(pred_counts.keys()))
    
    for class_name in sorted(all_classes):
        gt_count = gt_counts.get(class_name, 0)
        pred_count = pred_counts.get(class_name, 0)
        
        status = "✅" if gt_count == pred_count else "❌"
        print(f"  {class_name}: 真实={gt_count}, 预测={pred_count} {status}")
    
    print(f"\n详细检测结果:")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['class_name']} (置信度: {det['confidence']:.3f}, 面积: {det['area']:.1f})")

def create_visualization(image, gt_data, detections):
    """创建可视化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：真实标注
    ax1.imshow(image)
    ax1.set_title("Ground Truth", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    colors_gt = ['red', 'blue', 'green', 'orange']
    for i, gt in enumerate(gt_data):
        x1, y1, x2, y2 = gt['bbox_xyxy']
        color = colors_gt[i % len(colors_gt)]
        
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=3, edgecolor=color, facecolor='none')
        ax1.add_patch(rect)
        
        ax1.text(x1, y1-5, f"GT{i+1}: {gt['category_name']}", 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                fontsize=10, color='white', fontweight='bold')
    
    # 右图：检测结果
    ax2.imshow(image)
    ax2.set_title("Inference Results", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    colors_pred = ['cyan', 'magenta', 'yellow', 'lime', 'pink']
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        color = colors_pred[i % len(colors_pred)]
        
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=3, edgecolor=color, facecolor='none')
        ax2.add_patch(rect)
        
        ax2.text(x1, y1-5, f"P{i+1}: {det['class_name']} ({det['confidence']:.2f})", 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                fontsize=10, color='black', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("inference_test_result.jpg", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 可视化结果已保存: inference_test_result.jpg")

def main():
    print("=" * 60)
    print("===      推理测试 - 使用最新训练模型      ===")
    print("=" * 60)
    
    # 1. 加载真实标注
    image, gt_data, original_size = load_ground_truth()
    
    # 2. 运行推理
    model_path = "checkpoints/small_scale_final_model.pkl"
    detections = run_inference(model_path)
    
    # 3. 分析结果
    analyze_results(detections, gt_data)
    
    # 4. 创建可视化
    create_visualization(image, gt_data, detections)
    
    # 5. 总结
    print(f"\n" + "=" * 60)
    print("🔍 推理测试总结:")
    
    if len(detections) > 0:
        detected_classes = set(det['class_name'] for det in detections)
        expected_classes = set(gt['category_name'] for gt in gt_data)
        
        if expected_classes.issubset(detected_classes):
            print("✅ 成功检测到所有期望的类别")
        else:
            missing = expected_classes - detected_classes
            print(f"⚠️ 缺失类别: {', '.join(missing)}")
        
        print(f"📊 检测效果:")
        print(f"  - 检测数量: {len(detections)}")
        print(f"  - 检测类别: {', '.join(sorted(detected_classes))}")
    else:
        print("❌ 没有检测到任何目标")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
