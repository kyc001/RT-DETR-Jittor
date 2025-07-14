#!/usr/bin/env python3
"""
详细分析脚本 - 对比预测结果与真实标注
"""

import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'jittor_rt_detr'))

import jittor as jt
from src.nn.model import RTDETR

# 设置Jittor
jt.flags.use_cuda = 1

def load_ground_truth(img_path, ann_file, img_name):
    """加载真实标注数据"""
    print(">>> 加载真实标注数据")
    
    # 加载图片
    image = Image.open(img_path).convert('RGB')
    original_size = image.size
    print(f"图片尺寸: {original_size}")
    
    # 加载标注
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # 找到目标图片
    target_image = None
    for img in coco_data['images']:
        if img['file_name'] == img_name:
            target_image = img
            break
    
    # 找到标注
    annotations = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == target_image['id']:
            annotations.append(ann)
    
    # 创建类别映射
    categories = {cat['id']: cat for cat in coco_data['categories']}
    
    # 处理标注
    gt_data = []
    for i, ann in enumerate(annotations):
        x, y, w, h = ann['bbox']  # COCO格式：x,y,w,h
        category_name = categories[ann['category_id']]['name']
        
        gt_data.append({
            'id': i,
            'bbox_coco': [x, y, w, h],  # COCO格式
            'bbox_xyxy': [x, y, x+w, y+h],  # xyxy格式
            'category_name': category_name,
            'category_id': ann['category_id'],
            'area': w * h
        })
        
        print(f"真实标注 {i+1}:")
        print(f"  - 类别: {category_name}")
        print(f"  - COCO bbox: [{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}]")
        print(f"  - XYXY bbox: [{x:.1f}, {y:.1f}, {x+w:.1f}, {y+h:.1f}]")
        print(f"  - 面积: {w*h:.1f}")
    
    return image, gt_data, original_size

def load_model_and_predict(img_path, model_path, num_classes=2):
    """加载模型并进行预测"""
    print(f"\n>>> 加载模型并预测")
    
    # 预处理图片
    image = Image.open(img_path).convert('RGB')
    original_size = image.size
    
    resized_image = image.resize((640, 640), Image.LANCZOS)
    img_array = np.array(resized_image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_tensor = jt.array(img_array.transpose(2, 0, 1), dtype='float32').unsqueeze(0)
    
    # 加载模型
    model = RTDETR(num_classes=num_classes)
    model = model.float32()
    state_dict = jt.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 推理
    with jt.no_grad():
        outputs = model(img_tensor)
    
    logits, boxes, _, _ = outputs
    pred_logits = logits[-1][0]  # (num_queries, num_classes)
    pred_boxes = boxes[-1][0]    # (num_queries, 4)
    
    # 转换为numpy
    pred_logits = pred_logits.float32()
    pred_boxes = pred_boxes.float32()
    
    logits_np = pred_logits.stop_grad().numpy()
    boxes_np = pred_boxes.stop_grad().numpy()
    
    # 计算置信度
    exp_logits = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
    scores = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    max_scores = np.max(scores, axis=1)
    max_classes = np.argmax(scores, axis=1)
    
    print(f"模型输出统计:")
    print(f"  - 查询数量: {len(max_scores)}")
    print(f"  - 置信度范围: [{np.min(max_scores):.3f}, {np.max(max_scores):.3f}]")
    print(f"  - 平均置信度: {np.mean(max_scores):.3f}")
    
    # 分析不同阈值下的检测数量
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1]
    for threshold in thresholds:
        count = np.sum(max_scores > threshold)
        print(f"  - 阈值 {threshold}: {count} 个检测")
    
    # 使用较低阈值获取更多检测
    threshold = 0.3
    valid_mask = max_scores > threshold
    valid_indices = np.where(valid_mask)[0]
    
    predictions = []
    for idx in valid_indices:
        box = boxes_np[idx]  # cxcywh格式
        score = max_scores[idx]
        class_idx = max_classes[idx]
        
        # 转换为xyxy格式并缩放到原图尺寸
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
            predictions.append({
                'query_idx': int(idx),
                'bbox_xyxy': [x1, y1, x2, y2],
                'confidence': float(score),
                'class_idx': int(class_idx),
                'class_name': 'sports ball' if class_idx == 1 else 'person',
                'area': (x2-x1) * (y2-y1)
            })
    
    # 按置信度排序
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    print(f"\n预测结果 (阈值 {threshold}):")
    for i, pred in enumerate(predictions[:10]):
        print(f"预测 {i+1}:")
        print(f"  - 类别: {pred['class_name']}")
        print(f"  - 置信度: {pred['confidence']:.3f}")
        print(f"  - XYXY bbox: [{pred['bbox_xyxy'][0]:.1f}, {pred['bbox_xyxy'][1]:.1f}, {pred['bbox_xyxy'][2]:.1f}, {pred['bbox_xyxy'][3]:.1f}]")
        print(f"  - 面积: {pred['area']:.1f}")
    
    return predictions

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
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def analyze_predictions_vs_gt(predictions, gt_data):
    """详细分析预测结果与真实标注的对比"""
    print(f"\n>>> 详细对比分析")
    
    print(f"真实标注数量: {len(gt_data)}")
    print(f"预测结果数量: {len(predictions)}")
    
    # 按类别统计真实标注
    gt_by_class = {}
    for gt in gt_data:
        class_name = gt['category_name']
        if class_name not in gt_by_class:
            gt_by_class[class_name] = []
        gt_by_class[class_name].append(gt)
    
    # 按类别统计预测结果
    pred_by_class = {}
    for pred in predictions:
        class_name = pred['class_name']
        if class_name not in pred_by_class:
            pred_by_class[class_name] = []
        pred_by_class[class_name].append(pred)
    
    print(f"\n类别统计:")
    print(f"真实标注:")
    for class_name, gts in gt_by_class.items():
        print(f"  - {class_name}: {len(gts)} 个")
    
    print(f"预测结果:")
    for class_name, preds in pred_by_class.items():
        print(f"  - {class_name}: {len(preds)} 个")
    
    # 计算IoU匹配
    print(f"\nIoU匹配分析:")
    iou_threshold = 0.5
    
    matched_pairs = []
    for i, gt in enumerate(gt_data):
        best_iou = 0
        best_pred = None
        best_pred_idx = -1
        
        for j, pred in enumerate(predictions):
            # 只比较相同类别
            if gt['category_name'] == pred['class_name']:
                iou = calculate_iou(gt['bbox_xyxy'], pred['bbox_xyxy'])
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred
                    best_pred_idx = j
        
        print(f"GT {i+1} ({gt['category_name']}):")
        if best_iou > iou_threshold:
            print(f"  ✅ 匹配到预测 {best_pred_idx+1}, IoU: {best_iou:.3f}")
            matched_pairs.append((i, best_pred_idx, best_iou))
        else:
            print(f"  ❌ 未找到匹配 (最佳IoU: {best_iou:.3f})")
    
    # 分析未匹配的预测
    matched_pred_indices = set(pair[1] for pair in matched_pairs)
    unmatched_predictions = [i for i in range(len(predictions)) if i not in matched_pred_indices]
    
    if unmatched_predictions:
        print(f"\n未匹配的预测:")
        for i in unmatched_predictions[:5]:  # 只显示前5个
            pred = predictions[i]
            print(f"  预测 {i+1}: {pred['class_name']} (置信度: {pred['confidence']:.3f})")
    
    # 计算评估指标
    tp = len(matched_pairs)  # 真正例
    fp = len(predictions) - tp  # 假正例
    fn = len(gt_data) - tp  # 假负例
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n评估指标:")
    print(f"  - 真正例 (TP): {tp}")
    print(f"  - 假正例 (FP): {fp}")
    print(f"  - 假负例 (FN): {fn}")
    print(f"  - 精确率 (Precision): {precision:.3f}")
    print(f"  - 召回率 (Recall): {recall:.3f}")
    print(f"  - F1分数: {f1:.3f}")
    
    return matched_pairs

def create_detailed_visualization(image, gt_data, predictions, matched_pairs):
    """创建详细的可视化对比图"""
    print(f"\n>>> 创建详细可视化")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
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
    
    # 中图：预测结果
    ax2.imshow(image)
    ax2.set_title("Predictions", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    colors_pred = ['cyan', 'magenta', 'yellow', 'lime', 'pink']
    for i, pred in enumerate(predictions[:10]):
        x1, y1, x2, y2 = pred['bbox_xyxy']
        color = colors_pred[i % len(colors_pred)]
        
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=3, edgecolor=color, facecolor='none')
        ax2.add_patch(rect)
        
        ax2.text(x1, y1-5, f"P{i+1}: {pred['class_name']} ({pred['confidence']:.2f})", 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                fontsize=10, color='black', fontweight='bold')
    
    # 右图：匹配结果
    ax3.imshow(image)
    ax3.set_title("Matches", fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # 显示匹配的对
    for gt_idx, pred_idx, iou in matched_pairs:
        gt = gt_data[gt_idx]
        pred = predictions[pred_idx]
        
        # 绘制真实标注（绿色）
        x1, y1, x2, y2 = gt['bbox_xyxy']
        rect_gt = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                  linewidth=3, edgecolor='green', facecolor='none')
        ax3.add_patch(rect_gt)
        
        # 绘制预测结果（红色）
        x1, y1, x2, y2 = pred['bbox_xyxy']
        rect_pred = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                    linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
        ax3.add_patch(rect_pred)
        
        # 添加IoU标签
        ax3.text(x1, y1-5, f"IoU: {iou:.2f}", 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                fontsize=10, color='black', fontweight='bold')
    
    # 显示未匹配的真实标注
    matched_gt_indices = set(pair[0] for pair in matched_pairs)
    for i, gt in enumerate(gt_data):
        if i not in matched_gt_indices:
            x1, y1, x2, y2 = gt['bbox_xyxy']
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=3, edgecolor='orange', facecolor='none')
            ax3.add_patch(rect)
            ax3.text(x1, y1-5, f"MISSED: {gt['category_name']}", 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.8),
                    fontsize=10, color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("detailed_analysis_result.jpg", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 详细分析图已保存到: detailed_analysis_result.jpg")

def main():
    print("=" * 60)
    print("===      详细分析：预测结果 vs 真实标注      ===")
    print("=" * 60)
    
    # 参数设置
    img_path = "data/coco2017_50/train2017/000000225405.jpg"
    ann_file = "data/coco2017_50/annotations/instances_train2017.json"
    img_name = "000000225405.jpg"
    model_path = "checkpoints/correct_sanity_check_model.pkl"
    
    # 1. 加载真实标注
    image, gt_data, original_size = load_ground_truth(img_path, ann_file, img_name)
    
    # 2. 加载模型并预测
    predictions = load_model_and_predict(img_path, model_path, num_classes=2)
    
    # 3. 详细对比分析
    matched_pairs = analyze_predictions_vs_gt(predictions, gt_data)
    
    # 4. 创建详细可视化
    create_detailed_visualization(image, gt_data, predictions, matched_pairs)
    
    # 5. 问题总结
    print(f"\n" + "=" * 60)
    print("🔍 问题总结:")
    
    gt_classes = set(gt['category_name'] for gt in gt_data)
    pred_classes = set(pred['class_name'] for pred in predictions)
    
    print(f"1. 类别检测问题:")
    print(f"   - 真实类别: {', '.join(sorted(gt_classes))}")
    print(f"   - 预测类别: {', '.join(sorted(pred_classes))}")
    
    missing_classes = gt_classes - pred_classes
    if missing_classes:
        print(f"   ❌ 未检测到的类别: {', '.join(missing_classes)}")
    
    print(f"2. 定位精度问题:")
    if matched_pairs:
        avg_iou = np.mean([pair[2] for pair in matched_pairs])
        print(f"   - 平均IoU: {avg_iou:.3f}")
        if avg_iou < 0.7:
            print(f"   ❌ 定位精度偏低")
    else:
        print(f"   ❌ 没有成功匹配的检测")
    
    print(f"3. 检测数量问题:")
    print(f"   - 真实目标: {len(gt_data)} 个")
    print(f"   - 预测结果: {len(predictions)} 个")
    if len(predictions) > len(gt_data) * 2:
        print(f"   ❌ 检测数量过多，需要NMS后处理")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
