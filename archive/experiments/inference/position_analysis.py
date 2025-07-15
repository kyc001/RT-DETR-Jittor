#!/usr/bin/env python3
"""
位置分析脚本 - 详细分析检测框位置是否正确
验证模型是否真正学习到了特征
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

def load_detailed_ground_truth():
    """加载详细的真实标注信息"""
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
    
    print("=" * 60)
    print("===      真实标注详细信息      ===")
    print("=" * 60)
    
    for i, ann in enumerate(annotations):
        x, y, w, h = ann['bbox']  # COCO格式：x,y,w,h
        category_name = categories[ann['category_id']]['name']
        
        # 计算中心点和面积
        center_x = x + w/2
        center_y = y + h/2
        area = w * h
        
        gt_data.append({
            'id': i,
            'bbox_coco': [x, y, w, h],
            'bbox_xyxy': [x, y, x+w, y+h],
            'center': [center_x, center_y],
            'size': [w, h],
            'area': area,
            'category_name': category_name,
            'category_id': ann['category_id']
        })
        
        print(f"真实标注 {i+1}: {category_name}")
        print(f"  - COCO格式: [{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}]")
        print(f"  - XYXY格式: [{x:.1f}, {y:.1f}, {x+w:.1f}, {y+h:.1f}]")
        print(f"  - 中心点: [{center_x:.1f}, {center_y:.1f}]")
        print(f"  - 尺寸: {w:.1f} × {h:.1f}")
        print(f"  - 面积: {area:.1f}")
        print()
    
    return image, gt_data, original_size

def get_model_predictions():
    """获取模型预测结果"""
    img_path = "data/coco2017_50/train2017/000000225405.jpg"
    model_path = "checkpoints/correct_sanity_check_model.pkl"
    
    # 预处理
    image = Image.open(img_path).convert('RGB')
    original_size = image.size
    
    resized_image = image.resize((640, 640), Image.LANCZOS)
    img_array = np.array(resized_image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_tensor = jt.array(img_array.transpose(2, 0, 1), dtype='float32').unsqueeze(0)
    
    # 加载模型
    model = RTDETR(num_classes=2)
    model = model.float32()
    state_dict = jt.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 推理
    with jt.no_grad():
        outputs = model(img_tensor)
    
    logits, boxes, _, _ = outputs
    pred_logits = logits[-1][0]
    pred_boxes = boxes[-1][0]
    
    pred_logits = pred_logits.float32()
    pred_boxes = pred_boxes.float32()
    
    logits_np = pred_logits.stop_grad().numpy()
    boxes_np = pred_boxes.stop_grad().numpy()
    
    # 使用Sigmoid激活
    scores = 1.0 / (1.0 + np.exp(-logits_np))
    
    print("=" * 60)
    print("===      模型预测详细分析      ===")
    print("=" * 60)
    
    # 获取最高置信度的预测
    predictions = []
    
    for class_idx in range(2):
        class_scores = scores[:, class_idx]
        
        # 找到该类别的最高置信度预测
        max_idx = np.argmax(class_scores)
        max_score = class_scores[max_idx]
        max_box = boxes_np[max_idx]
        
        # 转换坐标
        cx, cy, w, h = max_box
        x1 = (cx - w/2) * original_size[0]
        y1 = (cy - h/2) * original_size[1]
        x2 = (cx + w/2) * original_size[0]
        y2 = (cy + h/2) * original_size[1]
        
        # 边界检查
        x1 = max(0, min(x1, original_size[0]))
        y1 = max(0, min(y1, original_size[1]))
        x2 = max(0, min(x2, original_size[0]))
        y2 = max(0, min(y2, original_size[1]))
        
        width = x2 - x1
        height = y2 - y1
        center_x = x1 + width/2
        center_y = y1 + height/2
        area = width * height
        
        class_name = 'person' if class_idx == 0 else 'sports ball'
        
        predictions.append({
            'class_idx': class_idx,
            'class_name': class_name,
            'confidence': float(max_score),
            'bbox_xyxy': [x1, y1, x2, y2],
            'center': [center_x, center_y],
            'size': [width, height],
            'area': area,
            'query_idx': int(max_idx)
        })
        
        print(f"预测 {class_idx+1}: {class_name}")
        print(f"  - 置信度: {max_score:.3f}")
        print(f"  - 查询索引: {max_idx}")
        print(f"  - 原始输出: cx={cx:.3f}, cy={cy:.3f}, w={w:.3f}, h={h:.3f}")
        print(f"  - XYXY格式: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        print(f"  - 中心点: [{center_x:.1f}, {center_y:.1f}]")
        print(f"  - 尺寸: {width:.1f} × {height:.1f}")
        print(f"  - 面积: {area:.1f}")
        print()
    
    return predictions

def calculate_position_accuracy(gt_data, predictions):
    """计算位置精度"""
    print("=" * 60)
    print("===      位置精度分析      ===")
    print("=" * 60)
    
    def calculate_iou(box1, box2):
        """计算IoU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
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
    
    def calculate_center_distance(center1, center2):
        """计算中心点距离"""
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    # 按类别匹配
    analysis_results = []
    
    for pred in predictions:
        pred_class = pred['class_name']
        
        # 找到相同类别的真实标注
        matching_gts = [gt for gt in gt_data if gt['category_name'] == pred_class]
        
        if not matching_gts:
            print(f"❌ 预测类别 '{pred_class}' 在真实标注中不存在")
            continue
        
        # 找到最佳匹配的真实标注
        best_gt = None
        best_iou = 0
        
        for gt in matching_gts:
            iou = calculate_iou(pred['bbox_xyxy'], gt['bbox_xyxy'])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt
        
        if best_gt is None:
            print(f"❌ 预测类别 '{pred_class}' 无法找到匹配的真实标注")
            continue
        
        # 计算详细指标
        center_distance = calculate_center_distance(pred['center'], best_gt['center'])
        
        # 计算尺寸差异
        size_diff_w = abs(pred['size'][0] - best_gt['size'][0])
        size_diff_h = abs(pred['size'][1] - best_gt['size'][1])
        size_diff_ratio_w = size_diff_w / best_gt['size'][0]
        size_diff_ratio_h = size_diff_h / best_gt['size'][1]
        
        # 计算面积差异
        area_diff = abs(pred['area'] - best_gt['area'])
        area_diff_ratio = area_diff / best_gt['area']
        
        result = {
            'class_name': pred_class,
            'iou': best_iou,
            'center_distance': center_distance,
            'size_diff_w': size_diff_w,
            'size_diff_h': size_diff_h,
            'size_diff_ratio_w': size_diff_ratio_w,
            'size_diff_ratio_h': size_diff_ratio_h,
            'area_diff': area_diff,
            'area_diff_ratio': area_diff_ratio,
            'prediction': pred,
            'ground_truth': best_gt
        }
        
        analysis_results.append(result)
        
        print(f"📊 {pred_class} 位置精度分析:")
        print(f"  - IoU: {best_iou:.3f}")
        print(f"  - 中心点距离: {center_distance:.1f} 像素")
        print(f"  - 宽度差异: {size_diff_w:.1f} 像素 ({size_diff_ratio_w:.1%})")
        print(f"  - 高度差异: {size_diff_h:.1f} 像素 ({size_diff_ratio_h:.1%})")
        print(f"  - 面积差异: {area_diff:.1f} 像素² ({area_diff_ratio:.1%})")
        
        # 评估精度
        if best_iou > 0.5:
            print(f"  ✅ 定位精度: 良好 (IoU > 0.5)")
        elif best_iou > 0.3:
            print(f"  ⚠️ 定位精度: 一般 (IoU > 0.3)")
        else:
            print(f"  ❌ 定位精度: 差 (IoU < 0.3)")
        
        print()
    
    return analysis_results

def create_detailed_visualization(image, gt_data, predictions, analysis_results):
    """创建详细的可视化分析图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 左上：真实标注
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
        
        # 添加中心点
        center_x, center_y = gt['center']
        ax1.plot(center_x, center_y, 'o', color=color, markersize=8)
        
        ax1.text(x1, y1-5, f"GT{i+1}: {gt['category_name']}", 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                fontsize=10, color='white', fontweight='bold')
    
    # 右上：预测结果
    ax2.imshow(image)
    ax2.set_title("Predictions", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    colors_pred = ['cyan', 'magenta']
    for i, pred in enumerate(predictions):
        x1, y1, x2, y2 = pred['bbox_xyxy']
        color = colors_pred[i % len(colors_pred)]
        
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=3, edgecolor=color, facecolor='none')
        ax2.add_patch(rect)
        
        # 添加中心点
        center_x, center_y = pred['center']
        ax2.plot(center_x, center_y, 'o', color=color, markersize=8)
        
        ax2.text(x1, y1-5, f"P{i+1}: {pred['class_name']} ({pred['confidence']:.2f})", 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                fontsize=10, color='black', fontweight='bold')
    
    # 左下：重叠对比
    ax3.imshow(image)
    ax3.set_title("Overlap Analysis", fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    for result in analysis_results:
        gt = result['ground_truth']
        pred = result['prediction']
        
        # 绘制真实标注（绿色）
        x1, y1, x2, y2 = gt['bbox_xyxy']
        rect_gt = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                  linewidth=3, edgecolor='green', facecolor='none', label='GT')
        ax3.add_patch(rect_gt)
        
        # 绘制预测结果（红色虚线）
        x1, y1, x2, y2 = pred['bbox_xyxy']
        rect_pred = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                    linewidth=2, edgecolor='red', facecolor='none', 
                                    linestyle='--', label='Pred')
        ax3.add_patch(rect_pred)
        
        # 添加IoU标签
        ax3.text(x1, y1-5, f"IoU: {result['iou']:.2f}", 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                fontsize=10, color='black', fontweight='bold')
    
    # 右下：精度统计
    ax4.axis('off')
    ax4.set_title("Accuracy Statistics", fontsize=14, fontweight='bold')
    
    stats_text = "位置精度统计:\n\n"
    
    if analysis_results:
        avg_iou = np.mean([r['iou'] for r in analysis_results])
        avg_center_dist = np.mean([r['center_distance'] for r in analysis_results])
        avg_area_diff_ratio = np.mean([r['area_diff_ratio'] for r in analysis_results])
        
        stats_text += f"平均IoU: {avg_iou:.3f}\n"
        stats_text += f"平均中心点距离: {avg_center_dist:.1f}px\n"
        stats_text += f"平均面积差异: {avg_area_diff_ratio:.1%}\n\n"
        
        for result in analysis_results:
            stats_text += f"{result['class_name']}:\n"
            stats_text += f"  IoU: {result['iou']:.3f}\n"
            stats_text += f"  中心距离: {result['center_distance']:.1f}px\n"
            stats_text += f"  面积差异: {result['area_diff_ratio']:.1%}\n\n"
    else:
        stats_text += "无有效匹配结果"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig("position_analysis_result.jpg", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 位置分析结果已保存到: position_analysis_result.jpg")

def main():
    print("=" * 60)
    print("===      位置精度分析 - 验证特征学习      ===")
    print("=" * 60)
    
    # 1. 加载真实标注
    image, gt_data, original_size = load_detailed_ground_truth()
    
    # 2. 获取模型预测
    predictions = get_model_predictions()
    
    # 3. 计算位置精度
    analysis_results = calculate_position_accuracy(gt_data, predictions)
    
    # 4. 创建详细可视化
    create_detailed_visualization(image, gt_data, predictions, analysis_results)
    
    # 5. 最终结论
    print("=" * 60)
    print("🔍 特征学习验证结论:")
    
    if analysis_results:
        avg_iou = np.mean([r['iou'] for r in analysis_results])
        avg_center_dist = np.mean([r['center_distance'] for r in analysis_results])
        
        print(f"平均IoU: {avg_iou:.3f}")
        print(f"平均中心点距离: {avg_center_dist:.1f} 像素")
        
        if avg_iou > 0.5:
            print("✅ 模型学习到了有意义的特征")
            print("  - 检测框位置基本准确")
            print("  - 特征学习效果良好")
        elif avg_iou > 0.3:
            print("⚠️ 模型部分学习到特征")
            print("  - 检测框位置有偏差")
            print("  - 特征学习不够充分")
        else:
            print("❌ 模型没有真正学习到特征")
            print("  - 检测框位置严重偏差")
            print("  - 可能只是记住了固定模式")
            print("  - 需要更多训练数据或调整策略")
    else:
        print("❌ 无法进行有效的位置分析")
        print("  - 预测结果与真实标注无法匹配")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
