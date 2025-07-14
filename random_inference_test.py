#!/usr/bin/env python3
"""
随机图片推理测试脚本
随机选择一张图片进行推理，并与真实标注对比可视化
"""

import os
import sys
import json
import random
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

def get_random_image():
    """随机选择一张图片"""
    data_dir = "data/coco2017_50/train2017"
    ann_file = "data/coco2017_50/annotations/instances_train2017.json"
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    
    # 随机选择一张图片
    selected_image = random.choice(image_files)
    img_path = os.path.join(data_dir, selected_image)
    
    print(f"🎲 随机选择的图片: {selected_image}")
    
    return img_path, selected_image, ann_file

def load_image_annotations(img_name, ann_file):
    """加载指定图片的标注"""
    # 加载标注文件
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # 找到目标图片
    target_image = None
    for img in coco_data['images']:
        if img['file_name'] == img_name:
            target_image = img
            break
    
    if target_image is None:
        print(f"❌ 未找到图片 {img_name} 的信息")
        return None, []
    
    # 收集该图片的标注
    annotations = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == target_image['id']:
            annotations.append(ann)
    
    # 处理标注信息
    categories = {cat['id']: cat for cat in coco_data['categories']}
    gt_data = []
    
    print(f"\n📋 图片信息:")
    print(f"  - 文件名: {target_image['file_name']}")
    print(f"  - 尺寸: {target_image['width']} × {target_image['height']}")
    print(f"  - 标注数量: {len(annotations)}")
    
    print(f"\n🎯 真实标注:")
    for i, ann in enumerate(annotations):
        x, y, w, h = ann['bbox']
        category_name = categories[ann['category_id']]['name']
        area = w * h
        
        gt_data.append({
            'bbox_coco': [x, y, w, h],
            'bbox_xyxy': [x, y, x+w, y+h],
            'category_name': category_name,
            'category_id': ann['category_id'],
            'area': area
        })
        
        print(f"  {i+1}. {category_name}: [{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}] (面积: {area:.1f})")
    
    return target_image, gt_data

def run_inference_on_image(img_path, model_path):
    """对指定图片运行推理"""
    print(f"\n🔍 运行推理...")
    
    # 加载图片
    image = Image.open(img_path).convert('RGB')
    original_size = image.size
    
    # 预处理
    resized_image = image.resize((640, 640), Image.LANCZOS)
    img_array = np.array(resized_image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_tensor = jt.array(img_array.transpose(2, 0, 1), dtype='float32').unsqueeze(0)
    
    # 加载模型 - 使用正确的类别数量
    # correct_sanity_check_model.pkl 是用2个类别训练的
    model = RTDETR(num_classes=2)  # person和sports ball
    model = model.float32()
    
    if os.path.exists(model_path):
        state_dict = jt.load(model_path)
        model.load_state_dict(state_dict)
        print("✅ 模型加载成功")
    else:
        print(f"❌ 模型文件不存在: {model_path}")
        return image, []
    
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
    
    # 使用Softmax激活 (多类别分类)
    exp_logits = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
    scores = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # 获取最高置信度的类别
    max_scores = np.max(scores, axis=1)
    max_classes = np.argmax(scores, axis=1)

    print(f"📊 模型输出分析:")
    print(f"  最高置信度: {np.max(max_scores):.3f}")
    print(f"  平均置信度: {np.mean(max_scores):.3f}")
    print(f"  >0.5置信度的数量: {np.sum(max_scores > 0.5)}")

    # 统计预测的类别分布
    unique_classes, counts = np.unique(max_classes, return_counts=True)
    print(f"  预测类别分布:")
    for class_idx, count in zip(unique_classes, counts):
        if count > 5:  # 只显示出现较多的类别
            print(f"    类别 {class_idx}: {count} 个预测")
    
    # 检测结果处理 - 使用置信度阈值
    detections = []

    # 简化的类别映射 - 只有2个类别
    # correct_sanity_check_model.pkl 只训练了person和sports ball
    model_idx_to_class_name = {
        0: 'person',
        1: 'sports ball'
    }

    # 使用置信度阈值筛选检测结果
    confidence_threshold = 0.5
    valid_mask = max_scores > confidence_threshold
    valid_indices = np.where(valid_mask)[0]

    print(f"  置信度>{confidence_threshold}: {len(valid_indices)} 个候选")
        
    # 收集所有有效检测
    all_detections = []
    for idx in valid_indices:
        box = boxes_np[idx]
        score = max_scores[idx]
        class_idx = max_classes[idx]

        # 转换模型索引到类别名称
        if class_idx in model_idx_to_class_name:
            class_name = model_idx_to_class_name[class_idx]
        else:
            continue  # 跳过无效的类别索引

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
            area = (x2-x1) * (y2-y1)
            all_detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(score),
                'class_name': class_name,
                'area': area
            })

    # 按置信度排序
    all_detections.sort(key=lambda x: x['confidence'], reverse=True)

    # 应用NMS
    filtered_detections = []
    for det in all_detections:
        is_duplicate = False
        for existing in filtered_detections:
            # 计算IoU
            iou = calculate_iou(det['bbox'], existing['bbox'])
            if iou > 0.5:  # 如果重叠度高，跳过
                is_duplicate = True
                break

        if not is_duplicate:
            filtered_detections.append(det)

            # 限制总检测数量
            if len(filtered_detections) >= 20:
                break

    detections = filtered_detections
    
    return image, detections

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

def create_comparison_visualization(image, gt_data, detections, img_name):
    """创建对比可视化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 左图：真实标注
    ax1.imshow(image)
    ax1.set_title(f"Ground Truth\n{img_name}", fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # 绘制真实标注
    colors_gt = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    for i, gt in enumerate(gt_data):
        x1, y1, x2, y2 = gt['bbox_xyxy']
        color = colors_gt[i % len(colors_gt)]
        
        # 绘制边界框
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=3, edgecolor=color, facecolor='none')
        ax1.add_patch(rect)
        
        # 添加标签
        ax1.text(x1, y1-5, f"GT{i+1}: {gt['category_name']}", 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                fontsize=12, color='white', fontweight='bold')
    
    # 右图：检测结果
    ax2.imshow(image)
    ax2.set_title(f"Inference Results\n{len(detections)} detections", fontsize=16, fontweight='bold')
    ax2.axis('off')
    
    # 绘制检测结果
    colors_pred = ['cyan', 'magenta', 'yellow', 'lime', 'orange', 'pink', 'lightblue', 'lightgreen']
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        color = colors_pred[i % len(colors_pred)]
        
        # 绘制边界框
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=3, edgecolor=color, facecolor='none')
        ax2.add_patch(rect)
        
        # 添加标签
        ax2.text(x1, y1-5, f"P{i+1}: {det['class_name']} ({det['confidence']:.2f})", 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                fontsize=12, color='black', fontweight='bold')
    
    plt.tight_layout()
    
    # 保存结果
    output_name = f"random_inference_{img_name.replace('.jpg', '')}.jpg"
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 对比可视化已保存: {output_name}")
    return output_name

def analyze_detection_results(gt_data, detections):
    """分析检测结果"""
    print(f"\n📊 检测结果分析:")
    
    # 统计真实标注
    gt_counts = {}
    for gt in gt_data:
        class_name = gt['category_name']
        gt_counts[class_name] = gt_counts.get(class_name, 0) + 1
    
    # 统计检测结果
    pred_counts = {}
    for det in detections:
        class_name = det['class_name']
        pred_counts[class_name] = pred_counts.get(class_name, 0) + 1
    
    print(f"检测数量对比:")
    all_classes = set(list(gt_counts.keys()) + list(pred_counts.keys()))
    
    for class_name in sorted(all_classes):
        gt_count = gt_counts.get(class_name, 0)
        pred_count = pred_counts.get(class_name, 0)
        
        if gt_count == pred_count:
            status = "✅"
        elif pred_count > 0:
            status = "⚠️"
        else:
            status = "❌"
            
        print(f"  {class_name}: 真实={gt_count}, 预测={pred_count} {status}")
    
    print(f"\n详细检测结果:")
    if detections:
        for i, det in enumerate(detections):
            print(f"  {i+1}. {det['class_name']} (置信度: {det['confidence']:.3f}, 面积: {det['area']:.1f})")
    else:
        print("  无检测结果")
    
    return gt_counts, pred_counts

def main():
    print("=" * 60)
    print("===      随机图片推理测试      ===")
    print("=" * 60)
    
    # 1. 随机选择图片
    img_path, img_name, ann_file = get_random_image()
    
    # 2. 加载图片标注
    img_info, gt_data = load_image_annotations(img_name, ann_file)
    
    if img_info is None:
        print("❌ 无法加载图片信息")
        return
    
    # 3. 运行推理
    model_path = "checkpoints/correct_sanity_check_model.pkl"
    image, detections = run_inference_on_image(img_path, model_path)
    
    # 4. 分析结果
    gt_counts, pred_counts = analyze_detection_results(gt_data, detections)
    
    # 5. 创建可视化
    output_file = create_comparison_visualization(image, gt_data, detections, img_name)
    
    # 6. 总结
    print(f"\n" + "=" * 60)
    print("🎯 随机推理测试总结:")
    print(f"  📷 测试图片: {img_name}")
    print(f"  📋 真实标注: {len(gt_data)} 个目标")
    print(f"  🔍 检测结果: {len(detections)} 个目标")
    
    if detections:
        detected_classes = set(det['class_name'] for det in detections)
        expected_classes = set(gt['category_name'] for gt in gt_data)
        
        if expected_classes.issubset(detected_classes):
            print(f"  ✅ 成功检测到所有期望的类别")
        else:
            missing = expected_classes - detected_classes
            print(f"  ⚠️ 缺失类别: {', '.join(missing)}")
        
        extra = detected_classes - expected_classes
        if extra:
            print(f"  ⚠️ 额外检测的类别: {', '.join(extra)}")
    else:
        print(f"  ❌ 没有检测到任何目标")
    
    print(f"  📊 可视化结果: {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
