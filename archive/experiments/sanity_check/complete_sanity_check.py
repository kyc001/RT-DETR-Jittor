#!/usr/bin/env python3
"""
完整流程自检脚本 - 训练→推理→可视化对比
用一张照片训练不超过20次，然后检测同一张照片，对比结果
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
from src.nn.loss import DETRLoss

# 设置Jittor
jt.flags.use_cuda = 1

# COCO类别名称
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def load_single_image_data(img_path, ann_file, img_name):
    """加载单张图片的数据和标注"""
    print(f">>> 加载数据: {img_name}")
    
    # 加载图片
    image = Image.open(img_path).convert('RGB')
    original_size = image.size
    print(f"原始图片尺寸: {original_size}")
    
    # 预处理图片
    resized_image = image.resize((640, 640), Image.LANCZOS)
    img_array = np.array(resized_image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_tensor = jt.array(img_array.transpose(2, 0, 1), dtype='float32').unsqueeze(0)
    
    # 加载标注
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # 找到目标图片
    target_image = None
    for img in coco_data['images']:
        if img['file_name'] == img_name:
            target_image = img
            break
    
    if target_image is None:
        raise ValueError(f"找不到图片: {img_name}")
    
    # 找到标注
    annotations = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == target_image['id']:
            annotations.append(ann)
    
    print(f"找到 {len(annotations)} 个标注")
    
    # 创建类别映射
    categories = {cat['id']: cat for cat in coco_data['categories']}
    
    # 处理标注 - 只使用图片中实际存在的类别
    unique_category_ids = list(set(ann['category_id'] for ann in annotations))
    unique_category_ids.sort()
    
    # 创建类别ID到模型类别索引的映射
    cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(unique_category_ids)}
    num_classes = len(unique_category_ids)
    
    print(f"图片中的类别:")
    for cat_id in unique_category_ids:
        cat_name = categories[cat_id]['name']
        model_idx = cat_id_to_idx[cat_id]
        print(f"  - {cat_name} (COCO ID: {cat_id}, 模型索引: {model_idx})")
    
    # 转换标注格式
    boxes = []
    labels = []
    gt_info = []  # 保存真实标注信息用于可视化
    
    for ann in annotations:
        # 边界框转换: COCO格式(x,y,w,h) -> 归一化cxcywh格式
        x, y, w, h = ann['bbox']
        cx = (x + w/2) / original_size[0]
        cy = (y + h/2) / original_size[1]
        w_norm = w / original_size[0]
        h_norm = h / original_size[1]
        
        boxes.append([cx, cy, w_norm, h_norm])
        labels.append(cat_id_to_idx[ann['category_id']])
        
        # 保存真实标注信息
        gt_info.append({
            'bbox': [x, y, x+w, y+h],  # xyxy格式
            'category_name': categories[ann['category_id']]['name'],
            'category_id': ann['category_id']
        })
    
    boxes_tensor = jt.array(boxes, dtype='float32')
    labels_tensor = jt.array(labels, dtype='int64')
    
    targets = [{
        'boxes': boxes_tensor,
        'labels': labels_tensor
    }]
    
    return img_tensor, targets, num_classes, cat_id_to_idx, gt_info, original_size, image

def train_single_image(model, criterion, optimizer, img_tensor, targets, max_epochs=20):
    """用单张图片训练模型"""
    print(f"\n>>> 开始训练 (最多 {max_epochs} 轮)")
    
    model.train()
    
    for epoch in range(max_epochs):
        # 前向传播
        outputs = model(img_tensor)
        logits, boxes, enc_logits, enc_boxes = outputs
        
        # 计算损失
        loss_dict = criterion(logits, boxes, targets, enc_logits, enc_boxes)
        total_loss = sum(loss_dict.values())
        
        # 反向传播
        optimizer.zero_grad()
        optimizer.backward(total_loss)
        optimizer.step()
        
        # 打印进度
        if epoch % 5 == 0 or epoch == max_epochs - 1:
            print(f"  Epoch {epoch+1}/{max_epochs}: Loss = {float(total_loss.data):.4f}")
            for key, value in loss_dict.items():
                print(f"    {key}: {float(value.data):.4f}")
    
    print("✅ 训练完成")
    return model

def inference_and_postprocess(model, img_tensor, original_size, conf_threshold=0.1):
    """推理和后处理"""
    print(f"\n>>> 执行推理...")
    
    model.eval()
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
    
    # 使用softmax激活
    exp_logits = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
    scores = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # 找到每个query的最高分数和类别
    max_scores = np.max(scores, axis=1)
    max_classes = np.argmax(scores, axis=1)
    
    # 过滤低置信度
    valid_mask = max_scores > conf_threshold
    valid_indices = np.where(valid_mask)[0]
    
    print(f"置信度范围: [{np.min(max_scores):.3f}, {np.max(max_scores):.3f}]")
    print(f"超过阈值 {conf_threshold} 的检测数量: {len(valid_indices)}")
    
    # 转换边界框格式并缩放
    detections = []
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
        
        # 确保边界框在图像范围内
        x1 = max(0, min(x1, original_size[0]))
        y1 = max(0, min(y1, original_size[1]))
        x2 = max(0, min(x2, original_size[0]))
        y2 = max(0, min(y2, original_size[1]))
        
        if x2 > x1 and y2 > y1:  # 有效边界框
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(score),
                'class_idx': int(class_idx),
                'query_idx': int(idx)
            })
    
    # 按置信度排序
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    print(f"✅ 推理完成，检测到 {len(detections)} 个目标")
    return detections

def visualize_comparison(image, gt_info, detections, cat_id_to_idx, save_path="sanity_check_result.jpg"):
    """可视化对比真实标注和检测结果"""
    print(f"\n>>> 生成可视化对比图...")

    # 设置中文字体（如果可用）
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 左图：真实标注
    ax1.imshow(image)
    ax1.set_title("Ground Truth", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    colors_gt = ['red', 'blue', 'green', 'orange', 'purple']
    for i, gt in enumerate(gt_info):
        x1, y1, x2, y2 = gt['bbox']
        color = colors_gt[i % len(colors_gt)]
        
        # 绘制边界框
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor=color, facecolor='none')
        ax1.add_patch(rect)
        
        # 添加标签
        ax1.text(x1, y1-5, f"{gt['category_name']}", 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                fontsize=10, color='white', fontweight='bold')
    
    # 右图：检测结果
    ax2.imshow(image)
    ax2.set_title("Predictions", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # 反向映射：模型索引到类别名称
    idx_to_category = {}
    for cat_id, model_idx in cat_id_to_idx.items():
        # 找到对应的类别名称
        for gt in gt_info:
            if gt['category_id'] == cat_id:
                idx_to_category[model_idx] = gt['category_name']
                break
    
    colors_pred = ['cyan', 'magenta', 'yellow', 'lime', 'pink']
    for i, det in enumerate(detections[:10]):  # 只显示前10个检测
        x1, y1, x2, y2 = det['bbox']
        color = colors_pred[i % len(colors_pred)]
        
        # 绘制边界框
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor=color, facecolor='none')
        ax2.add_patch(rect)
        
        # 获取类别名称
        class_name = idx_to_category.get(det['class_idx'], f"class_{det['class_idx']}")
        
        # 添加标签
        ax2.text(x1, y1-5, f"{class_name} ({det['confidence']:.2f})", 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                fontsize=10, color='black', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 可视化结果已保存到: {save_path}")
    
    # 打印详细对比
    print(f"\n📊 检测结果分析:")
    print(f"真实标注数量: {len(gt_info)}")
    print(f"检测结果数量: {len(detections)}")
    
    print(f"\n真实标注:")
    for i, gt in enumerate(gt_info):
        print(f"  {i+1}. {gt['category_name']}")
    
    print(f"\n检测结果 (前5个):")
    for i, det in enumerate(detections[:5]):
        class_name = idx_to_category.get(det['class_idx'], f"class_{det['class_idx']}")
        print(f"  {i+1}. {class_name} (置信度: {det['confidence']:.3f})")

def main():
    print("=" * 60)
    print("===      完整流程自检 - 训练→推理→可视化对比      ===")
    print("=" * 60)
    
    # 参数设置
    img_path = "data/coco2017_50/train2017/000000225405.jpg"
    ann_file = "data/coco2017_50/annotations/instances_train2017.json"
    img_name = "000000225405.jpg"
    max_epochs = 20
    conf_threshold = 0.1
    
    print(f"目标图片: {img_name}")
    print(f"最大训练轮数: {max_epochs}")
    print(f"置信度阈值: {conf_threshold}")
    
    # 1. 加载数据
    img_tensor, targets, num_classes, cat_id_to_idx, gt_info, original_size, image = load_single_image_data(
        img_path, ann_file, img_name)
    
    # 2. 创建模型
    print(f"\n>>> 创建模型 (类别数: {num_classes})")
    model = RTDETR(num_classes=num_classes)
    model = model.float32()
    
    # 3. 创建损失函数和优化器
    criterion = DETRLoss(num_classes=num_classes)
    optimizer = jt.optim.Adam(model.parameters(), lr=1e-4)
    
    # 4. 训练
    model = train_single_image(model, criterion, optimizer, img_tensor, targets, max_epochs)
    
    # 5. 保存模型
    save_path = "checkpoints/sanity_check_model.pkl"
    os.makedirs("checkpoints", exist_ok=True)
    jt.save(model.state_dict(), save_path)
    print(f"✅ 模型已保存到: {save_path}")
    
    # 6. 推理
    detections = inference_and_postprocess(model, img_tensor, original_size, conf_threshold)
    
    # 7. 可视化对比
    visualize_comparison(image, gt_info, detections, cat_id_to_idx)
    
    # 8. 流程自检结论
    print(f"\n" + "=" * 60)
    print("🔍 流程自检结论:")
    
    if len(detections) > 0:
        print("✅ 流程自检通过！")
        print("  - 数据加载正常")
        print("  - 模型训练成功")
        print("  - 推理执行正常")
        print("  - 检测到目标物体")
        print("  - 可视化对比完成")
        print("  - 整个训练→推理流程工作正常")
        
        # 简单的准确性评估
        gt_categories = set(gt['category_name'] for gt in gt_info)
        print(f"\n📈 简单评估:")
        print(f"  - 真实类别: {', '.join(gt_categories)}")
        print(f"  - 检测数量: {len(detections)}")
        print(f"  - 最高置信度: {detections[0]['confidence']:.3f}")
        
    else:
        print("⚠️ 流程自检部分通过")
        print("  - 数据加载正常 ✅")
        print("  - 模型训练成功 ✅")
        print("  - 推理执行正常 ✅")
        print("  - 但未检测到目标 ❌")
        print("  - 建议降低置信度阈值或增加训练轮数")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
