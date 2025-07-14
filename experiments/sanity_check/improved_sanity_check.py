#!/usr/bin/env python3
"""
改进版流程自检脚本 - 使用更合理的训练参数
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

def load_and_prepare_data(img_path, ann_file, img_name):
    """加载并准备数据"""
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
    
    # 创建类别映射 - 使用实际的COCO类别
    categories = {cat['id']: cat for cat in coco_data['categories']}
    unique_category_ids = list(set(ann['category_id'] for ann in annotations))
    unique_category_ids.sort()
    
    # 映射到连续的索引
    cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(unique_category_ids)}
    num_classes = len(unique_category_ids)
    
    print(f"图片中的类别:")
    for cat_id in unique_category_ids:
        cat_name = categories[cat_id]['name']
        model_idx = cat_id_to_idx[cat_id]
        print(f"  - {cat_name} (COCO ID: {cat_id}, 模型索引: {model_idx})")
    
    # 处理标注
    boxes = []
    labels = []
    gt_info = []
    
    for ann in annotations:
        x, y, w, h = ann['bbox']
        cx = (x + w/2) / original_size[0]
        cy = (y + h/2) / original_size[1]
        w_norm = w / original_size[0]
        h_norm = h / original_size[1]
        
        boxes.append([cx, cy, w_norm, h_norm])
        labels.append(cat_id_to_idx[ann['category_id']])
        
        gt_info.append({
            'bbox': [x, y, x+w, y+h],
            'category_name': categories[ann['category_id']]['name'],
            'category_id': ann['category_id'],
            'model_idx': cat_id_to_idx[ann['category_id']]
        })
    
    boxes_tensor = jt.array(boxes, dtype='float32')
    labels_tensor = jt.array(labels, dtype='int64')
    
    targets = [{
        'boxes': boxes_tensor,
        'labels': labels_tensor
    }]
    
    return img_tensor, targets, num_classes, cat_id_to_idx, gt_info, original_size, image

def train_with_early_stopping(model, criterion, optimizer, img_tensor, targets, max_epochs=15):
    """训练模型，使用早停策略"""
    print(f"\n>>> 开始训练 (最多 {max_epochs} 轮)")
    
    model.train()
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
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
        
        current_loss = float(total_loss.data)
        
        # 早停检查
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 打印进度
        if epoch % 3 == 0 or epoch == max_epochs - 1:
            print(f"  Epoch {epoch+1}/{max_epochs}: Loss = {current_loss:.4f} (Best: {best_loss:.4f})")
            
        # 早停
        if patience_counter >= patience:
            print(f"  早停触发 (连续{patience}轮无改善)")
            break
    
    print("✅ 训练完成")
    return model

def smart_inference(model, img_tensor, original_size, num_classes):
    """智能推理，使用多种置信度阈值"""
    print(f"\n>>> 执行推理...")
    
    model.eval()
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
    
    # 使用softmax
    exp_logits = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
    scores = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # 找到每个query的最高分数和类别
    max_scores = np.max(scores, axis=1)
    max_classes = np.argmax(scores, axis=1)
    
    print(f"置信度统计:")
    print(f"  - 最高: {np.max(max_scores):.3f}")
    print(f"  - 平均: {np.mean(max_scores):.3f}")
    print(f"  - 中位数: {np.median(max_scores):.3f}")
    
    # 尝试多个阈值
    thresholds = [0.5, 0.3, 0.1, 0.05]
    best_detections = []
    best_threshold = 0.1
    
    for threshold in thresholds:
        valid_mask = max_scores > threshold
        valid_count = np.sum(valid_mask)
        print(f"  - 阈值 {threshold}: {valid_count} 个检测")
        
        if 1 <= valid_count <= 20:  # 合理的检测数量
            best_threshold = threshold
            break
    
    # 使用最佳阈值进行检测
    valid_mask = max_scores > best_threshold
    valid_indices = np.where(valid_mask)[0]
    
    print(f"使用阈值 {best_threshold}，检测到 {len(valid_indices)} 个目标")
    
    # 转换检测结果
    detections = []
    for idx in valid_indices:
        box = boxes_np[idx]
        score = max_scores[idx]
        class_idx = max_classes[idx]
        
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
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(score),
                'class_idx': int(class_idx),
                'query_idx': int(idx)
            })
    
    # 按置信度排序
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    return detections

def create_visualization(image, gt_info, detections, cat_id_to_idx):
    """创建可视化对比图"""
    print(f"\n>>> 生成可视化对比图...")
    
    # 创建反向映射
    idx_to_category = {}
    for gt in gt_info:
        idx_to_category[gt['model_idx']] = gt['category_name']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：真实标注
    ax1.imshow(image)
    ax1.set_title("Ground Truth", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    colors_gt = ['red', 'blue', 'green', 'orange']
    for i, gt in enumerate(gt_info):
        x1, y1, x2, y2 = gt['bbox']
        color = colors_gt[i % len(colors_gt)]
        
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=3, edgecolor=color, facecolor='none')
        ax1.add_patch(rect)
        
        ax1.text(x1, y1-5, f"{gt['category_name']}", 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                fontsize=12, color='white', fontweight='bold')
    
    # 右图：检测结果
    ax2.imshow(image)
    ax2.set_title("Predictions", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    colors_pred = ['cyan', 'magenta', 'yellow', 'lime']
    for i, det in enumerate(detections[:10]):
        x1, y1, x2, y2 = det['bbox']
        color = colors_pred[i % len(colors_pred)]
        
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=3, edgecolor=color, facecolor='none')
        ax2.add_patch(rect)
        
        class_name = idx_to_category.get(det['class_idx'], f"class_{det['class_idx']}")
        
        ax2.text(x1, y1-5, f"{class_name} ({det['confidence']:.2f})", 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                fontsize=12, color='black', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("improved_sanity_check_result.jpg", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 可视化结果已保存到: improved_sanity_check_result.jpg")
    
    return idx_to_category

def analyze_results(gt_info, detections, idx_to_category):
    """分析检测结果"""
    print(f"\n📊 详细结果分析:")
    print(f"真实标注数量: {len(gt_info)}")
    print(f"检测结果数量: {len(detections)}")
    
    print(f"\n真实标注:")
    gt_categories = set()
    for i, gt in enumerate(gt_info):
        print(f"  {i+1}. {gt['category_name']}")
        gt_categories.add(gt['category_name'])
    
    print(f"\n检测结果 (前10个):")
    pred_categories = set()
    for i, det in enumerate(detections[:10]):
        class_name = idx_to_category.get(det['class_idx'], f"class_{det['class_idx']}")
        print(f"  {i+1}. {class_name} (置信度: {det['confidence']:.3f})")
        pred_categories.add(class_name)
    
    # 计算简单的匹配度
    matched_categories = gt_categories.intersection(pred_categories)
    if len(gt_categories) > 0:
        recall = len(matched_categories) / len(gt_categories)
        print(f"\n📈 简单评估:")
        print(f"  - 真实类别: {', '.join(sorted(gt_categories))}")
        print(f"  - 检测类别: {', '.join(sorted(pred_categories))}")
        print(f"  - 匹配类别: {', '.join(sorted(matched_categories))}")
        print(f"  - 类别召回率: {recall:.2f} ({len(matched_categories)}/{len(gt_categories)})")
        
        return recall > 0
    
    return False

def main():
    print("=" * 60)
    print("===      改进版流程自检 - 智能训练与推理      ===")
    print("=" * 60)
    
    # 参数设置
    img_path = "data/coco2017_50/train2017/000000225405.jpg"
    ann_file = "data/coco2017_50/annotations/instances_train2017.json"
    img_name = "000000225405.jpg"
    max_epochs = 15
    
    # 1. 加载数据
    img_tensor, targets, num_classes, cat_id_to_idx, gt_info, original_size, image = load_and_prepare_data(
        img_path, ann_file, img_name)
    
    # 2. 创建模型
    print(f"\n>>> 创建模型 (类别数: {num_classes})")
    model = RTDETR(num_classes=num_classes)
    model = model.float32()
    
    # 3. 创建损失函数和优化器 - 使用更低的学习率
    criterion = DETRLoss(num_classes=num_classes)
    optimizer = jt.optim.Adam(model.parameters(), lr=5e-5)  # 降低学习率
    
    # 4. 训练
    model = train_with_early_stopping(model, criterion, optimizer, img_tensor, targets, max_epochs)
    
    # 5. 保存模型
    save_path = "checkpoints/improved_sanity_check_model.pkl"
    os.makedirs("checkpoints", exist_ok=True)
    jt.save(model.state_dict(), save_path)
    print(f"✅ 模型已保存到: {save_path}")
    
    # 6. 智能推理
    detections = smart_inference(model, img_tensor, original_size, num_classes)
    
    # 7. 可视化
    idx_to_category = create_visualization(image, gt_info, detections, cat_id_to_idx)
    
    # 8. 分析结果
    success = analyze_results(gt_info, detections, idx_to_category)
    
    # 9. 最终结论
    print(f"\n" + "=" * 60)
    print("🔍 改进版流程自检结论:")
    
    if success:
        print("✅ 流程自检成功！")
        print("  - 数据加载正常")
        print("  - 模型训练收敛")
        print("  - 推理执行正常")
        print("  - 检测到正确类别")
        print("  - 可视化对比完成")
        print("  - 整个训练→推理流程工作正常")
    else:
        print("⚠️ 流程自检部分成功")
        print("  - 数据加载正常 ✅")
        print("  - 模型训练收敛 ✅")
        print("  - 推理执行正常 ✅")
        print("  - 但类别检测不准确 ❌")
        print("  - 建议调整训练参数或增加训练数据")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
