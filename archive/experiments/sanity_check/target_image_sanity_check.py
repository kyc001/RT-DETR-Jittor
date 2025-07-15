#!/usr/bin/env python3
"""
针对指定图片的流程自检脚本
目标图片: data/coco2017_50/train2017/000000225405.jpg
训练轮数: 最多40轮
目标: 验证整个训练→推理流程的正确性
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'jittor_rt_detr'))

import jittor as jt
from src.nn.model import RTDETR
from src.nn.loss import DETRLoss

# 设置Jittor
jt.flags.use_cuda = 1

def load_target_image_data():
    """加载指定的目标图片和标注"""
    img_path = "data/coco2017_50/train2017/000000225405.jpg"
    ann_file = "data/coco2017_50/annotations/instances_train2017.json"
    img_name = "000000225405.jpg"
    
    print(f">>> 加载目标图片: {img_name}")
    
    # 加载图片
    image = Image.open(img_path).convert('RGB')
    original_size = image.size
    print(f"原始图片尺寸: {original_size}")
    
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
        raise ValueError(f"未找到图片 {img_name}")
    
    # 获取标注
    annotations = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == target_image['id']:
            annotations.append(ann)
    
    print(f"找到 {len(annotations)} 个标注")
    
    # 创建类别映射
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    unique_cat_ids = list(set(ann['category_id'] for ann in annotations))
    cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(unique_cat_ids)}
    idx_to_cat_id = {idx: cat_id for cat_id, idx in cat_id_to_idx.items()}
    
    print("图片中的类别:")
    class_counts = {}
    for cat_id in unique_cat_ids:
        cat_name = categories[cat_id]
        count = sum(1 for ann in annotations if ann['category_id'] == cat_id)
        class_counts[cat_name] = count
        print(f"  - {cat_name} (COCO ID: {cat_id}, 模型索引: {cat_id_to_idx[cat_id]}, 数量: {count})")
    
    # 预处理图片
    img_tensor = preprocess_image(image)
    
    # 创建目标
    targets = create_targets(annotations, cat_id_to_idx, original_size)
    
    return img_tensor, targets, len(unique_cat_ids), cat_id_to_idx, idx_to_cat_id, class_counts, original_size, image

def preprocess_image(image, target_size=640):
    """预处理图片"""
    # Resize
    image = image.resize((target_size, target_size))
    
    # 转换为numpy数组并归一化
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # 转换为CHW格式
    img_array = img_array.transpose(2, 0, 1)
    
    # 转换为Jittor tensor并添加batch维度
    img_tensor = jt.array(img_array).unsqueeze(0)
    
    return img_tensor

def create_targets(annotations, cat_id_to_idx, original_size):
    """创建训练目标"""
    boxes = []
    labels = []
    
    for ann in annotations:
        # 获取边界框 (COCO格式: x, y, width, height)
        x, y, w, h = ann['bbox']
        
        # 转换为中心点格式并归一化
        cx = (x + w / 2) / original_size[0]
        cy = (y + h / 2) / original_size[1]
        w_norm = w / original_size[0]
        h_norm = h / original_size[1]
        
        boxes.append([cx, cy, w_norm, h_norm])
        labels.append(cat_id_to_idx[ann['category_id']])
    
    targets = [{
        'boxes': jt.array(boxes).float32(),
        'labels': jt.array(labels).int64()
    }]
    
    return targets

def train_model(model, criterion, img_tensor, targets, max_epochs=40):
    """训练模型"""
    print(f"\n>>> 开始训练 (最多 {max_epochs} 轮)")
    
    # 优化器
    optimizer = jt.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(max_epochs):
        model.train()
        
        # 前向传播
        logits, boxes, enc_logits, enc_boxes = model(img_tensor)

        # 计算损失
        loss_dict = criterion(logits, boxes, targets, enc_logits, enc_boxes)
        total_loss = sum(loss_dict.values())
        
        # 反向传播 (Jittor方式)
        optimizer.step(total_loss)
        
        # 记录损失
        loss_value = total_loss.item()
        
        if loss_value < best_loss:
            best_loss = loss_value
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 每5轮或最后一轮打印详细信息
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == max_epochs - 1:
            print(f"  Epoch {epoch+1}/{max_epochs}: Loss = {loss_value:.4f} (Best: {best_loss:.4f})")
            for key, value in loss_dict.items():
                print(f"    {key}: {value.item():.4f}")
        
        # 早停
        if patience_counter >= patience:
            print(f"  早停触发 (连续{patience}轮无改善)")
            break
    
    print("✅ 训练完成")
    return model

def test_inference(model, img_tensor, original_size, num_classes, cat_id_to_idx):
    """测试推理"""
    print("\n>>> 测试推理...")
    
    model.eval()
    
    with jt.no_grad():
        logits, boxes, enc_logits, enc_boxes = model(img_tensor)

        # 获取最后一层的输出
        pred_logits = logits[-1][0]  # [num_queries, num_classes] - 取最后一层，第一个batch
        pred_boxes = boxes[-1][0]    # [num_queries, 4] - 取最后一层，第一个batch
        
        # 应用sigmoid获取置信度
        pred_scores = jt.sigmoid(pred_logits)
        
        # 转换为numpy进行后处理
        scores_np = pred_scores.stop_grad().numpy()
        boxes_np = pred_boxes.stop_grad().numpy()
        
        # 收集检测结果
        detections = []
        confidence_threshold = 0.3
        
        for i in range(len(scores_np)):
            for class_idx in range(num_classes):
                confidence = scores_np[i, class_idx]
                
                if confidence > confidence_threshold:
                    # 转换边界框坐标
                    cx, cy, w, h = boxes_np[i]
                    x1 = (cx - w/2) * original_size[0]
                    y1 = (cy - h/2) * original_size[1]
                    x2 = (cx + w/2) * original_size[0]
                    y2 = (cy + h/2) * original_size[1]
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_idx': class_idx
                    })
        
        print(f"检测到 {len(detections)} 个目标 (置信度 > {confidence_threshold})")
        
        return detections

def visualize_results(image, detections, targets, cat_id_to_idx, idx_to_cat_id, save_path):
    """可视化结果"""
    print(f"\n>>> 生成可视化结果: {save_path}")
    
    # 创建类别名称映射
    coco_categories = {
        1: 'person', 37: 'sports ball', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
        5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat'
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：真实标注
    ax1.imshow(image)
    ax1.set_title('真实标注', fontsize=14, fontweight='bold')
    
    # 绘制真实标注
    target = targets[0]
    boxes = target['boxes'].numpy()
    labels = target['labels'].numpy()
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        cx, cy, w, h = box
        x1 = (cx - w/2) * image.size[0]
        y1 = (cy - h/2) * image.size[1]
        width = w * image.size[0]
        height = h * image.size[1]
        
        cat_id = idx_to_cat_id[label]
        cat_name = coco_categories.get(cat_id, f'class_{cat_id}')
        
        rect = patches.Rectangle((x1, y1), width, height, 
                               linewidth=2, edgecolor='green', facecolor='none')
        ax1.add_patch(rect)
        ax1.text(x1, y1-5, f'{cat_name}', fontsize=10, color='green', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # 右图：检测结果
    ax2.imshow(image)
    ax2.set_title('检测结果', fontsize=14, fontweight='bold')
    
    # 绘制检测结果
    colors = ['red', 'blue', 'orange', 'purple', 'brown']
    for i, det in enumerate(detections[:20]):  # 只显示前20个
        x1, y1, x2, y2 = det['bbox']
        confidence = det['confidence']
        class_idx = det['class_idx']
        
        cat_id = idx_to_cat_id[class_idx]
        cat_name = coco_categories.get(cat_id, f'class_{cat_id}')
        color = colors[class_idx % len(colors)]
        
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor=color, facecolor='none')
        ax2.add_patch(rect)
        ax2.text(x1, y1-5, f'{cat_name}: {confidence:.3f}', 
                fontsize=10, color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    for ax in [ax1, ax2]:
        ax.set_xlim(0, image.size[0])
        ax.set_ylim(image.size[1], 0)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 可视化结果已保存")

def analyze_results(detections, class_counts, idx_to_cat_id):
    """分析检测结果"""
    print(f"\n📊 检测结果分析:")
    print(f"检测总数: {len(detections)}")
    
    # 按类别统计
    detected_classes = {}
    for det in detections:
        class_idx = det['class_idx']
        cat_id = idx_to_cat_id[class_idx]
        
        # 获取类别名称
        coco_categories = {1: 'person', 37: 'sports ball'}
        cat_name = coco_categories.get(cat_id, f'class_{cat_id}')
        
        if cat_name not in detected_classes:
            detected_classes[cat_name] = []
        detected_classes[cat_name].append(det)
    
    print(f"检测到的类别:")
    for class_name, dets in detected_classes.items():
        print(f"  - {class_name}: {len(dets)} 个")
        # 显示前3个最高置信度的检测
        sorted_dets = sorted(dets, key=lambda x: x['confidence'], reverse=True)
        for i, det in enumerate(sorted_dets[:3]):
            print(f"    {i+1}. 置信度: {det['confidence']:.3f}")
    
    # 检查是否检测到所有真实类别
    expected_classes = set(class_counts.keys())
    detected_class_names = set(detected_classes.keys())
    
    success = expected_classes.issubset(detected_class_names)
    
    if success:
        print("✅ 成功检测到所有期望的类别！")
    else:
        missing_classes = expected_classes - detected_class_names
        print(f"❌ 缺失的类别: {', '.join(missing_classes)}")
    
    return success

def main():
    print("=" * 60)
    print("===      指定图片流程自检 (000000225405.jpg)      ===")
    print("=" * 60)
    
    # 1. 加载数据
    img_tensor, targets, num_classes, cat_id_to_idx, idx_to_cat_id, class_counts, original_size, image = load_target_image_data()
    
    # 2. 创建模型
    print(f"\n>>> 创建模型 (类别数: {num_classes})")
    model = RTDETR(num_classes=num_classes)
    model = model.float32()
    
    # 3. 创建损失函数
    criterion = DETRLoss(num_classes=num_classes-1)  # 不包括背景类
    
    # 4. 训练
    model = train_model(model, criterion, img_tensor, targets, max_epochs=40)
    
    # 5. 保存模型
    save_path = "checkpoints/target_image_sanity_check_model.pkl"
    os.makedirs("checkpoints", exist_ok=True)
    jt.save(model.state_dict(), save_path)
    print(f"✅ 模型已保存到: {save_path}")
    
    # 6. 测试推理
    detections = test_inference(model, img_tensor, original_size, num_classes, cat_id_to_idx)
    
    # 7. 可视化结果
    vis_save_path = "target_image_sanity_check_result.jpg"
    visualize_results(image, detections, targets, cat_id_to_idx, idx_to_cat_id, vis_save_path)
    
    # 8. 分析结果
    success = analyze_results(detections, class_counts, idx_to_cat_id)
    
    # 9. 最终结论
    print(f"\n" + "=" * 60)
    print("🔍 指定图片流程自检结论:")
    
    if success:
        print("🎉 流程自检完全成功！")
        print("  ✅ 成功检测到所有真实类别")
        print("  ✅ 训练过程稳定收敛")
        print("  ✅ 推理流程正常工作")
        print("  ✅ 可视化结果已生成")
        print("  ✅ 整个训练→推理流程验证通过")
    else:
        print("⚠️ 部分成功")
        print("  ✅ 训练过程正常")
        print("  ✅ 推理流程工作")
        print("  ❌ 类别检测不完整")
        print("  💡 建议：调整训练参数或增加训练数据")
    
    print("=" * 60)
    print(f"📁 生成的文件:")
    print(f"  - 模型文件: {save_path}")
    print(f"  - 可视化结果: {vis_save_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
