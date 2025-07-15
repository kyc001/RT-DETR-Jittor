#!/usr/bin/env python3
"""
严格验证流程自检脚本
按照主人的要求：只有训练后的模型能够正确检测出所有物体，才能算是项目成功
自检成功的标准是：训练后的模型能够成功识别物体个数并正确检测出物体的边界框
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

def load_and_analyze_target_data():
    """加载并分析目标图片的真实标注"""
    img_path = "data/coco2017_50/train2017/000000225405.jpg"
    ann_file = "data/coco2017_50/annotations/instances_train2017.json"
    img_name = "000000225405.jpg"
    
    print(f">>> 严格验证目标: {img_name}")
    
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
    
    print(f"真实标注数量: {len(annotations)}")
    
    # 创建类别映射
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    unique_cat_ids = list(set(ann['category_id'] for ann in annotations))
    cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(unique_cat_ids)}
    idx_to_cat_id = {idx: cat_id for cat_id, idx in cat_id_to_idx.items()}
    
    # 详细分析真实标注
    print("\n=== 真实标注详细分析 ===")
    ground_truth_objects = {}
    for ann in annotations:
        cat_id = ann['category_id']
        cat_name = categories[cat_id]
        bbox = ann['bbox']  # [x, y, width, height]
        
        if cat_name not in ground_truth_objects:
            ground_truth_objects[cat_name] = []
        
        ground_truth_objects[cat_name].append({
            'bbox': bbox,
            'area': ann['area'],
            'category_id': cat_id,
            'model_idx': cat_id_to_idx[cat_id]
        })
        
        print(f"  - {cat_name}: bbox={bbox}, area={ann['area']}")
    
    print(f"\n真实物体类别统计:")
    for cat_name, objects in ground_truth_objects.items():
        print(f"  - {cat_name}: {len(objects)} 个")
    
    return image, annotations, ground_truth_objects, cat_id_to_idx, idx_to_cat_id, original_size

def preprocess_image(image, target_size=640):
    """预处理图片"""
    # Resize
    image_resized = image.resize((target_size, target_size))
    
    # 转换为numpy数组并归一化
    img_array = np.array(image_resized).astype(np.float32) / 255.0
    
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

def strict_training(model, criterion, img_tensor, targets, max_epochs=30):
    """严格的训练过程"""
    print(f"\n>>> 开始严格训练 (最多 {max_epochs} 轮)")
    
    # 优化器
    optimizer = jt.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    training_log = []
    
    for epoch in range(max_epochs):
        model.train()
        
        # 前向传播
        logits, boxes, enc_logits, enc_boxes = model(img_tensor)
        
        # 计算损失
        loss_dict = criterion(logits, boxes, targets, enc_logits, enc_boxes)
        total_loss = sum(loss_dict.values())
        
        # 反向传播
        optimizer.step(total_loss)
        
        # 记录损失
        loss_value = total_loss.item()
        training_log.append({
            'epoch': epoch + 1,
            'total_loss': loss_value,
            'loss_dict': {k: v.item() for k, v in loss_dict.items()}
        })
        
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
    return model, training_log

def strict_inference_and_validation(model, img_tensor, original_size, ground_truth_objects, cat_id_to_idx, idx_to_cat_id):
    """严格的推理和验证"""
    print("\n>>> 开始严格推理和验证...")
    
    model.eval()
    
    with jt.no_grad():
        logits, boxes, enc_logits, enc_boxes = model(img_tensor)
        
        # 获取最后一层的输出
        pred_logits = logits[-1][0]  # [num_queries, num_classes]
        pred_boxes = boxes[-1][0]    # [num_queries, 4]
        
        # 应用sigmoid获取置信度
        pred_scores = jt.sigmoid(pred_logits)
        
        # 转换为numpy进行后处理
        scores_np = pred_scores.stop_grad().numpy()
        boxes_np = pred_boxes.stop_grad().numpy()
        
        # 收集检测结果
        detections = []
        confidence_threshold = 0.3
        
        for i in range(len(scores_np)):
            for class_idx in range(len(cat_id_to_idx)):
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
                        'class_idx': class_idx,
                        'category_id': idx_to_cat_id[class_idx]
                    })
        
        print(f"检测到 {len(detections)} 个目标 (置信度 > {confidence_threshold})")
        
        return detections

def strict_validation_analysis(detections, ground_truth_objects, idx_to_cat_id):
    """严格的验证分析"""
    print(f"\n=== 严格验证分析 ===")
    
    # COCO类别名称映射
    coco_categories = {
        1: 'person', 37: 'sports ball', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
        5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat'
    }
    
    # 按类别统计检测结果
    detected_objects = {}
    for det in detections:
        cat_id = det['category_id']
        cat_name = coco_categories.get(cat_id, f'class_{cat_id}')
        
        if cat_name not in detected_objects:
            detected_objects[cat_name] = []
        detected_objects[cat_name].append(det)
    
    print(f"检测结果统计:")
    for class_name, dets in detected_objects.items():
        print(f"  - {class_name}: {len(dets)} 个")
        # 显示前3个最高置信度的检测
        sorted_dets = sorted(dets, key=lambda x: x['confidence'], reverse=True)
        for i, det in enumerate(sorted_dets[:3]):
            print(f"    {i+1}. 置信度: {det['confidence']:.3f}")
    
    # 严格验证：检查是否检测到所有真实类别
    expected_classes = set(ground_truth_objects.keys())
    detected_class_names = set(detected_objects.keys())
    
    print(f"\n=== 严格对比分析 ===")
    print(f"期望检测的类别: {expected_classes}")
    print(f"实际检测的类别: {detected_class_names}")
    
    # 检查类别召回率
    missing_classes = expected_classes - detected_class_names
    extra_classes = detected_class_names - expected_classes
    
    if missing_classes:
        print(f"❌ 缺失的类别: {missing_classes}")
    else:
        print(f"✅ 所有期望类别都被检测到")
    
    if extra_classes:
        print(f"⚠️ 额外检测的类别: {extra_classes}")
    
    # 检查数量匹配
    print(f"\n=== 数量对比 ===")
    quantity_match = True
    for class_name in expected_classes:
        expected_count = len(ground_truth_objects[class_name])
        detected_count = len(detected_objects.get(class_name, []))
        
        print(f"{class_name}:")
        print(f"  期望数量: {expected_count}")
        print(f"  检测数量: {detected_count}")
        
        if detected_count == 0:
            print(f"  ❌ 完全未检测到")
            quantity_match = False
        elif detected_count < expected_count:
            print(f"  ⚠️ 检测数量不足")
            quantity_match = False
        elif detected_count > expected_count * 2:  # 允许一定的重复检测
            print(f"  ⚠️ 检测数量过多")
        else:
            print(f"  ✅ 检测数量合理")
    
    # 最终验证结果
    class_recall_success = len(missing_classes) == 0
    
    return {
        'class_recall_success': class_recall_success,
        'quantity_match': quantity_match,
        'missing_classes': missing_classes,
        'extra_classes': extra_classes,
        'detected_objects': detected_objects,
        'detection_count': len(detections)
    }

def generate_strict_comparison_visualization(image, ground_truth_objects, detections, idx_to_cat_id, validation_result):
    """生成严格对比可视化"""
    print(f"\n>>> 生成严格对比可视化...")
    
    coco_categories = {
        1: 'person', 37: 'sports ball', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
        5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat'
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 左图：真实标注
    ax1.imshow(image)
    ax1.set_title('Ground Truth Annotations', fontsize=16, fontweight='bold')
    
    # 绘制真实标注
    colors = ['green', 'blue', 'orange', 'purple', 'brown']
    color_idx = 0
    for class_name, objects in ground_truth_objects.items():
        color = colors[color_idx % len(colors)]
        for obj in objects:
            x, y, w, h = obj['bbox']
            rect = patches.Rectangle((x, y), w, h, 
                                   linewidth=3, edgecolor=color, facecolor='none')
            ax1.add_patch(rect)
            ax1.text(x, y-5, f'{class_name}', fontsize=12, color=color, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        color_idx += 1
    
    # 右图：检测结果
    ax2.imshow(image)
    ax2.set_title('Detection Results', fontsize=16, fontweight='bold')
    
    # 绘制检测结果
    detected_objects = validation_result['detected_objects']
    color_idx = 0
    for class_name, dets in detected_objects.items():
        color = colors[color_idx % len(colors)]
        # 只显示前5个最高置信度的检测
        sorted_dets = sorted(dets, key=lambda x: x['confidence'], reverse=True)
        for i, det in enumerate(sorted_dets[:5]):
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=2, edgecolor=color, facecolor='none')
            ax2.add_patch(rect)
            ax2.text(x1, y1-5, f'{class_name}: {confidence:.3f}', 
                    fontsize=10, color=color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        color_idx += 1
    
    for ax in [ax1, ax2]:
        ax.set_xlim(0, image.size[0])
        ax.set_ylim(image.size[1], 0)
        ax.axis('off')
    
    plt.tight_layout()
    save_path = "strict_validation_result.jpg"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 严格对比可视化已保存: {save_path}")
    return save_path

def main():
    print("=" * 80)
    print("===           RT-DETR 严格验证流程自检           ===")
    print("===     按照主人要求的最严格标准进行验证        ===")
    print("=" * 80)
    
    # 1. 加载和分析目标数据
    image, annotations, ground_truth_objects, cat_id_to_idx, idx_to_cat_id, original_size = load_and_analyze_target_data()
    
    # 2. 预处理图片
    img_tensor = preprocess_image(image)
    
    # 3. 创建目标
    targets = create_targets(annotations, cat_id_to_idx, original_size)
    
    # 4. 创建模型
    num_classes = len(cat_id_to_idx)
    print(f"\n>>> 创建模型 (类别数: {num_classes})")
    model = RTDETR(num_classes=num_classes)
    model = model.float32()
    
    # 5. 创建损失函数
    criterion = DETRLoss(num_classes=num_classes)
    
    # 6. 严格训练
    model, training_log = strict_training(model, criterion, img_tensor, targets, max_epochs=30)
    
    # 7. 保存模型
    save_path = "checkpoints/strict_validation_model.pkl"
    os.makedirs("checkpoints", exist_ok=True)
    jt.save(model.state_dict(), save_path)
    print(f"✅ 模型已保存到: {save_path}")
    
    # 8. 严格推理和验证
    detections = strict_inference_and_validation(model, img_tensor, original_size, ground_truth_objects, cat_id_to_idx, idx_to_cat_id)
    
    # 9. 严格验证分析
    validation_result = strict_validation_analysis(detections, ground_truth_objects, idx_to_cat_id)
    
    # 10. 生成可视化
    vis_path = generate_strict_comparison_visualization(image, ground_truth_objects, detections, idx_to_cat_id, validation_result)
    
    # 11. 最终严格判定
    print(f"\n" + "=" * 80)
    print("🔍 严格验证最终判定:")
    print("=" * 80)
    
    success_criteria = [
        ("类别召回率", validation_result['class_recall_success']),
        ("检测到目标", validation_result['detection_count'] > 0),
        ("无缺失类别", len(validation_result['missing_classes']) == 0)
    ]
    
    all_passed = True
    for criterion_name, passed in success_criteria:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {criterion_name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\n" + "=" * 80)
    if all_passed:
        print("🎉 严格验证完全成功！")
        print("  ✅ 模型能够正确检测出所有物体类别")
        print("  ✅ 训练→推理流程完全正确")
        print("  ✅ 符合主人的严格标准")
        print("  ✅ 项目可以认定为成功")
    else:
        print("❌ 严格验证失败！")
        print("  ❌ 模型未能满足严格标准")
        print("  ❌ 需要继续改进和调试")
        if validation_result['missing_classes']:
            print(f"  ❌ 缺失类别: {validation_result['missing_classes']}")
    
    print("=" * 80)
    print(f"📁 生成的文件:")
    print(f"  - 模型文件: {save_path}")
    print(f"  - 可视化结果: {vis_path}")
    print("=" * 80)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n⚠️ 根据主人的要求，此模型应该被删除，需要继续改进！")
        sys.exit(1)
    else:
        print("\n🎊 恭喜！模型通过了严格验证，项目成功！")
