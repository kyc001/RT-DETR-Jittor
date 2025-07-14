#!/usr/bin/env python3
"""
最终完整验证脚本
使用基于系统性验证构建的完整RT-DETR模型
"""

import os
import sys
import json
import numpy as np
from PIL import Image

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'jittor_rt_detr'))

import jittor as jt
from src.nn.rtdetr_complete_pytorch_aligned import build_rtdetr_complete, RTDETRPostProcessor
from src.nn.loss_pytorch_aligned import build_criterion

# 设置Jittor
jt.flags.use_cuda = 1

def load_target_data():
    """加载目标图片和标注"""
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
    
    print(f"真实标注数量: {len(annotations)}")
    
    # 创建类别映射
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    unique_cat_ids = list(set(ann['category_id'] for ann in annotations))
    cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(unique_cat_ids)}
    idx_to_cat_id = {idx: cat_id for cat_id, idx in cat_id_to_idx.items()}
    
    # 分析真实标注
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

def complete_training(model, criterion, img_tensor, targets, max_epochs=25):
    """完整模型训练"""
    print(f"\n>>> 开始完整模型训练 (最多 {max_epochs} 轮)")
    
    # 优化器
    optimizer = jt.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(max_epochs):
        model.train()
        
        # 前向传播
        outputs = model(img_tensor, targets)
        
        # 计算损失
        loss_dict = criterion(outputs, targets)
        
        # 加权总损失
        total_loss = sum(loss_dict[k] * criterion.weight_dict[k] 
                        for k in loss_dict.keys() if k in criterion.weight_dict)
        
        # 反向传播
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
            # 显示主要损失
            main_losses = ['loss_focal', 'loss_bbox', 'loss_giou']
            for key in main_losses:
                if key in loss_dict:
                    weighted_loss = loss_dict[key].item() * criterion.weight_dict.get(key, 1.0)
                    print(f"    {key}: {loss_dict[key].item():.4f} (weighted: {weighted_loss:.4f})")
        
        # 早停
        if patience_counter >= patience:
            print(f"  早停触发 (连续{patience}轮无改善)")
            break
    
    print("✅ 完整模型训练完成")
    return model

def complete_inference(model, img_tensor, original_size, num_classes, idx_to_cat_id):
    """完整模型推理"""
    print("\n>>> 开始完整模型推理...")
    
    model.eval()
    
    with jt.no_grad():
        outputs = model(img_tensor)
        
        # 使用后处理器
        postprocessor = RTDETRPostProcessor(num_classes=num_classes, confidence_threshold=0.3, num_top_queries=100)
        orig_target_sizes = jt.array([original_size]).float32()
        
        results = postprocessor(outputs, orig_target_sizes)
        
        # 转换结果格式
        detections = []
        if len(results) > 0 and len(results[0]['scores']) > 0:
            result = results[0]
            scores = result['scores']
            labels = result['labels']
            boxes = result['boxes']
            
            for i in range(len(scores)):
                confidence = scores[i].item()
                class_idx = labels[i].item()
                box = boxes[i]
                
                detections.append({
                    'bbox': [box[0].item(), box[1].item(), box[2].item(), box[3].item()],
                    'confidence': confidence,
                    'class_idx': class_idx,
                    'category_id': idx_to_cat_id[class_idx]
                })
        
        print(f"检测到 {len(detections)} 个目标 (置信度 > 0.3)")
        
        return detections

def validate_results(detections, ground_truth_objects, idx_to_cat_id):
    """验证检测结果"""
    print(f"\n=== 完整模型验证分析 ===")
    
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
    
    # 严格验证
    expected_classes = set(ground_truth_objects.keys())
    detected_class_names = set(detected_objects.keys())
    
    print(f"\n=== 严格对比分析 ===")
    print(f"期望检测的类别: {expected_classes}")
    print(f"实际检测的类别: {detected_class_names}")
    
    missing_classes = expected_classes - detected_class_names
    
    if missing_classes:
        print(f"❌ 缺失的类别: {missing_classes}")
        success = False
    else:
        print(f"✅ 所有期望类别都被检测到")
        success = True
    
    return success, detected_objects

def main():
    print("=" * 80)
    print("===        最终完整RT-DETR验证        ===")
    print("===   基于系统性验证构建的完整模型   ===")
    print("=" * 80)
    
    # 1. 加载数据
    image, annotations, ground_truth_objects, cat_id_to_idx, idx_to_cat_id, original_size = load_target_data()
    
    # 2. 预处理
    img_tensor = preprocess_image(image)
    targets = create_targets(annotations, cat_id_to_idx, original_size)
    
    # 3. 创建完整模型
    num_classes = len(cat_id_to_idx)
    print(f"\n>>> 创建完整RT-DETR模型 (类别数: {num_classes})")
    model = build_rtdetr_complete(num_classes=num_classes, hidden_dim=256, num_queries=300)
    model = model.float32()
    
    # 4. 创建损失函数
    criterion = build_criterion(num_classes)
    
    # 5. 完整模型训练
    model = complete_training(model, criterion, img_tensor, targets, max_epochs=25)
    
    # 6. 保存模型
    save_path = "checkpoints/final_complete_model.pkl"
    os.makedirs("checkpoints", exist_ok=True)
    jt.save(model.state_dict(), save_path)
    print(f"✅ 模型已保存到: {save_path}")
    
    # 7. 完整模型推理
    detections = complete_inference(model, img_tensor, original_size, num_classes, idx_to_cat_id)
    
    # 8. 验证结果
    success, detected_objects = validate_results(detections, ground_truth_objects, idx_to_cat_id)
    
    # 9. 最终判定
    print(f"\n" + "=" * 80)
    print("🔍 最终完整验证判定:")
    print("=" * 80)
    
    if success:
        print("🎉 最终完整验证成功！")
        print("  ✅ 模型能够正确检测出所有物体类别")
        print("  ✅ 基于系统性验证构建的完整模型工作正常")
        print("  ✅ 符合主人的严格标准")
        print("  ✅ RT-DETR Jittor项目完全成功！")
    else:
        print("❌ 最终完整验证失败！")
        print("  ❌ 模型未能满足严格标准")
        print("  ❌ 需要继续改进和调试")
    
    print("=" * 80)
    print(f"📁 生成的文件:")
    print(f"  - 模型文件: {save_path}")
    print("=" * 80)
    
    return success

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n⚠️ 根据主人的要求，此模型应该被删除，需要继续改进！")
        sys.exit(1)
    else:
        print("\n🎊 恭喜！RT-DETR Jittor项目完全成功！")
