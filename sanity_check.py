#!/usr/bin/env python3
"""
流程自检脚本 - 最小可行性验证
使用单张图片进行过拟合训练，然后验证推理结果是否正确
"""

import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'jittor_rt_detr'))

import jittor as jt
from src.nn.model import RTDETR
from src.nn.loss import DETRLoss

def load_single_image_data(img_path, ann_file, target_image_name):
    """加载单张图片的数据和标注"""
    print(f"=== 加载图片数据: {target_image_name} ===")
    
    # 加载COCO标注
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # 找到目标图片
    target_image = None
    for img in coco_data['images']:
        if img['file_name'] == target_image_name:
            target_image = img
            break
    
    if target_image is None:
        raise ValueError(f"找不到图片: {target_image_name}")
    
    # 找到该图片的所有标注
    image_annotations = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == target_image['id']:
            image_annotations.append(ann)
    
    # 创建类别映射
    cat_id_to_name = {}
    for cat in coco_data['categories']:
        cat_id_to_name[cat['id']] = cat['name']
    
    print(f"✅ 图片信息: {target_image['width']} x {target_image['height']}")
    print(f"📋 标注数量: {len(image_annotations)}")
    
    for i, ann in enumerate(image_annotations):
        cat_name = cat_id_to_name[ann['category_id']]
        bbox = ann['bbox']
        print(f"  {i+1}. {cat_name} (ID:{ann['category_id']}) - 边界框: {bbox}")
    
    return target_image, image_annotations, cat_id_to_name

def preprocess_single_image(img_path, annotations, cat_id_to_name, target_size=(640, 640)):
    """预处理单张图片和标注"""
    # 加载图片
    image = Image.open(img_path).convert('RGB')
    original_size = image.size
    
    # Resize图片
    resized_image = image.resize(target_size, Image.LANCZOS)
    
    # 转换为tensor
    img_array = np.array(resized_image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    img_tensor = jt.array(img_array.transpose(2, 0, 1)).unsqueeze(0)  # (1, 3, H, W)
    
    # 处理标注
    boxes = []
    labels = []
    unique_cat_ids = sorted(set(ann['category_id'] for ann in annotations))
    cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(unique_cat_ids)}
    
    scale_x = target_size[0] / original_size[0]
    scale_y = target_size[1] / original_size[1]
    
    for ann in annotations:
        # COCO格式: [x, y, width, height] -> 缩放后的cxcywh格式
        x, y, w, h = ann['bbox']
        
        # 缩放到目标尺寸
        x_scaled = x * scale_x
        y_scaled = y * scale_y
        w_scaled = w * scale_x
        h_scaled = h * scale_y
        
        # 转换为中心点坐标并归一化
        cx = (x_scaled + w_scaled / 2) / target_size[0]
        cy = (y_scaled + h_scaled / 2) / target_size[1]
        w_norm = w_scaled / target_size[0]
        h_norm = h_scaled / target_size[1]
        
        boxes.append([cx, cy, w_norm, h_norm])
        labels.append(cat_id_to_idx[ann['category_id']])
        
        cat_name = cat_id_to_name[ann['category_id']]
        print(f"  标注转换: {cat_name} -> 归一化cxcywh: [{cx:.3f}, {cy:.3f}, {w_norm:.3f}, {h_norm:.3f}]")
    
    # 确保tensor有正确的形状
    if len(boxes) == 0:
        boxes_tensor = jt.zeros((0, 4), dtype='float32')
        labels_tensor = jt.zeros((0,), dtype='int64')
    else:
        # 创建tensor
        boxes_tensor = jt.array(boxes, dtype='float32')  # (num_objects, 4)
        labels_tensor = jt.array(labels, dtype='int64')  # (num_objects,)

        # 确保是正确的形状
        if boxes_tensor.ndim == 1:
            boxes_tensor = boxes_tensor.unsqueeze(0)
        if labels_tensor.ndim == 0:
            labels_tensor = labels_tensor.unsqueeze(0)

        print(f"✅ 数据准备完成: boxes_tensor.shape={boxes_tensor.shape}, labels_tensor.shape={labels_tensor.shape}")
    
    # 准备目标字典
    targets = [{
        'boxes': boxes_tensor,
        'labels': labels_tensor
    }]


    return img_tensor, targets, cat_id_to_idx

def train_single_image(model, criterion, img_tensor, targets, epochs=100, lr=1e-4):
    """在单张图片上进行过拟合训练"""
    print(f"\n=== 开始过拟合训练 (epochs={epochs}) ===")
    
    optimizer = jt.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(img_tensor)
        
        # 计算损失
        logits, boxes, enc_logits, enc_boxes = outputs
        loss_dict = criterion(logits, boxes, targets, enc_logits, enc_boxes)
        total_loss = sum(loss_dict.values())
        
        # 反向传播
        optimizer.backward(total_loss)
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}")
            for key, value in loss_dict.items():
                print(f"  {key}: {value.item():.4f}")
    
    print("✅ 过拟合训练完成")
    return model

def inference_and_check(model, img_tensor, targets, cat_id_to_idx, cat_id_to_name, 
                       original_size, conf_threshold=0.3):
    """推理并检查结果"""
    print(f"\n=== 推理验证 ===")
    
    model.eval()
    with jt.no_grad():
        outputs = model(img_tensor)
    
    # 解包输出
    logits, boxes, _, _ = outputs
    pred_logits = logits[-1][0]  # (num_queries, num_classes)
    pred_boxes = boxes[-1][0]    # (num_queries, 4)
    
    # 计算分数
    scores = jt.sigmoid(pred_logits)
    scores_max = jt.max(scores, dim=-1)[0]
    labels_pred = jt.argmax(scores, dim=-1)
    
    # 过滤低置信度预测
    keep_mask = scores_max > conf_threshold
    keep_indices = jt.where(keep_mask)[0]
    
    if len(keep_indices) == 0:
        print("❌ 没有检测到任何目标！")
        return False
    
    final_boxes = pred_boxes[keep_indices]
    final_scores = scores_max[keep_indices]
    final_labels = labels_pred[keep_indices]
    
    print(f"📊 检测结果: {len(final_boxes)} 个目标")
    
    # 创建反向映射
    idx_to_cat_id = {idx: cat_id for cat_id, idx in cat_id_to_idx.items()}
    
    detected_objects = []
    for i in range(len(final_boxes)):
        box = final_boxes[i].numpy()
        score = final_scores[i].item()
        label_idx = final_labels[i].item()
        
        # 转换回像素坐标
        cx, cy, w, h = box
        x1 = (cx - w/2) * original_size[0]
        y1 = (cy - h/2) * original_size[1]
        x2 = (cx + w/2) * original_size[0]
        y2 = (cy + h/2) * original_size[1]
        
        if label_idx in idx_to_cat_id:
            cat_id = idx_to_cat_id[label_idx]
            cat_name = cat_id_to_name[cat_id]
        else:
            cat_name = f"unknown_{label_idx}"
        
        detected_objects.append({
            'class': cat_name,
            'confidence': score,
            'bbox': [x1, y1, x2, y2]
        })
        
        print(f"  {i+1}. {cat_name}: 置信度={score:.3f}, 边界框=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    
    # 验证是否检测到了所有真实目标
    gt_classes = set()
    for target in targets:
        for label_idx in target['labels'].numpy():
            cat_id = idx_to_cat_id[label_idx]
            gt_classes.add(cat_id_to_name[cat_id])
    
    detected_classes = set(obj['class'] for obj in detected_objects)
    
    print(f"\n📋 验证结果:")
    print(f"  真实目标类别: {sorted(gt_classes)}")
    print(f"  检测到的类别: {sorted(detected_classes)}")
    
    missing_classes = gt_classes - detected_classes
    extra_classes = detected_classes - gt_classes
    
    success = True
    if missing_classes:
        print(f"  ❌ 漏检的类别: {sorted(missing_classes)}")
        success = False
    if extra_classes:
        print(f"  ⚠️  额外检测的类别: {sorted(extra_classes)}")
    
    if success and len(detected_objects) >= len(targets[0]['labels']):
        print("  ✅ 流程自检通过！模型能够正确学习和检测目标")
    else:
        print("  ❌ 流程自检失败！存在漏检或其他问题")
    
    return success, detected_objects

def visualize_results(img_path, detected_objects, output_path):
    """可视化检测结果"""
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    colors = ["red", "blue", "green", "yellow", "purple", "orange"]
    
    for i, obj in enumerate(detected_objects):
        x1, y1, x2, y2 = obj['bbox']
        color = colors[i % len(colors)]
        
        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # 绘制标签
        text = f"{obj['class']}: {obj['confidence']:.2f}"
        draw.text((x1, y1-25), text, fill=color, font=font)
    
    img.save(output_path)
    print(f"✅ 可视化结果已保存: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="RT-DETR流程自检")
    parser.add_argument('--img_path', type=str, required=True, help='测试图片路径')
    parser.add_argument('--ann_file', type=str, required=True, help='COCO标注文件路径')
    parser.add_argument('--img_name', type=str, required=True, help='图片文件名')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--output_path', type=str, default='sanity_check_result.jpg', help='输出路径')
    
    args = parser.parse_args()
    
    try:
        # 1. 加载数据
        image_info, annotations, cat_id_to_name = load_single_image_data(
            args.img_path, args.ann_file, args.img_name
        )
        
        # 2. 预处理
        img_tensor, targets, cat_id_to_idx = preprocess_single_image(
            args.img_path, annotations, cat_id_to_name
        )
        
        # 3. 创建模型
        num_classes = len(cat_id_to_idx)
        print(f"\n📊 模型配置: {num_classes} 个类别")
        
        model = RTDETR(num_classes=num_classes)
        criterion = DETRLoss(num_classes=num_classes)
        
        # 4. 过拟合训练
        model = train_single_image(model, criterion, img_tensor, targets, args.epochs)
        
        # 5. 推理验证
        success, detected_objects = inference_and_check(
            model, img_tensor, targets, cat_id_to_idx, cat_id_to_name, 
            (image_info['width'], image_info['height'])
        )
        
        # 6. 可视化
        if detected_objects:
            visualize_results(args.img_path, detected_objects, args.output_path)
        
        # 7. 总结
        print(f"\n🎯 流程自检结果: {'✅ 通过' if success else '❌ 失败'}")
        
        if success:
            print("恭喜！训练和推理流程工作正常，可以进行更大规模的训练。")
        else:
            print("流程存在问题，需要进一步调试模型、损失函数或数据处理部分。")
            
    except Exception as e:
        print(f"❌ 流程自检出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
