#!/usr/bin/env python3
"""
检查坐标转换是否正确
对比真实标注和模型预测的坐标
"""

import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt
from jittor_rt_detr.src.nn.rtdetr_complete_pytorch_aligned import build_rtdetr_complete

# 设置Jittor
jt.flags.use_cuda = 1
jt.flags.auto_mixed_precision_level = 0

def safe_float32(tensor):
    if isinstance(tensor, jt.Var):
        return tensor.float32()
    elif isinstance(tensor, np.ndarray):
        return jt.array(tensor.astype(np.float32))
    else:
        return jt.array(tensor, dtype=jt.float32)

def load_coco_data():
    """加载COCO数据"""
    ann_file = "data/coco2017_50/annotations/instances_train2017.json"
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    return coco_data

def analyze_single_image(image_path, coco_data):
    """分析单张图片的坐标转换"""
    image_name = os.path.basename(image_path)
    
    # 找到对应的图片信息
    image_info = None
    for img in coco_data['images']:
        if img['file_name'] == image_name:
            image_info = img
            break
    
    if image_info is None:
        print(f"❌ 找不到图片信息: {image_name}")
        return
    
    print(f"\n" + "="*60)
    print(f"分析图片: {image_name}")
    print(f"图片ID: {image_info['id']}")
    print(f"原始尺寸: {image_info['width']} x {image_info['height']}")
    print("="*60)
    
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    print(f"实际尺寸: {original_size[0]} x {original_size[1]}")
    
    # 获取该图片的标注
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_info['id']]
    print(f"标注数量: {len(annotations)}")
    
    # 类别映射
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    
    print(f"\n=== 原始COCO标注 ===")
    for i, ann in enumerate(annotations):
        x, y, w, h = ann['bbox']  # COCO格式：[x, y, width, height]
        cat_name = cat_id_to_name[ann['category_id']]
        print(f"{i+1}. {cat_name}: COCO格式 [{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}]")
        print(f"   左上角: ({x:.1f}, {y:.1f}), 右下角: ({x+w:.1f}, {y+h:.1f})")
    
    print(f"\n=== 训练时的坐标转换 ===")
    training_boxes = []
    training_labels = []
    
    for i, ann in enumerate(annotations):
        x, y, w, h = ann['bbox']
        cat_name = cat_id_to_name[ann['category_id']]
        
        # 训练时的转换逻辑
        cx = np.float32(x + w / 2) / np.float32(original_size[0])
        cy = np.float32(y + h / 2) / np.float32(original_size[1])
        w_norm = np.float32(w) / np.float32(original_size[0])
        h_norm = np.float32(h) / np.float32(original_size[1])
        
        training_boxes.append([cx, cy, w_norm, h_norm])
        training_labels.append(cat_id_to_idx[ann['category_id']])
        
        print(f"{i+1}. {cat_name}: 归一化格式 [{cx:.4f}, {cy:.4f}, {w_norm:.4f}, {h_norm:.4f}]")
        print(f"   中心点: ({cx:.4f}, {cy:.4f}), 尺寸: ({w_norm:.4f}, {h_norm:.4f})")
    
    # 加载模型进行预测
    print(f"\n=== 模型预测结果 ===")
    model = build_rtdetr_complete(num_classes=80, hidden_dim=256, num_queries=300)
    state_dict = jt.load("checkpoints/rtdetr_jittor.pkl")
    model.load_state_dict(state_dict)
    model.eval()
    
    # 预处理图片
    image_resized = image.resize((640, 640))
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    img_tensor = safe_float32(img_array).unsqueeze(0)
    
    # 推理
    with jt.no_grad():
        outputs = model(img_tensor)
    
    pred_logits = outputs['pred_logits'][0]
    pred_boxes = outputs['pred_boxes'][0]
    
    # 后处理
    pred_probs = jt.sigmoid(pred_logits)
    pred_scores = pred_probs.max(dim=-1)[0]
    pred_labels = pred_probs.argmax(dim=-1)
    
    # 找到高置信度预测
    keep = pred_scores > 0.1
    if keep.sum() > 0:
        filtered_scores = pred_scores[keep].numpy()
        filtered_labels = pred_labels[keep].numpy().astype(int)
        filtered_boxes = pred_boxes[keep].numpy()
        
        idx_to_name = {idx: cat['name'] for idx, cat in enumerate(coco_data['categories'])}
        
        print(f"高置信度预测 (>0.1): {len(filtered_scores)} 个")
        for i, (score, label, box) in enumerate(zip(filtered_scores, filtered_labels, filtered_boxes)):
            cx, cy, w, h = box
            class_name = idx_to_name.get(label, f'class_{label}')
            print(f"{i+1}. {class_name}: 置信度={score:.4f}, 归一化格式 [{cx:.4f}, {cy:.4f}, {w:.4f}, {h:.4f}]")
    else:
        print("没有高置信度预测")
    
    # 创建可视化对比
    print(f"\n=== 创建可视化对比 ===")
    
    # 创建对比图：左侧真实标注，右侧模型预测
    fig_width = original_size[0] * 2
    fig_height = original_size[1]
    comparison_img = Image.new('RGB', (fig_width, fig_height), 'white')
    
    # 左侧：真实标注
    gt_img = image.copy()
    gt_draw = ImageDraw.Draw(gt_img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for ann in annotations:
        x, y, w, h = ann['bbox']
        cat_name = cat_id_to_name[ann['category_id']]
        
        # 绘制真实边界框
        gt_draw.rectangle([x, y, x+w, y+h], outline='green', width=3)
        gt_draw.text((x, y-20), f'GT: {cat_name}', fill='green', font=font)
    
    # 右侧：模型预测
    pred_img = image.copy()
    pred_draw = ImageDraw.Draw(pred_img)
    
    if keep.sum() > 0:
        for i, (score, label, box) in enumerate(zip(filtered_scores, filtered_labels, filtered_boxes)):
            cx, cy, w, h = box
            class_name = idx_to_name.get(label, f'class_{label}')
            
            # 转换回原始图片坐标
            cx_pixel = cx * original_size[0]
            cy_pixel = cy * original_size[1]
            w_pixel = w * original_size[0]
            h_pixel = h * original_size[1]
            
            x1 = cx_pixel - w_pixel / 2
            y1 = cy_pixel - h_pixel / 2
            x2 = cx_pixel + w_pixel / 2
            y2 = cy_pixel + h_pixel / 2
            
            # 绘制预测边界框
            color = 'red' if i == 0 else 'blue'
            pred_draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            pred_draw.text((x1, y1-20), f'Pred: {class_name} ({score:.2f})', fill=color, font=font)
    
    # 合并图片
    comparison_img.paste(gt_img, (0, 0))
    comparison_img.paste(pred_img, (original_size[0], 0))
    
    # 添加标题
    title_draw = ImageDraw.Draw(comparison_img)
    title_draw.text((10, 10), "Ground Truth", fill='green', font=font)
    title_draw.text((original_size[0] + 10, 10), "Model Prediction", fill='red', font=font)
    
    # 保存对比图
    os.makedirs("results/coordinate_check", exist_ok=True)
    save_path = f"results/coordinate_check/comparison_{image_name}"
    comparison_img.save(save_path)
    print(f"✅ 对比图保存到: {save_path}")
    
    # 分析问题
    print(f"\n=== 问题分析 ===")
    
    if len(annotations) == 0:
        print("⚠️ 该图片没有标注数据")
        return
    
    if keep.sum() == 0:
        print("❌ 模型没有任何高置信度预测")
        print("   可能原因：模型训练不充分")
        return
    
    # 检查预测是否合理
    gt_centers = []
    for ann in annotations:
        x, y, w, h = ann['bbox']
        gt_cx = (x + w/2) / original_size[0]
        gt_cy = (y + h/2) / original_size[1]
        gt_centers.append((gt_cx, gt_cy))
    
    pred_centers = []
    for box in filtered_boxes:
        cx, cy, w, h = box
        pred_centers.append((cx, cy))
    
    print(f"真实中心点 (归一化):")
    for i, (cx, cy) in enumerate(gt_centers):
        print(f"  GT{i+1}: ({cx:.4f}, {cy:.4f})")
    
    print(f"预测中心点 (归一化):")
    for i, (cx, cy) in enumerate(pred_centers):
        print(f"  Pred{i+1}: ({cx:.4f}, {cy:.4f})")
    
    # 检查是否都集中在中心
    center_threshold = 0.1  # 距离图片中心0.1的范围内
    pred_in_center = 0
    for cx, cy in pred_centers:
        if abs(cx - 0.5) < center_threshold and abs(cy - 0.5) < center_threshold:
            pred_in_center += 1
    
    if pred_in_center == len(pred_centers) and len(pred_centers) > 0:
        print("❌ 所有预测都集中在图片中心区域")
        print("   问题：模型没有学习到真实的位置信息")
        print("   建议：检查训练数据、增加训练轮次、调整损失权重")
    else:
        print("✅ 预测位置分布相对合理")

def main():
    print("坐标转换检查工具")
    print("="*60)
    
    # 加载COCO数据
    coco_data = load_coco_data()
    
    # 测试几张图片
    test_image_dir = "data/coco2017_50/train2017"
    image_files = [f for f in os.listdir(test_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for i, image_file in enumerate(image_files[:3]):  # 检查前3张图片
        image_path = os.path.join(test_image_dir, image_file)
        analyze_single_image(image_path, coco_data)
        
        if i < 2:
            input("\n按Enter继续检查下一张图片...")

if __name__ == "__main__":
    main()
