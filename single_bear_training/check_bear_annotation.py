#!/usr/bin/env python3
"""
检查熊图片的真实标注位置
用于验证训练数据的正确性
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def check_bear_annotation():
    """检查000000000285.jpg中熊的真实标注"""
    
    img_path = 'data/coco/val2017/000000000285.jpg'
    ann_file = 'data/coco/annotations/instances_val2017.json'
    
    print("=== 检查熊的真实标注位置 ===")
    
    # 加载图片
    image = Image.open(img_path).convert('RGB')
    orig_w, orig_h = image.size
    print(f"图片尺寸: {orig_w} x {orig_h}")
    
    # 加载COCO标注
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # 找到图片信息
    target_image = None
    for img in coco_data['images']:
        if img['file_name'] == '000000000285.jpg':
            target_image = img
            break
    
    if target_image is None:
        print("❌ 找不到图片")
        return
    
    print(f"图片ID: {target_image['id']}")
    
    # 找到熊的标注
    bear_annotations = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == target_image['id']:
            # 找到类别名称
            for cat in coco_data['categories']:
                if cat['id'] == ann['category_id'] and cat['name'] == 'bear':
                    bear_annotations.append(ann)
                    break
    
    print(f"找到 {len(bear_annotations)} 个熊标注")
    
    if len(bear_annotations) == 0:
        print("❌ 没有找到熊标注")
        return
    
    # 分析每个熊标注
    for i, ann in enumerate(bear_annotations):
        print(f"\n🐻 熊标注 {i+1}:")
        
        # COCO格式: [x, y, width, height] (左上角坐标 + 宽高)
        x, y, w, h = ann['bbox']
        print(f"  COCO bbox: [x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}]")
        
        # 转换为左上角右下角格式
        x1, y1, x2, y2 = x, y, x + w, y + h
        print(f"  左上右下: [x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}]")
        
        # 计算中心点
        cx = x + w / 2
        cy = y + h / 2
        print(f"  中心点: [cx={cx:.1f}, cy={cy:.1f}]")
        
        # 计算相对于图片的位置
        cx_rel = cx / orig_w
        cy_rel = cy / orig_h
        w_rel = w / orig_w
        h_rel = h / orig_h
        print(f"  归一化坐标: [cx={cx_rel:.3f}, cy={cy_rel:.3f}, w={w_rel:.3f}, h={h_rel:.3f}]")
        
        # 判断位置
        if cx_rel < 0.3:
            pos_x = "左侧"
        elif cx_rel > 0.7:
            pos_x = "右侧"
        else:
            pos_x = "中央"
            
        if cy_rel < 0.3:
            pos_y = "上方"
        elif cy_rel > 0.7:
            pos_y = "下方"
        else:
            pos_y = "中央"
            
        print(f"  位置分析: {pos_x}{pos_y}")
    
    # 可视化真实标注
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for i, ann in enumerate(bear_annotations):
        x, y, w, h = ann['bbox']
        x1, y1, x2, y2 = x, y, x + w, y + h
        
        # 绘制真实标注框（绿色）
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.text((x1, max(0, y1-25)), f"🐻 真实熊 {i+1}", fill="green", font=font)
        
        # 绘制中心点
        cx, cy = x + w/2, y + h/2
        draw.ellipse([cx-5, cy-5, cx+5, cy+5], fill="green")
    
    # 保存结果
    output_path = 'single_bear_training/bear_true_annotation.jpg'
    image.save(output_path)
    print(f"\n✅ 真实标注可视化已保存: {output_path}")

if __name__ == "__main__":
    check_bear_annotation()
