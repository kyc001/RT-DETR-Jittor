#!/usr/bin/env python3
"""
分析10张图像训练中的问题
"""

import os
import sys
import json
import numpy as np
from PIL import Image
from collections import Counter

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

def analyze_10_images_data():
    """分析10张图像的数据分布"""
    print("🔍 分析10张图像的数据分布问题")
    print("=" * 60)
    
    # 数据路径
    data_dir = '/home/kyc/project/RT-DETR/data/coco2017_50'
    images_dir = os.path.join(data_dir, "train2017")
    annotations_file = os.path.join(data_dir, "annotations", "instances_train2017.json")
    
    # COCO类别映射
    COCO_CLASSES = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
        21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
        27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
        34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
        39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
        43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup',
        48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
        53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
        58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
        63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
        70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
        76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
        80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
        85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
        89: 'hair drier', 90: 'toothbrush'
    }
    
    # 加载标注
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # 构建图像ID到标注的映射
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # 获取前10张图像的信息
    selected_images = []
    for img_info in coco_data['images'][:20]:
        image_id = img_info['id']
        annotations = image_annotations.get(image_id, [])
        if annotations:
            selected_images.append((img_info, annotations))
            if len(selected_images) >= 10:
                break
    
    print(f"📊 10张图像详细分析:")
    
    all_categories = []
    all_positions = []
    
    for i, (img_info, annotations) in enumerate(selected_images):
        image_id = img_info['id']
        file_name = img_info['file_name']
        
        print(f"\n图像 {i+1}: {file_name} (ID: {image_id})")
        print(f"   尺寸: {img_info['width']}x{img_info['height']}")
        print(f"   目标数: {len(annotations)}")
        
        # 统计类别
        categories = []
        positions = []
        
        for ann in annotations:
            category_id = ann['category_id']
            class_name = COCO_CLASSES.get(category_id, f'class_{category_id}')
            categories.append(class_name)
            all_categories.append(class_name)
            
            # 分析位置分布
            x, y, w, h = ann['bbox']
            cx = (x + w/2) / img_info['width']  # 中心点x（归一化）
            cy = (y + h/2) / img_info['height']  # 中心点y（归一化）
            positions.append((cx, cy))
            all_positions.append((cx, cy))
        
        # 显示类别分布
        category_counts = Counter(categories)
        print(f"   类别分布:")
        for class_name, count in category_counts.most_common():
            print(f"     {class_name}: {count}个")
        
        # 显示位置分布（前5个）
        print(f"   位置分布（前5个）:")
        for j, (cx, cy) in enumerate(positions[:5]):
            print(f"     目标{j+1}: 中心点({cx:.3f}, {cy:.3f})")
        if len(positions) > 5:
            print(f"     ... 还有{len(positions)-5}个目标")
    
    # 总体统计
    print(f"\n📈 总体统计:")
    category_counts = Counter(all_categories)
    print(f"   总目标数: {len(all_categories)}")
    print(f"   类别分布:")
    for class_name, count in category_counts.most_common():
        percentage = count / len(all_categories) * 100
        print(f"     {class_name}: {count}个 ({percentage:.1f}%)")
    
    # 分析位置分布
    if all_positions:
        positions_array = np.array(all_positions)
        mean_x = np.mean(positions_array[:, 0])
        mean_y = np.mean(positions_array[:, 1])
        std_x = np.std(positions_array[:, 0])
        std_y = np.std(positions_array[:, 1])
        
        print(f"\n📍 位置分布分析:")
        print(f"   平均中心点: ({mean_x:.3f}, {mean_y:.3f})")
        print(f"   标准差: x={std_x:.3f}, y={std_y:.3f}")
        
        # 检查是否有位置偏向
        if mean_x > 0.6:
            print(f"   ⚠️ 目标偏向图像右侧")
        elif mean_x < 0.4:
            print(f"   ⚠️ 目标偏向图像左侧")
        
        if mean_y > 0.6:
            print(f"   ⚠️ 目标偏向图像下方")
        elif mean_y < 0.4:
            print(f"   ⚠️ 目标偏向图像上方")
    
    # 检查数据不平衡问题
    print(f"\n⚠️ 潜在问题分析:")
    
    # 1. 类别不平衡
    max_category = category_counts.most_common(1)[0]
    if max_category[1] / len(all_categories) > 0.5:
        print(f"   1. 严重类别不平衡: {max_category[0]}占{max_category[1]/len(all_categories)*100:.1f}%")
    
    # 2. 位置集中
    if std_x < 0.2 and std_y < 0.2:
        print(f"   2. 目标位置过于集中: 标准差x={std_x:.3f}, y={std_y:.3f}")
    
    # 3. 图像复杂度差异
    object_counts = [len(annotations) for _, annotations in selected_images]
    if max(object_counts) / min(object_counts) > 10:
        print(f"   3. 图像复杂度差异巨大: {min(object_counts)}-{max(object_counts)}个目标")
    
    return selected_images, category_counts, all_positions

def suggest_fixes():
    """建议修复方案"""
    print(f"\n🔧 修复建议:")
    print(f"=" * 60)
    
    print(f"1. **数据平衡**:")
    print(f"   - 选择类别更平衡的图像")
    print(f"   - 限制单一类别的最大比例")
    print(f"   - 确保每个类别至少有2-3个样本")
    
    print(f"\n2. **位置多样化**:")
    print(f"   - 选择目标位置分布更广的图像")
    print(f"   - 避免所有目标都在图像中心")
    
    print(f"\n3. **训练策略**:")
    print(f"   - 降低学习率，避免过快收敛到局部最优")
    print(f"   - 增加数据增强（随机裁剪、翻转）")
    print(f"   - 使用更强的正则化")
    
    print(f"\n4. **模型架构**:")
    print(f"   - 检查查询数量是否足够")
    print(f"   - 调整损失函数权重")

def main():
    print("🔍 RT-DETR 10张图像问题分析")
    print("分析为什么模型输出固定模式")
    print("=" * 60)
    
    selected_images, category_counts, all_positions = analyze_10_images_data()
    suggest_fixes()

if __name__ == "__main__":
    main()
