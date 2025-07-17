#!/usr/bin/env python3
"""
分析训练数据的类别分布
"""

import os
import json
from collections import Counter

def analyze_train_data():
    """分析训练数据的类别分布"""
    print("🔍 分析训练数据类别分布")
    print("=" * 60)
    
    # 数据路径
    data_dir = '/home/kyc/project/RT-DETR/data/coco2017_50'
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
    
    print(f"📊 训练数据集概览:")
    print(f"   图像数量: {len(coco_data['images'])}")
    print(f"   标注数量: {len(coco_data['annotations'])}")
    print(f"   类别数量: {len(coco_data['categories'])}")
    
    # 统计类别分布
    category_counts = Counter()
    image_categories = {}
    
    for ann in coco_data['annotations']:
        category_id = ann['category_id']
        image_id = ann['image_id']
        
        category_counts[category_id] += 1
        
        if image_id not in image_categories:
            image_categories[image_id] = set()
        image_categories[image_id].add(category_id)
    
    print(f"\n🏷️ 训练集类别分布:")
    sorted_categories = category_counts.most_common()
    for i, (category_id, count) in enumerate(sorted_categories):
        class_name = COCO_CLASSES.get(category_id, f'class_{category_id}')
        percentage = count / len(coco_data['annotations']) * 100
        print(f"   {i+1:2d}: {class_name:15} (ID {category_id:2d}): {count:3d} 个标注 ({percentage:.1f}%)")
    
    # 检查dining table的分布
    dining_table_id = 67
    dining_table_count = category_counts.get(dining_table_id, 0)
    
    print(f"\n🍽️ Dining Table 分析:")
    print(f"   标注数量: {dining_table_count}")
    print(f"   占总标注比例: {dining_table_count/len(coco_data['annotations'])*100:.1f}%")
    
    # 统计包含dining table的图像
    images_with_dining_table = 0
    for image_id, categories in image_categories.items():
        if dining_table_id in categories:
            images_with_dining_table += 1
    
    print(f"   包含dining table的图像: {images_with_dining_table}/{len(coco_data['images'])}")
    print(f"   图像包含比例: {images_with_dining_table/len(coco_data['images'])*100:.1f}%")
    
    # 分析前10张图像的标注
    print(f"\n📋 前10张训练图像的真实标注:")
    
    # 构建图像ID到标注的映射
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # 获取前10张图像
    first_10_images = coco_data['images'][:10]
    
    for i, img_info in enumerate(first_10_images):
        image_id = img_info['id']
        file_name = img_info['file_name']
        
        annotations = image_annotations.get(image_id, [])
        
        print(f"   {i+1:2d}: {file_name} (ID {image_id})")
        
        if annotations:
            # 统计这张图像的类别
            img_categories = Counter()
            for ann in annotations:
                img_categories[ann['category_id']] += 1
            
            # 显示类别
            categories_str = []
            for category_id, count in img_categories.most_common():
                class_name = COCO_CLASSES.get(category_id, f'class_{category_id}')
                if count > 1:
                    categories_str.append(f"{class_name}({count})")
                else:
                    categories_str.append(class_name)
            
            print(f"       真实类别: {', '.join(categories_str)}")
            
            # 检查是否包含dining table
            if dining_table_id in img_categories:
                print(f"       ✅ 包含 dining table")
            else:
                print(f"       ❌ 不包含 dining table")
        else:
            print(f"       无标注")
    
    # 总结
    print(f"\n📊 总结:")
    print(f"   训练集中dining table占比: {dining_table_count/len(coco_data['annotations'])*100:.1f}%")
    print(f"   这解释了为什么模型倾向于预测dining table")
    print(f"   建议:")
    print(f"   1. 使用更平衡的训练数据")
    print(f"   2. 增加其他类别的权重")
    print(f"   3. 使用更多的训练轮数")

if __name__ == "__main__":
    analyze_train_data()
