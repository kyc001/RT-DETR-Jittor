#!/usr/bin/env python3
"""
调试坐标格式问题 - 对比单张图像和5张图像的差异
"""

import os
import sys
import json
import numpy as np
from PIL import Image

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt

# 设置Jittor
jt.flags.use_cuda = 1

def load_single_image_ultimate_way():
    """按照ultimate_sanity_check.py的方式加载单张图像"""
    print("🔍 按照ultimate_sanity_check.py的方式加载单张图像")
    
    # 数据路径
    image_path = "/home/kyc/project/RT-DETR/data/coco2017_50/train2017/000000055150.jpg"
    annotation_path = "/home/kyc/project/RT-DETR/data/coco2017_50/annotations/instances_train2017.json"
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    original_width, original_height = image.size
    print(f"原始图像尺寸: {original_width}x{original_height}")

    image_resized = image.resize((640, 640), Image.LANCZOS)

    # 转换为张量
    img_array = np.array(image_resized).astype(np.float32) / 255.0
    img_tensor = jt.array(img_array.transpose(2, 0, 1)).float32().unsqueeze(0)

    # 加载标注 - 修复版本
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    image_id = 55150
    annotations = []
    labels = []

    print(f"查找图像ID {image_id} 的标注...")

    for ann in coco_data['annotations']:
        if ann['image_id'] == image_id:
            x, y, w, h = ann['bbox']
            category_id = ann['category_id']

            print(f"找到标注: 类别{category_id}, 边界框[{x},{y},{w},{h}]")

            # 归一化坐标 - 使用正确的原始尺寸
            x1, y1 = x / original_width, y / original_height
            x2, y2 = (x + w) / original_width, (y + h) / original_height

            # 确保坐标有效
            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= 1 and y2 <= 1:
                annotations.append([x1, y1, x2, y2])
                labels.append(category_id)
                print(f"   归一化后: [{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}]")

    # 创建目标 - 修复COCO类别映射问题
    if annotations:
        # COCO类别ID需要转换为0-based索引
        mapped_labels = []
        for label in labels:
            if label == 1:  # person
                mapped_labels.append(0)  # 映射到索引0
            elif label == 3:  # car
                mapped_labels.append(2)  # 映射到索引2
            elif label == 27:  # backpack
                mapped_labels.append(26)  # 映射到索引26
            elif label == 33:  # suitcase
                mapped_labels.append(32)  # 映射到索引32
            elif label == 84:  # book
                mapped_labels.append(83)  # 映射到索引83
            else:
                mapped_labels.append(label - 1)  # 其他类别减1

        target = {
            'boxes': jt.array(annotations, dtype=jt.float32),
            'labels': jt.array(mapped_labels, dtype=jt.int64)  # 使用映射后的类别
        }
        
        print(f"✅ 单张图像加载成功")
        print(f"   目标数量: {len(annotations)}")
        print(f"   边界框格式: [x1, y1, x2, y2] (左上右下)")
        print(f"   边界框: {annotations}")
        print(f"   类别: {mapped_labels}")
        
        return img_tensor, [target], annotations, mapped_labels
    
    return None, None, None, None

def load_5_images_old_way():
    """按照之前错误的方式加载5张图像"""
    print("\n🔍 按照之前错误的方式加载5张图像")
    
    # 数据路径
    data_dir = '/home/kyc/project/RT-DETR/data/coco2017_50'
    images_dir = os.path.join(data_dir, "train2017")
    annotations_file = os.path.join(data_dir, "annotations", "instances_train2017.json")
    
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
    
    # 选择第一张图像（与单张图像测试相同）
    target_image_id = 55150
    
    for img_info in coco_data['images']:
        if img_info['id'] == target_image_id:
            image_path = os.path.join(images_dir, img_info['file_name'])
            
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            
            # 调整图像大小到640x640
            image_resized = image.resize((640, 640))
            image_array = np.array(image_resized).astype(np.float32) / 255.0
            image_tensor = jt.array(image_array.transpose(2, 0, 1)).unsqueeze(0)
            
            # 获取标注
            annotations = image_annotations.get(target_image_id, [])
            
            # 处理标注 - 错误的中心点格式
            boxes = []
            labels = []
            
            for ann in annotations:
                # COCO格式: [x, y, width, height] -> 归一化的中心点格式（错误！）
                x, y, w, h = ann['bbox']
                
                # 转换为归一化坐标
                cx = (x + w/2) / original_size[0]
                cy = (y + h/2) / original_size[1]
                nw = w / original_size[0]
                nh = h / original_size[1]
                
                boxes.append([cx, cy, nw, nh])
                
                # 类别映射
                category_id = ann['category_id']
                if category_id == 1:
                    mapped_label = 0  # person
                elif category_id == 3:
                    mapped_label = 2  # car
                elif category_id == 27:
                    mapped_label = 26  # backpack
                elif category_id == 33:
                    mapped_label = 32  # suitcase
                elif category_id == 84:
                    mapped_label = 83  # book
                else:
                    mapped_label = category_id - 1  # 其他类别减1
                
                labels.append(mapped_label)
            
            if boxes:
                target = {
                    'boxes': jt.array(boxes),
                    'labels': jt.array(labels)
                }
                
                print(f"✅ 5张图像方式加载成功（错误格式）")
                print(f"   目标数量: {len(boxes)}")
                print(f"   边界框格式: [cx, cy, w, h] (中心点+宽高) - 错误！")
                print(f"   边界框: {boxes}")
                print(f"   类别: {labels}")
                
                return image_tensor, [target], boxes, labels
            
            break
    
    return None, None, None, None

def load_5_images_fixed_way():
    """按照修复后的正确方式加载5张图像"""
    print("\n🔍 按照修复后的正确方式加载5张图像")
    
    # 数据路径
    data_dir = '/home/kyc/project/RT-DETR/data/coco2017_50'
    images_dir = os.path.join(data_dir, "train2017")
    annotations_file = os.path.join(data_dir, "annotations", "instances_train2017.json")
    
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
    
    # 选择第一张图像（与单张图像测试相同）
    target_image_id = 55150
    
    for img_info in coco_data['images']:
        if img_info['id'] == target_image_id:
            image_path = os.path.join(images_dir, img_info['file_name'])
            
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            
            # 调整图像大小到640x640
            image_resized = image.resize((640, 640))
            image_array = np.array(image_resized).astype(np.float32) / 255.0
            image_tensor = jt.array(image_array.transpose(2, 0, 1)).unsqueeze(0)
            
            # 获取标注
            annotations = image_annotations.get(target_image_id, [])
            
            # 处理标注 - 正确的左上右下格式（完全按照ultimate_sanity_check.py）
            boxes = []
            labels = []
            
            for ann in annotations:
                # COCO格式: [x, y, width, height] -> 归一化的左上右下格式
                x, y, w, h = ann['bbox']
                
                # 转换为归一化坐标 - 使用正确的原始尺寸
                x1, y1 = x / original_size[0], y / original_size[1]
                x2, y2 = (x + w) / original_size[0], (y + h) / original_size[1]
                
                # 确保坐标有效
                if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= 1 and y2 <= 1:
                    boxes.append([x1, y1, x2, y2])
                    
                    # 类别映射
                    category_id = ann['category_id']
                    if category_id == 1:
                        mapped_label = 0  # person
                    elif category_id == 3:
                        mapped_label = 2  # car
                    elif category_id == 27:
                        mapped_label = 26  # backpack
                    elif category_id == 33:
                        mapped_label = 32  # suitcase
                    elif category_id == 84:
                        mapped_label = 83  # book
                    else:
                        mapped_label = category_id - 1  # 其他类别减1
                    
                    labels.append(mapped_label)
            
            if boxes:
                target = {
                    'boxes': jt.array(boxes),
                    'labels': jt.array(labels)
                }
                
                print(f"✅ 5张图像方式加载成功（修复格式）")
                print(f"   目标数量: {len(boxes)}")
                print(f"   边界框格式: [x1, y1, x2, y2] (左上右下) - 正确！")
                print(f"   边界框: {boxes}")
                print(f"   类别: {labels}")
                
                return image_tensor, [target], boxes, labels
            
            break
    
    return None, None, None, None

def compare_formats():
    """对比三种格式的差异"""
    print("🔍 坐标格式对比分析")
    print("=" * 80)
    
    # 1. 单张图像方式（正确）
    single_tensor, single_targets, single_boxes, single_labels = load_single_image_ultimate_way()
    
    # 2. 5张图像方式（错误）
    multi_old_tensor, multi_old_targets, multi_old_boxes, multi_old_labels = load_5_images_old_way()
    
    # 3. 5张图像方式（修复）
    multi_fixed_tensor, multi_fixed_targets, multi_fixed_boxes, multi_fixed_labels = load_5_images_fixed_way()
    
    print("\n📊 对比结果:")
    print("=" * 80)
    
    if single_boxes and multi_old_boxes and multi_fixed_boxes:
        print(f"单张图像方式 (正确): {single_boxes[0]}")
        print(f"5张图像方式 (错误): {multi_old_boxes[0]}")
        print(f"5张图像方式 (修复): {multi_fixed_boxes[0]}")
        
        # 检查是否一致
        single_box = np.array(single_boxes[0])
        fixed_box = np.array(multi_fixed_boxes[0])
        
        if np.allclose(single_box, fixed_box, atol=1e-6):
            print("\n✅ 修复成功！单张图像和修复后的5张图像格式完全一致")
        else:
            print("\n❌ 修复失败！格式仍然不一致")
            print(f"   差异: {single_box - fixed_box}")
    
    print("\n🎯 结论:")
    print("   问题根源: 边界框格式不一致")
    print("   单张图像: [x1, y1, x2, y2] (左上右下)")
    print("   5张图像(错误): [cx, cy, w, h] (中心点+宽高)")
    print("   5张图像(修复): [x1, y1, x2, y2] (左上右下)")
    print("   这就是为什么检测框位置完全错误的原因！")

def main():
    print("🔍 RT-DETR坐标格式调试")
    print("分析为什么5张图像训练的检测框位置完全错误")
    print("=" * 80)
    
    compare_formats()

if __name__ == "__main__":
    main()
