#!/usr/bin/env python3
"""
数据加载组件
提供COCO数据集加载功能，参考ultimate_sanity_check.py的验证实现
"""

import os
import sys
import json
import numpy as np
from PIL import Image

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt

# COCO类别名称映射
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

class COCODataset:
    """
    COCO数据集加载器
    参考ultimate_sanity_check.py的验证实现
    """
    def __init__(self, img_dir, ann_file, augment_factor=1):
        self.img_dir = img_dir
        self.augment_factor = augment_factor
        
        # 加载COCO标注
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # 创建映射
        self.img_id_to_filename = {img['id']: img['file_name'] for img in self.coco_data['images']}
        self.img_ids = list(self.img_id_to_filename.keys())
        
        # 数据增强：重复数据
        if augment_factor > 1:
            self.img_ids = self.img_ids * augment_factor
        
        print(f"📊 数据集加载完成:")
        print(f"   原始图像数: {len(self.img_id_to_filename)}")
        print(f"   增强后数据: {len(self.img_ids)}")
        print(f"   标注数量: {len(self.coco_data['annotations'])}")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        filename = self.img_id_to_filename[img_id]
        
        # 加载图像
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert('RGB')
        original_width, original_height = image.size
        
        # 数据增强（简单的随机翻转）
        if self.augment_factor > 1 and np.random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 调整图像大小到640x640
        image_resized = image.resize((640, 640), Image.LANCZOS)
        img_array = np.array(image_resized).astype(np.float32) / 255.0
        img_tensor = jt.array(img_array.transpose(2, 0, 1)).float32()
        
        # 获取标注 - 使用验证过的类别映射
        annotations = []
        labels = []
        
        for ann in self.coco_data['annotations']:
            if ann['image_id'] == img_id:
                x, y, w, h = ann['bbox']
                category_id = ann['category_id']
                
                # 归一化坐标
                x1, y1 = x / original_width, y / original_height
                x2, y2 = (x + w) / original_width, (y + h) / original_height
                
                if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= 1 and y2 <= 1:
                    annotations.append([x1, y1, x2, y2])
                    
                    # 使用验证过的类别映射
                    mapped_label = self._map_category_id(category_id)
                    labels.append(mapped_label)
        
        # 创建目标
        if annotations:
            target = {
                'boxes': jt.array(annotations, dtype=jt.float32),
                'labels': jt.array(labels, dtype=jt.int64)
            }
        else:
            target = {
                'boxes': jt.zeros((0, 4), dtype=jt.float32),
                'labels': jt.zeros((0,), dtype=jt.int64)
            }
        
        return img_tensor, target
    
    def _map_category_id(self, category_id):
        """
        COCO类别ID映射
        参考ultimate_sanity_check.py的实现
        """
        if category_id == 1:  # person
            return 0
        elif category_id == 3:  # car
            return 2
        elif category_id == 27:  # backpack
            return 26
        elif category_id == 33:  # suitcase
            return 32
        elif category_id == 84:  # book
            return 83
        else:
            return category_id - 1
    
    def get_dataset_stats(self):
        """获取数据集统计信息"""
        # 统计类别分布
        category_counts = {}
        for ann in self.coco_data['annotations']:
            cat_id = ann['category_id']
            category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
        
        # 统计边界框大小
        bbox_areas = []
        for ann in self.coco_data['annotations']:
            x, y, w, h = ann['bbox']
            area = w * h
            bbox_areas.append(area)
        
        bbox_areas = np.array(bbox_areas)
        
        return {
            'total_images': len(self.img_id_to_filename),
            'total_annotations': len(self.coco_data['annotations']),
            'category_counts': category_counts,
            'bbox_stats': {
                'mean_area': bbox_areas.mean(),
                'min_area': bbox_areas.min(),
                'max_area': bbox_areas.max(),
                'std_area': bbox_areas.std()
            }
        }
    
    def print_dataset_info(self):
        """打印数据集信息"""
        stats = self.get_dataset_stats()
        
        print("📊 数据集统计信息:")
        print(f"   图像数量: {stats['total_images']}")
        print(f"   标注数量: {stats['total_annotations']}")
        print(f"   平均每张图像标注数: {stats['total_annotations']/stats['total_images']:.1f}")
        
        print(f"\n📦 边界框统计:")
        bbox_stats = stats['bbox_stats']
        print(f"   平均面积: {bbox_stats['mean_area']:.2f}")
        print(f"   最小面积: {bbox_stats['min_area']:.2f}")
        print(f"   最大面积: {bbox_stats['max_area']:.2f}")
        
        print(f"\n🏷️ 类别分布 (前10个):")
        sorted_cats = sorted(stats['category_counts'].items(), key=lambda x: x[1], reverse=True)[:10]
        for cat_id, count in sorted_cats:
            cat_name = COCO_CLASSES.get(cat_id, f"unknown_{cat_id}")
            print(f"   {cat_name} (ID:{cat_id}): {count}个标注")

def create_coco_dataset(data_root, split='train', augment_factor=1):
    """
    创建COCO数据集的工厂函数
    
    Args:
        data_root: 数据根目录
        split: 数据集分割 ('train' 或 'val')
        augment_factor: 数据增强倍数
    
    Returns:
        dataset: COCO数据集
    """
    if split == 'train':
        img_dir = os.path.join(data_root, 'train2017')
        ann_file = os.path.join(data_root, 'annotations', 'instances_train2017.json')
    elif split == 'val':
        img_dir = os.path.join(data_root, 'val2017')
        ann_file = os.path.join(data_root, 'annotations', 'instances_val2017.json')
    else:
        raise ValueError(f"不支持的数据集分割: {split}")
    
    # 检查文件是否存在
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"图像目录不存在: {img_dir}")
    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"标注文件不存在: {ann_file}")
    
    # 创建数据集
    dataset = COCODataset(img_dir, ann_file, augment_factor=augment_factor)
    dataset.print_dataset_info()
    
    return dataset

if __name__ == "__main__":
    # 测试数据集加载
    print("🧪 测试COCO数据集组件")
    print("=" * 50)
    
    data_root = "/home/kyc/project/RT-DETR/data/coco2017_50"
    
    try:
        # 创建训练数据集
        train_dataset = create_coco_dataset(data_root, split='train', augment_factor=2)
        
        # 测试数据加载
        img_tensor, target = train_dataset[0]
        print(f"\n✅ 数据加载测试成功!")
        print(f"   图像形状: {img_tensor.shape}")
        print(f"   边界框数量: {len(target['boxes'])}")
        print(f"   标签数量: {len(target['labels'])}")
        
        print(f"\n🎉 COCO数据集组件验证完成!")
        
    except Exception as e:
        print(f"❌ 数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
