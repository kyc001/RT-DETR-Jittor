#!/usr/bin/env python3
"""
准备小规模数据集脚本
支持从现有COCO数据创建小规模训练集，或下载PASCAL VOC
"""

import os
import json
import random
import shutil
from pathlib import Path
import requests
from tqdm import tqdm

def create_coco_subset(source_dir, source_ann, target_dir, target_ann, max_images=1000, min_objects_per_image=1):
    """从现有COCO数据创建子集"""
    print(f">>> 创建COCO子集 (最多{max_images}张图片)")
    
    # 加载原始标注
    with open(source_ann, 'r') as f:
        coco_data = json.load(f)
    
    # 统计每张图片的目标数量
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # 筛选有足够目标的图片
    valid_images = []
    for img in coco_data['images']:
        img_id = img['id']
        if img_id in annotations_by_image:
            num_objects = len(annotations_by_image[img_id])
            if num_objects >= min_objects_per_image:
                valid_images.append(img)
    
    print(f"找到 {len(valid_images)} 张有效图片 (每张至少{min_objects_per_image}个目标)")
    
    # 随机选择图片
    if len(valid_images) > max_images:
        selected_images = random.sample(valid_images, max_images)
    else:
        selected_images = valid_images
    
    selected_image_ids = set(img['id'] for img in selected_images)
    
    print(f"选择了 {len(selected_images)} 张图片")
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 复制图片
    print("复制图片...")
    copied_images = []
    for img in tqdm(selected_images):
        source_path = os.path.join(source_dir, img['file_name'])
        target_path = os.path.join(target_dir, img['file_name'])
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            copied_images.append(img)
        else:
            print(f"警告: 图片不存在 {source_path}")
    
    copied_image_ids = set(img['id'] for img in copied_images)
    
    # 筛选标注
    selected_annotations = []
    for ann in coco_data['annotations']:
        if ann['image_id'] in copied_image_ids:
            selected_annotations.append(ann)
    
    # 创建新的标注文件
    new_coco_data = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': coco_data['categories'],
        'images': copied_images,
        'annotations': selected_annotations
    }
    
    # 保存标注文件
    with open(target_ann, 'w') as f:
        json.dump(new_coco_data, f, indent=2)
    
    # 统计信息
    class_counts = {}
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    for ann in selected_annotations:
        cat_name = categories[ann['category_id']]
        class_counts[cat_name] = class_counts.get(cat_name, 0) + 1
    
    print(f"\n✅ COCO子集创建完成:")
    print(f"  - 图片数量: {len(copied_images)}")
    print(f"  - 标注数量: {len(selected_annotations)}")
    print(f"  - 类别分布:")
    for cat_name, count in sorted(class_counts.items()):
        print(f"    - {cat_name}: {count} 个")
    
    return len(copied_images), len(selected_annotations)

def download_pascal_voc():
    """下载PASCAL VOC数据集"""
    print(">>> 下载PASCAL VOC数据集")
    
    # PASCAL VOC 2007 下载链接
    voc_urls = {
        'trainval': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'test': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar'
    }
    
    voc_dir = "data/pascal_voc"
    os.makedirs(voc_dir, exist_ok=True)
    
    print("注意: PASCAL VOC数据集较大，建议手动下载")
    print("下载链接:")
    for name, url in voc_urls.items():
        print(f"  - {name}: {url}")
    
    print(f"下载后请解压到: {voc_dir}")
    print("然后运行 convert_voc_to_coco() 函数转换格式")

def convert_voc_to_coco(voc_dir, output_dir):
    """将PASCAL VOC格式转换为COCO格式"""
    print(">>> 转换PASCAL VOC到COCO格式")
    
    # 这里需要实现VOC到COCO的转换
    # 由于代码较长，建议使用现有的转换工具
    print("建议使用以下工具进行转换:")
    print("1. https://github.com/yukkyo/voc2coco")
    print("2. https://github.com/Tony607/voc2coco")
    print("3. 或使用 roboflow 在线转换")

def create_balanced_subset(source_dir, source_ann, target_dir, target_ann, 
                          images_per_class=50, max_total_images=1000):
    """创建类别平衡的子集"""
    print(f">>> 创建类别平衡子集 (每类{images_per_class}张图片)")
    
    # 加载原始标注
    with open(source_ann, 'r') as f:
        coco_data = json.load(f)
    
    # 按类别分组图片
    images_by_category = {}
    annotations_by_image = {}
    
    # 建立标注索引
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
        
        if cat_id not in images_by_category:
            images_by_category[cat_id] = set()
        images_by_category[cat_id].add(img_id)
    
    # 为每个类别选择图片
    selected_image_ids = set()
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    for cat_id, img_ids in images_by_category.items():
        cat_name = categories[cat_id]
        img_list = list(img_ids)
        
        # 随机选择图片
        num_select = min(images_per_class, len(img_list))
        selected = random.sample(img_list, num_select)
        selected_image_ids.update(selected)
        
        print(f"  - {cat_name}: 选择了 {num_select}/{len(img_list)} 张图片")
    
    # 限制总图片数量
    if len(selected_image_ids) > max_total_images:
        selected_image_ids = set(random.sample(list(selected_image_ids), max_total_images))
        print(f"  - 限制总数为 {max_total_images} 张图片")
    
    # 获取选中的图片信息
    selected_images = []
    images_dict = {img['id']: img for img in coco_data['images']}
    
    for img_id in selected_image_ids:
        if img_id in images_dict:
            selected_images.append(images_dict[img_id])
    
    # 复制图片和创建标注
    os.makedirs(target_dir, exist_ok=True)
    
    print("复制图片...")
    copied_images = []
    for img in tqdm(selected_images):
        source_path = os.path.join(source_dir, img['file_name'])
        target_path = os.path.join(target_dir, img['file_name'])
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            copied_images.append(img)
    
    copied_image_ids = set(img['id'] for img in copied_images)
    
    # 筛选标注
    selected_annotations = []
    for ann in coco_data['annotations']:
        if ann['image_id'] in copied_image_ids:
            selected_annotations.append(ann)
    
    # 创建新的标注文件
    new_coco_data = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': coco_data['categories'],
        'images': copied_images,
        'annotations': selected_annotations
    }
    
    with open(target_ann, 'w') as f:
        json.dump(new_coco_data, f, indent=2)
    
    # 统计最终结果
    final_class_counts = {}
    for ann in selected_annotations:
        cat_name = categories[ann['category_id']]
        final_class_counts[cat_name] = final_class_counts.get(cat_name, 0) + 1
    
    print(f"\n✅ 类别平衡子集创建完成:")
    print(f"  - 图片数量: {len(copied_images)}")
    print(f"  - 标注数量: {len(selected_annotations)}")
    print(f"  - 最终类别分布:")
    for cat_name, count in sorted(final_class_counts.items()):
        print(f"    - {cat_name}: {count} 个")
    
    return len(copied_images), len(selected_annotations)

def main():
    print("=" * 60)
    print("===      小规模数据集准备工具      ===")
    print("=" * 60)
    
    # 配置
    source_dir = "data/coco2017_50/train2017"
    source_ann = "data/coco2017_50/annotations/instances_train2017.json"
    
    print("选择数据集准备方式:")
    print("1. 从现有COCO数据创建随机子集")
    print("2. 从现有COCO数据创建类别平衡子集")
    print("3. 下载PASCAL VOC数据集信息")
    
    choice = input("请选择 (1/2/3): ").strip()
    
    if choice == "1":
        # 创建随机子集
        target_dir = "data/coco_small/train2017"
        target_ann = "data/coco_small/annotations/instances_train2017.json"
        
        os.makedirs("data/coco_small/annotations", exist_ok=True)
        
        create_coco_subset(
            source_dir=source_dir,
            source_ann=source_ann,
            target_dir=target_dir,
            target_ann=target_ann,
            max_images=1000,
            min_objects_per_image=2
        )
        
        print(f"\n🎯 使用方法:")
        print(f"python small_scale_training.py")
        print(f"# 修改脚本中的数据路径为:")
        print(f"# data_dir = '{target_dir}'")
        print(f"# ann_file = '{target_ann}'")
    
    elif choice == "2":
        # 创建类别平衡子集
        target_dir = "data/coco_balanced/train2017"
        target_ann = "data/coco_balanced/annotations/instances_train2017.json"
        
        os.makedirs("data/coco_balanced/annotations", exist_ok=True)
        
        create_balanced_subset(
            source_dir=source_dir,
            source_ann=source_ann,
            target_dir=target_dir,
            target_ann=target_ann,
            images_per_class=30,
            max_total_images=800
        )
        
        print(f"\n🎯 使用方法:")
        print(f"python small_scale_training.py")
        print(f"# 修改脚本中的数据路径为:")
        print(f"# data_dir = '{target_dir}'")
        print(f"# ann_file = '{target_ann}'")
    
    elif choice == "3":
        # PASCAL VOC信息
        download_pascal_voc()
        
        print(f"\n🎯 PASCAL VOC使用建议:")
        print(f"1. 手动下载PASCAL VOC 2007数据集")
        print(f"2. 使用在线工具转换为COCO格式:")
        print(f"   - Roboflow: https://roboflow.com/")
        print(f"   - 或使用转换脚本")
        print(f"3. 然后使用 small_scale_training.py 训练")
    
    else:
        print("无效选择")

if __name__ == "__main__":
    main()
