#!/usr/bin/env python3
"""
检查COCO数据集的类别信息
"""

import json

def check_coco_categories():
    ann_file = "data/coco2017_50/annotations/instances_train2017.json"
    
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    print("COCO数据集类别信息:")
    print(f"总类别数: {len(coco_data['categories'])}")
    
    # 创建类别映射
    categories = {}
    for cat in coco_data['categories']:
        categories[cat['id']] = cat
    
    print("\n类别列表:")
    for cat_id in sorted(categories.keys()):
        cat = categories[cat_id]
        print(f"  ID {cat_id:2d}: {cat['name']}")
    
    # 检查数据中实际使用的类别
    used_categories = set()
    for ann in coco_data['annotations']:
        used_categories.add(ann['category_id'])
    
    print(f"\n数据中实际使用的类别数: {len(used_categories)}")
    print("实际使用的类别:")
    for cat_id in sorted(used_categories):
        cat_name = categories[cat_id]['name']
        print(f"  ID {cat_id:2d}: {cat_name}")

if __name__ == "__main__":
    check_coco_categories()
