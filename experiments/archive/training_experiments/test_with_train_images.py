#!/usr/bin/env python3
"""
使用训练集图像测试模型性能，验证模型是否正常工作
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from datetime import datetime

# 设置matplotlib支持中文字体，避免中文字符警告
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt

# 设置Jittor
jt.flags.use_cuda = 1

# 导入测试脚本的函数
from test_trained_model import (
    load_trained_model, postprocess_outputs, convert_to_coco_class,
    nms_filter, visualize_detections
)

def load_train_dataset(data_dir, num_images=5):
    """加载训练数据集的前几张图像"""
    print(f"🔄 加载训练数据集前{num_images}张图像...")
    
    # 数据路径
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
    
    # 获取前几张图像
    selected_images = coco_data['images'][:num_images]
    image_info = {img['id']: img for img in selected_images}
    
    print(f"✅ 加载完成: {len(selected_images)}张图像")
    
    return image_info, image_annotations, images_dir

def preprocess_image(image_path):
    """预处理图像"""
    try:
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # 调整图像大小到640x640
        image_resized = image.resize((640, 640))
        image_array = np.array(image_resized).astype(np.float32) / 255.0
        image_tensor = jt.array(image_array.transpose(2, 0, 1)).unsqueeze(0)
        
        return image_tensor, image, original_size
        
    except Exception as e:
        print(f"❌ 图像预处理失败: {e}")
        return None, None, None

def test_train_images():
    """测试训练集图像"""
    print("🧪 使用训练集图像测试模型")
    print("=" * 60)
    
    # 配置
    config = {
        'data_dir': '/home/kyc/project/RT-DETR/data/coco2017_50',
        'model_path': '/home/kyc/project/RT-DETR/results/full_training/rtdetr_trained.pkl',
        'results_dir': '/home/kyc/project/RT-DETR/results/train_test',
        'num_test_images': 5,
        'score_threshold': 0.15
    }
    
    # 创建结果目录
    os.makedirs(config['results_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['results_dir'], 'visualizations'), exist_ok=True)
    
    # 加载模型
    model, checkpoint = load_trained_model(config['model_path'])
    if model is None:
        print("❌ 无法加载模型")
        return
    
    # 加载训练数据
    image_info, image_annotations, images_dir = load_train_dataset(
        config['data_dir'], config['num_test_images'])
    
    print("🎯 开始测试...")
    print("=" * 60)
    
    results = []
    
    for i, (image_id, img_info) in enumerate(image_info.items()):
        print(f"\n🔍 测试图像 {i+1}/{len(image_info)} (ID: {image_id})")
        print(f"   文件名: {img_info['file_name']}")
        
        try:
            # 加载图像
            image_path = os.path.join(images_dir, img_info['file_name'])
            image_tensor, original_image, original_size = preprocess_image(image_path)
            
            if image_tensor is None:
                continue
            
            # 模型推理
            start_time = datetime.now()
            with jt.no_grad():
                features = model.backbone(image_tensor)
                outputs = model.transformer(features)
            
            # 后处理
            detections = postprocess_outputs(outputs, original_size, config['score_threshold'])
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 获取真实标注
            gt_annotations = image_annotations.get(image_id, [])
            
            print(f"   ✅ 检测到 {len(detections)} 个目标")
            print(f"   ⏱️ 处理时间: {processing_time:.3f}s")
            print(f"   📊 真实标注: {len(gt_annotations)} 个目标")
            
            # 显示检测结果
            if detections:
                print(f"   🎯 检测结果:")
                for j, det in enumerate(detections[:5]):  # 显示前5个
                    print(f"      {j+1}: {det['class_name']} (置信度: {det['confidence']:.3f})")
                if len(detections) > 5:
                    print(f"      ... 还有 {len(detections)-5} 个检测")
            
            # 显示真实标注类别
            if gt_annotations:
                print(f"   📋 真实类别:")
                gt_classes = set()
                for ann in gt_annotations:
                    category_id = ann['category_id']
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
                    class_name = COCO_CLASSES.get(category_id, f'class_{category_id}')
                    gt_classes.add(class_name)
                
                print(f"      {', '.join(sorted(gt_classes))}")
            
            # 可视化
            vis_save_path = os.path.join(
                config['results_dir'], 'visualizations', 
                f"train_test_{i+1:02d}_{img_info['file_name']}"
            )
            
            fig = visualize_detections(
                original_image, detections, gt_annotations, vis_save_path
            )
            plt.close(fig)
            
            # 记录结果
            results.append({
                'image_id': image_id,
                'file_name': img_info['file_name'],
                'detections': len(detections),
                'gt_annotations': len(gt_annotations),
                'processing_time': processing_time,
                'detected_classes': [d['class_name'] for d in detections],
                'gt_classes': list(gt_classes) if gt_annotations else []
            })
            
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            continue
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 训练集测试总结")
    print("=" * 60)
    
    if results:
        total_detections = sum(r['detections'] for r in results)
        total_gt = sum(r['gt_annotations'] for r in results)
        avg_time = np.mean([r['processing_time'] for r in results])
        
        print(f"✅ 成功测试: {len(results)} 张图像")
        print(f"📈 总检测数: {total_detections}")
        print(f"📋 总真实标注: {total_gt}")
        print(f"⏱️ 平均处理时间: {avg_time:.3f}s")
        print(f"🚀 平均推理速度: {1/avg_time:.1f} FPS")
        
        # 统计检测类别
        all_detected_classes = []
        for r in results:
            all_detected_classes.extend(r['detected_classes'])
        
        if all_detected_classes:
            from collections import Counter
            class_counts = Counter(all_detected_classes)
            print(f"\n🏷️ 检测到的类别:")
            for class_name, count in class_counts.most_common():
                print(f"   {class_name}: {count} 次")
        
        print(f"\n📊 详细结果已保存到: {config['results_dir']}")
    
    print("\n🎉 训练集测试完成！")

if __name__ == "__main__":
    test_train_images()
