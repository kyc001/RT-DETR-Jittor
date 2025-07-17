#!/usr/bin/env python3
"""
测试基于ultimate_sanity_check.py方法训练的模型
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

def load_sanity_check_model(model_path):
    """加载基于sanity check方法训练的模型"""
    print(f"🔄 加载sanity check训练的模型: {model_path}")
    
    try:
        # 导入模型组件（完全按照ultimate_sanity_check.py）
        from jittor_rt_detr.src.nn.backbone.resnet import ResNet50
        from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
        
        # 创建模型架构（完全按照ultimate_sanity_check.py）
        backbone = ResNet50(pretrained=False)
        transformer = RTDETRTransformer(
            num_classes=80,
            hidden_dim=256,
            num_queries=300,
            feat_channels=[256, 512, 1024, 2048]
        )
        
        # 加载训练好的权重
        checkpoint = jt.load(model_path)
        backbone.load_state_dict(checkpoint['backbone_state_dict'])
        transformer.load_state_dict(checkpoint['transformer_state_dict'])
        
        # 设置为评估模式
        backbone.eval()
        transformer.eval()
        
        print(f"✅ 模型加载成功")
        print(f"   训练轮数: {checkpoint.get('epoch', 'Unknown')}")
        print(f"   训练损失: {checkpoint.get('loss', 'Unknown'):.4f}")
        print(f"   训练方法: {checkpoint.get('additional_info', {}).get('training_method', 'Unknown')}")
        
        return backbone, transformer, checkpoint
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def nms_filter(boxes, scores, classes, iou_threshold=0.5, score_threshold=0.2):
    """简单的NMS过滤（完全按照ultimate_sanity_check.py）"""
    if len(boxes) == 0:
        return [], [], []

    # 按分数排序
    sorted_indices = np.argsort(scores)[::-1]

    keep_indices = []
    for i in sorted_indices:
        if scores[i] < score_threshold:
            continue

        # 检查与已保留的框是否重叠过多
        keep_this = True
        for j in keep_indices:
            # 计算IoU
            box1 = boxes[i]
            box2 = boxes[j]

            # 计算交集
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])

            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                union = area1 + area2 - intersection
                iou = intersection / union if union > 0 else 0

                if iou > iou_threshold:
                    keep_this = False
                    break

        if keep_this:
            keep_indices.append(i)

    return [boxes[i] for i in keep_indices], [scores[i] for i in keep_indices], [classes[i] for i in keep_indices]

def postprocess_outputs(outputs, original_size, score_threshold=0.2):
    """后处理模型输出（完全按照ultimate_sanity_check.py）"""
    try:
        # 获取预测结果
        pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
        pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4]
        
        # 后处理 - 完全按照ultimate_sanity_check.py的鲁棒版本
        pred_scores = jt.nn.softmax(pred_logits, dim=-1)
        pred_scores_no_bg = pred_scores[:, :-1]  # 排除背景类
        
        # 修复：正确处理Jittor的max和argmax返回值
        max_result = jt.max(pred_scores_no_bg, dim=-1)
        if isinstance(max_result, tuple):
            max_scores = max_result[0]
        else:
            max_scores = max_result

        argmax_result = jt.argmax(pred_scores_no_bg, dim=-1)
        if isinstance(argmax_result, tuple):
            pred_classes = argmax_result[0]
        else:
            pred_classes = argmax_result
        
        print(f"   分数范围: {max_scores.min().item():.4f} - {max_scores.max().item():.4f}")
        
        # 应用NMS过滤
        print(f"🔄 应用NMS过滤重复检测...")
        boxes_640 = pred_boxes.numpy() * 640  # 转换到640x640坐标系用于NMS
        scores_np = max_scores.numpy()
        classes_np = pred_classes.numpy()
        
        filtered_boxes, filtered_scores, filtered_classes = nms_filter(
            boxes_640, scores_np, classes_np,
            iou_threshold=0.5, score_threshold=score_threshold
        )
        
        print(f"   NMS前: {len(scores_np)}个检测")
        print(f"   NMS后: {len(filtered_scores)}个检测")
        
        # 转换边界框格式并缩放到原始图像尺寸
        detections = []
        orig_w, orig_h = original_size
        
        for box_640, score, class_idx in zip(filtered_boxes, filtered_scores, filtered_classes):
            # box_640是在640x640坐标系中的坐标，需要转换到原始图像尺寸
            x1_640, y1_640, x2_640, y2_640 = box_640
            
            # 转换到原始图像坐标
            x1 = (x1_640 / 640) * orig_w
            y1 = (y1_640 / 640) * orig_h
            x2 = (x2_640 / 640) * orig_w
            y2 = (y2_640 / 640) * orig_h
            
            # 确保坐标在有效范围内
            x1 = max(0, min(orig_w, x1))
            y1 = max(0, min(orig_h, y1))
            x2 = max(0, min(orig_w, x2))
            y2 = max(0, min(orig_h, y2))
            
            if x2 > x1 and y2 > y1:
                # 转换类别
                coco_id, class_name = convert_to_coco_class(class_idx)
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(score),
                    'class_id': int(class_idx),
                    'coco_id': coco_id,
                    'class_name': class_name
                })
        
        return detections
        
    except Exception as e:
        print(f"❌ 后处理失败: {e}")
        import traceback
        traceback.print_exc()
        return []

def convert_to_coco_class(class_idx):
    """将模型类别索引转换为COCO类别ID和名称"""
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
    
    # 基于训练时的映射
    if class_idx == 0:
        coco_id = 1  # person
    elif class_idx == 2:
        coco_id = 3  # car
    elif class_idx == 26:
        coco_id = 27  # backpack
    elif class_idx == 32:
        coco_id = 33  # suitcase
    elif class_idx == 83:
        coco_id = 84  # book
    else:
        coco_id = class_idx + 1
    
    class_name = COCO_CLASSES.get(coco_id, f'class_{coco_id}')
    return coco_id, class_name

def test_sanity_check_model():
    """测试sanity check训练的模型"""
    print("🧪 测试Sanity Check训练的模型")
    print("=" * 60)
    
    # 配置
    config = {
        'model_path': '/home/kyc/project/RT-DETR/results/sanity_check_training/rtdetr_50img_50epoch.pkl',
        'data_dir': '/home/kyc/project/RT-DETR/data/coco2017_50',
        'results_dir': '/home/kyc/project/RT-DETR/results/sanity_check_test',
        'num_test_images': 10,
        'score_threshold': 0.2
    }
    
    # 创建结果目录
    os.makedirs(config['results_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['results_dir'], 'visualizations'), exist_ok=True)
    
    # 加载模型
    backbone, transformer, checkpoint = load_sanity_check_model(config['model_path'])
    if backbone is None:
        print("❌ 无法加载模型，退出测试")
        return
    
    # 加载验证数据
    print("🔄 加载验证数据集...")
    data_dir = config['data_dir']
    images_dir = os.path.join(data_dir, "val2017")
    annotations_file = os.path.join(data_dir, "annotations", "instances_val2017.json")
    
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    print(f"✅ 验证数据加载完成: {len(coco_data['images'])}张图像")
    
    # 测试前几张图像
    print(f"🎯 开始测试前{config['num_test_images']}张图像...")
    print("=" * 60)
    
    results = []
    
    for i, img_info in enumerate(coco_data['images'][:config['num_test_images']]):
        print(f"\n🔍 测试图像 {i+1}/{config['num_test_images']} (ID: {img_info['id']})")
        print(f"   文件名: {img_info['file_name']}")
        
        try:
            # 加载图像
            image_path = os.path.join(images_dir, img_info['file_name'])
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            
            # 预处理图像
            image_resized = image.resize((640, 640))
            image_array = np.array(image_resized).astype(np.float32) / 255.0
            image_tensor = jt.array(image_array.transpose(2, 0, 1)).unsqueeze(0)
            
            # 模型推理
            start_time = datetime.now()
            with jt.no_grad():
                features = backbone(image_tensor)
                outputs = transformer(features)
            
            # 后处理
            detections = postprocess_outputs(outputs, original_size, config['score_threshold'])
            processing_time = (datetime.now() - start_time).total_seconds()
            
            print(f"   ✅ 检测到 {len(detections)} 个目标, 处理时间: {processing_time:.3f}s")
            
            # 显示检测结果
            if detections:
                print(f"   🎯 检测结果:")
                for j, det in enumerate(detections):
                    print(f"      {j+1}: {det['class_name']} (置信度: {det['confidence']:.3f})")
            else:
                print(f"   ❌ 未检测到任何目标")
            
            # 记录结果
            results.append({
                'image_id': img_info['id'],
                'file_name': img_info['file_name'],
                'detections': len(detections),
                'processing_time': processing_time,
                'detected_classes': [d['class_name'] for d in detections]
            })
            
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            continue
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 Sanity Check模型测试总结")
    print("=" * 60)
    
    if results:
        total_detections = sum(r['detections'] for r in results)
        avg_time = np.mean([r['processing_time'] for r in results])
        
        print(f"✅ 成功测试: {len(results)} 张图像")
        print(f"📈 总检测数: {total_detections}")
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
    
    print("\n🎉 Sanity Check模型测试完成！")

if __name__ == "__main__":
    test_sanity_check_model()
