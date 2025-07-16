#!/usr/bin/env python3
"""
测试训练好的RT-DETR模型性能
使用val2017数据集进行推理测试和评估，生成可视化结果
"""

import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from datetime import datetime

# 设置matplotlib支持中文字体，避免中文字符警告
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用英文字体避免中文警告
matplotlib.rcParams['axes.unicode_minus'] = False

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt

# 设置Jittor
jt.flags.use_cuda = 1

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

def load_trained_model(model_path):
    """加载训练好的模型（完全按照ultimate_sanity_check.py的方式）"""
    print(f"🔄 加载训练好的模型: {model_path}")

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
        print(f"   训练损失: {checkpoint.get('loss', 'Unknown')}")

        return backbone, transformer, checkpoint

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def load_validation_data(data_dir):
    """加载验证数据集"""
    print("🔄 加载验证数据集...")
    
    # 数据路径
    images_dir = os.path.join(data_dir, "val2017")
    annotations_file = os.path.join(data_dir, "annotations", "instances_val2017.json")
    
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
    
    # 构建图像信息映射
    image_info = {}
    for img in coco_data['images']:
        image_info[img['id']] = img
    
    print(f"✅ 验证数据加载完成: {len(image_info)}张图像, {len(coco_data['annotations'])}个标注")
    
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

def postprocess_outputs(outputs, original_size, score_threshold=0.2):
    """后处理模型输出（基于ultimate_sanity_check.py的成功方法）"""
    try:
        # 获取预测结果
        pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
        pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4]

        # 后处理 - 基于ultimate_sanity_check.py的鲁棒版本
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

        # 转换为numpy
        scores_np = max_scores.numpy()
        classes_np = pred_classes.numpy()
        boxes_np = pred_boxes.numpy()

        print(f"   分数范围: {scores_np.min():.4f} - {scores_np.max():.4f}")
        print(f"   类别索引范围: {classes_np.min()} - {classes_np.max()}")

        # 显示前5个最高分数的预测
        top_indices = np.argsort(scores_np)[::-1][:5]
        print(f"   前10个预测:")
        for i, idx in enumerate(top_indices[:10]):
            class_idx = classes_np[idx]
            score = scores_np[idx]
            coco_id, class_name = convert_to_coco_class(class_idx)
            print(f"     {i+1}: 类别{class_idx}({class_name}), 分数{score:.4f}")

        # 转换边界框格式：从中心点格式到左上右下格式，并缩放到640x640
        boxes_640_xyxy = []
        for box in boxes_np:
            cx, cy, w, h = box
            x1 = (cx - w/2) * 640
            y1 = (cy - h/2) * 640
            x2 = (cx + w/2) * 640
            y2 = (cy + h/2) * 640
            boxes_640_xyxy.append([x1, y1, x2, y2])

        # 应用NMS过滤
        filtered_boxes, filtered_scores, filtered_classes = nms_filter(
            boxes_640_xyxy, scores_np, classes_np, 0.5, score_threshold
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
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(score),
                    'class_id': int(class_idx),
                    'coco_id': int(coco_id),
                    'class_name': str(class_name)
                })

        return detections
        
    except Exception as e:
        print(f"❌ 后处理失败: {e}")
        return []

def convert_to_coco_class(class_idx):
    """将模型类别索引转换为COCO类别ID和名称"""
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

def nms_filter(boxes, scores, classes, iou_threshold=0.5, score_threshold=0.3):
    """改进的NMS过滤，保留多个不同类别的检测"""
    if len(boxes) == 0:
        return [], [], []

    # 按分数排序
    sorted_indices = np.argsort(scores)[::-1]

    keep_indices = []
    kept_classes = set()

    for i in sorted_indices:
        if scores[i] < score_threshold:
            continue

        current_class = classes[i]

        # 如果已经保留了3个不同类别，且当前类别已存在，跳过
        if len(kept_classes) >= 3 and current_class in kept_classes:
            continue

        # 检查与已保留的框是否重叠过多
        keep_this = True
        for j in keep_indices:
            # 只与同类别的框计算IoU
            if classes[j] == current_class:
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
            kept_classes.add(current_class)

            # 限制最多保留5个检测
            if len(keep_indices) >= 5:
                break

    return [boxes[i] for i in keep_indices], [scores[i] for i in keep_indices], [classes[i] for i in keep_indices]

def visualize_detections(image, detections, gt_annotations=None, save_path=None):
    """可视化检测结果"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    # 绘制检测结果
    for det in detections:
        bbox = det['bbox']
        confidence = det['confidence']
        class_name = det['class_name']
        
        # 绘制边界框
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        # 添加标签
        label = f"{class_name}: {confidence:.2f}"
        ax.text(bbox[0], bbox[1] - 5, label, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7),
                fontsize=10, color='white', weight='bold')
    
    # 绘制真实标注（如果提供）
    if gt_annotations:
        for ann in gt_annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            
            # 绘制真实边界框
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor='green', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)
            
            # 添加真实标签
            category_id = ann['category_id']
            class_name = COCO_CLASSES.get(category_id, f'class_{category_id}')
            ax.text(x, y - 20, f"GT: {class_name}", 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.7),
                    fontsize=10, color='white', weight='bold')
    
    ax.set_title(f'Detection Results (Count: {len(detections)})', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 可视化结果已保存: {save_path}")
    
    plt.tight_layout()
    return fig

def main():
    print("🧪 RT-DETR训练模型性能测试")
    print("=" * 80)
    
    # 配置参数
    config = {
        'model_path': '/home/kyc/project/RT-DETR/results/full_training/rtdetr_trained.pkl',
        'data_dir': '/home/kyc/project/RT-DETR/data/coco2017_50',
        'results_dir': '/home/kyc/project/RT-DETR/results/model_evaluation',
        'num_test_images': 20,  # 测试图像数量
        'score_threshold': 0.10,  # 置信度阈值（进一步降低以检测更多类别）
        'iou_threshold': 0.5     # NMS IoU阈值
    }
    
    print(f"📋 测试配置:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # 创建结果目录
    os.makedirs(config['results_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['results_dir'], 'visualizations'), exist_ok=True)
    
    # 加载训练好的模型
    backbone, transformer, checkpoint = load_trained_model(config['model_path'])
    if backbone is None:
        print("❌ 无法加载模型，退出测试")
        return
    
    # 加载验证数据
    image_info, image_annotations, images_dir = load_validation_data(config['data_dir'])
    
    print("🎯 开始模型推理测试...")
    print("=" * 80)

    # 测试统计
    test_results = {
        'total_images': 0,
        'successful_inferences': 0,
        'total_detections': 0,
        'detection_stats': {},
        'confidence_scores': [],
        'processing_times': [],
        'per_image_results': []
    }

    # 选择测试图像
    image_ids = list(image_info.keys())[:config['num_test_images']]

    print(f"📊 测试 {len(image_ids)} 张图像...")

    for i, image_id in enumerate(image_ids):
        print(f"\n🔍 处理图像 {i+1}/{len(image_ids)} (ID: {image_id})")

        try:
            # 获取图像信息
            img_info = image_info[image_id]
            image_path = os.path.join(images_dir, img_info['file_name'])

            if not os.path.exists(image_path):
                print(f"   ⚠️ 图像文件不存在: {image_path}")
                continue

            # 预处理图像
            start_time = datetime.now()
            image_tensor, original_image, original_size = preprocess_image(image_path)

            if image_tensor is None:
                print(f"   ❌ 图像预处理失败")
                continue

            # 模型推理（完全按照ultimate_sanity_check.py的方式）
            with jt.no_grad():
                features = backbone(image_tensor)
                outputs = transformer(features)

            # 后处理（已包含NMS过滤）
            detections = postprocess_outputs(outputs, original_size, config['score_threshold'])

            processing_time = (datetime.now() - start_time).total_seconds()

            # 获取真实标注
            gt_annotations = image_annotations.get(image_id, [])

            # 统计结果
            test_results['total_images'] += 1
            test_results['successful_inferences'] += 1
            test_results['total_detections'] += len(detections)
            test_results['processing_times'].append(processing_time)

            # 统计检测类别
            for det in detections:
                class_name = det['class_name']
                confidence = det['confidence']

                if class_name not in test_results['detection_stats']:
                    test_results['detection_stats'][class_name] = 0
                test_results['detection_stats'][class_name] += 1
                test_results['confidence_scores'].append(confidence)

            # 记录单张图像结果
            image_result = {
                'image_id': image_id,
                'file_name': img_info['file_name'],
                'detections': len(detections),
                'gt_annotations': len(gt_annotations),
                'processing_time': processing_time,
                'detection_details': detections
            }
            test_results['per_image_results'].append(image_result)

            print(f"   ✅ 检测到 {len(detections)} 个目标, 处理时间: {processing_time:.3f}s")

            # 可视化结果（前10张图像）
            if i < 10:
                vis_save_path = os.path.join(
                    config['results_dir'], 'visualizations',
                    f"detection_{i+1:02d}_{img_info['file_name']}"
                )

                fig = visualize_detections(
                    original_image, detections, gt_annotations, vis_save_path
                )
                plt.close(fig)  # 释放内存

            # 显示检测详情
            if detections:
                for j, det in enumerate(detections[:3]):  # 只显示前3个
                    print(f"      检测{j+1}: {det['class_name']} (置信度: {det['confidence']:.3f})")
                if len(detections) > 3:
                    print(f"      ... 还有 {len(detections)-3} 个检测")

        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            continue

    # 生成评估报告
    print("\n📊 生成评估报告...")
    generate_evaluation_report(test_results, config)

    print("✅ 模型性能测试完成！")

def generate_evaluation_report(test_results, config):
    """生成评估报告"""
    print("\n" + "=" * 80)
    print("📊 模型性能评估报告")
    print("=" * 80)

    # 基本统计
    total_images = test_results['total_images']
    successful_inferences = test_results['successful_inferences']
    total_detections = test_results['total_detections']

    print(f"\n📈 基本统计:")
    print(f"   测试图像数: {total_images}")
    print(f"   成功推理数: {successful_inferences}")
    print(f"   推理成功率: {successful_inferences/max(total_images,1)*100:.1f}%")
    print(f"   总检测数: {total_detections}")
    print(f"   平均每张图像检测数: {total_detections/max(successful_inferences,1):.2f}")

    # 处理时间统计
    if test_results['processing_times']:
        avg_time = np.mean(test_results['processing_times'])
        min_time = np.min(test_results['processing_times'])
        max_time = np.max(test_results['processing_times'])

        print(f"\n⏱️ 处理时间统计:")
        print(f"   平均处理时间: {avg_time:.3f}s")
        print(f"   最快处理时间: {min_time:.3f}s")
        print(f"   最慢处理时间: {max_time:.3f}s")
        print(f"   推理速度: {1/avg_time:.1f} FPS")

    # 置信度统计
    if test_results['confidence_scores']:
        avg_conf = np.mean(test_results['confidence_scores'])
        min_conf = np.min(test_results['confidence_scores'])
        max_conf = np.max(test_results['confidence_scores'])

        print(f"\n🎯 置信度统计:")
        print(f"   平均置信度: {avg_conf:.3f}")
        print(f"   最低置信度: {min_conf:.3f}")
        print(f"   最高置信度: {max_conf:.3f}")

    # 检测类别统计
    if test_results['detection_stats']:
        print(f"\n🏷️ 检测类别统计:")
        sorted_classes = sorted(test_results['detection_stats'].items(),
                               key=lambda x: x[1], reverse=True)
        for class_name, count in sorted_classes[:10]:  # 显示前10个
            print(f"   {class_name}: {count} 次")

        if len(sorted_classes) > 10:
            print(f"   ... 还有 {len(sorted_classes)-10} 个类别")

    # 保存详细报告
    report_path = os.path.join(config['results_dir'], 'evaluation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)

    print(f"\n📋 详细报告已保存: {report_path}")

    # 生成统计图表
    generate_statistics_plots(test_results, config)

def generate_statistics_plots(test_results, config):
    """生成统计图表"""
    print("📈 生成统计图表...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 每张图像检测数分布
    ax1 = axes[0, 0]
    detections_per_image = [r['detections'] for r in test_results['per_image_results']]
    ax1.hist(detections_per_image, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Detections per Image Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Number of Detections')
    ax1.set_ylabel('Number of Images')
    ax1.grid(True, alpha=0.3)

    # 2. 置信度分布
    ax2 = axes[0, 1]
    if test_results['confidence_scores']:
        ax2.hist(test_results['confidence_scores'], bins=20, alpha=0.7,
                color='lightgreen', edgecolor='black')
        ax2.set_title('Confidence Score Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Number of Detections')
        ax2.grid(True, alpha=0.3)

    # 3. 处理时间分布
    ax3 = axes[1, 0]
    if test_results['processing_times']:
        ax3.hist(test_results['processing_times'], bins=15, alpha=0.7,
                color='orange', edgecolor='black')
        ax3.set_title('Processing Time Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Processing Time (seconds)')
        ax3.set_ylabel('Number of Images')
        ax3.grid(True, alpha=0.3)

    # 4. 检测类别统计（前10个）
    ax4 = axes[1, 1]
    if test_results['detection_stats']:
        sorted_classes = sorted(test_results['detection_stats'].items(),
                               key=lambda x: x[1], reverse=True)[:10]
        classes, counts = zip(*sorted_classes)

        bars = ax4.bar(range(len(classes)), counts, alpha=0.7, color='lightcoral')
        ax4.set_title('Detection Class Statistics (Top 10)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Class')
        ax4.set_ylabel('Detection Count')
        ax4.set_xticks(range(len(classes)))
        ax4.set_xticklabels(classes, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)

        # 添加数值标签
        for bar, count in zip(bars, counts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # 保存图表
    plots_path = os.path.join(config['results_dir'], 'evaluation_statistics.png')
    plt.savefig(plots_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 统计图表已保存: {plots_path}")

if __name__ == "__main__":
    main()
