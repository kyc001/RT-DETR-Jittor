#!/usr/bin/env python3
"""
基于成功的5张图像方法，扩展到10张图像过拟合训练
优化参数：提高置信度阈值，增加训练轮数
"""

import os
import sys
import json
import numpy as np
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from collections import Counter

# 设置matplotlib支持中文字体，避免中文字符警告
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt
from jittor import nn

# 设置Jittor
jt.flags.use_cuda = 1

def load_10_images():
    """加载10张训练图像（基于成功的5张图像方法）"""
    print("🔄 加载10张训练图像...")
    
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
    
    # 选择前10张有标注的图像
    selected_data = []
    
    for img_info in coco_data['images'][:20]:  # 从前20张中选择10张有标注的
        image_id = img_info['id']
        image_path = os.path.join(images_dir, img_info['file_name'])
        
        # 获取标注
        annotations = image_annotations.get(image_id, [])
        if not annotations:
            continue
        
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            
            # 调整图像大小到640x640
            image_resized = image.resize((640, 640))
            image_array = np.array(image_resized).astype(np.float32) / 255.0
            image_tensor = jt.array(image_array.transpose(2, 0, 1)).unsqueeze(0)
            
            # 处理标注 - 使用修复后的正确格式（左上右下）
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
                    
                    # 类别映射（COCO ID -> 0-based索引）
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
                else:
                    print(f"   ⚠️ 跳过无效边界框: [{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}]")
                    continue
            
            if boxes:
                target = {
                    'boxes': jt.array(boxes),
                    'labels': jt.array(labels)
                }
                
                selected_data.append({
                    'image_tensor': image_tensor,
                    'target': target,
                    'image_id': image_id,
                    'file_name': img_info['file_name'],
                    'num_objects': len(boxes)
                })
                
                print(f"   ✅ 图像 {len(selected_data)}: {img_info['file_name']} ({len(boxes)}个目标)")
                
                # 只要10张图像
                if len(selected_data) >= 10:
                    break
                    
        except Exception as e:
            print(f"   ⚠️ 跳过图像 {img_info['file_name']}: {e}")
            continue
    
    print(f"✅ 加载完成: {len(selected_data)}张训练图像")
    return selected_data

def create_model():
    """创建模型（完全按照ultimate_sanity_check.py）"""
    print("🔄 创建模型...")
    
    try:
        from jittor_rt_detr.src.nn.backbone.resnet import ResNet50
        from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
        from jittor_rt_detr.src.nn.criterion.rtdetr_criterion import build_criterion
        
        # 创建模型
        backbone = ResNet50(pretrained=False)
        transformer = RTDETRTransformer(
            num_classes=80,
            hidden_dim=256,
            num_queries=300,
            feat_channels=[256, 512, 1024, 2048]
        )
        criterion = build_criterion(num_classes=80)
        
        print(f"✅ 模型创建成功")
        return backbone, transformer, criterion
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return None, None, None

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

def fix_batchnorm(module):
    """修复BatchNorm（完全按照ultimate_sanity_check.py）"""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()
            # 确保BatchNorm参数可训练
            if hasattr(m, 'weight') and m.weight is not None:
                m.weight.requires_grad = True
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = True

def nms_filter(boxes, scores, classes, iou_threshold=0.5, score_threshold=0.3):
    """简单的NMS过滤"""
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

def intensive_training_10_images(backbone, transformer, criterion, train_data, num_epochs=150):
    """10张图像过拟合训练（增加训练轮数）"""
    print(f"\n🚀 开始10张图像过拟合训练 ({num_epochs}轮)")
    print("=" * 60)
    print("目标: 模型必须能够完美记住这10张图像的所有目标")
    
    try:
        # 设置模型为训练模式并修复BatchNorm问题
        backbone.train()
        transformer.train()
        
        # 修复BatchNorm
        fix_batchnorm(backbone)
        fix_batchnorm(transformer)
        
        # 收集所有需要梯度的参数
        all_params = []
        for module in [backbone, transformer]:
            for param in module.parameters():
                if param.requires_grad:
                    all_params.append(param)
        
        # 创建优化器 - 使用稍高的学习率
        optimizer = jt.optim.Adam(all_params, lr=1.5e-4, weight_decay=0)
        
        # 检查参数数量
        total_params = sum(p.numel() for p in backbone.parameters()) + sum(p.numel() for p in transformer.parameters())
        trainable_elements = sum(p.numel() for p in all_params)
        
        print(f"📊 模型配置:")
        print(f"   总参数: {total_params:,}")
        print(f"   可训练参数张量: {len(all_params)}")
        print(f"   可训练参数元素: {trainable_elements:,}")
        print(f"   学习率: 1.5e-4 (稍微提高)")
        
        # 显示训练数据信息
        total_objects = sum(data['num_objects'] for data in train_data)
        print(f"📋 训练数据:")
        for i, data in enumerate(train_data):
            print(f"   图像{i+1:2d}: {data['file_name']} ({data['num_objects']:2d}个目标)")
        print(f"   总目标数: {total_objects}")
        
        # 检查初始损失
        print(f"\n🔍 检查初始损失...")
        initial_losses = []
        for i, data in enumerate(train_data):
            img_tensor = data['image_tensor']
            target = data['target']
            
            feats = backbone(img_tensor)
            outputs = transformer(feats)
            loss_dict = criterion(outputs, [target])
            total_loss = sum(loss_dict.values())
            initial_losses.append(total_loss.numpy().item())
            
            print(f"   图像{i+1:2d}: {total_loss.numpy().item():.4f}")
        
        avg_initial_loss = np.mean(initial_losses)
        print(f"   平均初始损失: {avg_initial_loss:.4f}")
        
        # 训练循环
        print(f"\n🎯 开始训练...")
        all_losses = []
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # 确保模型在训练模式
            backbone.train()
            transformer.train()
            
            # 对每张图像进行训练
            for data in train_data:
                img_tensor = data['image_tensor']
                target = data['target']
                
                # 前向传播
                feats = backbone(img_tensor)
                outputs = transformer(feats)
                
                # 损失计算
                loss_dict = criterion(outputs, [target])
                total_loss = sum(loss_dict.values())
                
                # 反向传播和参数更新
                optimizer.step(total_loss)
                
                epoch_losses.append(total_loss.numpy().item())
            
            # 记录平均损失
            avg_loss = np.mean(epoch_losses)
            all_losses.append(avg_loss)
            
            # 打印进度
            if epoch % 30 == 0 or epoch < 10 or epoch >= num_epochs - 10:
                print(f"   Epoch {epoch+1:3d}/{num_epochs}: 平均损失 = {avg_loss:.4f}")
        
        print(f"\n✅ 训练完成!")
        print(f"   初始损失: {avg_initial_loss:.4f}")
        print(f"   最终损失: {all_losses[-1]:.4f}")
        print(f"   损失下降: {avg_initial_loss - all_losses[-1]:.4f}")
        print(f"   下降比例: {(avg_initial_loss - all_losses[-1])/avg_initial_loss*100:.1f}%")
        
        return backbone, transformer, all_losses
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def visualize_results(image_data, detections, gt_annotations, save_path, title="Detection Results"):
    """可视化检测结果"""
    try:
        # 重新加载原始图像用于可视化
        data_dir = '/home/kyc/project/RT-DETR/data/coco2017_50'
        images_dir = os.path.join(data_dir, "train2017")
        image_path = os.path.join(images_dir, image_data['file_name'])

        image = Image.open(image_path).convert('RGB')

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # 左侧：真实标注
        ax1.imshow(image)
        ax1.set_title(f'Ground Truth ({len(gt_annotations)} objects)', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # 绘制真实边界框
        for ann in gt_annotations:
            x, y, w, h = ann['bbox']
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none')
            ax1.add_patch(rect)

            # 添加类别标签
            category_id = ann['category_id']
            _, class_name = convert_to_coco_class(category_id - 1)  # 转换为0-based索引
            ax1.text(x, y-5, class_name, fontsize=10, color='green', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        # 右侧：检测结果
        ax2.imshow(image)
        ax2.set_title(f'Detection Results ({len(detections)} detections)', fontsize=14, fontweight='bold')
        ax2.axis('off')

        # 绘制检测边界框
        colors = ['red', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            w, h = x2 - x1, y2 - y1

            color = colors[i % len(colors)]
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=color, facecolor='none')
            ax2.add_patch(rect)

            # 添加类别和置信度标签
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            ax2.text(x1, y1-5, label, fontsize=10, color=color, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 保存图像
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return True

    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        return False

def postprocess_outputs(outputs, original_size, score_threshold=0.3):
    """后处理模型输出（提高置信度阈值到0.3）"""
    try:
        # 获取预测结果
        pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
        pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4]

        # 后处理
        pred_scores = jt.nn.softmax(pred_logits, dim=-1)
        pred_scores_no_bg = pred_scores[:, :-1]  # 排除背景类

        # 获取最高分数的类别
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

        # 应用NMS过滤
        boxes_640 = pred_boxes.numpy() * 640  # 转换到640x640坐标系用于NMS
        scores_np = max_scores.numpy()
        classes_np = pred_classes.numpy()

        filtered_boxes, filtered_scores, filtered_classes = nms_filter(
            boxes_640, scores_np, classes_np, 0.5, score_threshold
        )

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
        return []

def test_overfitting_with_visualization(backbone, transformer, train_data, score_threshold=0.3):
    """测试过拟合效果并生成可视化结果（提高置信度阈值）"""
    print(f"\n🧪 测试过拟合效果并生成可视化")
    print("=" * 60)
    print(f"验证: 模型是否能在训练图像上检测到目标 (置信度阈值: {score_threshold})")

    try:
        # 设置为评估模式
        backbone.eval()
        transformer.eval()

        # 创建可视化结果目录
        vis_dir = '/home/kyc/project/RT-DETR/results/sanity_check_10_images/visualizations'
        os.makedirs(vis_dir, exist_ok=True)

        success_count = 0
        total_gt_objects = 0
        total_detected_objects = 0
        all_detections = []

        # 加载原始标注数据用于可视化
        data_dir = '/home/kyc/project/RT-DETR/data/coco2017_50'
        annotations_file = os.path.join(data_dir, "annotations", "instances_train2017.json")

        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)

        # 构建图像ID到标注的映射
        image_annotations = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(ann)

        for i, data in enumerate(train_data):
            img_tensor = data['image_tensor']
            target = data['target']
            gt_boxes = target['boxes'].numpy()
            image_id = data['image_id']

            print(f"\n📊 测试图像 {i+1:2d}: {data['file_name']}")
            print(f"   真实目标: {len(gt_boxes)}个")

            with jt.no_grad():
                # 前向传播
                feats = backbone(img_tensor)
                outputs = transformer(feats)

                # 获取原始图像尺寸
                data_dir = '/home/kyc/project/RT-DETR/data/coco2017_50'
                images_dir = os.path.join(data_dir, "train2017")
                image_path = os.path.join(images_dir, data['file_name'])
                original_image = Image.open(image_path).convert('RGB')
                original_size = original_image.size

                # 后处理
                detections = postprocess_outputs(outputs, original_size, score_threshold)

                print(f"   检测到: {len(detections)}个目标")
                if detections:
                    # 只显示前5个检测结果
                    for j, det in enumerate(detections[:5]):
                        print(f"      {j+1}: {det['class_name']} (置信度: {det['confidence']:.3f})")
                    if len(detections) > 5:
                        print(f"      ... 还有 {len(detections)-5} 个检测")

                # 获取真实标注用于可视化
                gt_annotations = image_annotations.get(image_id, [])

                # 生成可视化
                vis_save_path = os.path.join(vis_dir, f"image_{i+1:02d}_{data['file_name']}")
                title = f"Image {i+1}: {data['file_name']}"

                visualize_results(data, detections, gt_annotations, vis_save_path, title)
                print(f"   💾 可视化已保存: {vis_save_path}")

                # 统计
                total_gt_objects += len(gt_boxes)
                total_detected_objects += len(detections)
                all_detections.extend(detections)

                # 判断是否成功（检测数量接近真实目标数量）
                if len(detections) >= len(gt_boxes) * 0.3:  # 至少检测到30%的目标
                    success_count += 1
                    print(f"   ✅ 过拟合成功")
                else:
                    print(f"   ❌ 过拟合不足")

        # 统计检测类别
        detected_classes = [det['class_name'] for det in all_detections]
        class_counts = Counter(detected_classes)

        # 总结
        success_rate = success_count / len(train_data) * 100
        detection_rate = total_detected_objects / total_gt_objects * 100 if total_gt_objects > 0 else 0

        print(f"\n📊 过拟合测试总结:")
        print(f"   成功图像: {success_count}/{len(train_data)} ({success_rate:.1f}%)")
        print(f"   总体检测率: {total_detected_objects}/{total_gt_objects} ({detection_rate:.1f}%)")

        if class_counts:
            print(f"   🏷️ 检测到的类别:")
            for class_name, count in class_counts.most_common():
                print(f"      {class_name}: {count} 次")

        print(f"   📁 可视化结果保存在: {vis_dir}")

        if success_rate >= 70:
            print(f"   🎉 过拟合测试通过！模型成功记住了训练图像")
            return True
        else:
            print(f"   ⚠️ 过拟合测试未通过，需要更多训练")
            return False

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_model(backbone, transformer, losses, save_path):
    """保存训练好的模型"""
    print(f"💾 保存模型到: {save_path}")

    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        checkpoint = {
            'backbone_state_dict': backbone.state_dict(),
            'transformer_state_dict': transformer.state_dict(),
            'epoch': len(losses),
            'loss': losses[-1] if losses else 0.0,
            'timestamp': datetime.now().isoformat(),
            'training_losses': losses,
            'additional_info': {
                'training_method': 'sanity_check_10_images',
                'num_images': 10,
                'num_epochs': len(losses),
                'learning_rate': 1.5e-4,
                'optimizer': 'Adam',
                'score_threshold': 0.3
            }
        }

        jt.save(checkpoint, save_path)
        print(f"✅ 模型保存成功")

    except Exception as e:
        print(f"❌ 模型保存失败: {e}")

def main():
    print("🧪 RT-DETR 10张图像过拟合训练")
    print("基于成功的5张图像方法，优化参数扩展")
    print("=" * 60)

    # 1. 加载10张训练图像
    train_data = load_10_images()
    if len(train_data) < 10:
        print(f"❌ 只加载到{len(train_data)}张图像，需要10张")
        return

    # 2. 创建模型
    backbone, transformer, criterion = create_model()
    if backbone is None:
        print("❌ 模型创建失败，退出")
        return

    # 3. 过拟合训练（增加训练轮数）
    trained_backbone, trained_transformer, losses = intensive_training_10_images(
        backbone, transformer, criterion, train_data, num_epochs=150
    )

    if trained_backbone is None:
        print("❌ 训练失败，退出")
        return

    # 4. 测试过拟合效果并生成可视化（提高置信度阈值）
    overfitting_success = test_overfitting_with_visualization(
        trained_backbone, trained_transformer, train_data, score_threshold=0.3
    )

    # 5. 保存模型
    save_path = '/home/kyc/project/RT-DETR/results/sanity_check_training/rtdetr_10img_150epoch.pkl'
    save_model(trained_backbone, trained_transformer, losses, save_path)

    # 6. 总结
    print("\n" + "=" * 60)
    print("🎉 10张图像过拟合训练完成！")
    print("=" * 60)

    if overfitting_success:
        print("✅ 过拟合测试通过，可以进行下一步扩展")
        print("💡 建议: 扩展到20张图像或更多")
    else:
        print("⚠️ 过拟合测试未通过，建议:")
        print("   1. 增加训练轮数到200轮")
        print("   2. 降低置信度阈值")
        print("   3. 调整学习率")

    print(f"模型已保存到: {save_path}")
    print(f"可视化结果保存在: /home/kyc/project/RT-DETR/results/sanity_check_10_images/visualizations/")

if __name__ == "__main__":
    main()
