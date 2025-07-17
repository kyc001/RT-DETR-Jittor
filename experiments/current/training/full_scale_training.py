#!/usr/bin/env python3
"""
RT-DETR完整训练脚本
基于ultimate_sanity_check.py的成功架构，扩展到完整数据集训练
"""

import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
# 设置matplotlib支持中文字体，避免中文字符警告
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt
import jittor.nn as nn
from datetime import datetime
import pickle

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

def nms_filter(boxes, scores, classes, iou_threshold=0.5, score_threshold=0.3):
    """简单的NMS过滤，去除重复检测"""
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

def convert_to_coco_class(class_idx):
    """将模型类别索引转换为COCO类别ID和名称"""
    # 基于ultimate_sanity_check.py中验证的映射
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

def load_coco_dataset(data_dir, split='train'):
    """加载COCO数据集"""
    print(f"🔄 加载{split}数据集...")
    
    # 数据路径
    images_dir = os.path.join(data_dir, f"{split}2017")
    annotations_file = os.path.join(data_dir, "annotations", f"instances_{split}2017.json")
    
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
    
    print(f"✅ 加载完成: {len(image_info)}张图像, {len(coco_data['annotations'])}个标注")
    
    return image_info, image_annotations, images_dir

def create_model():
    """创建RT-DETR模型（基于ultimate_sanity_check.py的成功方法）"""
    print("🔄 创建RT-DETR模型...")

    try:
        # 导入模型组件（基于ultimate_sanity_check.py的成功导入）
        from jittor_rt_detr.src.nn.backbone.resnet import ResNet50
        from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
        from jittor_rt_detr.src.nn.criterion.rtdetr_criterion import build_criterion

        # 创建backbone
        backbone = ResNet50(pretrained=False)

        # 创建transformer
        transformer = RTDETRTransformer(
            num_classes=80,
            hidden_dim=256,
            num_queries=300,
            feat_channels=[256, 512, 1024, 2048]
        )

        # 创建完整模型
        class RTDETRModel(jt.nn.Module):
            def __init__(self, backbone, transformer):
                super().__init__()
                self.backbone = backbone
                self.transformer = transformer

            def execute(self, x):
                # 提取特征
                features = self.backbone(x)

                # Transformer处理
                outputs = self.transformer(features)

                return outputs

        model = RTDETRModel(backbone, transformer)
        model.train()

        print("✅ 模型创建成功")
        return model

    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_model(model, save_path, epoch, loss, additional_info=None):
    """保存模型"""
    print(f"💾 保存模型到: {save_path}")

    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        # 使用Jittor的保存方法
        jt.save({
            'backbone_state_dict': model.backbone.state_dict(),
            'transformer_state_dict': model.transformer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'timestamp': datetime.now().isoformat(),
            'additional_info': additional_info or {}
        }, save_path)

        print(f"✅ 模型保存成功")

    except Exception as e:
        print(f"❌ 模型保存失败: {e}")
        # 备用方案：分别保存各个组件
        try:
            backup_dir = save_path.replace('.pkl', '_backup')
            os.makedirs(backup_dir, exist_ok=True)

            jt.save(model.backbone.state_dict(), os.path.join(backup_dir, 'backbone.pkl'))
            jt.save(model.transformer.state_dict(), os.path.join(backup_dir, 'transformer.pkl'))

            # 保存元信息
            meta_info = {
                'epoch': epoch,
                'loss': loss,
                'timestamp': datetime.now().isoformat(),
                'additional_info': additional_info or {}
            }

            with open(os.path.join(backup_dir, 'meta.json'), 'w') as f:
                json.dump(meta_info, f, indent=2)

            print(f"✅ 模型备用保存成功到: {backup_dir}")

        except Exception as e2:
            print(f"❌ 备用保存也失败: {e2}")

def load_model(model, load_path):
    """加载模型"""
    print(f"🔄 从{load_path}加载模型...")

    try:
        # 尝试使用Jittor加载
        checkpoint = jt.load(load_path)

        if 'backbone_state_dict' in checkpoint and 'transformer_state_dict' in checkpoint:
            model.backbone.load_state_dict(checkpoint['backbone_state_dict'])
            model.transformer.load_state_dict(checkpoint['transformer_state_dict'])
        else:
            # 兼容旧格式
            model.load_state_dict(checkpoint['model_state_dict'])

        print(f"✅ 模型加载成功")
        print(f"   训练轮数: {checkpoint.get('epoch', 'Unknown')}")
        print(f"   损失: {checkpoint.get('loss', 'Unknown')}")
        print(f"   保存时间: {checkpoint.get('timestamp', 'Unknown')}")

        return checkpoint

    except Exception as e:
        print(f"❌ 主要加载失败: {e}")

        # 尝试备用加载方案
        try:
            backup_dir = load_path.replace('.pkl', '_backup')

            if os.path.exists(backup_dir):
                backbone_path = os.path.join(backup_dir, 'backbone.pkl')
                transformer_path = os.path.join(backup_dir, 'transformer.pkl')
                meta_path = os.path.join(backup_dir, 'meta.json')

                model.backbone.load_state_dict(jt.load(backbone_path))
                model.transformer.load_state_dict(jt.load(transformer_path))

                with open(meta_path, 'r') as f:
                    meta_info = json.load(f)

                print(f"✅ 备用加载成功")
                print(f"   训练轮数: {meta_info.get('epoch', 'Unknown')}")
                print(f"   损失: {meta_info.get('loss', 'Unknown')}")

                return meta_info
            else:
                print(f"❌ 备用目录不存在: {backup_dir}")
                return None

        except Exception as e2:
            print(f"❌ 备用加载也失败: {e2}")
            return None

class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.train_losses = []
        self.train_epochs = []
        self.val_results = []
        
    def log_training(self, epoch, loss):
        """记录训练损失"""
        self.train_epochs.append(epoch)
        self.train_losses.append(loss)
        
    def log_validation(self, epoch, results):
        """记录验证结果"""
        results['epoch'] = epoch
        self.val_results.append(results)
        
    def save_logs(self):
        """保存日志"""
        # 保存训练日志
        train_log = {
            'epochs': self.train_epochs,
            'losses': self.train_losses
        }
        
        with open(os.path.join(self.log_dir, 'training_log.json'), 'w') as f:
            json.dump(train_log, f, indent=2)
        
        # 保存验证日志
        with open(os.path.join(self.log_dir, 'validation_log.json'), 'w') as f:
            json.dump(self.val_results, f, indent=2)
        
        print(f"📊 日志已保存到: {self.log_dir}")
        
    def plot_training_curve(self):
        """绘制训练曲线"""
        if not self.train_losses:
            return
            
        plt.figure(figsize=(12, 8))
        
        # 训练损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(self.train_epochs, self.train_losses, 'b-', linewidth=2)
        plt.title('Training Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        # 如果有验证结果，绘制验证指标
        if self.val_results:
            epochs = [r['epoch'] for r in self.val_results]
            
            # 检测数量
            plt.subplot(2, 2, 2)
            detections = [r.get('total_detections', 0) for r in self.val_results]
            plt.plot(epochs, detections, 'g-', linewidth=2)
            plt.title('Total Detections', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            
            # 平均置信度
            plt.subplot(2, 2, 3)
            avg_conf = [r.get('avg_confidence', 0) for r in self.val_results]
            plt.plot(epochs, avg_conf, 'r-', linewidth=2)
            plt.title('Average Confidence', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Confidence')
            plt.grid(True, alpha=0.3)
            
            # 类别多样性
            plt.subplot(2, 2, 4)
            diversity = [r.get('class_diversity', 0) for r in self.val_results]
            plt.plot(epochs, diversity, 'm-', linewidth=2)
            plt.title('Class Diversity', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Unique Classes')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.log_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 训练曲线已保存到: {save_path}")

def main():
    print("🚀 RT-DETR完整训练开始")
    print("=" * 80)
    
    # 配置参数
    config = {
        'data_dir': '/home/kyc/project/RT-DETR/data/coco2017_50',
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'save_dir': '/home/kyc/project/RT-DETR/results/full_training',
        'log_dir': '/home/kyc/project/RT-DETR/results/full_training/logs',
        'model_save_path': '/home/kyc/project/RT-DETR/results/full_training/rtdetr_trained.pkl'
    }
    
    print(f"📋 训练配置:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # 初始化日志记录器
    logger = TrainingLogger(config['log_dir'])
    
    # 加载数据集
    train_images, train_annotations, train_dir = load_coco_dataset(config['data_dir'], 'train')
    val_images, val_annotations, val_dir = load_coco_dataset(config['data_dir'], 'val')
    
    # 创建模型
    model = create_model()
    if model is None:
        print("❌ 无法创建模型，退出")
        return
    
    print("🎯 开始训练...")
    print("=" * 80)

    # 训练逻辑
    try:
        # 导入必要的模块（基于ultimate_sanity_check.py的成功方法）
        from jittor_rt_detr.src.nn.criterion.rtdetr_criterion import build_criterion

        # 创建损失函数
        criterion = build_criterion(num_classes=80)

        # 创建优化器
        optimizer = jt.optim.Adam(model.parameters(), lr=config['learning_rate'])

        # 训练循环
        for epoch in range(config['num_epochs']):
            print(f"\n📅 Epoch {epoch+1}/{config['num_epochs']}")

            # 训练一个epoch
            epoch_loss = train_one_epoch(model, train_images, train_annotations, train_dir,
                                       criterion, optimizer, epoch)

            # 记录训练损失
            logger.log_training(epoch, epoch_loss)

            # 每5个epoch进行一次验证
            if (epoch + 1) % 5 == 0:
                print(f"🔍 进行验证...")
                val_results = validate_model(model, val_images, val_annotations, val_dir)
                logger.log_validation(epoch, val_results)

                # 保存模型
                save_model(model, config['model_save_path'], epoch, epoch_loss,
                          {'validation_results': val_results})

            # 每10个epoch保存一次训练曲线
            if (epoch + 1) % 10 == 0:
                logger.plot_training_curve()

        # 训练完成
        print("\n🎉 训练完成！")

        # 最终验证
        print("🔍 进行最终验证...")
        final_val_results = validate_model(model, val_images, val_annotations, val_dir)
        logger.log_validation(config['num_epochs']-1, final_val_results)

        # 保存最终模型
        save_model(model, config['model_save_path'], config['num_epochs']-1, epoch_loss,
                  {'final_validation_results': final_val_results})

        # 保存所有日志
        logger.save_logs()
        logger.plot_training_curve()

        # 生成最终报告
        generate_final_report(config, logger, final_val_results)

        print("✅ 完整训练流程结束！")

    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()

def train_one_epoch(model, images_info, annotations, images_dir, criterion, optimizer, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    # 获取所有图像ID
    image_ids = list(images_info.keys())

    print(f"   处理 {len(image_ids)} 张图像...")

    for i, image_id in enumerate(image_ids):
        try:
            # 加载图像和标注（基于ultimate_sanity_check.py的成功方法）
            image_data, targets = load_single_image_data(
                image_id, images_info, annotations, images_dir)

            if image_data is None or targets is None:
                continue

            # 前向传播（基于ultimate_sanity_check.py的成功方法）
            features = model.backbone(image_data)
            outputs = model.transformer(features)

            # 计算损失
            loss_dict = criterion(outputs, targets)
            total_loss_value = sum(loss_dict.values())

            # 反向传播（使用Jittor的正确API）
            optimizer.step(total_loss_value)

            total_loss += total_loss_value.numpy().item()
            num_batches += 1

            # 打印进度
            if (i + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"     进度: {i+1}/{len(image_ids)}, 平均损失: {avg_loss:.4f}")

        except Exception as e:
            print(f"     ⚠️ 图像 {image_id} 处理失败: {e}")
            continue

    avg_epoch_loss = total_loss / max(num_batches, 1)
    print(f"   ✅ Epoch {epoch+1} 完成, 平均损失: {avg_epoch_loss:.4f}")

    return avg_epoch_loss

def load_single_image_data(image_id, images_info, annotations, images_dir):
    """加载单张图像数据（基于ultimate_sanity_check.py的成功方法）"""
    try:
        # 获取图像信息
        img_info = images_info[image_id]
        image_path = os.path.join(images_dir, img_info['file_name'])

        # 加载图像
        image = Image.open(image_path).convert('RGB')
        original_width, original_height = image.size

        # 调整图像大小到640x640
        image = image.resize((640, 640))
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = jt.array(image_array.transpose(2, 0, 1)).unsqueeze(0)

        # 获取标注
        if image_id not in annotations:
            return None, None

        image_annotations = annotations[image_id]

        # 处理边界框和标签
        boxes = []
        labels = []

        for ann in image_annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            category_id = ann['category_id']

            # 转换为归一化坐标
            x1 = bbox[0] / original_width
            y1 = bbox[1] / original_height
            x2 = (bbox[0] + bbox[2]) / original_width
            y2 = (bbox[1] + bbox[3]) / original_height

            # 确保坐标有效
            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= 1 and y2 <= 1:
                boxes.append([x1, y1, x2, y2])

                # COCO类别映射（基于ultimate_sanity_check.py的成功映射）
                if category_id == 1:  # person
                    labels.append(0)
                elif category_id == 3:  # car
                    labels.append(2)
                elif category_id == 27:  # backpack
                    labels.append(26)
                elif category_id == 33:  # suitcase
                    labels.append(32)
                elif category_id == 84:  # book
                    labels.append(83)
                else:
                    labels.append(category_id - 1)  # 其他类别减1

        if not boxes:
            return None, None

        # 创建目标字典
        targets = [{
            'boxes': jt.array(boxes, dtype=jt.float32),
            'labels': jt.array(labels, dtype=jt.int64)
        }]

        return image_tensor, targets

    except Exception as e:
        return None, None

def validate_model(model, images_info, annotations, images_dir):
    """验证模型性能"""
    model.eval()

    print("   🔍 开始验证...")

    total_detections = 0
    total_confidence = 0
    detected_classes = set()
    validation_results = []

    # 获取验证图像ID（取前100张进行快速验证）
    image_ids = list(images_info.keys())[:100]

    with jt.no_grad():
        for i, image_id in enumerate(image_ids):
            try:
                # 加载图像
                image_data, _ = load_single_image_data(
                    image_id, images_info, annotations, images_dir)

                if image_data is None:
                    continue

                # 推理（基于ultimate_sanity_check.py的成功方法）
                features = model.backbone(image_data)
                outputs = model.transformer(features)

                # 后处理（基于ultimate_sanity_check.py的成功方法）
                detections = postprocess_outputs(outputs)

                # 统计结果
                for detection in detections:
                    total_detections += 1
                    total_confidence += detection['confidence']
                    detected_classes.add(detection['class_name'])

                validation_results.append({
                    'image_id': image_id,
                    'detections': len(detections),
                    'avg_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0
                })

                if (i + 1) % 20 == 0:
                    print(f"     验证进度: {i+1}/{len(image_ids)}")

            except Exception as e:
                continue

    # 计算验证指标
    avg_confidence = total_confidence / max(total_detections, 1)
    class_diversity = len(detected_classes)

    results = {
        'total_detections': total_detections,
        'avg_confidence': avg_confidence,
        'class_diversity': class_diversity,
        'detected_classes': list(detected_classes),
        'per_image_results': validation_results
    }

    print(f"   ✅ 验证完成:")
    print(f"      总检测数: {total_detections}")
    print(f"      平均置信度: {avg_confidence:.3f}")
    print(f"      检测类别数: {class_diversity}")

    return results

def postprocess_outputs(outputs):
    """后处理模型输出（基于ultimate_sanity_check.py的成功方法）"""
    try:
        # 获取预测结果
        pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
        pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4]

        # 转换为numpy
        scores = jt.nn.softmax(pred_logits, dim=-1).numpy()
        boxes = pred_boxes.numpy()

        # 获取最高分数的类别
        class_scores = np.max(scores[:, :-1], axis=1)  # 排除背景类
        class_indices = np.argmax(scores[:, :-1], axis=1)

        # 过滤低置信度检测
        valid_mask = class_scores > 0.3

        if not np.any(valid_mask):
            return []

        valid_scores = class_scores[valid_mask]
        valid_classes = class_indices[valid_mask]
        valid_boxes = boxes[valid_mask]

        # 转换边界框格式（从中心点格式到左上右下格式）
        converted_boxes = []
        for box in valid_boxes:
            cx, cy, w, h = box
            x1 = (cx - w/2) * 640
            y1 = (cy - h/2) * 640
            x2 = (cx + w/2) * 640
            y2 = (cy + h/2) * 640

            # 确保坐标在有效范围内
            x1 = max(0, min(640, x1))
            y1 = max(0, min(640, y1))
            x2 = max(0, min(640, x2))
            y2 = max(0, min(640, y2))

            if x2 > x1 and y2 > y1:
                converted_boxes.append([x1, y1, x2, y2])
            else:
                converted_boxes.append([0, 0, 1, 1])  # 默认框

        # NMS过滤
        filtered_boxes, filtered_scores, filtered_classes = nms_filter(
            converted_boxes, valid_scores.tolist(), valid_classes.tolist())

        # 构建检测结果
        detections = []
        for box, score, class_idx in zip(filtered_boxes, filtered_scores, filtered_classes):
            coco_id, class_name = convert_to_coco_class(class_idx)

            detections.append({
                'bbox': box,
                'confidence': score,
                'class_id': class_idx,
                'coco_id': coco_id,
                'class_name': class_name
            })

        return detections

    except Exception as e:
        return []

def generate_final_report(config, logger, final_results):
    """生成最终训练报告"""
    print("\n📊 生成最终报告...")

    report_path = os.path.join(config['save_dir'], 'training_report.md')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# RT-DETR完整训练报告\n\n")
        f.write(f"**训练时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 训练配置\n\n")
        for key, value in config.items():
            f.write(f"- **{key}**: {value}\n")
        f.write("\n")

        f.write("## 训练结果\n\n")
        if logger.train_losses:
            f.write(f"- **总训练轮数**: {len(logger.train_losses)}\n")
            f.write(f"- **最终训练损失**: {logger.train_losses[-1]:.4f}\n")
            f.write(f"- **最低训练损失**: {min(logger.train_losses):.4f}\n")

        f.write("\n## 验证结果\n\n")
        if final_results:
            f.write(f"- **总检测数**: {final_results['total_detections']}\n")
            f.write(f"- **平均置信度**: {final_results['avg_confidence']:.3f}\n")
            f.write(f"- **检测类别数**: {final_results['class_diversity']}\n")
            f.write(f"- **检测到的类别**: {', '.join(final_results['detected_classes'])}\n")

        f.write("\n## 文件位置\n\n")
        f.write(f"- **训练模型**: {config['model_save_path']}\n")
        f.write(f"- **训练日志**: {config['log_dir']}/training_log.json\n")
        f.write(f"- **验证日志**: {config['log_dir']}/validation_log.json\n")
        f.write(f"- **训练曲线**: {config['log_dir']}/training_curves.png\n")

    print(f"📋 报告已保存到: {report_path}")

if __name__ == "__main__":
    main()
