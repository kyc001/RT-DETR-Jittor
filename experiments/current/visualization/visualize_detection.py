#!/usr/bin/env python3
"""
RT-DETR推理结果可视化工具
可视化模型的检测结果，包括边界框、类别标签和置信度
"""

import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
import jittor as jt

# 添加项目路径
sys.path.append('/home/kyc/project/RT-DETR')

# 使用与sanity check相同的导入方式
from jittor_rt_detr.src.nn.backbone.resnet import ResNet
from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
from jittor_rt_detr.src.nn.criterion import SetCriterion

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

def load_model():
    """加载RT-DETR模型"""
    print("🔄 加载RT-DETR模型...")

    # 创建backbone
    backbone = ResNet(depth=18, variant='d', return_idx=[1, 2, 3], act='relu', freeze_at=-1)

    # 创建transformer
    transformer = RTDETRTransformer(
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        position_embed_type='sine',
        feat_channels=[128, 256, 512],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_decoder_layers=6,
        pe_temperature=10000,
        eval_spatial_size=None,
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0
    )

    # 设置为评估模式
    backbone.eval()
    transformer.eval()

    print("✅ 模型加载成功")
    return backbone, transformer

def load_and_preprocess_image(image_path, target_size=640):
    """加载和预处理图像"""
    print(f"🔄 加载图像: {image_path}")
    
    # 加载原始图像
    original_image = Image.open(image_path).convert('RGB')
    original_width, original_height = original_image.size
    
    # 调整大小
    resized_image = original_image.resize((target_size, target_size), Image.LANCZOS)
    
    # 转换为张量
    img_array = np.array(resized_image).astype(np.float32) / 255.0
    img_tensor = jt.array(img_array.transpose(2, 0, 1)).float32().unsqueeze(0)
    
    print(f"✅ 图像预处理完成: {original_width}x{original_height} -> {target_size}x{target_size}")
    
    return original_image, img_tensor, (original_width, original_height)

def run_inference(backbone, transformer, img_tensor, conf_threshold=0.3):
    """运行推理"""
    print("🔄 运行推理...")

    with jt.no_grad():
        # 前向传播
        feats = backbone(img_tensor)
        outputs = transformer(feats)
        
        # 获取预测结果
        pred_logits = outputs['pred_logits'][0]  # [300, 80]
        pred_boxes = outputs['pred_boxes'][0]    # [300, 4]
        
        # 计算置信度分数
        pred_scores = jt.nn.softmax(pred_logits, dim=-1)
        pred_scores_no_bg = pred_scores[:, :-1]  # 移除背景类
        
        # 获取最大分数和对应类别
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
        
        # 过滤低置信度预测
        valid_mask = scores_np > conf_threshold
        valid_scores = scores_np[valid_mask]
        valid_classes = classes_np[valid_mask]
        valid_boxes = boxes_np[valid_mask]
        
        print(f"✅ 推理完成，找到 {len(valid_scores)} 个高置信度检测结果")
        
        return valid_scores, valid_classes, valid_boxes

def convert_to_coco_class(class_idx):
    """将模型类别索引转换为COCO类别ID和名称"""
    if class_idx == 0:
        coco_id = 1  # person
    elif class_idx == 35:
        coco_id = 37  # sports ball (COCO中sports ball是37号)
    else:
        coco_id = class_idx + 1

    class_name = COCO_CLASSES.get(coco_id, f'class_{coco_id}')
    return coco_id, class_name

def visualize_detections(original_image, scores, classes, boxes, original_size, save_path=None):
    """可视化检测结果"""
    print("🔄 生成可视化结果...")
    
    # 创建matplotlib图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 左侧：原始图像
    ax1.imshow(original_image)
    ax1.set_title('原始图像', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # 右侧：检测结果
    ax2.imshow(original_image)
    ax2.set_title(f'检测结果 ({len(scores)} 个目标)', fontsize=16, fontweight='bold')
    ax2.axis('off')
    
    # 颜色映射
    color_map = plt.cm.Set3(np.linspace(0, 1, 12))
    
    # 绘制检测框
    original_width, original_height = original_size
    
    for i, (score, class_idx, box) in enumerate(zip(scores, classes, boxes)):
        # 转换坐标到原始图像尺寸
        x1, y1, x2, y2 = box
        x1 = x1 * original_width
        y1 = y1 * original_height
        x2 = x2 * original_width
        y2 = y2 * original_height
        
        # 获取类别信息
        coco_id, class_name = convert_to_coco_class(class_idx)
        
        # 选择颜色
        color = color_map[i % len(color_map)]
        
        # 绘制边界框
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3, edgecolor=color, facecolor='none'
        )
        ax2.add_patch(rect)
        
        # 添加标签
        label = f'{class_name}\n{score:.3f}'
        ax2.text(
            x1, y1 - 10, label,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8),
            fontsize=12, fontweight='bold', color='black'
        )
        
        print(f"   检测 {i+1}: {class_name} (COCO ID: {coco_id}), 置信度: {score:.3f}, 位置: [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
    
    plt.tight_layout()
    
    # 保存结果
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 可视化结果已保存到: {save_path}")
    
    plt.show()
    
    return fig

def train_and_visualize():
    """训练模型并可视化结果"""
    print("🎯 RT-DETR训练后推理可视化")
    print("=" * 60)

    # 配置
    image_path = "/home/kyc/project/RT-DETR/data/coco2017_50/train2017/000000036539.jpg"
    output_path = "/home/kyc/project/RT-DETR/experiments/trained_detection_result.png"
    conf_threshold = 0.2  # 降低阈值，因为训练后的模型可能置信度不同

    try:
        # 1. 加载模型
        backbone, transformer = load_model()

        # 2. 加载和预处理图像
        original_image, img_tensor, original_size = load_and_preprocess_image(image_path)

        # 3. 简单训练（类似sanity check中的训练）
        print("🔄 进行简单的过拟合训练...")

        # 加载真实标注用于训练
        annotation_path = "/home/kyc/project/RT-DETR/data/coco2017_50/annotations/instances_train2017.json"
        with open(annotation_path, 'r') as f:
            coco_data = json.load(f)

        # 构建训练目标
        image_id = 36539
        annotations = []
        labels = []

        for ann in coco_data['annotations']:
            if ann['image_id'] == image_id:
                x, y, w, h = ann['bbox']
                category_id = ann['category_id']

                # 归一化坐标
                x1, y1 = x / 427, y / 640  # 原始图像尺寸
                x2, y2 = (x + w) / 427, (y + h) / 640

                if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= 1 and y2 <= 1:
                    annotations.append([x1, y1, x2, y2])
                    # 映射COCO类别到模型索引
                    if category_id == 1:  # person
                        labels.append(0)
                    elif category_id == 37:  # sports ball
                        labels.append(35)
                    else:
                        labels.append(category_id - 1)

        if annotations:
            targets = [{
                'boxes': jt.array(annotations, dtype=jt.float32),
                'labels': jt.array(labels, dtype=jt.int64)
            }]

            # 简单训练
            criterion = SetCriterion(
                num_classes=80,
                matcher=None,
                weight_dict={'loss_focal': 1, 'loss_bbox': 5, 'loss_giou': 2},
                losses=['focal', 'boxes']
            )

            backbone.train()
            transformer.train()
            all_params = list(backbone.parameters()) + list(transformer.parameters())
            trainable_params = [p for p in all_params if p.requires_grad]
            optimizer = jt.optim.Adam(trainable_params, lr=1e-3)

            print(f"   训练参数数量: {len(trainable_params)}")

            # 训练20个epoch
            for epoch in range(20):
                feats = backbone(img_tensor)
                outputs = transformer(feats)
                loss_dict = criterion(outputs, targets)
                total_loss = sum(loss_dict.values())

                optimizer.step(total_loss)

                if epoch % 5 == 0:
                    print(f"   Epoch {epoch}: 损失={total_loss.numpy().item():.4f}")

            print("✅ 训练完成")

        # 4. 切换到评估模式并运行推理
        backbone.eval()
        transformer.eval()
        scores, classes, boxes = run_inference(backbone, transformer, img_tensor, conf_threshold)

        # 5. 可视化结果
        if len(scores) > 0:
            fig = visualize_detections(original_image, scores, classes, boxes, original_size, output_path)
            print("\n🎉 训练后可视化完成！")
        else:
            print(f"⚠️ 没有找到置信度大于 {conf_threshold} 的检测结果")

    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🎯 RT-DETR推理结果可视化")
    print("=" * 60)

    # 配置
    image_path = "/home/kyc/project/RT-DETR/data/coco2017_50/train2017/000000036539.jpg"
    output_path = "/home/kyc/project/RT-DETR/experiments/detection_result.png"
    conf_threshold = 0.3

    try:
        # 1. 加载模型
        backbone, transformer = load_model()

        # 2. 加载和预处理图像
        original_image, img_tensor, original_size = load_and_preprocess_image(image_path)

        # 3. 运行推理
        scores, classes, boxes = run_inference(backbone, transformer, img_tensor, conf_threshold)

        # 4. 可视化结果
        if len(scores) > 0:
            fig = visualize_detections(original_image, scores, classes, boxes, original_size, output_path)
            print("\n🎉 可视化完成！")
        else:
            print(f"⚠️ 没有找到置信度大于 {conf_threshold} 的检测结果")

    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # 训练后可视化
        train_and_visualize()
    else:
        # 直接推理可视化
        main()
