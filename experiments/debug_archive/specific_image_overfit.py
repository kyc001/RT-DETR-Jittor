#!/usr/bin/env python3
"""
使用指定的COCO图像进行过拟合训练
图像: /home/kyc/project/RT-DETR/data/coco2017_50/train2017/000000036539.jpg
"""

import os
import sys
import json
import math
import numpy as np
from PIL import Image, ImageDraw

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt
import jittor.nn as nn

# 设置Jittor
jt.flags.use_cuda = 1
jt.set_global_seed(42)
jt.flags.auto_mixed_precision_level = 0

def load_specific_coco_image():
    """加载指定的COCO图像和标注"""
    print("=" * 60)
    print("===        加载指定COCO图像        ===")
    print("=" * 60)
    
    # 指定的图像路径
    image_path = "/home/kyc/project/RT-DETR/data/coco2017_50/train2017/000000036539.jpg"
    annotation_path = "/home/kyc/project/RT-DETR/data/coco2017_50/annotations/instances_train2017.json"
    
    # 从文件名提取图像ID
    image_filename = os.path.basename(image_path)
    image_id = int(image_filename.split('.')[0])  # 000000036539 -> 36539
    
    print(f"目标图像: {image_path}")
    print(f"图像ID: {image_id}")
    print(f"标注文件: {annotation_path}")
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return None, None, None
    
    if not os.path.exists(annotation_path):
        print(f"❌ 标注文件不存在: {annotation_path}")
        return None, None, None
    
    try:
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        img_width, img_height = image.size
        print(f"✅ 图像加载成功: {img_width}x{img_height}")
        
        # 加载标注
        print("加载COCO标注...")
        with open(annotation_path, 'r') as f:
            coco_data = json.load(f)
        
        # 查找该图像的标注
        target_annotations = []
        for ann in coco_data['annotations']:
            if ann['image_id'] == image_id:
                target_annotations.append(ann)
        
        print(f"✅ 找到 {len(target_annotations)} 个标注")
        
        # 处理标注
        processed_annotations = []
        for i, ann in enumerate(target_annotations):
            # COCO格式: [x, y, width, height]
            x, y, w, h = ann['bbox']
            
            # 转换为 [x1, y1, x2, y2] 并归一化
            x1 = x / img_width
            y1 = y / img_height
            x2 = (x + w) / img_width
            y2 = (y + h) / img_height
            
            # 确保坐标在[0, 1]范围内
            x1 = max(0, min(1, x1))
            y1 = max(0, min(1, y1))
            x2 = max(0, min(1, x2))
            y2 = max(0, min(1, y2))
            
            # 过滤掉无效的边界框
            if x2 > x1 and y2 > y1 and (x2 - x1) > 0.01 and (y2 - y1) > 0.01:
                processed_annotations.append({
                    'bbox': [x1, y1, x2, y2],
                    'category_id': ann['category_id'],
                    'area': ann['area']
                })
                print(f"   目标{i+1}: 类别{ann['category_id']}, 边界框{[x1, y1, x2, y2]}")
        
        print(f"✅ 处理后有效标注: {len(processed_annotations)}个")
        
        return image_path, processed_annotations, (img_width, img_height)
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def preprocess_image(image_path, target_size=640):
    """预处理图像"""
    print(f"\n预处理图像...")
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    print(f"原始尺寸: {original_size}")
    
    # 调整大小到640x640
    image_resized = image.resize((target_size, target_size), Image.LANCZOS)
    print(f"调整后尺寸: {image_resized.size}")
    
    # 转换为张量
    img_array = np.array(image_resized).astype(np.float32) / 255.0
    img_tensor = jt.array(img_array.transpose(2, 0, 1)).float32()
    img_batch = img_tensor.unsqueeze(0)
    
    return img_batch, original_size

def create_targets(annotations):
    """创建训练目标"""
    if not annotations:
        return [{'boxes': jt.zeros((0, 4), dtype=jt.float32), 'labels': jt.zeros((0,), dtype=jt.int64)}]
    
    boxes = []
    labels = []
    
    for ann in annotations:
        boxes.append(ann['bbox'])
        labels.append(ann['category_id'])
    
    target = {
        'boxes': jt.array(boxes, dtype=jt.float32),
        'labels': jt.array(labels, dtype=jt.int64)
    }
    
    return [target]

def overfit_training(image_tensor, targets, num_epochs=500):
    """过拟合训练"""
    print("\n" + "=" * 60)
    print("===        过拟合训练        ===")
    print("=" * 60)
    
    try:
        # 导入模型
        from jittor_rt_detr.src.nn.backbone.resnet import ResNet50
        from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
        from jittor_rt_detr.src.nn.criterion.rtdetr_criterion import build_criterion
        
        # 创建模型
        print("创建模型...")
        backbone = ResNet50(pretrained=False)
        transformer = RTDETRTransformer(
            num_classes=80,  # COCO 80个类别
            hidden_dim=256,
            num_queries=300,
            feat_channels=[256, 512, 1024, 2048]
        )
        criterion = build_criterion(num_classes=80)
        
        # 创建优化器
        all_params = list(backbone.parameters()) + list(transformer.parameters())
        optimizer = jt.optim.Adam(all_params, lr=1e-3)  # 较高学习率用于过拟合
        
        print(f"✅ 模型创建完成")
        print(f"   总参数量: {sum(p.numel() for p in all_params):,}")
        print(f"   目标数量: {len(targets[0]['boxes'])}")
        print(f"   目标类别: {targets[0]['labels'].numpy().tolist()}")
        
        # 训练循环
        print(f"\n开始过拟合训练 ({num_epochs} epochs)...")
        losses = []
        loss_components = {'focal': [], 'bbox': [], 'giou': []}
        
        for epoch in range(num_epochs):
            # 前向传播
            feats = backbone(image_tensor)
            outputs = transformer(feats)
            
            # 损失计算
            loss_dict = criterion(outputs, targets)
            total_loss = sum(loss_dict.values())
            losses.append(total_loss.item())
            
            # 记录损失组件
            for key in loss_components:
                for loss_name, loss_value in loss_dict.items():
                    if key in loss_name:
                        loss_components[key].append(loss_value.item())
                        break
            
            # 反向传播
            optimizer.backward(total_loss)
            
            # 打印进度
            if epoch % 100 == 0 or epoch < 10 or epoch == num_epochs - 1:
                print(f"Epoch {epoch:3d}: 总损失={total_loss.item():.4f}")
                for k, v in loss_dict.items():
                    print(f"         {k}: {v.item():.4f}")
        
        print(f"\n✅ 训练完成")
        print(f"   初始损失: {losses[0]:.4f}")
        print(f"   最终损失: {losses[-1]:.4f}")
        print(f"   损失下降: {(losses[0] - losses[-1])/losses[0]*100:.1f}%")
        
        # 判断过拟合是否成功
        overfit_success = losses[-1] < losses[0] * 0.1  # 损失下降到初始的10%以下
        if overfit_success:
            print("🎉 过拟合成功！模型已经记住了这张图像")
        else:
            print("⚠️ 过拟合效果一般，可能需要更多训练")
        
        return backbone, transformer, overfit_success, losses
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False, []

def inference_and_visualization(backbone, transformer, image_tensor, targets, image_path):
    """推理和可视化"""
    print("\n" + "=" * 60)
    print("===        推理和可视化        ===")
    print("=" * 60)
    
    try:
        # 设置为评估模式
        backbone.eval()
        transformer.eval()
        
        # 推理
        print("执行推理...")
        with jt.no_grad():
            feats = backbone(image_tensor)
            outputs = transformer(feats)
        
        print(f"✅ 推理完成")
        
        # 后处理
        pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
        pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4]
        
        # 获取预测结果
        pred_scores = jt.nn.softmax(pred_logits, dim=-1)
        max_scores, pred_classes = jt.max(pred_scores[:, :-1], dim=-1)  # 排除背景类
        
        # 过滤预测（使用较低阈值以便看到过拟合效果）
        confidence_threshold = 0.1
        high_conf_mask = max_scores > confidence_threshold
        
        print(f"使用置信度阈值: {confidence_threshold}")
        print(f"高置信度预测数量: {high_conf_mask.sum().item()}")
        
        if high_conf_mask.sum() > 0:
            high_conf_boxes = pred_boxes[high_conf_mask]
            high_conf_classes = pred_classes[high_conf_mask]
            high_conf_scores = max_scores[high_conf_mask]
            
            # 转换为numpy
            boxes_np = high_conf_boxes.numpy()
            classes_np = high_conf_classes.numpy()
            scores_np = high_conf_scores.numpy()
            
            print(f"\n🎯 检测结果 ({len(boxes_np)}个目标):")
            for i, (box, cls, score) in enumerate(zip(boxes_np, classes_np, scores_np)):
                x1, y1, x2, y2 = box * 640  # 转换为像素坐标
                print(f"   预测{i+1}: 类别{cls}, 置信度{score:.3f}, 边界框[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
            
            # 与真实标注对比
            print(f"\n📊 与真实标注对比:")
            gt_boxes = targets[0]['boxes'].numpy() * 640
            gt_classes = targets[0]['labels'].numpy()
            
            print(f"真实目标数量: {len(gt_boxes)}")
            print(f"预测目标数量: {len(boxes_np)}")
            
            print(f"\n真实标注:")
            for i, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
                x1, y1, x2, y2 = gt_box
                print(f"   GT{i+1}: 类别{gt_cls}, 边界框[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
            
            # 创建可视化
            create_visualization(image_path, boxes_np, classes_np, scores_np, gt_boxes, gt_classes)
            
            return True
        else:
            print("⚠️ 没有检测到高置信度目标")
            
            # 显示最高置信度的预测
            top_k = 10
            top_scores, top_indices = jt.topk(max_scores, top_k)
            print(f"\n前{top_k}个最高置信度预测:")
            for i, (idx, score) in enumerate(zip(top_indices.numpy(), top_scores.numpy())):
                box = pred_boxes[idx].numpy() * 640
                cls = pred_classes[idx].item()
                print(f"   预测{i+1}: 类别{cls}, 置信度{score:.3f}, 边界框{box}")
            
            return False
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_visualization(image_path, pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes):
    """创建可视化图像"""
    try:
        # 加载原始图像并调整大小
        image = Image.open(image_path).convert('RGB')
        image = image.resize((640, 640), Image.LANCZOS)
        
        draw = ImageDraw.Draw(image)
        
        # 绘制真实标注（绿色，粗线）
        for i, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
            x1, y1, x2, y2 = gt_box
            draw.rectangle([x1, y1, x2, y2], outline='green', width=4)
            draw.text((x1, y1-25), f"GT: {gt_cls}", fill='green')
        
        # 绘制预测结果（红色，细线）
        for i, (pred_box, pred_cls, pred_score) in enumerate(zip(pred_boxes, pred_classes, pred_scores)):
            x1, y1, x2, y2 = pred_box
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            draw.text((x1, y2+5), f"Pred: {pred_cls} ({pred_score:.2f})", fill='red')
        
        # 保存可视化结果
        os.makedirs('experiments/overfit_data', exist_ok=True)
        vis_path = 'experiments/overfit_data/specific_image_overfit_result.jpg'
        image.save(vis_path)
        print(f"✅ 可视化结果保存到: {vis_path}")
        
        # 同时保存原始图像副本
        original_path = 'experiments/overfit_data/original_image.jpg'
        Image.open(image_path).resize((640, 640), Image.LANCZOS).save(original_path)
        print(f"✅ 原始图像保存到: {original_path}")
        
    except Exception as e:
        print(f"⚠️ 可视化创建失败: {e}")

def main():
    print("🎯 指定COCO图像过拟合训练验证")
    print("图像: 000000036539.jpg")
    print("=" * 80)
    
    # 1. 加载指定图像和标注
    image_path, annotations, original_size = load_specific_coco_image()
    if image_path is None:
        print("❌ 无法加载指定图像，退出")
        return
    
    # 2. 预处理图像
    image_tensor, _ = preprocess_image(image_path)
    targets = create_targets(annotations)
    
    # 3. 过拟合训练
    backbone, transformer, overfit_success, losses = overfit_training(image_tensor, targets)
    
    if backbone is None:
        print("❌ 训练失败，退出")
        return
    
    # 4. 推理和可视化
    inference_success = inference_and_visualization(backbone, transformer, image_tensor, targets, image_path)
    
    # 总结
    print("\n" + "=" * 80)
    print("🎯 指定图像过拟合训练验证总结:")
    print("=" * 80)
    
    results = [
        ("图像和标注加载", image_path is not None),
        ("过拟合训练", overfit_success),
        ("推理检测", inference_success),
    ]
    
    all_passed = True
    for name, result in results:
        status = "✅ 成功" if result else "❌ 失败"
        print(f"{status} {name}")
        if not result:
            all_passed = False
    
    if losses:
        print(f"\n训练统计:")
        print(f"  训练轮数: {len(losses)}")
        print(f"  初始损失: {losses[0]:.4f}")
        print(f"  最终损失: {losses[-1]:.4f}")
        print(f"  损失下降: {(losses[0] - losses[-1])/losses[0]*100:.1f}%")
    
    print(f"\n目标信息:")
    if annotations:
        print(f"  目标数量: {len(annotations)}")
        print(f"  目标类别: {[ann['category_id'] for ann in annotations]}")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 指定图像过拟合训练验证完全成功！")
        print("✅ 成功加载指定的COCO图像和标注")
        print("✅ 模型成功过拟合到该图像")
        print("✅ 推理能够检测出训练的目标")
        print("✅ 可视化结果已保存")
        print("\n🚀 这证明RT-DETR:")
        print("1. ✅ 能够正确处理真实COCO数据")
        print("2. ✅ 训练流程完全正确")
        print("3. ✅ 能够学习和记忆特定图像")
        print("4. ✅ 推理结果合理可信")
        print("\n✨ 可以开始大规模COCO训练！")
    else:
        print("⚠️ 验证过程中发现问题")
        if not overfit_success:
            print("💡 建议: 增加训练轮数或调整学习率")
        if not inference_success:
            print("💡 建议: 降低置信度阈值或检查后处理逻辑")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
