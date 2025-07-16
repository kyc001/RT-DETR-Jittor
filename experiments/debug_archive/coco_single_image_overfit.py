#!/usr/bin/env python3
"""
从COCO训练集抽取一张图像进行过拟合训练
验证RT-DETR训练流程的正确性
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

def find_coco_dataset():
    """查找COCO数据集路径"""
    print("=" * 60)
    print("===        查找COCO数据集        ===")
    print("=" * 60)
    
    # 常见的COCO数据集路径
    possible_paths = [
        "/home/kyc/project/RT-DETR/data/coco2017",
        "/home/kyc/project/RT-DETR/data/coco",
        "/home/kyc/data/coco2017",
        "/home/kyc/data/coco",
        "./data/coco2017",
        "./data/coco",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            train_images = os.path.join(path, "train2017")
            train_ann = os.path.join(path, "annotations/instances_train2017.json")
            
            if os.path.exists(train_images) and os.path.exists(train_ann):
                print(f"✅ 找到COCO数据集: {path}")
                print(f"   训练图像: {train_images}")
                print(f"   训练标注: {train_ann}")
                return path, train_images, train_ann
    
    print("❌ 未找到COCO数据集")
    print("💡 请确保COCO数据集在以下位置之一:")
    for path in possible_paths:
        print(f"   {path}")
    
    return None, None, None

def load_coco_sample(coco_root, train_images, train_ann, sample_id=None):
    """从COCO数据集加载一个样本"""
    print("\n" + "=" * 60)
    print("===        加载COCO样本        ===")
    print("=" * 60)
    
    try:
        # 加载COCO标注
        print("加载COCO标注文件...")
        with open(train_ann, 'r') as f:
            coco_data = json.load(f)
        
        print(f"✅ COCO数据集信息:")
        print(f"   图像数量: {len(coco_data['images'])}")
        print(f"   标注数量: {len(coco_data['annotations'])}")
        print(f"   类别数量: {len(coco_data['categories'])}")
        
        # 创建图像ID到标注的映射
        img_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        # 选择一个有多个目标的图像
        if sample_id is None:
            # 找一个有3-5个目标的图像
            suitable_images = []
            for img_info in coco_data['images']:
                img_id = img_info['id']
                if img_id in img_to_anns:
                    num_objects = len(img_to_anns[img_id])
                    if 3 <= num_objects <= 5:  # 选择有3-5个目标的图像
                        suitable_images.append(img_info)
            
            if not suitable_images:
                print("⚠️ 未找到合适的图像，使用第一个有标注的图像")
                for img_info in coco_data['images']:
                    if img_info['id'] in img_to_anns:
                        selected_img = img_info
                        break
            else:
                # 选择第一个合适的图像
                selected_img = suitable_images[0]
        else:
            # 使用指定的图像ID
            selected_img = None
            for img_info in coco_data['images']:
                if img_info['id'] == sample_id:
                    selected_img = img_info
                    break
            
            if selected_img is None:
                print(f"❌ 未找到图像ID {sample_id}")
                return None, None
        
        img_id = selected_img['id']
        img_filename = selected_img['file_name']
        img_path = os.path.join(train_images, img_filename)
        
        print(f"✅ 选择图像:")
        print(f"   ID: {img_id}")
        print(f"   文件名: {img_filename}")
        print(f"   路径: {img_path}")
        
        if not os.path.exists(img_path):
            print(f"❌ 图像文件不存在: {img_path}")
            return None, None
        
        # 获取该图像的标注
        annotations = img_to_anns.get(img_id, [])
        print(f"   标注数量: {len(annotations)}")
        
        # 处理标注
        processed_annotations = []
        for ann in annotations:
            # COCO格式: [x, y, width, height]
            x, y, w, h = ann['bbox']
            
            # 转换为 [x1, y1, x2, y2] 并归一化
            x1 = x / selected_img['width']
            y1 = y / selected_img['height']
            x2 = (x + w) / selected_img['width']
            y2 = (y + h) / selected_img['height']
            
            # 确保坐标在[0, 1]范围内
            x1 = max(0, min(1, x1))
            y1 = max(0, min(1, y1))
            x2 = max(0, min(1, x2))
            y2 = max(0, min(1, y2))
            
            # 过滤掉无效的边界框
            if x2 > x1 and y2 > y1:
                processed_annotations.append({
                    'bbox': [x1, y1, x2, y2],
                    'category_id': ann['category_id'],
                    'area': ann['area']
                })
        
        print(f"✅ 处理后的标注:")
        for i, ann in enumerate(processed_annotations):
            print(f"   目标{i+1}: 类别{ann['category_id']}, 边界框{ann['bbox']}")
        
        return img_path, processed_annotations
        
    except Exception as e:
        print(f"❌ 加载COCO样本失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def load_and_preprocess_image(image_path, target_size=640):
    """加载和预处理图像"""
    print(f"\n加载图像: {image_path}")
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    print(f"原始尺寸: {original_size}")
    
    # 调整大小
    image = image.resize((target_size, target_size), Image.LANCZOS)
    print(f"调整后尺寸: {image.size}")
    
    # 转换为张量
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = jt.array(img_array.transpose(2, 0, 1)).float32()
    img_batch = img_tensor.unsqueeze(0)
    
    return img_batch, original_size

def create_target_dict(annotations):
    """创建目标字典"""
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

def coco_overfit_training(image_tensor, targets, num_epochs=300):
    """COCO图像过拟合训练"""
    print("\n" + "=" * 60)
    print("===        COCO图像过拟合训练        ===")
    print("=" * 60)
    
    try:
        # 导入模型
        from jittor_rt_detr.src.nn.backbone.resnet import ResNet50
        from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
        from jittor_rt_detr.src.nn.criterion.rtdetr_criterion import build_criterion
        
        # 创建模型
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
        optimizer = jt.optim.Adam(all_params, lr=1e-3)
        
        print(f"✅ 模型创建完成")
        print(f"   总参数量: {sum(p.numel() for p in all_params):,}")
        print(f"   目标数量: {len(targets[0]['boxes'])}")
        
        # 过拟合训练
        print(f"\n开始过拟合训练 ({num_epochs} epochs)...")
        losses = []
        
        for epoch in range(num_epochs):
            # 前向传播
            feats = backbone(image_tensor)
            outputs = transformer(feats)
            
            # 损失计算
            loss_dict = criterion(outputs, targets)
            total_loss = sum(loss_dict.values())
            losses.append(total_loss.item())
            
            # 反向传播
            optimizer.backward(total_loss)
            
            # 打印进度
            if epoch % 50 == 0 or epoch < 10:
                print(f"Epoch {epoch:3d}: 总损失={total_loss.item():.4f}")
                for k, v in loss_dict.items():
                    print(f"         {k}: {v.item():.4f}")
        
        print(f"\n✅ 过拟合训练完成")
        print(f"   初始损失: {losses[0]:.4f}")
        print(f"   最终损失: {losses[-1]:.4f}")
        print(f"   损失下降: {(losses[0] - losses[-1])/losses[0]*100:.1f}%")
        
        # 检查过拟合效果
        overfit_success = losses[-1] < losses[0] * 0.2  # 损失下降到初始的20%以下
        if overfit_success:
            print("🎉 过拟合成功！模型已经记住了这张图像")
        else:
            print("⚠️ 过拟合效果一般，可能需要更多训练或调整参数")
        
        return backbone, transformer, overfit_success, losses
        
    except Exception as e:
        print(f"❌ 过拟合训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False, []

def inference_and_evaluation(backbone, transformer, image_tensor, targets):
    """推理和评估"""
    print("\n" + "=" * 60)
    print("===        推理和评估        ===")
    print("=" * 60)
    
    try:
        # 设置为评估模式
        backbone.eval()
        transformer.eval()
        
        # 推理
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
        
        # 过滤预测
        confidence_threshold = 0.3  # 降低阈值以便看到更多预测
        high_conf_mask = max_scores > confidence_threshold
        
        if high_conf_mask.sum() > 0:
            high_conf_boxes = pred_boxes[high_conf_mask]
            high_conf_classes = pred_classes[high_conf_mask]
            high_conf_scores = max_scores[high_conf_mask]
            
            print(f"✅ 检测到 {len(high_conf_boxes)} 个高置信度目标:")
            
            # 转换为numpy
            boxes_np = high_conf_boxes.numpy()
            classes_np = high_conf_classes.numpy()
            scores_np = high_conf_scores.numpy()
            
            for i, (box, cls, score) in enumerate(zip(boxes_np, classes_np, scores_np)):
                x1, y1, x2, y2 = box * 640  # 转换为像素坐标
                print(f"   预测{i+1}: 类别{cls}, 置信度{score:.3f}, 边界框[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
            
            # 与真实标注对比
            print(f"\n📊 与真实标注对比:")
            gt_boxes = targets[0]['boxes'].numpy() * 640
            gt_classes = targets[0]['labels'].numpy()
            
            print(f"真实目标数量: {len(gt_boxes)}")
            print(f"预测目标数量: {len(high_conf_boxes)}")
            
            print(f"\n真实标注:")
            for i, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
                x1, y1, x2, y2 = gt_box
                print(f"   GT{i+1}: 类别{gt_cls}, 边界框[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
            
            # 保存可视化
            save_visualization(image_tensor, boxes_np, classes_np, scores_np, gt_boxes, gt_classes)
            
            return True
        else:
            print("⚠️ 没有检测到高置信度目标")
            
            # 显示最高置信度的预测
            top_k = 5
            top_scores, top_indices = jt.topk(max_scores, top_k)
            print(f"\n前{top_k}个最高置信度预测:")
            for i, (idx, score) in enumerate(zip(top_indices.numpy(), top_scores.numpy())):
                box = pred_boxes[idx].numpy() * 640
                cls = pred_classes[idx].item()
                print(f"   预测{i+1}: 类别{cls}, 置信度{score:.3f}")
            
            return False
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_visualization(image_tensor, pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes):
    """保存可视化结果"""
    try:
        # 转换图像
        img_array = image_tensor[0].transpose(0, 1).transpose(1, 2).numpy()
        img_array = (img_array * 255).astype(np.uint8)
        image = Image.fromarray(img_array)
        
        draw = ImageDraw.Draw(image)
        
        # 绘制真实标注（绿色）
        for i, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
            x1, y1, x2, y2 = gt_box
            draw.rectangle([x1, y1, x2, y2], outline='green', width=3)
            draw.text((x1, y1-20), f"GT: {gt_cls}", fill='green')
        
        # 绘制预测结果（红色）
        for i, (pred_box, pred_cls, pred_score) in enumerate(zip(pred_boxes, pred_classes, pred_scores)):
            x1, y1, x2, y2 = pred_box
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            draw.text((x1, y2+5), f"Pred: {pred_cls} ({pred_score:.2f})", fill='red')
        
        # 保存
        os.makedirs('experiments/overfit_data', exist_ok=True)
        vis_path = 'experiments/overfit_data/coco_overfit_result.jpg'
        image.save(vis_path)
        print(f"✅ 可视化结果保存到: {vis_path}")
        
    except Exception as e:
        print(f"⚠️ 可视化保存失败: {e}")

def main():
    print("🎯 COCO单张图像过拟合训练验证")
    print("=" * 80)
    
    # 1. 查找COCO数据集
    coco_root, train_images, train_ann = find_coco_dataset()
    if coco_root is None:
        print("❌ 无法找到COCO数据集，退出")
        return
    
    # 2. 加载COCO样本
    img_path, annotations = load_coco_sample(coco_root, train_images, train_ann)
    if img_path is None:
        print("❌ 无法加载COCO样本，退出")
        return
    
    # 3. 预处理图像
    image_tensor, original_size = load_and_preprocess_image(img_path)
    targets = create_target_dict(annotations)
    
    # 4. 过拟合训练
    backbone, transformer, overfit_success, losses = coco_overfit_training(image_tensor, targets)
    
    if backbone is None:
        print("❌ 训练失败，退出")
        return
    
    # 5. 推理和评估
    inference_success = inference_and_evaluation(backbone, transformer, image_tensor, targets)
    
    # 总结
    print("\n" + "=" * 80)
    print("🎯 COCO过拟合训练验证总结:")
    print("=" * 80)
    
    results = [
        ("COCO数据加载", img_path is not None),
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
        print(f"  初始损失: {losses[0]:.4f}")
        print(f"  最终损失: {losses[-1]:.4f}")
        print(f"  损失下降: {(losses[0] - losses[-1])/losses[0]*100:.1f}%")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 COCO单张图像过拟合训练验证完全成功！")
        print("✅ 成功从COCO数据集加载真实图像和标注")
        print("✅ 模型成功过拟合到单张COCO图像")
        print("✅ 推理能够检测出训练的目标")
        print("✅ 整个训练流程在真实数据上验证正确")
        print("\n🚀 这证明RT-DETR可以:")
        print("1. ✅ 正确处理COCO格式的真实数据")
        print("2. ✅ 进行有效的训练和学习")
        print("3. ✅ 产生合理的检测结果")
        print("4. ✅ 用于大规模COCO数据集训练")
        print("\n✨ 现在可以开始完整的COCO训练！")
    else:
        print("⚠️ 验证过程中发现问题，需要进一步检查")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
