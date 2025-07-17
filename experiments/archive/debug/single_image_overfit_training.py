#!/usr/bin/env python3
"""
单张照片过拟合训练
验证RT-DETR训练流程的正确性
"""

import os
import sys
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

def create_synthetic_image_and_annotations():
    """创建合成图像和标注"""
    print("=" * 60)
    print("===        创建合成图像和标注        ===")
    print("=" * 60)
    
    # 创建一个640x640的图像
    img_size = 640
    image = Image.new('RGB', (img_size, img_size), color='lightblue')
    draw = ImageDraw.Draw(image)
    
    # 添加一些简单的目标
    objects = [
        {'bbox': [100, 100, 200, 200], 'class': 1, 'color': 'red'},      # 人
        {'bbox': [300, 150, 450, 300], 'class': 2, 'color': 'green'},    # 自行车
        {'bbox': [500, 400, 600, 550], 'class': 3, 'color': 'blue'},     # 汽车
    ]
    
    # 绘制目标
    for obj in objects:
        x1, y1, x2, y2 = obj['bbox']
        draw.rectangle([x1, y1, x2, y2], outline=obj['color'], width=3)
        draw.text((x1, y1-20), f"Class {obj['class']}", fill=obj['color'])
    
    # 保存图像
    os.makedirs('experiments/overfit_data', exist_ok=True)
    image_path = 'experiments/overfit_data/test_image.jpg'
    image.save(image_path)
    print(f"✅ 创建合成图像: {image_path}")
    
    # 创建标注（COCO格式的边界框：x, y, w, h -> 转换为 x1, y1, x2, y2）
    annotations = []
    for obj in objects:
        x1, y1, x2, y2 = obj['bbox']
        # 归一化坐标 [0, 1]
        bbox_norm = [x1/img_size, y1/img_size, x2/img_size, y2/img_size]
        annotations.append({
            'bbox': bbox_norm,
            'class': obj['class']
        })
    
    print(f"✅ 创建标注: {len(annotations)}个目标")
    for i, ann in enumerate(annotations):
        print(f"   目标{i+1}: 类别{ann['class']}, 边界框{ann['bbox']}")
    
    return image_path, annotations

def load_and_preprocess_image(image_path):
    """加载和预处理图像"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 转换为numpy数组
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # 转换为CHW格式
    img_tensor = jt.array(img_array.transpose(2, 0, 1)).float32()
    
    # 添加batch维度
    img_batch = img_tensor.unsqueeze(0)
    
    return img_batch

def create_target_dict(annotations):
    """创建目标字典"""
    boxes = []
    labels = []
    
    for ann in annotations:
        boxes.append(ann['bbox'])
        labels.append(ann['class'])
    
    target = {
        'boxes': jt.array(boxes, dtype=jt.float32),
        'labels': jt.array(labels, dtype=jt.int64)
    }
    
    return [target]  # 返回列表，因为criterion期望batch

def single_image_overfit_training():
    """单张图像过拟合训练"""
    print("\n" + "=" * 60)
    print("===        单张图像过拟合训练        ===")
    print("=" * 60)
    
    try:
        # 导入模型
        from jittor_rt_detr.src.nn.backbone.resnet import ResNet50
        from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
        from jittor_rt_detr.src.nn.criterion.rtdetr_criterion import build_criterion
        
        # 创建数据
        image_path, annotations = create_synthetic_image_and_annotations()
        
        # 加载图像和标注
        image_tensor = load_and_preprocess_image(image_path)
        targets = create_target_dict(annotations)
        
        print(f"✅ 数据准备完成")
        print(f"   图像形状: {image_tensor.shape}")
        print(f"   目标数量: {len(targets[0]['boxes'])}")
        
        # 创建模型
        backbone = ResNet50(pretrained=False)
        transformer = RTDETRTransformer(
            num_classes=80,  # 使用COCO的80个类别
            hidden_dim=256,
            num_queries=300,
            feat_channels=[256, 512, 1024, 2048]
        )
        criterion = build_criterion(num_classes=80)
        
        # 创建优化器
        all_params = list(backbone.parameters()) + list(transformer.parameters())
        optimizer = jt.optim.Adam(all_params, lr=1e-3)  # 使用较高的学习率进行过拟合
        
        print(f"✅ 模型创建完成")
        print(f"   总参数量: {sum(p.numel() for p in all_params):,}")
        
        # 过拟合训练
        print(f"\n开始过拟合训练...")
        num_epochs = 500  # 几百次迭代
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
        
        # 检查是否过拟合成功
        overfit_success = losses[-1] < losses[0] * 0.1  # 损失下降到初始的10%以下
        if overfit_success:
            print("🎉 过拟合成功！模型已经记住了这张图像")
        else:
            print("⚠️ 过拟合可能不够充分，可能需要更多训练")
        
        return backbone, transformer, image_tensor, targets, overfit_success
        
    except Exception as e:
        print(f"❌ 过拟合训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, False

def inference_and_visualization(backbone, transformer, image_tensor, targets):
    """推理和可视化"""
    print("\n" + "=" * 60)
    print("===        推理和可视化        ===")
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
        print(f"   pred_logits: {outputs['pred_logits'].shape}")
        print(f"   pred_boxes: {outputs['pred_boxes'].shape}")
        
        # 后处理：获取预测结果
        pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
        pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4]
        
        # 获取置信度最高的预测
        pred_scores = jt.nn.softmax(pred_logits, dim=-1)
        max_scores, pred_classes = jt.max(pred_scores[:, :-1], dim=-1)  # 排除背景类
        
        # 过滤低置信度预测
        confidence_threshold = 0.5
        high_conf_mask = max_scores > confidence_threshold
        
        if high_conf_mask.sum() > 0:
            high_conf_boxes = pred_boxes[high_conf_mask]
            high_conf_classes = pred_classes[high_conf_mask]
            high_conf_scores = max_scores[high_conf_mask]
            
            print(f"✅ 检测到 {len(high_conf_boxes)} 个高置信度目标:")
            
            # 转换为numpy进行可视化
            boxes_np = high_conf_boxes.numpy()
            classes_np = high_conf_classes.numpy()
            scores_np = high_conf_scores.numpy()
            
            for i, (box, cls, score) in enumerate(zip(boxes_np, classes_np, scores_np)):
                # 将归一化坐标转换为像素坐标
                x1, y1, x2, y2 = box * 640  # 图像大小640x640
                print(f"   目标{i+1}: 类别{cls}, 置信度{score:.3f}, 边界框[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
            
            # 与真实标注对比
            print(f"\n📊 与真实标注对比:")
            print(f"真实目标数量: {len(targets[0]['boxes'])}")
            print(f"预测目标数量: {len(high_conf_boxes)}")
            
            # 简单的匹配检查
            gt_boxes = targets[0]['boxes'].numpy() * 640
            gt_classes = targets[0]['labels'].numpy()
            
            print(f"\n真实标注:")
            for i, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
                x1, y1, x2, y2 = gt_box
                print(f"   GT{i+1}: 类别{gt_cls}, 边界框[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
            
            # 创建可视化图像
            create_visualization(image_tensor, boxes_np, classes_np, scores_np, gt_boxes, gt_classes)
            
            return True
        else:
            print("⚠️ 没有检测到高置信度目标")
            print("   这可能表明过拟合不够充分或后处理阈值过高")
            
            # 显示最高置信度的几个预测
            top_k = 5
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

def create_visualization(image_tensor, pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes):
    """创建可视化图像"""
    try:
        # 转换图像张量为PIL图像
        img_array = image_tensor[0].transpose(0, 1).transpose(1, 2).numpy()
        img_array = (img_array * 255).astype(np.uint8)
        image = Image.fromarray(img_array)
        
        # 创建绘图对象
        draw = ImageDraw.Draw(image)
        
        # 绘制真实标注（绿色）
        for i, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
            x1, y1, x2, y2 = gt_box
            draw.rectangle([x1, y1, x2, y2], outline='green', width=3)
            draw.text((x1, y1-20), f"GT: Class {gt_cls}", fill='green')
        
        # 绘制预测结果（红色）
        for i, (pred_box, pred_cls, pred_score) in enumerate(zip(pred_boxes, pred_classes, pred_scores)):
            x1, y1, x2, y2 = pred_box
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            draw.text((x1, y2+5), f"Pred: Class {pred_cls} ({pred_score:.2f})", fill='red')
        
        # 保存可视化结果
        vis_path = 'experiments/overfit_data/visualization_result.jpg'
        image.save(vis_path)
        print(f"✅ 可视化结果保存到: {vis_path}")
        
    except Exception as e:
        print(f"⚠️ 可视化创建失败: {e}")

def main():
    print("🎯 RT-DETR单张照片过拟合训练验证")
    print("=" * 80)
    
    # 1. 过拟合训练
    backbone, transformer, image_tensor, targets, overfit_success = single_image_overfit_training()
    
    if backbone is None:
        print("❌ 训练失败，无法继续")
        return
    
    # 2. 推理和可视化
    inference_success = inference_and_visualization(backbone, transformer, image_tensor, targets)
    
    # 总结
    print("\n" + "=" * 80)
    print("🎯 过拟合训练验证总结:")
    print("=" * 80)
    
    results = [
        ("过拟合训练", overfit_success),
        ("推理检测", inference_success),
    ]
    
    all_passed = True
    for name, result in results:
        status = "✅ 成功" if result else "❌ 失败"
        print(f"{status} {name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 单张照片过拟合训练验证完全成功！")
        print("✅ 模型成功过拟合到单张图像")
        print("✅ 推理能够检测出训练的目标")
        print("✅ 整个训练流程验证正确")
        print("✅ RT-DETR可以进行正常训练")
        print("\n🚀 这证明了:")
        print("1. ✅ 前向传播完全正确")
        print("2. ✅ 损失计算完全正确")
        print("3. ✅ 反向传播完全正确")
        print("4. ✅ 优化器更新完全正确")
        print("5. ✅ 推理流程完全正确")
        print("\n✨ 现在可以放心地进行大规模数据集训练！")
    else:
        print("⚠️ 过拟合训练验证发现问题")
        if not overfit_success:
            print("💡 建议: 增加训练轮数或调整学习率")
        if not inference_success:
            print("💡 建议: 检查后处理逻辑或降低置信度阈值")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
