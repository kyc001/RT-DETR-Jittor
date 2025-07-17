#!/usr/bin/env python3
"""
单张图片过拟合训练脚本 (最新版本)
使用最新的rtdetr_complete_pytorch_aligned模型和loss_pytorch_aligned损失函数
"""

import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt
from jittor_rt_detr.src.nn.rtdetr_complete_pytorch_aligned import build_rtdetr_complete
from jittor_rt_detr.src.nn.loss_pytorch_aligned import SetCriterion, build_criterion

# 设置Jittor
jt.flags.use_cuda = 1
jt.set_global_seed(42)
jt.flags.auto_mixed_precision_level = 0

def safe_float32(tensor):
    """确保张量为float32类型"""
    if isinstance(tensor, jt.Var):
        return tensor.float32()
    elif isinstance(tensor, np.ndarray):
        return jt.array(tensor.astype(np.float32))
    else:
        return jt.array(tensor, dtype=jt.float32)

def safe_int64(tensor):
    """确保张量为int64类型"""
    if isinstance(tensor, jt.Var):
        return tensor.int64()
    elif isinstance(tensor, np.ndarray):
        return jt.array(tensor.astype(np.int64))
    else:
        return jt.array(np.array(tensor, dtype=np.int64))

def load_single_image_data(image_name="000000343218.jpg"):
    """加载单张图片的数据"""
    # 加载COCO数据
    ann_file = "data/coco2017_50/annotations/instances_train2017.json"
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # 找到指定图片
    image_info = None
    for img in coco_data['images']:
        if img['file_name'] == image_name:
            image_info = img
            break
    
    if image_info is None:
        raise ValueError(f"找不到图片: {image_name}")
    
    # 获取该图片的标注
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_info['id']]
    
    if not annotations:
        raise ValueError(f"图片 {image_name} 没有标注")
    
    print(f"选择图片: {image_name}")
    print(f"图片尺寸: {image_info['width']} x {image_info['height']}")
    print(f"标注数量: {len(annotations)}")
    
    # 显示标注信息
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    print("标注目标:")
    for i, ann in enumerate(annotations):
        cat_name = cat_id_to_name[ann['category_id']]
        x, y, w, h = ann['bbox']
        print(f"  {i+1}. {cat_name}: [{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}]")
    
    return image_info, annotations, coco_data

def prepare_single_image_batch(image_info, annotations, coco_data):
    """准备单张图片的训练批次"""
    image_path = f"data/coco2017_50/train2017/{image_info['file_name']}"
    
    # 加载和预处理图片
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    image_resized = image.resize((640, 640))
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    img_tensor = safe_float32(img_array).unsqueeze(0)  # batch_size=1
    
    # 创建目标
    cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    boxes, labels = [], []
    
    for ann in annotations:
        x, y, w, h = ann['bbox']
        cx = np.float32(x + w / 2) / np.float32(original_size[0])
        cy = np.float32(y + h / 2) / np.float32(original_size[1])
        w_norm = np.float32(w) / np.float32(original_size[0])
        h_norm = np.float32(h) / np.float32(original_size[1])
        
        boxes.append([cx, cy, w_norm, h_norm])
        labels.append(cat_id_to_idx[ann['category_id']])
    
    targets = [{
        'boxes': safe_float32(np.array(boxes, dtype=np.float32)),
        'labels': safe_int64(np.array(labels, dtype=np.int64))
    }]
    
    return img_tensor, targets, image, original_size

def test_inference(model, img_tensor, targets, original_image, original_size, coco_data, epoch):
    """测试推理结果"""
    model.eval()

    with jt.no_grad():
        outputs = model(img_tensor)

    # 确保所有输出都是float32类型
    pred_logits = safe_float32(outputs['pred_logits'][0])
    pred_boxes = safe_float32(outputs['pred_boxes'][0])

    # 后处理
    pred_probs = jt.sigmoid(pred_logits)

    # 获取最大概率和对应的类别
    pred_scores = pred_probs.max(dim=-1)[0]
    pred_labels = pred_probs.argmax(dim=-1)

    # 过滤高置信度预测
    keep_indices = []
    for i in range(len(pred_scores)):
        if pred_scores[i].item() > 0.1:
            keep_indices.append(i)

    if len(keep_indices) == 0:
        print(f"Epoch {epoch}: 没有高置信度预测")
        return 0

    # 提取保留的预测
    filtered_scores = [pred_scores[i].item() for i in keep_indices]
    filtered_labels = [pred_labels[i].item() for i in keep_indices]
    filtered_boxes = [pred_boxes[i].numpy() for i in keep_indices]

    # 转换为numpy数组
    filtered_scores = np.array(filtered_scores)
    filtered_labels = np.array(filtered_labels, dtype=np.int32)
    filtered_boxes = np.array(filtered_boxes)
    
    # 类别映射
    idx_to_name = {idx: cat['name'] for idx, cat in enumerate(coco_data['categories'])}
    
    print(f"Epoch {epoch}: 检测到 {len(filtered_scores)} 个目标")
    for i, (score, label, box) in enumerate(zip(filtered_scores, filtered_labels, filtered_boxes)):
        class_name = idx_to_name.get(label, f'class_{label}')
        cx, cy, w, h = box
        print(f"  {i+1}. {class_name}: {score:.3f} at ({cx:.3f}, {cy:.3f}, {w:.3f}, {h:.3f})")
    
    # 创建可视化
    if epoch % 50 == 0 or epoch == 1 or epoch >= 295:  # 每50轮或第1轮或最后几轮保存可视化
        vis_img = original_image.copy()
        draw = ImageDraw.Draw(vis_img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # 绘制真实标注（绿色）
        for target in targets:
            boxes = target['boxes'].numpy()
            labels = target['labels'].numpy()
            
            for box, label in zip(boxes, labels):
                cx, cy, w, h = box
                class_name = idx_to_name[label]
                
                # 转换到像素坐标
                cx_pixel = cx * original_size[0]
                cy_pixel = cy * original_size[1]
                w_pixel = w * original_size[0]
                h_pixel = h * original_size[1]
                
                x1 = cx_pixel - w_pixel / 2
                y1 = cy_pixel - h_pixel / 2
                x2 = cx_pixel + w_pixel / 2
                y2 = cy_pixel + h_pixel / 2
                
                draw.rectangle([x1, y1, x2, y2], outline='green', width=2)
                draw.text((x1, y1-20), f'GT: {class_name}', fill='green', font=font)
        
        # 绘制预测结果（红色）
        for i, (score, label, box) in enumerate(zip(filtered_scores, filtered_labels, filtered_boxes)):
            class_name = idx_to_name.get(label, f'class_{label}')
            cx, cy, w, h = box
            
            # 转换到像素坐标
            cx_pixel = cx * original_size[0]
            cy_pixel = cy * original_size[1]
            w_pixel = w * original_size[0]
            h_pixel = h * original_size[1]
            
            x1 = cx_pixel - w_pixel / 2
            y1 = cy_pixel - h_pixel / 2
            x2 = cx_pixel + w_pixel / 2
            y2 = cy_pixel + h_pixel / 2
            
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            draw.text((x1, y1+20), f'Pred: {class_name} ({score:.2f})', fill='red', font=font)
        
        # 保存可视化
        os.makedirs("results/overfit_training_v2", exist_ok=True)
        save_path = f"results/overfit_training_v2/epoch_{epoch:03d}.jpg"
        vis_img.save(save_path)
        print(f"  可视化保存到: {save_path}")
    
    return len(filtered_scores)

def main():
    print("=" * 60)
    print("===        单张图片过拟合训练 V2        ===")
    print("=" * 60)
    
    # 选择一张有多个目标的图片进行过拟合
    image_name = "000000343218.jpg"  # 这张图片有23个目标，包含多种类别
    
    try:
        # 加载单张图片数据
        image_info, annotations, coco_data = load_single_image_data(image_name)
        img_tensor, targets, original_image, original_size = prepare_single_image_batch(image_info, annotations, coco_data)
        
        # 创建模型
        num_classes = len(coco_data['categories'])
        model = build_rtdetr_complete(num_classes=num_classes, hidden_dim=256, num_queries=300)
        
        # 创建损失函数 - 使用build_criterion函数
        criterion = build_criterion(num_classes)
        
        # 优化器 - 使用较小的学习率进行精细训练
        optimizer = jt.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-6)
        
        print(f"\n开始过拟合训练 - 目标：让模型完美记住这张图片")
        print(f"训练轮数: 300")
        print(f"学习率: 5e-5")
        print(f"边界框损失权重: 10")
        
        # 训练历史
        training_history = {'losses': [], 'detections': []}
        
        # 初始测试
        print(f"\n=== 训练前测试 ===")
        initial_detections = test_inference(model, img_tensor, targets, original_image, original_size, coco_data, 0)
        training_history['detections'].append(initial_detections)
        
        # 过拟合训练
        model.train()
        for epoch in range(1, 301):
            # 前向传播
            outputs = model(img_tensor)
            
            # 计算损失
            loss_dict = criterion(outputs, targets)
            total_loss = sum(loss_dict.values())
            
            # 反向传播
            optimizer.step(total_loss)
            
            training_history['losses'].append(total_loss.item())
            
            # 定期测试
            if epoch % 25 == 0 or epoch == 1:
                print(f"\n=== Epoch {epoch} ===")
                print(f"Loss: {total_loss.item():.4f}")
                detections = test_inference(model, img_tensor, targets, original_image, original_size, coco_data, epoch)
                training_history['detections'].append(detections)
            elif epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")
        
        # 最终测试
        print(f"\n=== 最终测试 (Epoch 300) ===")
        final_detections = test_inference(model, img_tensor, targets, original_image, original_size, coco_data, 300)
        
        # 保存模型
        os.makedirs("checkpoints", exist_ok=True)
        jt.save(model.state_dict(), "checkpoints/rtdetr_overfitted_v2.pkl")
        print(f"\n✅ 过拟合模型已保存到: checkpoints/rtdetr_overfitted_v2.pkl")
        
        # 绘制训练曲线
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(training_history['losses'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        epochs = [0] + list(range(1, 301, 25)) + [300]
        detections = training_history['detections']
        if len(detections) < len(epochs):
            detections = detections + [detections[-1]] * (len(epochs) - len(detections))
        plt.plot(epochs, detections, 'o-')
        plt.title('Number of Detections')
        plt.xlabel('Epoch')
        plt.ylabel('Detections (confidence > 0.1)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("results/overfit_training_v2/training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 训练曲线保存到: results/overfit_training_v2/training_curves.png")
        
        # 总结
        print(f"\n" + "=" * 60)
        print("🎯 过拟合训练总结:")
        print("=" * 60)
        print(f"目标图片: {image_name}")
        print(f"真实目标数: {len(annotations)}")
        print(f"训练前检测数: {initial_detections}")
        print(f"训练后检测数: {final_detections}")
        print(f"最终损失: {training_history['losses'][-1]:.6f}")
        
        if final_detections >= len(annotations) * 0.8:  # 检测到80%以上的目标
            print("✅ 过拟合成功！模型能够很好地记住这张图片")
        elif final_detections > initial_detections:
            print("⚠️ 部分成功：模型有所改善但未完全过拟合")
        else:
            print("❌ 过拟合失败：可能存在架构或训练问题")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 过拟合训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
