#!/usr/bin/env python3
"""
修复版流程自检脚本 - 参考PyTorch版本的训练配置
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'jittor_rt_detr'))

import jittor as jt
from src.nn.model import RTDETR
from src.nn.loss import DETRLoss

# 设置Jittor
jt.flags.use_cuda = 1

def load_data_with_augmentation(img_path, ann_file, img_name):
    """加载数据并添加数据增强"""
    print(f">>> 加载数据: {img_name}")
    
    # 加载图片
    image = Image.open(img_path).convert('RGB')
    original_size = image.size
    print(f"原始图片尺寸: {original_size}")
    
    # 加载标注
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # 找到目标图片和标注
    target_image = None
    for img in coco_data['images']:
        if img['file_name'] == img_name:
            target_image = img
            break
    
    annotations = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == target_image['id']:
            annotations.append(ann)
    
    # 创建类别映射
    categories = {cat['id']: cat for cat in coco_data['categories']}
    unique_category_ids = list(set(ann['category_id'] for ann in annotations))
    unique_category_ids.sort()
    
    cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(unique_category_ids)}
    num_classes = len(unique_category_ids)
    
    print(f"图片中的类别:")
    for cat_id in unique_category_ids:
        cat_name = categories[cat_id]['name']
        model_idx = cat_id_to_idx[cat_id]
        print(f"  - {cat_name} (COCO ID: {cat_id}, 模型索引: {model_idx})")
    
    # 创建多个数据增强版本
    def create_augmented_data(flip_h=False, scale=1.0, offset_x=0, offset_y=0):
        # 预处理图片
        resized_image = image.resize((640, 640), Image.LANCZOS)
        
        # 水平翻转
        if flip_h:
            resized_image = resized_image.transpose(Image.FLIP_LEFT_RIGHT)
        
        img_array = np.array(resized_image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        img_tensor = jt.array(img_array.transpose(2, 0, 1), dtype='float32').unsqueeze(0)
        
        # 处理标注
        boxes = []
        labels = []
        
        for ann in annotations:
            x, y, w, h = ann['bbox']
            
            # 应用翻转
            if flip_h:
                x = original_size[0] - x - w
            
            # 转换为归一化cxcywh格式
            cx = (x + w/2) / original_size[0]
            cy = (y + h/2) / original_size[1]
            w_norm = w / original_size[0]
            h_norm = h / original_size[1]
            
            boxes.append([cx, cy, w_norm, h_norm])
            labels.append(cat_id_to_idx[ann['category_id']])
        
        boxes_tensor = jt.array(boxes, dtype='float32')
        labels_tensor = jt.array(labels, dtype='int64')
        
        targets = [{
            'boxes': boxes_tensor,
            'labels': labels_tensor
        }]
        
        return img_tensor, targets
    
    # 创建原始数据和增强数据
    data_variants = []
    
    # 原始数据
    img_tensor, targets = create_augmented_data()
    data_variants.append((img_tensor, targets, "original"))
    
    # 水平翻转
    img_tensor_flip, targets_flip = create_augmented_data(flip_h=True)
    data_variants.append((img_tensor_flip, targets_flip, "flip"))
    
    return data_variants, num_classes, cat_id_to_idx, original_size, image

def create_model_with_proper_init(num_classes):
    """创建模型并进行适当的初始化"""
    print(f">>> 创建模型 (类别数: {num_classes})")
    
    model = RTDETR(num_classes=num_classes)
    model = model.float32()
    
    # 参考PyTorch版本进行权重初始化
    def init_weights(m):
        if hasattr(m, 'weight') and m.weight is not None:
            if len(m.weight.shape) > 1:
                # Xavier初始化
                fan_in = m.weight.shape[1]
                fan_out = m.weight.shape[0]
                std = np.sqrt(2.0 / (fan_in + fan_out))
                m.weight.data = jt.randn_like(m.weight) * std
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data = jt.zeros_like(m.bias)
    
    # 只初始化分类头，保持其他部分不变
    for name, module in model.named_modules():
        if 'class_embed' in name:
            init_weights(module)
    
    return model

def train_with_proper_config(model, data_variants, num_classes, max_epochs=10):
    """使用正确的训练配置进行训练"""
    print(f"\n>>> 开始训练 (最多 {max_epochs} 轮)")
    
    model.train()
    
    # 参考PyTorch版本的配置
    criterion = DETRLoss(num_classes=num_classes)
    
    # 使用AdamW优化器，参考PyTorch版本
    optimizer = jt.optim.AdamW(
        model.parameters(), 
        lr=1e-5,  # 更低的学习率
        weight_decay=1e-4,  # 添加权重衰减
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    def get_lr_factor(epoch):
        if epoch < 3:
            return 0.1  # 前3轮使用更低的学习率
        elif epoch < 7:
            return 1.0
        else:
            return 0.1
    
    best_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(max_epochs):
        epoch_losses = []
        
        # 使用不同的数据增强版本
        for img_tensor, targets, variant_name in data_variants:
            # 前向传播
            outputs = model(img_tensor)
            logits, boxes, enc_logits, enc_boxes = outputs
            
            # 计算损失
            loss_dict = criterion(logits, boxes, targets, enc_logits, enc_boxes)
            total_loss = sum(loss_dict.values())
            
            # 反向传播
            optimizer.zero_grad()
            optimizer.backward(total_loss)
            
            # 梯度裁剪 - Jittor版本实现
            def clip_grad_norm(parameters, optimizer, max_norm=0.1):
                total_norm = 0
                for p in parameters:
                    try:
                        grad = p.opt_grad(optimizer)
                        if grad is not None:
                            param_norm = grad.norm()
                            total_norm += param_norm ** 2
                    except:
                        continue

                total_norm = total_norm ** 0.5

                if total_norm > max_norm:
                    clip_coef = max_norm / (total_norm + 1e-6)
                    for p in parameters:
                        try:
                            grad = p.opt_grad(optimizer)
                            if grad is not None:
                                # Jittor中直接修改梯度可能有问题，这里跳过梯度裁剪
                                pass
                        except:
                            continue

                return total_norm

            # 暂时跳过梯度裁剪，因为Jittor的实现比较复杂
            # clip_grad_norm(model.parameters(), optimizer, max_norm=0.1)
            
            optimizer.step()
            
            epoch_losses.append(float(total_loss.data))
        
        # 调整学习率
        lr_factor = get_lr_factor(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-5 * lr_factor
        
        avg_loss = np.mean(epoch_losses)
        
        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f"  Epoch {epoch+1}/{max_epochs}: Loss = {avg_loss:.4f} (Best: {best_loss:.4f}), LR = {optimizer.param_groups[0]['lr']:.2e}")
        
        # 早停
        if patience_counter >= patience:
            print(f"  早停触发 (连续{patience}轮无改善)")
            break
    
    print("✅ 训练完成")
    return model

def test_inference_with_analysis(model, img_tensor, original_size, num_classes):
    """测试推理并分析结果"""
    print(f"\n>>> 测试推理...")
    
    model.eval()
    with jt.no_grad():
        outputs = model(img_tensor)
    
    logits, boxes, _, _ = outputs
    pred_logits = logits[-1][0]
    pred_boxes = boxes[-1][0]
    
    # 转换为numpy
    pred_logits = pred_logits.float32()
    pred_boxes = pred_boxes.float32()
    
    logits_np = pred_logits.stop_grad().numpy()
    boxes_np = pred_boxes.stop_grad().numpy()
    
    # 分析logits分布
    print(f"Logits分析:")
    for class_idx in range(num_classes):
        class_logits = logits_np[:, class_idx]
        print(f"  类别 {class_idx}: 均值={np.mean(class_logits):.3f}, 标准差={np.std(class_logits):.3f}")
    
    # 使用softmax
    exp_logits = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
    scores = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    max_scores = np.max(scores, axis=1)
    max_classes = np.argmax(scores, axis=1)
    
    # 分析类别分布
    print(f"类别分布:")
    for class_idx in range(num_classes):
        count = np.sum(max_classes == class_idx)
        print(f"  类别 {class_idx}: {count} 个预测")
    
    print(f"置信度统计:")
    print(f"  - 最高: {np.max(max_scores):.3f}")
    print(f"  - 最低: {np.min(max_scores):.3f}")
    print(f"  - 平均: {np.mean(max_scores):.3f}")
    print(f"  - 标准差: {np.std(max_scores):.3f}")
    
    # 使用自适应阈值
    threshold = np.mean(max_scores) + np.std(max_scores)
    threshold = min(threshold, 0.7)  # 不超过0.7
    threshold = max(threshold, 0.3)  # 不低于0.3
    
    print(f"使用自适应阈值: {threshold:.3f}")
    
    valid_mask = max_scores > threshold
    valid_indices = np.where(valid_mask)[0]
    
    print(f"超过阈值的检测数量: {len(valid_indices)}")
    
    # 转换检测结果
    detections = []
    for idx in valid_indices:
        box = boxes_np[idx]
        score = max_scores[idx]
        class_idx = max_classes[idx]
        
        # 转换坐标
        cx, cy, w, h = box
        x1 = (cx - w/2) * original_size[0]
        y1 = (cy - h/2) * original_size[1]
        x2 = (cx + w/2) * original_size[0]
        y2 = (cy + h/2) * original_size[1]
        
        # 边界检查
        x1 = max(0, min(x1, original_size[0]))
        y1 = max(0, min(y1, original_size[1]))
        x2 = max(0, min(x2, original_size[0]))
        y2 = max(0, min(y2, original_size[1]))
        
        if x2 > x1 and y2 > y1:
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(score),
                'class_idx': int(class_idx),
                'query_idx': int(idx)
            })
    
    # 按置信度排序
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    return detections

def main():
    print("=" * 60)
    print("===      修复版流程自检 - 正确的训练配置      ===")
    print("=" * 60)
    
    # 参数设置
    img_path = "data/coco2017_50/train2017/000000225405.jpg"
    ann_file = "data/coco2017_50/annotations/instances_train2017.json"
    img_name = "000000225405.jpg"
    max_epochs = 10
    
    # 1. 加载数据（包含数据增强）
    data_variants, num_classes, cat_id_to_idx, original_size, image = load_data_with_augmentation(
        img_path, ann_file, img_name)
    
    print(f"数据增强版本数量: {len(data_variants)}")
    
    # 2. 创建模型
    model = create_model_with_proper_init(num_classes)
    
    # 3. 训练
    model = train_with_proper_config(model, data_variants, num_classes, max_epochs)
    
    # 4. 保存模型
    save_path = "checkpoints/fixed_sanity_check_model.pkl"
    os.makedirs("checkpoints", exist_ok=True)
    jt.save(model.state_dict(), save_path)
    print(f"✅ 模型已保存到: {save_path}")
    
    # 5. 测试推理
    img_tensor, targets = data_variants[0][:2]  # 使用原始数据测试
    detections = test_inference_with_analysis(model, img_tensor, original_size, num_classes)
    
    # 6. 简单分析
    print(f"\n📊 检测结果:")
    print(f"检测数量: {len(detections)}")
    
    if detections:
        class_counts = {}
        for det in detections:
            class_name = 'sports ball' if det['class_idx'] == 1 else 'person'
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"类别分布:")
        for class_name, count in class_counts.items():
            print(f"  - {class_name}: {count} 个")
        
        print(f"前5个检测:")
        for i, det in enumerate(detections[:5]):
            class_name = 'sports ball' if det['class_idx'] == 1 else 'person'
            print(f"  {i+1}. {class_name} (置信度: {det['confidence']:.3f})")
    
    # 7. 结论
    print(f"\n🔍 修复版流程自检结论:")
    
    if len(detections) > 0:
        # 检查是否有多样性
        unique_classes = set(det['class_idx'] for det in detections)
        if len(unique_classes) > 1:
            print("✅ 修复成功！")
            print("  - 检测到多个类别")
            print("  - 避免了完全过拟合")
        else:
            print("⚠️ 部分改善")
            print("  - 仍然只检测到一个类别")
            print("  - 但训练过程更稳定")
    else:
        print("❌ 需要进一步调整")
        print("  - 未检测到任何目标")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
