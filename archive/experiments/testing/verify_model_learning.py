#!/usr/bin/env python3
"""
验证模型是否能够学习的简单测试
使用一个极简单的例子：单个目标，固定位置
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt
from jittor_rt_detr.src.nn.rtdetr_complete_pytorch_aligned import build_rtdetr_complete

# 设置Jittor
jt.flags.use_cuda = 1
jt.set_global_seed(42)
jt.flags.auto_mixed_precision_level = 0

def safe_float32(tensor):
    if isinstance(tensor, jt.Var):
        return tensor.float32()
    elif isinstance(tensor, np.ndarray):
        return jt.array(tensor.astype(np.float32))
    else:
        return jt.array(tensor, dtype=jt.float32)

def safe_int64(tensor):
    if isinstance(tensor, jt.Var):
        return tensor.int64()
    elif isinstance(tensor, np.ndarray):
        return jt.array(tensor.astype(np.int64))
    else:
        return jt.array(tensor, dtype=jt.int64)

class SimpleFocalLoss(jt.nn.Module):
    """简化版Focal Loss"""
    
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
    
    def execute(self, src_logits, target_classes):
        """计算focal loss"""
        src_logits = safe_float32(src_logits)
        target_classes = safe_int64(target_classes)
        
        # 创建one-hot编码
        target_onehot = jt.zeros(src_logits.shape, dtype=jt.float32)
        target_onehot.scatter_(-1, target_classes.unsqueeze(-1), safe_float32(1.0))
        
        # 简化版交叉熵损失
        sigmoid_p = jt.sigmoid(src_logits).float32()
        ce_loss = -(target_onehot * jt.log(sigmoid_p + 1e-8) + 
                   (1 - target_onehot) * jt.log(1 - sigmoid_p + 1e-8)).float32()
        
        # 简化版focal loss
        loss = ce_loss.mean()
        return safe_float32(loss)

class SimpleL1Loss(jt.nn.Module):
    """简化版L1损失"""
    
    def execute(self, pred_boxes, target_boxes):
        """计算L1损失"""
        pred_boxes = safe_float32(pred_boxes)
        target_boxes = safe_float32(target_boxes)
        
        loss = jt.abs(pred_boxes - target_boxes).mean()
        return safe_float32(loss)

def create_synthetic_data():
    """创建合成数据：一个固定位置的目标"""
    # 创建一个空白图像
    img = np.zeros((3, 640, 640), dtype=np.float32)
    
    # 目标位置：中心点(0.7, 0.3)，宽高(0.2, 0.2)
    target_box = np.array([[0.7, 0.3, 0.2, 0.2]], dtype=np.float32)
    target_label = np.array([0], dtype=np.int64)  # 类别0
    
    # 创建目标
    targets = [{
        'boxes': safe_float32(target_box),
        'labels': safe_int64(target_label)
    }]
    
    # 创建可视化
    vis_img = Image.new('RGB', (640, 640), color='black')
    draw = ImageDraw.Draw(vis_img)
    
    # 绘制目标
    cx, cy, w, h = target_box[0]
    x1 = int((cx - w/2) * 640)
    y1 = int((cy - h/2) * 640)
    x2 = int((cx + w/2) * 640)
    y2 = int((cy + h/2) * 640)
    draw.rectangle([x1, y1, x2, y2], outline='green', width=2)
    
    # 保存可视化
    os.makedirs("results/verify_model", exist_ok=True)
    vis_img.save("results/verify_model/synthetic_data.png")
    
    return safe_float32(img).unsqueeze(0), targets, vis_img

def test_model_prediction(model, img_tensor, epoch=0):
    """测试模型预测"""
    model.eval()
    
    with jt.no_grad():
        outputs = model(img_tensor)
    
    pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
    pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4]
    
    # 找到最高置信度的预测
    pred_probs = jt.sigmoid(pred_logits)
    pred_scores = pred_probs.max(dim=-1)[0]
    pred_labels = pred_probs.argmax(dim=-1)
    
    # 获取最高置信度的预测
    max_score_idx = pred_scores.argmax(dim=0)[0].item()
    max_score = pred_scores[max_score_idx].item()
    max_label = pred_labels[max_score_idx].item()
    max_box = pred_boxes[max_score_idx].numpy()
    
    print(f"Epoch {epoch}: 最高置信度 = {max_score:.4f}, 类别 = {max_label}")
    print(f"预测框: [{max_box[0]:.4f}, {max_box[1]:.4f}, {max_box[2]:.4f}, {max_box[3]:.4f}]")
    
    # 创建可视化
    if epoch % 50 == 0 or epoch == 1 or epoch >= 195:
        vis_img = Image.new('RGB', (640, 640), color='black')
        draw = ImageDraw.Draw(vis_img)
        
        # 绘制真实目标（绿色）
        cx, cy, w, h = 0.7, 0.3, 0.2, 0.2
        x1 = int((cx - w/2) * 640)
        y1 = int((cy - h/2) * 640)
        x2 = int((cx + w/2) * 640)
        y2 = int((cy + h/2) * 640)
        draw.rectangle([x1, y1, x2, y2], outline='green', width=2)
        draw.text((x1, y1-20), "GT: class 0", fill='green')
        
        # 绘制预测（红色）
        cx, cy, w, h = max_box
        x1 = int((cx - w/2) * 640)
        y1 = int((cy - h/2) * 640)
        x2 = int((cx + w/2) * 640)
        y2 = int((cy + h/2) * 640)
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        draw.text((x1, y1+20), f"Pred: class {max_label} ({max_score:.2f})", fill='red')
        
        # 保存可视化
        save_path = f"results/verify_model/epoch_{epoch:03d}.png"
        vis_img.save(save_path)
        print(f"  可视化保存到: {save_path}")
    
    return max_score, max_label, max_box

def main():
    print("=" * 60)
    print("===        验证模型学习能力        ===")
    print("=" * 60)
    
    # 创建合成数据
    img_tensor, targets, vis_img = create_synthetic_data()
    print("✅ 创建合成数据：一个位于(0.7, 0.3)的目标，类别0")
    
    # 创建模型
    num_classes = 80
    model = build_rtdetr_complete(num_classes=num_classes, hidden_dim=256, num_queries=300)
    print("✅ 创建模型")
    
    # 创建简化版损失函数
    focal_loss = SimpleFocalLoss(num_classes)
    l1_loss = SimpleL1Loss()
    print("✅ 创建简化损失函数")
    
    # 优化器
    optimizer = jt.optim.Adam(model.parameters(), lr=1e-3)
    print("✅ 创建优化器，学习率=1e-3")
    
    # 初始测试
    print("\n=== 训练前测试 ===")
    initial_score, initial_label, initial_box = test_model_prediction(model, img_tensor, 0)
    
    # 训练历史
    history = {
        'losses': [],
        'scores': [initial_score],
        'boxes': [np.mean(np.abs(initial_box - np.array([0.7, 0.3, 0.2, 0.2])))],
    }
    
    # 训练
    print("\n开始训练 - 200轮")
    num_epochs = 200
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        
        # 前向传播
        outputs = model(img_tensor)
        
        # 计算损失
        # 1. 分类损失
        pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
        target_classes = jt.full((300,), num_classes, dtype=jt.int64)
        target_classes[0] = 0  # 第一个查询的目标是类别0
        class_loss = focal_loss(pred_logits, target_classes)
        
        # 2. 边界框损失
        pred_boxes = outputs['pred_boxes'][0]  # [num_queries, 4]
        target_box = jt.array([[0.7, 0.3, 0.2, 0.2]], dtype=jt.float32)
        box_loss = l1_loss(pred_boxes[0:1], target_box)
        
        # 总损失
        total_loss = class_loss + 5.0 * box_loss  # 增加边界框损失权重
        
        # 反向传播
        optimizer.step(total_loss)
        
        # 记录损失
        history['losses'].append(total_loss.item())
        
        # 定期测试
        if epoch % 10 == 0 or epoch == 1:
            print(f"\n=== Epoch {epoch} ===")
            print(f"Loss: {total_loss.item():.4f} (Class: {class_loss.item():.4f}, Box: {box_loss.item():.4f})")
            
            score, label, box = test_model_prediction(model, img_tensor, epoch)
            history['scores'].append(score)
            history['boxes'].append(np.mean(np.abs(box - np.array([0.7, 0.3, 0.2, 0.2]))))
    
    # 最终测试
    print("\n=== 最终测试 ===")
    final_score, final_label, final_box = test_model_prediction(model, img_tensor, num_epochs)
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['losses'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    epochs = [0] + list(range(1, num_epochs + 1, 10)) + [num_epochs]
    plt.plot(epochs, history['scores'], 'o-')
    plt.title('Max Confidence Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['boxes'], 'o-')
    plt.title('Box L1 Error')
    plt.xlabel('Epoch')
    plt.ylabel('Average L1 Error')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/verify_model/training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 训练曲线保存到: results/verify_model/training_curves.png")
    
    # 总结
    print("\n" + "=" * 60)
    print("🎯 验证结果:")
    print("=" * 60)
    print(f"初始置信度: {initial_score:.4f}")
    print(f"最终置信度: {final_score:.4f}")
    print(f"初始框误差: {history['boxes'][0]:.4f}")
    print(f"最终框误差: {history['boxes'][-1]:.4f}")
    
    if final_score > 0.5 and history['boxes'][-1] < 0.1:
        print("✅ 验证成功！模型能够学习")
    else:
        print("❌ 验证失败：模型无法正常学习")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
