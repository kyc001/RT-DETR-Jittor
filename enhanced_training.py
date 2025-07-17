#!/usr/bin/env python3
"""
增强的RT-DETR训练脚本
增加训练量，调整超参数，改善训练效果
"""

import os
import sys
import json
import time
import numpy as np
from PIL import Image

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt
from jittor_rt_detr.src.nn.backbone.resnet import ResNet50
from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
import jittor.nn as nn

jt.flags.use_cuda = 1

class EnhancedCOCODataset:
    """增强的COCO数据集加载器"""
    def __init__(self, img_dir, ann_file):
        self.img_dir = img_dir
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.img_id_to_filename = {img['id']: img['file_name'] for img in self.coco_data['images']}
        self.img_ids = list(self.img_id_to_filename.keys())
        
        # 数据增强：每张图片重复多次
        self.img_ids = self.img_ids * 5  # 5倍数据增强
        
        print(f"📊 增强后数据集大小: {len(self.img_ids)} 张图像")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        filename = self.img_id_to_filename[img_id]
        
        # 加载图像
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert('RGB')
        original_width, original_height = image.size
        
        # 数据增强：随机翻转
        if np.random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 调整图像大小到640x640
        image_resized = image.resize((640, 640), Image.LANCZOS)
        img_array = np.array(image_resized).astype(np.float32) / 255.0
        
        # 随机亮度调整
        brightness_factor = np.random.uniform(0.8, 1.2)
        img_array = np.clip(img_array * brightness_factor, 0, 1)
        
        img_tensor = jt.array(img_array.transpose(2, 0, 1)).float32()
        
        # 获取标注
        annotations = []
        labels = []
        
        for ann in self.coco_data['annotations']:
            if ann['image_id'] == img_id:
                x, y, w, h = ann['bbox']
                category_id = ann['category_id']
                
                # 归一化坐标
                x1, y1 = x / original_width, y / original_height
                x2, y2 = (x + w) / original_width, (y + h) / original_height
                
                if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= 1 and y2 <= 1:
                    annotations.append([x1, y1, x2, y2])
                    
                    # 简化类别映射
                    mapped_label = min(category_id - 1, 79)  # 确保在0-79范围内
                    labels.append(mapped_label)
        
        # 创建目标
        if annotations:
            target = {
                'boxes': jt.array(annotations).float32(),
                'labels': jt.array(labels).int64()
            }
        else:
            target = {
                'boxes': jt.zeros((0, 4)).float32(),
                'labels': jt.zeros((0,)).int64()
            }
        
        return img_tensor, target

class EnhancedRTDETRModel(nn.Module):
    """增强的RT-DETR模型"""
    def __init__(self, num_classes=80):
        super().__init__()
        self.backbone = ResNet50(pretrained=True)
        self.transformer = RTDETRTransformer(
            num_classes=num_classes,
            hidden_dim=256,
            num_queries=300,
            feat_channels=[256, 512, 1024, 2048]
        )
        
        # 初始化输出头的权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # Jittor的权重初始化方式不同，跳过手动初始化
        pass
    
    def execute(self, x, targets=None):
        features = self.backbone(x)
        return self.transformer(features, targets)

class EnhancedCriterion(nn.Module):
    """增强的损失函数"""
    def __init__(self, num_classes=80, alpha=0.25, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        
    def focal_loss(self, pred_logits, targets):
        """改进的Focal Loss"""
        batch_size, num_queries, num_classes = pred_logits.shape
        
        # 创建目标类别
        target_classes = jt.full((batch_size, num_queries), num_classes - 1, dtype=jt.int64)
        
        # 改进的目标分配：分配更多查询给真实目标
        for batch_idx, target in enumerate(targets):
            if len(target['labels']) > 0:
                num_targets = min(len(target['labels']), num_queries // 2)  # 使用更多查询
                if num_targets > 0:
                    target_labels = target['labels'][:num_targets]
                    target_classes[batch_idx, :num_targets] = target_labels
        
        # 计算交叉熵损失
        pred_logits_flat = pred_logits.view(-1, num_classes)
        target_classes_flat = target_classes.view(-1)
        
        ce_loss = nn.cross_entropy_loss(pred_logits_flat, target_classes_flat, reduction='none')
        ce_loss = ce_loss.view(batch_size, num_queries)
        
        # 计算focal loss
        p_t = jt.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        return focal_loss.mean()
    
    def bbox_loss(self, pred_boxes, targets):
        """改进的边界框损失"""
        total_loss = jt.zeros(1)
        total_loss.requires_grad = True
        
        for batch_idx, target in enumerate(targets):
            if len(target['boxes']) > 0:
                num_targets = min(len(target['boxes']), pred_boxes.shape[1] // 2)
                if num_targets > 0:
                    pred_subset = pred_boxes[batch_idx, :num_targets]
                    target_subset = target['boxes'][:num_targets]
                    
                    bbox_loss = nn.l1_loss(pred_subset, target_subset)
                    total_loss = total_loss + bbox_loss
        
        return total_loss
    
    def execute(self, outputs, targets):
        """计算总损失"""
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        loss_focal = self.focal_loss(pred_logits, targets)
        loss_bbox = self.bbox_loss(pred_boxes, targets) * 2.0  # 降低bbox损失权重
        
        zero_giou = jt.zeros(1)
        zero_giou.requires_grad = True
        
        return {
            'loss_focal': loss_focal,
            'loss_bbox': loss_bbox,
            'loss_giou': zero_giou
        }

def enhanced_training():
    """增强训练"""
    print("🚀 增强RT-DETR训练 - 更多数据，更多轮数")
    print("=" * 60)
    
    # 数据路径
    img_dir = "/home/kyc/project/RT-DETR/data/coco2017_50/train2017"
    ann_file = "/home/kyc/project/RT-DETR/data/coco2017_50/annotations/instances_train2017.json"
    
    # 创建增强数据集
    print("🔄 加载增强训练数据...")
    dataset = EnhancedCOCODataset(img_dir, ann_file)
    print(f"✅ 增强数据加载完成: {len(dataset)}张训练图像")
    
    # 创建模型
    print("🔄 创建增强模型...")
    model = EnhancedRTDETRModel(num_classes=80)
    criterion = EnhancedCriterion(num_classes=80)
    print("✅ 增强模型创建成功")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 模型总参数: {total_params:,}")
    
    # 创建优化器 - 使用更高的学习率
    optimizer = nn.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # 训练配置 - 增加训练轮数
    num_epochs = 100  # 增加到100轮
    losses = []
    
    print(f"\n🚀 开始增强训练 {len(dataset)} 张图像，{num_epochs} 轮")
    print("=" * 60)
    
    # 开始训练
    model.train()
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # 每轮随机打乱数据
        indices = np.random.permutation(len(dataset))
        
        for i in range(len(dataset)):
            idx = indices[i]
            images, targets = dataset[idx]
            
            # 添加batch维度
            images = images.unsqueeze(0)
            targets = [targets]
            
            # 前向传播
            outputs = model(images, targets)
            
            # 损失计算
            loss_dict = criterion(outputs, targets)
            total_loss = sum(loss_dict.values())
            
            # 反向传播
            optimizer.step(total_loss)
            
            epoch_losses.append(total_loss.item())
        
        # 计算平均损失
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        # 每10轮或最后几轮打印进度
        if (epoch + 1) % 10 == 0 or epoch < 5 or epoch >= num_epochs - 5:
            print(f"   Epoch {epoch + 1:3d}/{num_epochs}: 平均损失 = {avg_loss:.4f}")
        
        # 学习率衰减
        if (epoch + 1) % 30 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"   学习率衰减到: {optimizer.param_groups[0]['lr']:.6f}")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n✅ 增强训练完成!")
    print(f"   初始损失: {losses[0]:.4f}")
    print(f"   最终损失: {losses[-1]:.4f}")
    print(f"   损失下降: {losses[0] - losses[-1]:.4f}")
    print(f"   训练时间: {training_time:.1f}秒")
    
    # 保存模型
    save_dir = "/home/kyc/project/RT-DETR/results/enhanced_training"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "rtdetr_enhanced_training.pkl")
    
    model.save(save_path)
    
    # 保存训练结果
    training_results = {
        'losses': losses,
        'training_time': training_time,
        'num_epochs': num_epochs,
        'final_loss': losses[-1],
        'loss_reduction': losses[0] - losses[-1],
        'total_params': total_params,
        'dataset_size': len(dataset)
    }
    
    with open(os.path.join(save_dir, "enhanced_training_results.pkl"), 'wb') as f:
        import pickle
        pickle.dump(training_results, f)
    
    print(f"💾 增强模型保存到: {save_path}")
    print(f"📊 训练结果保存到: {save_dir}/enhanced_training_results.pkl")
    print(f"\n🎉 增强训练完成！")
    
    return losses, training_time, total_params

if __name__ == "__main__":
    enhanced_training()
