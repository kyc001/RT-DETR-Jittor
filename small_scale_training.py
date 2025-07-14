#!/usr/bin/env python3
"""
小规模数据集训练脚本 - 适配4060显卡的训练策略
支持PASCAL VOC和COCO子集训练
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import random
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'jittor_rt_detr'))

import jittor as jt
from src.nn.model import RTDETR
from src.nn.loss import DETRLoss

# 设置Jittor
jt.flags.use_cuda = 1

class SmallScaleDataLoader:
    """小规模数据加载器"""
    
    def __init__(self, data_dir, ann_file, batch_size=2, max_images=1000):
        self.data_dir = data_dir
        self.ann_file = ann_file
        self.batch_size = batch_size
        self.max_images = max_images
        
        # 加载数据
        self.load_data()
        
    def load_data(self):
        """加载COCO格式数据"""
        print(f">>> 加载数据: {self.ann_file}")
        
        with open(self.ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # 创建类别映射
        self.categories = {cat['id']: cat for cat in coco_data['categories']}
        self.category_ids = list(self.categories.keys())
        self.num_classes = len(self.category_ids)
        self.cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(self.category_ids)}
        
        print(f"类别数量: {self.num_classes}")
        for cat_id, cat in self.categories.items():
            print(f"  - {cat['name']} (ID: {cat_id}, 索引: {self.cat_id_to_idx[cat_id]})")
        
        # 收集图片和标注
        images_dict = {img['id']: img for img in coco_data['images']}
        
        # 按图片分组标注
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # 筛选有标注的图片
        valid_image_ids = []
        for img_id, anns in annotations_by_image.items():
            if len(anns) > 0 and img_id in images_dict:
                valid_image_ids.append(img_id)
        
        # 限制图片数量
        if len(valid_image_ids) > self.max_images:
            valid_image_ids = random.sample(valid_image_ids, self.max_images)
        
        print(f"有效图片数量: {len(valid_image_ids)} (限制: {self.max_images})")
        
        # 准备训练数据
        self.train_data = []
        for img_id in valid_image_ids:
            img_info = images_dict[img_id]
            anns = annotations_by_image[img_id]
            
            self.train_data.append({
                'image_id': img_id,
                'image_info': img_info,
                'annotations': anns
            })
        
        print(f"训练样本数量: {len(self.train_data)}")
        
        # 统计类别分布
        class_counts = {}
        for data in self.train_data:
            for ann in data['annotations']:
                cat_name = self.categories[ann['category_id']]['name']
                class_counts[cat_name] = class_counts.get(cat_name, 0) + 1
        
        print(f"类别分布:")
        for cat_name, count in sorted(class_counts.items()):
            print(f"  - {cat_name}: {count} 个")
    
    def __len__(self):
        return len(self.train_data) // self.batch_size
    
    def __iter__(self):
        # 打乱数据
        random.shuffle(self.train_data)
        
        for i in range(0, len(self.train_data), self.batch_size):
            batch_data = self.train_data[i:i+self.batch_size]
            
            # 处理batch
            images = []
            targets_list = []
            
            for data in batch_data:
                img_tensor, targets = self.process_single_image(data)
                images.append(img_tensor)
                targets_list.append(targets)
            
            # 合并batch
            batch_images = jt.concat(images, dim=0)
            
            yield batch_images, targets_list
    
    def process_single_image(self, data):
        """处理单张图片"""
        img_info = data['image_info']
        annotations = data['annotations']
        
        # 加载图片
        img_path = os.path.join(self.data_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        original_size = image.size
        
        # 预处理图片
        resized_image = image.resize((640, 640), Image.LANCZOS)
        img_array = np.array(resized_image, dtype=np.float32) / 255.0
        
        # 数据增强
        if random.random() > 0.5:
            img_array = np.fliplr(img_array)  # 水平翻转
            flip_h = True
        else:
            flip_h = False
        
        # 归一化
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
            
            # 边界检查
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))
            
            boxes.append([cx, cy, w_norm, h_norm])
            labels.append(self.cat_id_to_idx[ann['category_id']])
        
        if len(boxes) == 0:
            # 如果没有有效标注，创建一个虚拟标注
            boxes = [[0.5, 0.5, 0.1, 0.1]]
            labels = [0]
        
        boxes_tensor = jt.array(boxes, dtype='float32')
        labels_tensor = jt.array(labels, dtype='int64')
        
        targets = {
            'boxes': boxes_tensor,
            'labels': labels_tensor
        }
        
        return img_tensor, targets

def create_small_scale_trainer(num_classes, learning_rate=1e-4):
    """创建小规模训练器"""
    print(f">>> 创建训练器 (类别数: {num_classes})")
    
    # 创建模型 - 使用较小的配置以适应4060显卡
    model = RTDETR(num_classes=num_classes)
    model = model.float32()
    
    # 创建损失函数
    criterion = DETRLoss(
        num_classes=num_classes,
        lambda_cls=2.0,
        lambda_bbox=5.0,
        lambda_giou=2.0,
        eos_coef=0.1
    )
    
    # 创建优化器 - 使用较小的学习率
    optimizer = jt.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    return model, criterion, optimizer

def train_small_scale(data_loader, model, criterion, optimizer, num_epochs=50):
    """小规模训练"""
    print(f">>> 开始小规模训练 ({num_epochs} 轮)")
    
    model.train()
    
    # 学习率调度 - 适配50轮训练
    def get_lr_factor(epoch):
        if epoch < 5:
            return 0.1  # 前5轮预热
        elif epoch < 35:
            return 1.0  # 主要训练阶段
        else:
            return 0.1  # 后期降低学习率
    
    best_loss = float('inf')
    patience = 15  # 增加patience，适合50轮训练
    patience_counter = 0
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # 调整学习率
        lr_factor = get_lr_factor(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-4 * lr_factor
        
        print(f"\nEpoch {epoch+1}/{num_epochs} (LR: {optimizer.param_groups[0]['lr']:.2e})")
        
        # 训练一个epoch
        for batch_idx, (images, targets_list) in enumerate(data_loader):
            # 前向传播
            outputs = model(images)
            logits, boxes, enc_logits, enc_boxes = outputs
            
            # 计算损失 - 对batch中的每个样本分别计算
            batch_loss = 0
            for i in range(len(targets_list)):
                sample_logits = logits[:, i:i+1, :, :]  # 取第i个样本
                sample_boxes = boxes[:, i:i+1, :, :]
                sample_enc_logits = enc_logits[i:i+1, :, :] if enc_logits is not None else None
                sample_enc_boxes = enc_boxes[i:i+1, :, :] if enc_boxes is not None else None
                sample_targets = [targets_list[i]]
                
                loss_dict = criterion(sample_logits, sample_boxes, sample_targets, 
                                    sample_enc_logits, sample_enc_boxes)
                sample_loss = sum(loss_dict.values())
                batch_loss += sample_loss
            
            batch_loss = batch_loss / len(targets_list)  # 平均损失
            
            # 反向传播
            optimizer.zero_grad()
            optimizer.backward(batch_loss)
            optimizer.step()
            
            epoch_losses.append(float(batch_loss.data))
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: Loss = {float(batch_loss.data):.4f}")
        
        # Epoch统计
        avg_loss = np.mean(epoch_losses)
        print(f"  Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
        
        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            # 保存最佳模型
            save_path = f"checkpoints/rt_detr_80class_best_model.pkl"
            os.makedirs("checkpoints", exist_ok=True)
            jt.save(model.state_dict(), save_path)
            print(f"  ✅ 保存最佳模型: {save_path}")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= patience:
            print(f"  早停触发 (连续{patience}轮无改善)")
            break
    
    print("✅ 训练完成")
    return model

def main():
    print("=" * 60)
    print("===      小规模数据集训练 - 4060显卡优化      ===")
    print("=" * 60)
    
    # 配置参数 - 使用现有的小规模数据
    data_dir = "data/coco2017_50/train2017"
    ann_file = "data/coco2017_50/annotations/instances_train2017.json"
    batch_size = 2  # 4060显卡适合的batch size
    max_images = 50  # 使用现有的50张图片
    num_epochs = 50  # 50轮训练
    learning_rate = 1e-4  # 适中的学习率
    
    print(f"训练配置:")
    print(f"  - 数据目录: {data_dir}")
    print(f"  - 标注文件: {ann_file}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 最大图片数: {max_images}")
    print(f"  - 训练轮数: {num_epochs}")
    print(f"  - 学习率: {learning_rate}")
    
    # 1. 创建数据加载器
    data_loader = SmallScaleDataLoader(
        data_dir=data_dir,
        ann_file=ann_file,
        batch_size=batch_size,
        max_images=max_images
    )
    
    # 2. 创建训练器
    model, criterion, optimizer = create_small_scale_trainer(
        num_classes=data_loader.num_classes,
        learning_rate=learning_rate
    )
    
    # 3. 开始训练
    trained_model = train_small_scale(
        data_loader=data_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs
    )
    
    # 4. 保存最终模型
    final_save_path = "checkpoints/rt_detr_80class_final_model.pkl"
    jt.save(trained_model.state_dict(), final_save_path)
    print(f"✅ 最终模型已保存: {final_save_path}")
    
    print("=" * 60)
    print("🎉 小规模训练完成！")
    print("💡 建议:")
    print("  - 使用训练好的模型进行推理测试")
    print("  - 在验证集上评估性能")
    print("  - 根据结果调整超参数")
    print("=" * 60)

if __name__ == "__main__":
    main()
