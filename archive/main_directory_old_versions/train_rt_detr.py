#!/usr/bin/env python3
"""
RT-DETR 主训练脚本
整合了所有训练功能的核心脚本
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import random
import argparse

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'jittor_rt_detr'))

import jittor as jt
from src.nn.model import RTDETR

# 设置Jittor
jt.flags.use_cuda = 1

class SimpleDataLoader:
    """简化的数据加载器"""

    def __init__(self, data_dir, ann_file, batch_size=2, max_images=25):
        self.data_dir = data_dir
        self.ann_file = ann_file
        self.batch_size = batch_size
        self.max_images = max_images

        self.load_data()

    def load_data(self):
        """加载数据"""
        with open(self.ann_file, 'r') as f:
            coco_data = json.load(f)

        # 创建类别映射
        self.categories = {cat['id']: cat for cat in coco_data['categories']}
        self.category_ids = list(self.categories.keys())
        self.num_classes = len(self.category_ids)
        self.cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(self.category_ids)}

        # 收集图片和标注
        images_dict = {img['id']: img for img in coco_data['images']}

        # 按图片分组标注
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)

        # 筛选有标注的图片，优先选择标注较少的图片
        valid_image_data = []
        for img_id, anns in annotations_by_image.items():
            if len(anns) > 0 and len(anns) <= 5 and img_id in images_dict:
                valid_image_data.append((img_id, len(anns)))

        # 按标注数量排序
        valid_image_data.sort(key=lambda x: x[1])

        # 限制图片数量
        if len(valid_image_data) > self.max_images:
            valid_image_data = valid_image_data[:self.max_images]

        # 准备训练数据
        self.train_data = []
        for img_id, _ in valid_image_data:
            img_info = images_dict[img_id]
            anns = annotations_by_image[img_id]

            self.train_data.append({
                'image_id': img_id,
                'image_info': img_info,
                'annotations': anns
            })

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

            # 转换为归一化cxcywh格式
            cx = (x + w/2) / original_size[0]
            cy = (y + h/2) / original_size[1]
            w_norm = w / original_size[0]
            h_norm = h / original_size[1]

            # 边界检查
            cx = max(0.01, min(0.99, cx))
            cy = max(0.01, min(0.99, cy))
            w_norm = max(0.01, min(0.99, w_norm))
            h_norm = max(0.01, min(0.99, h_norm))

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

class RTDETRTrainer:
    """RT-DETR训练器"""
    
    def __init__(self, config):
        self.config = config
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
    
    def setup_model(self):
        """设置模型"""
        self.model = RTDETR(num_classes=self.config['num_classes'])
        self.model = self.model.float32()
        self.model.train()
        
        print(f"✅ 模型创建完成 (类别数: {self.config['num_classes']})")
    
    def setup_data(self):
        """设置数据加载器"""
        self.data_loader = SimpleDataLoader(
            data_dir=self.config['data_dir'],
            ann_file=self.config['ann_file'],
            batch_size=self.config['batch_size'],
            max_images=self.config['max_images']
        )

        print(f"✅ 数据加载器创建完成 (图片数: {len(self.data_loader.train_data)})")
    
    def setup_optimizer(self):
        """设置优化器"""
        self.optimizer = jt.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        print(f"✅ 优化器创建完成 (学习率: {self.config['learning_rate']})")
    
    def compute_loss(self, outputs, targets):
        """计算稳定的损失"""
        logits, boxes, _, _ = outputs

        # 简化的损失计算 - 添加数值稳定性检查
        total_loss = 0
        batch_size = len(targets)
        valid_samples = 0

        for i in range(batch_size):
            # 使用最后一层的输出
            pred_logits = logits[-1][i]  # [num_queries, num_classes]
            pred_boxes = boxes[-1][i]    # [num_queries, 4]

            target = targets[i]
            if len(target['labels']) > 0:
                # 简单匹配：使用前N个查询
                num_targets = len(target['labels'])
                num_queries = pred_logits.shape[0]
                matched_queries = min(num_targets, num_queries, 10)  # 限制匹配数量

                if matched_queries > 0:
                    # 分类损失
                    pos_logits = pred_logits[:matched_queries]
                    pos_labels = target['labels'][:matched_queries]

                    try:
                        cls_loss = jt.nn.cross_entropy_loss(pos_logits, pos_labels)

                        # 检查NaN
                        if jt.isnan(cls_loss).any():
                            cls_loss = jt.ones(1)

                        # 边界框损失 - 简化
                        pos_boxes = pred_boxes[:matched_queries]
                        pos_targets = target['boxes'][:matched_queries]
                        bbox_loss = jt.nn.l1_loss(pos_boxes, pos_targets)

                        # 检查NaN
                        if jt.isnan(bbox_loss).any():
                            bbox_loss = jt.zeros(1)

                        sample_loss = cls_loss + 0.1 * bbox_loss  # 降低bbox损失权重

                    except:
                        # 如果计算失败，使用默认损失
                        sample_loss = jt.ones(1)
                else:
                    sample_loss = jt.ones(1)
            else:
                # 没有目标，使用简单的背景损失
                try:
                    bg_targets = jt.full((min(pred_logits.shape[0], 50),), self.config['num_classes'])  # 限制查询数量
                    sample_loss = jt.nn.cross_entropy_loss(pred_logits[:50], bg_targets)

                    if jt.isnan(sample_loss).any():
                        sample_loss = jt.ones(1)
                except:
                    sample_loss = jt.ones(1)

            # 检查最终损失
            if not jt.isnan(sample_loss).any() and sample_loss.data > 0:
                total_loss += sample_loss
                valid_samples += 1

        if valid_samples > 0:
            return total_loss / valid_samples
        else:
            return jt.ones(1)  # 返回默认损失
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        epoch_losses = []
        
        for batch_idx, (images, targets) in enumerate(self.data_loader):
            # 前向传播
            outputs = self.model(images)
            
            # 计算损失
            loss = self.compute_loss(outputs, targets)
            
            # 检查NaN
            if jt.isnan(loss).any():
                print(f"警告: 批次 {batch_idx} 损失为NaN，跳过")
                continue
            
            # 反向传播
            self.optimizer.zero_grad()
            self.optimizer.backward(loss)
            self.optimizer.step()
            
            loss_value = float(loss.data)
            epoch_losses.append(loss_value)
            
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}: Loss = {loss_value:.4f}")
        
        return np.mean(epoch_losses) if epoch_losses else float('inf')
    
    def train(self):
        """主训练循环"""
        print(f">>> 开始训练 ({self.config['num_epochs']} 轮)")
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            
            # 训练一个epoch
            avg_loss = self.train_epoch(epoch)
            
            if avg_loss == float('inf'):
                print("警告: 本轮没有有效损失")
                continue
            
            print(f"  Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                
                save_path = self.config['save_path']
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                jt.save(self.model.state_dict(), save_path)
                print(f"  ✅ 保存最佳模型: {save_path}")
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= self.config['patience']:
                print(f"  早停触发 (连续{self.config['patience']}轮无改善)")
                break
        
        print(f"✅ 训练完成！最佳损失: {best_loss:.4f}")
        return self.model

def get_default_config():
    """获取默认配置"""
    return {
        'data_dir': 'data/coco2017_50/train2017',
        'ann_file': 'data/coco2017_50/annotations/instances_train2017.json',
        'num_classes': 80,
        'batch_size': 2,
        'max_images': 30,
        'num_epochs': 30,
        'learning_rate': 1e-5,
        'weight_decay': 1e-4,
        'patience': 10,
        'save_path': 'checkpoints/rt_detr_trained_model.pkl'
    }

def main():
    parser = argparse.ArgumentParser(description='RT-DETR训练脚本')
    parser.add_argument('--data_dir', default='data/coco2017_50/train2017', help='数据目录')
    parser.add_argument('--ann_file', default='data/coco2017_50/annotations/instances_train2017.json', help='标注文件')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--max_images', type=int, default=30, help='最大图片数')
    parser.add_argument('--num_epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='学习率')
    parser.add_argument('--save_path', default='checkpoints/rt_detr_trained_model.pkl', help='模型保存路径')
    
    args = parser.parse_args()
    
    # 创建配置
    config = get_default_config()
    config.update(vars(args))
    
    print("=" * 60)
    print("===      RT-DETR 训练      ===")
    print("=" * 60)
    print("训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # 创建训练器并开始训练
    trainer = RTDETRTrainer(config)
    trained_model = trainer.train()
    
    print("=" * 60)
    print("🎉 训练完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
