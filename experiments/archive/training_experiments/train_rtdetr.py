#!/usr/bin/env python3
"""
RT-DETR Jittor版本训练主脚本
使用方法: python train_rtdetr.py
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt
from jittor_rt_detr.src.nn.rtdetr_complete_pytorch_aligned import build_rtdetr_complete
from config import *

def setup_jittor():
    """设置Jittor环境"""
    jt.flags.use_cuda = JITTOR_CONFIG['use_cuda']
    jt.set_global_seed(JITTOR_CONFIG['seed'])
    jt.flags.auto_mixed_precision_level = JITTOR_CONFIG['auto_mixed_precision_level']

def safe_float32(tensor):
    """安全地将tensor转换为float32"""
    if isinstance(tensor, jt.Var):
        return tensor.float32()
    elif isinstance(tensor, np.ndarray):
        return jt.array(tensor.astype(np.float32))
    else:
        return jt.array(tensor, dtype=jt.float32)

def safe_int64(tensor):
    """安全地将tensor转换为int64"""
    if isinstance(tensor, jt.Var):
        return tensor.int64()
    elif isinstance(tensor, np.ndarray):
        return jt.array(tensor.astype(np.int64))
    else:
        return jt.array(tensor, dtype=jt.int64)

class FixedFocalLoss(jt.nn.Module):
    """修复数据类型问题的Focal Loss"""
    
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=80):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
    
    def execute(self, src_logits, target_classes, num_boxes):
        src_logits = safe_float32(src_logits)
        target_classes = safe_int64(target_classes)
        
        target_onehot = jt.zeros(src_logits.shape, dtype=jt.float32)
        target_onehot.scatter_(-1, target_classes.unsqueeze(-1), safe_float32(1.0))
        
        sigmoid_p = jt.sigmoid(src_logits).float32()
        ce_loss = -(target_onehot * jt.log(sigmoid_p + 1e-8) + 
                   (1 - target_onehot) * jt.log(1 - sigmoid_p + 1e-8)).float32()
        p_t = (sigmoid_p * target_onehot + (1 - sigmoid_p) * (1 - target_onehot)).float32()
        focal_weight = ((1 - p_t) ** self.gamma).float32()
        loss = (ce_loss * focal_weight).float32()
        
        if self.alpha >= 0:
            alpha_t = (self.alpha * target_onehot + (1 - self.alpha) * (1 - target_onehot)).float32()
            loss = (alpha_t * loss).float32()
        
        return (loss.mean(1).sum() * src_logits.shape[1] / num_boxes).float32()

class RTDETRCriterion(jt.nn.Module):
    """RT-DETR损失函数"""
    
    def __init__(self, num_classes, weight_dict):
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.focal_loss = FixedFocalLoss(**FOCAL_LOSS_CONFIG, num_classes=num_classes)
    
    def loss_labels(self, outputs, targets, indices, num_boxes, suffix=""):
        src_logits = safe_float32(outputs['pred_logits'])
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = jt.concat([safe_int64(t["labels"][J]) for t, (_, J) in zip(targets, indices)])
        target_classes = jt.full(src_logits.shape[:2], self.num_classes, dtype=jt.int64)
        target_classes[idx] = target_classes_o
        
        loss_focal = self.focal_loss(src_logits, target_classes, num_boxes)
        return {f'loss_focal{suffix}': safe_float32(loss_focal)}
    
    def loss_boxes(self, outputs, targets, indices, num_boxes, suffix=""):
        src_boxes = safe_float32(outputs['pred_boxes'])
        idx = self._get_src_permutation_idx(indices)
        src_boxes = src_boxes[idx]
        target_boxes = jt.concat([safe_float32(t['boxes'][i]) for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = jt.nn.l1_loss(src_boxes, target_boxes) / num_boxes
        loss_giou = safe_float32(jt.array(0.5))
        
        return {f'loss_bbox{suffix}': safe_float32(loss_bbox), f'loss_giou{suffix}': loss_giou}
    
    def _get_src_permutation_idx(self, indices):
        batch_idx = jt.concat([jt.full_like(safe_int64(src), i) for i, (src, _) in enumerate(indices)])
        src_idx = jt.concat([safe_int64(src) for (src, _) in indices])
        return batch_idx, src_idx
    
    def execute(self, outputs, targets):
        indices = []
        for i, target in enumerate(targets):
            num_targets = len(target['labels'])
            if num_targets > 0:
                src_idx = list(range(min(num_targets, outputs['pred_logits'].shape[1])))
                tgt_idx = list(range(len(src_idx)))
                indices.append((jt.array(src_idx), jt.array(tgt_idx)))
            else:
                indices.append((jt.array([]), jt.array([])))
        
        num_boxes = max(1, sum(len(t["labels"]) for t in targets))
        losses = {}
        
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                aux_outputs_clean = {
                    'pred_logits': safe_float32(aux_outputs['pred_logits']),
                    'pred_boxes': safe_float32(aux_outputs['pred_boxes'])
                }
                losses.update(self.loss_labels(aux_outputs_clean, targets, indices, num_boxes, f'_aux_{i}'))
                losses.update(self.loss_boxes(aux_outputs_clean, targets, indices, num_boxes, f'_aux_{i}'))
        
        return losses

def load_data():
    """加载数据"""
    with open(DATA_CONFIG['ann_file'], 'r') as f:
        coco_data = json.load(f)
    return coco_data

def preprocess_image(image_path):
    """预处理图片"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    image_resized = image.resize((DATA_CONFIG['target_size'], DATA_CONFIG['target_size']))
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    
    return safe_float32(img_array), original_size

def create_targets(image_id, annotations, categories, original_size):
    """创建训练目标"""
    image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]
    if not image_annotations:
        return None
    
    cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(categories)}
    boxes, labels = [], []
    
    for ann in image_annotations:
        x, y, w, h = ann['bbox']
        cx = np.float32(x + w / 2) / np.float32(original_size[0])
        cy = np.float32(y + h / 2) / np.float32(original_size[1])
        w_norm = np.float32(w) / np.float32(original_size[0])
        h_norm = np.float32(h) / np.float32(original_size[1])
        
        boxes.append([cx, cy, w_norm, h_norm])
        labels.append(cat_id_to_idx[ann['category_id']])
    
    return {
        'boxes': safe_float32(np.array(boxes, dtype=np.float32)),
        'labels': safe_int64(np.array(labels, dtype=np.int64))
    }

def create_dataloader(coco_data):
    """创建数据加载器"""
    max_images = DATA_CONFIG['max_images'] or len(coco_data['images'])
    images = coco_data['images'][:max_images]
    annotations = coco_data['annotations']
    categories = coco_data['categories']
    
    processed_data = []
    
    for img_info in tqdm(images, desc="预处理图片"):
        image_path = os.path.join(DATA_CONFIG['data_dir'], img_info['file_name'])
        
        if not os.path.exists(image_path):
            continue
        
        img_tensor, original_size = preprocess_image(image_path)
        targets = create_targets(img_info['id'], annotations, categories, original_size)
        
        if targets is not None:
            processed_data.append((img_tensor, targets))
    
    # 创建批次
    batches = []
    batch_size = DATA_CONFIG['batch_size']
    for i in range(0, len(processed_data), batch_size):
        batch_data = processed_data[i:i+batch_size]
        batch_images = jt.stack([item[0] for item in batch_data])
        batch_targets = [item[1] for item in batch_data]
        batches.append((batch_images, batch_targets))
    
    return batches, len(categories)

def train_model(model, criterion, dataloader):
    """训练模型"""
    config = TRAINING_CONFIG
    optimizer = jt.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    training_history = {'epoch_losses': [], 'best_loss': float('inf')}
    
    print(f"开始训练 - 轮数: {config['num_epochs']}")
    
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_losses = []
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            # 确保数据类型
            images = safe_float32(images)
            for target in targets:
                target['boxes'] = safe_float32(target['boxes'])
                target['labels'] = safe_int64(target['labels'])
            
            # 前向传播
            outputs = model(images, targets)
            
            # 确保输出类型
            for key in outputs:
                if isinstance(outputs[key], jt.Var):
                    outputs[key] = safe_float32(outputs[key])
                elif isinstance(outputs[key], list):
                    for item in outputs[key]:
                        if isinstance(item, dict):
                            for k, v in item.items():
                                if isinstance(v, jt.Var):
                                    item[k] = safe_float32(v)
            
            # 计算损失
            loss_dict = criterion(outputs, targets)
            total_loss = safe_float32(0.0)
            for k, v in loss_dict.items():
                if k in criterion.weight_dict:
                    total_loss = total_loss + safe_float32(v) * criterion.weight_dict[k]
            
            # 反向传播
            optimizer.step(total_loss)
            epoch_losses.append(total_loss.item())
            
            if batch_idx % config['print_freq'] == 0:
                print(f"Epoch {epoch+1}/{config['num_epochs']}, Batch {batch_idx+1}: Loss = {total_loss.item():.4f}")
        
        avg_loss = np.mean(epoch_losses)
        training_history['epoch_losses'].append(avg_loss)
        
        if avg_loss < training_history['best_loss']:
            training_history['best_loss'] = avg_loss
        
        print(f"✅ Epoch {epoch+1}: Avg Loss = {avg_loss:.4f} (Best: {training_history['best_loss']:.4f})")
    
    return model, training_history

def save_results(model, training_history):
    """保存结果"""
    # 保存模型
    os.makedirs(os.path.dirname(SAVE_CONFIG['model_save_path']), exist_ok=True)
    jt.save(model.state_dict(), SAVE_CONFIG['model_save_path'])
    print(f"✅ 模型已保存到: {SAVE_CONFIG['model_save_path']}")
    
    # 保存训练历史图
    if training_history['epoch_losses']:
        plt.figure(figsize=(10, 6))
        plt.plot(training_history['epoch_losses'], 'b-', linewidth=2, label='Training Loss')
        plt.axhline(y=training_history['best_loss'], color='r', linestyle='--', 
                    label=f'Best Loss: {training_history["best_loss"]:.4f}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('RT-DETR Jittor Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        os.makedirs(os.path.dirname(SAVE_CONFIG['history_save_path']), exist_ok=True)
        plt.savefig(SAVE_CONFIG['history_save_path'], dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 训练历史图已保存到: {SAVE_CONFIG['history_save_path']}")

def main():
    print("=" * 60)
    print("===        RT-DETR Jittor版本训练        ===")
    print("=" * 60)
    
    # 设置环境
    setup_jittor()
    
    # 加载数据
    coco_data = load_data()
    dataloader, num_classes = create_dataloader(coco_data)
    print(f"数据加载完成 - 类别数: {num_classes}, 批次数: {len(dataloader)}")
    
    # 创建模型
    model = build_rtdetr_complete(num_classes=num_classes, **MODEL_CONFIG)
    
    # 创建损失函数
    weight_dict = LOSS_WEIGHTS.copy()
    weight_dict.update({f'loss_focal_aux_{i}': LOSS_WEIGHTS['loss_focal'] for i in range(6)})
    weight_dict.update({f'loss_bbox_aux_{i}': LOSS_WEIGHTS['loss_bbox'] for i in range(6)})
    weight_dict.update({f'loss_giou_aux_{i}': LOSS_WEIGHTS['loss_giou'] for i in range(6)})
    
    criterion = RTDETRCriterion(num_classes, weight_dict)
    print("✅ 模型和损失函数创建完成")
    
    # 训练
    model, training_history = train_model(model, criterion, dataloader)
    
    # 保存结果
    save_results(model, training_history)
    
    print(f"\n🎯 训练完成 - 最佳损失: {training_history['best_loss']:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
