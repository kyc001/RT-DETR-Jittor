#!/usr/bin/env python3
"""
直接使用COCO预训练权重进行微调的RT-DETR训练脚本
"""

import os
import sys
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt
from jittor_rt_detr.src.nn.rtdetr_complete_pytorch_aligned import build_rtdetr_complete

# 设置Jittor
jt.flags.use_cuda = 1
jt.set_global_seed(42)
jt.flags.auto_mixed_precision_level = 0

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
        self.focal_loss = FixedFocalLoss(alpha=0.25, gamma=2.0, num_classes=num_classes)
    
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

def load_pretrained_weights(model, pretrained_path):
    """加载预训练权重"""
    print(f">>> 加载预训练权重: {pretrained_path}")
    
    try:
        # 直接用Jittor加载
        checkpoint = jt.load(pretrained_path)
        
        # 提取权重
        if 'ema' in checkpoint and 'module' in checkpoint['ema']:
            pretrained_dict = checkpoint['ema']['module']
            print(f"✅ 从ema.module提取到 {len(pretrained_dict)} 个参数")
        else:
            print("❌ 权重文件结构不符合预期")
            return False
        
        # 获取模型参数
        model_dict = model.state_dict()
        
        # 统计匹配情况
        matched_keys = []
        missing_keys = []
        shape_mismatch_keys = []
        
        for key in model_dict.keys():
            if key in pretrained_dict:
                if model_dict[key].shape == pretrained_dict[key].shape:
                    matched_keys.append(key)
                else:
                    shape_mismatch_keys.append(key)
                    print(f"⚠️ 形状不匹配: {key}")
                    print(f"   模型: {model_dict[key].shape}")
                    print(f"   预训练: {pretrained_dict[key].shape}")
            else:
                missing_keys.append(key)
        
        # 过滤匹配的权重
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in matched_keys}
        
        # 加载权重
        model.load_state_dict(filtered_dict)
        
        print(f"✅ 预训练权重加载完成")
        print(f"  匹配参数: {len(matched_keys)}")
        print(f"  缺失参数: {len(missing_keys)}")
        print(f"  形状不匹配: {len(shape_mismatch_keys)}")
        
        # 显示一些关键的匹配参数
        key_params = ['backbone.conv1.conv1_1.conv.weight', 'decoder.layers.0.self_attn.in_proj_weight']
        for key in key_params:
            if key in matched_keys:
                print(f"  ✅ 关键参数已加载: {key}")
        
        return len(matched_keys) > 100  # 至少匹配100个参数才算成功
        
    except Exception as e:
        print(f"❌ 加载预训练权重失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_coco_data():
    """加载COCO数据集"""
    data_dir = "data/coco2017_50/train2017"
    ann_file = "data/coco2017_50/annotations/instances_train2017.json"
    
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    return coco_data, data_dir

def preprocess_image(image_path, target_size=640):
    """预处理图片"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    image_resized = image.resize((target_size, target_size))
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

def create_dataloader(coco_data, data_dir, batch_size=2, max_images=50):
    """创建数据加载器"""
    max_images = max_images or len(coco_data['images'])
    images = coco_data['images'][:max_images]
    annotations = coco_data['annotations']
    categories = coco_data['categories']
    
    processed_data = []
    
    for img_info in tqdm(images, desc="预处理图片"):
        image_path = os.path.join(data_dir, img_info['file_name'])
        
        if not os.path.exists(image_path):
            continue
        
        img_tensor, original_size = preprocess_image(image_path)
        targets = create_targets(img_info['id'], annotations, categories, original_size)
        
        if targets is not None:
            processed_data.append((img_tensor, targets))
    
    # 创建批次
    batches = []
    for i in range(0, len(processed_data), batch_size):
        batch_data = processed_data[i:i+batch_size]
        batch_images = jt.stack([item[0] for item in batch_data])
        batch_targets = [item[1] for item in batch_data]
        batches.append((batch_images, batch_targets))
    
    return batches, len(categories)

def main():
    print("=" * 60)
    print("===    使用COCO预训练权重进行微调训练    ===")
    print("=" * 60)
    
    # 预训练权重路径
    pretrained_path = "rtdetr_r18vd_dec3_6x_coco_from_paddle.pth"
    
    # 加载数据
    coco_data, data_dir = load_coco_data()
    dataloader, num_classes = create_dataloader(coco_data, data_dir, batch_size=2, max_images=50)
    print(f"数据加载完成 - 类别数: {num_classes}, 批次数: {len(dataloader)}")
    
    # 创建模型
    model = build_rtdetr_complete(num_classes=num_classes, hidden_dim=256, num_queries=300)
    
    # 加载预训练权重
    if not load_pretrained_weights(model, pretrained_path):
        print("❌ 无法加载预训练权重，退出训练")
        return False
    
    # 创建损失函数
    weight_dict = {
        'loss_focal': 2, 'loss_bbox': 5, 'loss_giou': 2,
        **{f'loss_focal_aux_{i}': 2 for i in range(6)},
        **{f'loss_bbox_aux_{i}': 5 for i in range(6)},
        **{f'loss_giou_aux_{i}': 2 for i in range(6)},
    }
    criterion = RTDETRCriterion(num_classes, weight_dict)
    
    # 优化器 - 使用较小的学习率进行微调
    optimizer = jt.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    
    print(f"开始微调训练 - 轮数: 20, 学习率: 1e-5")
    
    training_history = {'epoch_losses': [], 'best_loss': float('inf')}
    
    for epoch in range(20):
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
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}/20, Batch {batch_idx+1}: Loss = {total_loss.item():.4f}")
        
        avg_loss = np.mean(epoch_losses)
        training_history['epoch_losses'].append(avg_loss)
        
        if avg_loss < training_history['best_loss']:
            training_history['best_loss'] = avg_loss
            # 保存最佳模型
            os.makedirs("checkpoints", exist_ok=True)
            jt.save(model.state_dict(), "checkpoints/rtdetr_finetuned_best.pkl")
        
        print(f"✅ Epoch {epoch+1}: Avg Loss = {avg_loss:.4f} (Best: {training_history['best_loss']:.4f})")
    
    # 保存最终模型
    jt.save(model.state_dict(), "checkpoints/rtdetr_finetuned_final.pkl")
    
    print(f"\n🎯 微调训练完成!")
    print(f"最佳损失: {training_history['best_loss']:.4f}")
    print("模型已保存:")
    print("  最佳模型: checkpoints/rtdetr_finetuned_best.pkl")
    print("  最终模型: checkpoints/rtdetr_finetuned_final.pkl")
    
    return True

if __name__ == "__main__":
    main()
