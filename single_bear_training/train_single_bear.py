#!/usr/bin/env python3
"""
单图片熊检测训练脚本
专门用于训练RT-DETR模型识别000000000285.jpg中的熊
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import json

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR/jittor_rt_detr')

import jittor as jt
from jittor import nn
from src.nn.model import RTDETR

# 简单的数据变换
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

class Resize:
    def __init__(self, size):
        self.size = size
    
    def __call__(self, image):
        return image.resize(self.size, Image.LANCZOS)

class ToTensor:
    def __call__(self, image):
        image_array = np.array(image, dtype=np.float32) / 255.0
        return jt.array(image_array.transpose(2, 0, 1))

class Normalize:
    def __init__(self, mean, std):
        self.mean = jt.array(mean).view(-1, 1, 1)
        self.std = jt.array(std).view(-1, 1, 1)
    
    def __call__(self, tensor):
        return (tensor - self.mean) / self.std

def load_single_image_data(img_path, ann_file, target_image_name="000000000285.jpg"):
    """加载单张图片的数据和标注"""
    print(f"=== 加载单张图片数据: {target_image_name} ===")
    
    # 加载COCO标注
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # 找到目标图片
    target_image = None
    for img in coco_data['images']:
        if img['file_name'] == target_image_name:
            target_image = img
            break
    
    if target_image is None:
        raise ValueError(f"找不到图片: {target_image_name}")
    
    print(f"✅ 找到图片: {target_image['file_name']}")
    print(f"   尺寸: {target_image['width']} x {target_image['height']}")
    print(f"   ID: {target_image['id']}")
    
    # 找到该图片的所有标注
    image_annotations = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == target_image['id']:
            image_annotations.append(ann)
    
    print(f"✅ 找到 {len(image_annotations)} 个标注")
    
    # 分析标注中的类别
    category_counts = {}
    for ann in image_annotations:
        cat_id = ann['category_id']
        if cat_id not in category_counts:
            category_counts[cat_id] = 0
        category_counts[cat_id] += 1
    
    # 获取类别名称
    cat_id_to_name = {}
    for cat in coco_data['categories']:
        cat_id_to_name[cat['id']] = cat['name']
    
    print("📊 图片中的类别分布:")
    bear_found = False
    for cat_id, count in category_counts.items():
        cat_name = cat_id_to_name.get(cat_id, f"Unknown_{cat_id}")
        print(f"   {cat_name}: {count} 个实例")
        if cat_name == 'bear':
            bear_found = True
    
    if bear_found:
        print("🐻 ✅ 确认图片中包含熊！")
    else:
        print("❌ 图片中没有熊标注")
    
    return target_image, image_annotations, cat_id_to_name

class SingleImageDataset(jt.dataset.Dataset):
    """单图片数据集"""
    
    def __init__(self, img_path, image_info, annotations, cat_id_to_name, transforms=None):
        super().__init__()
        self.img_path = img_path
        self.image_info = image_info
        self.annotations = annotations
        self.cat_id_to_name = cat_id_to_name
        self.transforms = transforms
        
        # 创建类别ID到连续索引的映射
        unique_cat_ids = sorted(set(ann['category_id'] for ann in annotations))
        self.cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(unique_cat_ids)}
        self.idx_to_cat_id = {idx: cat_id for cat_id, idx in self.cat_id_to_idx.items()}
        
        print(f"📋 类别映射:")
        for cat_id, idx in self.cat_id_to_idx.items():
            cat_name = self.cat_id_to_name.get(cat_id, f"Unknown_{cat_id}")
            print(f"   {cat_name} (ID:{cat_id}) -> 索引:{idx}")
    
    def __len__(self):
        return 1  # 只有一张图片
    
    def __getitem__(self, idx):
        # 加载图片
        image = Image.open(self.img_path).convert('RGB')
        original_size = image.size
        
        # 准备标注
        boxes = []
        labels = []
        
        for ann in self.annotations:
            # COCO格式: [x, y, width, height] -> RT-DETR格式 [cx, cy, w, h] (归一化)
            x, y, w, h = ann['bbox']
            
            # 转换为中心点坐标并归一化
            cx = (x + w / 2) / original_size[0]  # 归一化到[0,1]
            cy = (y + h / 2) / original_size[1]  # 归一化到[0,1]
            w_norm = w / original_size[0]        # 归一化宽度
            h_norm = h / original_size[1]        # 归一化高度
            
            boxes.append([cx, cy, w_norm, h_norm])
            print(f"  标注转换: COCO[{x:.1f},{y:.1f},{w:.1f},{h:.1f}] -> RT-DETR[{cx:.3f},{cy:.3f},{w_norm:.3f},{h_norm:.3f}]")
            
            # 转换类别ID到连续索引
            cat_id = ann['category_id']
            label_idx = self.cat_id_to_idx[cat_id]
            labels.append(label_idx)
        
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # 应用变换
        if self.transforms:
            image = self.transforms(image)
        
        # 准备目标字典
        target = {
            'boxes': jt.array(boxes),
            'labels': jt.array(labels),
            'image_id': jt.array([self.image_info['id']]),
            'orig_size': jt.array([original_size[1], original_size[0]]),  # [height, width]
            'size': jt.array([640, 640])  # 变换后的尺寸
        }
        
        return image, target

def main():
    parser = argparse.ArgumentParser(description="单图片训练 - 熊图片")
    parser.add_argument('--img_path', type=str, default='data/coco/val2017/000000000285.jpg', help='图片路径')
    parser.add_argument('--ann_file', type=str, default='data/coco/annotations/instances_val2017.json', help='标注文件')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--output_dir', type=str, default='single_bear_training/checkpoints', help='模型保存目录')
    
    args = parser.parse_args()
    
    print("=== 单图片训练：熊图片识别 ===")
    print(f"配置参数:")
    print(f"  - 图片路径: {args.img_path}")
    print(f"  - 标注文件: {args.ann_file}")
    print(f"  - 训练轮数: {args.epochs}")
    print(f"  - 学习率: {args.lr}")
    print(f"  - 输出目录: {args.output_dir}")
    
    # 检查文件是否存在
    if not os.path.exists(args.img_path):
        raise FileNotFoundError(f"图片文件不存在: {args.img_path}")
    if not os.path.exists(args.ann_file):
        raise FileNotFoundError(f"标注文件不存在: {args.ann_file}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载单张图片数据
    image_info, annotations, cat_id_to_name = load_single_image_data(
        args.img_path, args.ann_file, "000000000285.jpg"
    )
    
    # 数据预处理
    transform = Compose([
        Resize((640, 640)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    dataset = SingleImageDataset(
        args.img_path, image_info, annotations, cat_id_to_name, transforms=transform
    )
    
    print(f"✅ 数据集创建完成，包含 {len(dataset)} 张图片")
    
    # 创建数据加载器
    dataloader = jt.dataset.DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 获取类别数量
    num_classes = len(dataset.cat_id_to_idx)
    print(f"📊 类别数量: {num_classes}")
    
    # 创建模型
    model = RTDETR(num_classes=num_classes)
    
    # 优化器
    optimizer = jt.optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"✅ 开始训练...")
    
    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # 前向传播
            logits, boxes, enc_logits, enc_boxes = model(images)
            
            # 正确的损失计算
            target_boxes = targets['boxes']  # (num_targets, 4)
            target_labels = targets['labels']  # (num_targets,)
            
            # 使用最后一层的预测
            pred_logits = logits[-1]  # (B, num_queries, num_classes)
            pred_boxes = boxes[-1]    # (B, num_queries, 4)
            
            # 计算损失
            if len(target_labels) > 0:
                batch_size = pred_logits.shape[0]
                num_queries = pred_logits.shape[1]
                num_targets = len(target_labels)
                
                # 为每个query计算与目标的匹配损失
                total_loss = jt.array(0.0)
                
                # 简化的匈牙利匹配：找到与真实目标最接近的query
                target_box = target_boxes[0]  # 只有一个目标
                target_label = target_labels[0]  # 只有一个标签
                
                # 计算所有query与目标的距离
                pred_boxes_flat = pred_boxes[0]  # (num_queries, 4)
                
                # L1距离
                box_distances = jt.abs(pred_boxes_flat - target_box.unsqueeze(0)).sum(dim=1)
                
                # 找到最接近的query (转换为numpy处理)
                box_distances_np = box_distances.numpy()
                best_query_idx = int(np.argmin(box_distances_np))
                
                # 对最佳匹配的query计算损失
                # 分类损失
                pred_logit_best = pred_logits[0, best_query_idx:best_query_idx+1, :]  # (1, num_classes)
                loss_cls = nn.cross_entropy_loss(pred_logit_best, target_label.unsqueeze(0))
                
                # 回归损失
                pred_box_best = pred_boxes_flat[best_query_idx]  # (4,)
                loss_bbox = nn.l1_loss(pred_box_best, target_box)
                
                # 背景损失：其他query应该预测背景
                other_queries_mask = jt.ones(num_queries, dtype=jt.bool)
                other_queries_mask[best_query_idx] = False
                
                if other_queries_mask.sum() > 0:
                    # 为其他query创建背景标签（假设背景类别是0，但我们只有1个类别）
                    # 对于单类别模型，我们希望其他query输出低置信度
                    other_logits = pred_logits[0, other_queries_mask, :]  # (num_other, num_classes)
                    # 使用sigmoid损失，希望其他query输出接近0的logits
                    loss_bg = jt.mean(jt.sigmoid(other_logits) ** 2)  # 惩罚高置信度
                else:
                    loss_bg = jt.array(0.0)
                
                # 总损失
                loss = loss_cls + 5.0 * loss_bbox + 0.1 * loss_bg
                
                # 简化打印
                pass
            else:
                loss = jt.array(0.0)
            
            # 反向传播
            optimizer.backward(loss)
            optimizer.step()
            
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item():.4f}")
        
        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            save_path = f'{args.output_dir}/bear_model_epoch_{epoch+1}.pkl'
            jt.save(model.state_dict(), save_path)
            print(f"✅ 模型已保存: {save_path}")
    
    # 保存最终模型
    final_save_path = f'{args.output_dir}/bear_model_final.pkl'
    jt.save(model.state_dict(), final_save_path)
    print(f"🎉 最终模型已保存: {final_save_path}")
    
    print("🎉 训练完成！")
    print(f"现在可以使用模型进行推理测试")

if __name__ == "__main__":
    main()
