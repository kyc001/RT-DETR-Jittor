#!/usr/bin/env python3
"""
RT-DETR 改进版训练脚本
基于原始train.py，修复了一些潜在问题并添加了改进功能
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import random
from datetime import datetime

# --- Jittor and Project Imports ---
sys.path.insert(0, '/home/kyc/project/RT-DETR')
import jittor as jt
import jittor.nn as nn
from jittor.dataset import Dataset, DataLoader

from jittor_rt_detr.src.nn.backbone.resnet import ResNet50
from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
from jittor_rt_detr.src.nn.criterion.rtdetr_criterion import build_criterion

# --- 设置随机种子确保可复现性 ---
jt.flags.use_cuda = 1
jt.set_global_seed(42)
random.seed(42)  # 🔧 修复：确保数据集分割的可复现性
np.random.seed(42)

class ImprovedCocoDataset(Dataset):
    """改进的COCO数据集类，支持数据增强和验证集分割"""
    
    def __init__(self, image_dir, ann_file, transform=None, is_training=True, train_ratio=0.8):
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.is_training = is_training
        
        print(f"🔄 正在加载标注文件: {ann_file}...")
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # 类别映射
        categories = sorted(coco_data['categories'], key=lambda x: x['id'])
        self.cat_id_to_contiguous_id = {cat['id']: i for i, cat in enumerate(categories)}
        self.contiguous_id_to_cat_id = {i: cat['id'] for i, cat in enumerate(categories)}
        self.num_classes = len(categories)
        print(f"✅ 找到 {self.num_classes} 个类别。")

        # 构建图像到标注的映射
        self.img_id_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)
        
        # 只保留有标注的图像
        self.images = [img for img in coco_data['images'] if img['id'] in self.img_id_to_anns]
        
        # 训练/验证集分割
        random.shuffle(self.images)
        split_idx = int(len(self.images) * train_ratio)
        
        if is_training:
            self.images = self.images[:split_idx]
            print(f"✅ 训练集: {len(self.images)} 张图像")
        else:
            self.images = self.images[split_idx:]
            print(f"✅ 验证集: {len(self.images)} 张图像")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        获取单个数据样本，包含图像和标注
        修复了数据增强与标注不匹配的问题
        """
        img_info = self.images[idx]
        img_id = img_info['id']
        file_name = img_info['file_name']

        image_path = os.path.join(self.image_dir, file_name)
        image = Image.open(image_path).convert('RGB')
        original_width, original_height = image.size

        # 获取标注
        annotations = self.img_id_to_anns.get(img_id, [])

        # 🔧 修复：先提取所有标注，再应用数据增强
        boxes = []
        labels = []
        for ann in annotations:
            x, y, w, h = ann['bbox']

            # 归一化坐标 [x1, y1, x2, y2]
            x1 = x / original_width
            y1 = y / original_height
            x2 = (x + w) / original_width
            y2 = (y + h) / original_height

            # 确保坐标有效
            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= 1 and y2 <= 1:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.cat_id_to_contiguous_id[ann['category_id']])

        # 🔧 修复：应用数据增强，同时更新标注坐标
        if self.is_training:
            # 随机水平翻转
            h_flip = random.random() > 0.5
            if h_flip:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                # 更新所有边界框坐标
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    # 水平翻转时，x坐标需要变换: x' = 1 - x
                    boxes[i] = [1.0 - x2, y1, 1.0 - x1, y2]

            # 随机亮度调整 (不影响坐标)
            if random.random() > 0.5:
                factor = random.uniform(0.8, 1.2)
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(factor)

        # 转换为Jittor张量
        boxes = jt.array(boxes, dtype=jt.float32) if boxes else jt.zeros((0, 4), dtype=jt.float32)
        labels = jt.array(labels, dtype=jt.int64) if labels else jt.zeros((0,), dtype=jt.int64)

        target = {'boxes': boxes, 'labels': labels}

        # 图像预处理
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_resized = image.resize((640, 640), Image.LANCZOS)
            img_array = np.array(image_resized).astype(np.float32) / 255.0
            image_tensor = jt.array(img_array.transpose(2, 0, 1)).float32()

        return image_tensor, target

    def collate_batch(self, batch):
        """自定义批次整理函数"""
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        images = jt.stack(images, dim=0)
        return images, targets

def fix_batchnorm(module):
    """修复BatchNorm设置"""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()
            # 确保BatchNorm参数可训练
            if hasattr(m, 'weight') and m.weight is not None:
                m.weight.requires_grad = True
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = True

def train_one_epoch(model, criterion, optimizer, data_loader, epoch):
    """改进的训练函数"""
    model.train()
    criterion.train()
    
    # 修复BatchNorm
    fix_batchnorm(model)

    total_loss = 0
    loss_components = {}
    
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1} Training")
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        try:
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # 记录损失组件
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = []
                loss_components[key].append(value.item())
            
            optimizer.step(losses)
            
            total_loss += losses.item()
            progress_bar.set_postfix(loss=f"{losses.item():.4f}")
            
        except Exception as e:
            print(f"❌ 训练批次 {batch_idx} 失败: {e}")
            continue
        
    avg_loss = total_loss / len(data_loader)
    
    # 打印详细损失信息
    print(f"Epoch {epoch+1} Training Results:")
    print(f"  Average Total Loss: {avg_loss:.4f}")
    for key, values in loss_components.items():
        avg_component = np.mean(values)
        print(f"  Average {key}: {avg_component:.4f}")
    
    return avg_loss

def evaluate(model, criterion, data_loader):
    """改进的评估函数"""
    model.eval()
    criterion.eval()
    
    total_loss = 0
    loss_components = {}
    
    with jt.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating")
        for batch_idx, (images, targets) in enumerate(progress_bar):
            try:
                outputs = model(images)
                loss_dict = criterion(outputs, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # 记录损失组件
                for key, value in loss_dict.items():
                    if key not in loss_components:
                        loss_components[key] = []
                    loss_components[key].append(value.item())
                
                total_loss += losses.item()
                progress_bar.set_postfix(loss=f"{losses.item():.4f}")
                
            except Exception as e:
                print(f"❌ 验证批次 {batch_idx} 失败: {e}")
                continue

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else float('inf')
    
    # 打印详细损失信息
    print(f"Validation Results:")
    print(f"  Average Total Loss: {avg_loss:.4f}")
    for key, values in loss_components.items():
        avg_component = np.mean(values)
        print(f"  Average {key}: {avg_component:.4f}")
    
    return avg_loss

class RTDETRModel(nn.Module):
    """改进的RT-DETR模型包装器"""
    def __init__(self, num_classes, pretrained_backbone=True):
        super().__init__()
        self.backbone = ResNet50(pretrained=pretrained_backbone)
        self.transformer = RTDETRTransformer(
            num_classes=num_classes,
            hidden_dim=256,
            num_queries=300,
            feat_channels=[256, 512, 1024, 2048]
        )

    def execute(self, x):
        features = self.backbone(x)
        return self.transformer(features)

def create_shared_dataset_split(image_dir, ann_file, train_ratio=0.8):
    """
    🔧 修复：创建共享的数据集分割，确保训练集和验证集不重叠
    """
    print(f"🔄 正在加载标注文件进行数据集分割: {ann_file}...")
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)

    # 构建图像到标注的映射
    img_id_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)

    # 只保留有标注的图像
    valid_images = [img for img in coco_data['images'] if img['id'] in img_id_to_anns]

    # 固定随机分割
    random.shuffle(valid_images)
    split_idx = int(len(valid_images) * train_ratio)

    train_images = valid_images[:split_idx]
    val_images = valid_images[split_idx:]

    print(f"✅ 数据集分割完成: 训练集 {len(train_images)} 张, 验证集 {len(val_images)} 张")

    return train_images, val_images, coco_data

class FixedCocoDataset(Dataset):
    """
    🔧 修复：使用预分割的数据集，避免重复分割问题
    """
    def __init__(self, image_dir, images_list, coco_data, is_training=True):
        super().__init__()
        self.image_dir = image_dir
        self.images = images_list
        self.is_training = is_training

        # 类别映射
        categories = sorted(coco_data['categories'], key=lambda x: x['id'])
        self.cat_id_to_contiguous_id = {cat['id']: i for i, cat in enumerate(categories)}
        self.contiguous_id_to_cat_id = {i: cat['id'] for i, cat in enumerate(categories)}
        self.num_classes = len(categories)

        # 构建图像到标注的映射
        self.img_id_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)

        mode = "训练" if is_training else "验证"
        print(f"✅ {mode}数据集创建完成: {len(self.images)} 张图像, {self.num_classes} 个类别")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """获取单个数据样本，包含正确的数据增强"""
        img_info = self.images[idx]
        img_id = img_info['id']
        file_name = img_info['file_name']

        image_path = os.path.join(self.image_dir, file_name)
        image = Image.open(image_path).convert('RGB')
        original_width, original_height = image.size

        # 获取标注
        annotations = self.img_id_to_anns.get(img_id, [])

        # 先提取所有标注
        boxes = []
        labels = []
        for ann in annotations:
            x, y, w, h = ann['bbox']

            # 归一化坐标 [x1, y1, x2, y2]
            x1 = x / original_width
            y1 = y / original_height
            x2 = (x + w) / original_width
            y2 = (y + h) / original_height

            # 确保坐标有效
            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= 1 and y2 <= 1:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.cat_id_to_contiguous_id[ann['category_id']])

        # 应用数据增强，同时更新标注坐标
        if self.is_training:
            # 随机水平翻转
            h_flip = random.random() > 0.5
            if h_flip:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                # 更新所有边界框坐标
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    # 水平翻转时，x坐标需要变换: x' = 1 - x
                    boxes[i] = [1.0 - x2, y1, 1.0 - x1, y2]

            # 随机亮度调整 (不影响坐标)
            if random.random() > 0.5:
                factor = random.uniform(0.8, 1.2)
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(factor)

        # 转换为Jittor张量
        boxes = jt.array(boxes, dtype=jt.float32) if boxes else jt.zeros((0, 4), dtype=jt.float32)
        labels = jt.array(labels, dtype=jt.int64) if labels else jt.zeros((0,), dtype=jt.int64)

        target = {'boxes': boxes, 'labels': labels}

        # 图像预处理
        image_resized = image.resize((640, 640), Image.LANCZOS)
        img_array = np.array(image_resized).astype(np.float32) / 255.0
        image_tensor = jt.array(img_array.transpose(2, 0, 1)).float32()

        return image_tensor, target

    def collate_batch(self, batch):
        """自定义批次整理函数"""
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        images = jt.stack(images, dim=0)
        return images, targets

def save_training_log(log_data, save_path):
    """保存训练日志"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"📊 训练日志已保存: {save_path}")

def plot_training_curves(train_losses, val_losses, save_path):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')

    plt.title('RT-DETR Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📈 训练曲线已保存: {save_path}")

def main(args):
    print("🎯 RT-DETR 改进版训练脚本")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # --- 1. 🔧 修复：创建共享的数据集分割 ---
    print("📂 创建数据集分割...")
    train_images, val_images, coco_data = create_shared_dataset_split(
        image_dir=os.path.join(args.data_path, 'train2017'),
        ann_file=os.path.join(args.data_path, 'annotations', 'instances_train2017.json'),
        train_ratio=args.train_ratio
    )

    # 创建数据集
    train_dataset = FixedCocoDataset(
        image_dir=os.path.join(args.data_path, 'train2017'),
        images_list=train_images,
        coco_data=coco_data,
        is_training=True
    )

    val_dataset = FixedCocoDataset(
        image_dir=os.path.join(args.data_path, 'train2017'),
        images_list=val_images,
        coco_data=coco_data,
        is_training=False
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    num_classes = train_dataset.num_classes
    print("=" * 80)

    # --- 2. 创建模型 ---
    print("🔧 创建模型...")
    model = RTDETRModel(num_classes=num_classes, pretrained_backbone=args.pretrained_backbone)
    criterion = build_criterion(num_classes=num_classes)

    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✅ 模型创建成功")
    print(f"   总参数: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print("=" * 80)

    # --- 3. 创建优化器 ---
    print("⚙️ 创建优化器...")
    optimizer = jt.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("=" * 80)

    # --- 4. 训练和验证循环 ---
    print("🚀 开始训练...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    training_log = {
        'start_time': datetime.now().isoformat(),
        'args': vars(args),
        'model_info': {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'num_classes': num_classes
        },
        'epochs': []
    }

    for epoch in range(args.epochs):
        print(f"\n📅 Epoch {epoch+1}/{args.epochs}")
        print("-" * 40)

        # 训练
        train_loss = train_one_epoch(model, criterion, optimizer, train_loader, epoch)
        train_losses.append(train_loss)

        # 验证
        val_loss = evaluate(model, criterion, val_loader)
        val_losses.append(val_loss)

        # 记录日志
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': args.lr,
            'timestamp': datetime.now().isoformat()
        }
        training_log['epochs'].append(epoch_log)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.output_dir, "best_model.pkl")
            jt.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'num_classes': num_classes,
                'args': vars(args)
            }, save_path)
            print(f"🎉 新的最优模型已保存 (Val Loss: {val_loss:.4f})")

        # 定期保存checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pkl")
            jt.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'num_classes': num_classes,
                'args': vars(args)
            }, checkpoint_path)
            print(f"💾 Checkpoint已保存: checkpoint_epoch_{epoch+1}.pkl")

        print(f"📊 Epoch {epoch+1} 总结: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    # --- 5. 保存训练结果 ---
    training_log['end_time'] = datetime.now().isoformat()
    training_log['best_val_loss'] = best_val_loss
    training_log['total_epochs'] = args.epochs

    # 保存训练日志
    log_path = os.path.join(args.output_dir, "training_log.json")
    save_training_log(training_log, log_path)

    # 绘制训练曲线
    curve_path = os.path.join(args.output_dir, "training_curves.png")
    plot_training_curves(train_losses, val_losses, curve_path)

    print("\n" + "=" * 80)
    print("✅ 训练完成！")
    print(f"🏆 最佳验证损失: {best_val_loss:.4f}")
    print(f"📁 结果保存在: {args.output_dir}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RT-DETR 改进版训练脚本 - 修复了数据增强Bug')

    # 数据相关
    parser.add_argument('--data_path', type=str, default='/home/kyc/project/RT-DETR/data/coco2017_50',
                       help='数据集根目录')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例 (0.8 = 80%训练, 20%验证)')

    # 模型相关
    parser.add_argument('--pretrained_backbone', action='store_true', default=True,
                       help='使用预训练backbone')

    # 训练相关
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--epochs', type=int, default=50, help='训练总轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')

    # 系统相关
    parser.add_argument('--num_workers', type=int, default=2, help='数据加载器的工作线程数')
    parser.add_argument('--output_dir', type=str, default='./rtdetr_checkpoints_improved',
                       help='模型权重保存目录')
    parser.add_argument('--save_interval', type=int, default=5, help='多少个epoch保存一次checkpoint')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("🐾 RT-DETR 改进版训练脚本启动")
    print("🔧 主要修复:")
    print("   ✅ 数据增强与标注坐标同步更新")
    print("   ✅ 训练/验证集固定分割，确保可复现性")
    print("   ✅ 异常处理，提高训练稳定性")
    print("   ✅ 详细的损失组件记录")
    print("   ✅ 完整的训练日志和可视化")
    print()

    main(args)
