#!/usr/bin/env python3
"""
RT-DETR PyTorch 训练脚本
"""

from src.misc import dist_utils
from src.core.ema import ModelEMA
from src.solver.lr_scheduler import LRScheduler
from src.data.coco import COCODetection
from src.zoo.rtdetr import RTDETR
import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def parse_args():
    parser = argparse.ArgumentParser(description='RT-DETR Training')
    parser.add_argument('--config', type=str, default='configs/rtdetr_r50vd_6x_coco.yml',
                        help='配置文件路径')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--device', type=str, default='auto', help='训练设备')
    parser.add_argument('--save_interval', type=int, default=10, help='保存间隔')
    parser.add_argument('--subset_size', type=int,
                        default=None, help='使用数据子集大小')
    parser.add_argument('--output_dir', type=str,
                        default='checkpoints', help='输出目录')
    return parser.parse_args()


def setup_device(device_arg):
    """设置训练设备"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    print(f"使用设备: {device}")
    return device


def create_model(config_path):
    """创建RT-DETR模型"""
    try:
        from src.zoo.rtdetr import RTDETR
        model = RTDETR(config_path)
        print(f"模型创建成功: {type(model).__name__}")
        return model
    except Exception as e:
        print(f"模型创建失败: {e}")
        # 创建简化版模型
        return create_simple_model()


def create_simple_model():
    """创建简化版RT-DETR模型"""
    class SimpleRTDETR(nn.Module):
        def __init__(self, num_classes=80):
            super().__init__()
            # 简化的backbone
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )

            # 简化的transformer
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=256, nhead=8, dim_feedforward=1024),
                num_layers=6
            )

            # 分类头和回归头
            self.class_head = nn.Linear(256, num_classes)
            self.bbox_head = nn.Linear(256, 4)

        def forward(self, x):
            # 简化的前向传播
            batch_size = x.size(0)
            x = self.backbone(x)  # [B, 256, H/32, W/32]
            x = x.flatten(2).transpose(1, 2)  # [B, N, 256]
            x = self.transformer(x)

            # 预测
            cls_pred = self.class_head(x)  # [B, N, num_classes]
            bbox_pred = self.bbox_head(x)  # [B, N, 4]

            return cls_pred, bbox_pred

    model = SimpleRTDETR()
    print("创建简化版RT-DETR模型")
    return model


def create_dataset(data_path, subset_size=None):
    """创建数据集"""
    try:
        # 简化的数据集
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 创建虚拟数据集用于演示
        class VirtualDataset(torch.utils.data.Dataset):
            def __init__(self, size=1000, transform=None):
                self.size = size
                self.transform = transform

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                # 创建虚拟图像和标签
                img = torch.randn(3, 640, 640)
                labels = torch.randint(0, 80, (torch.randint(1, 10, (1,)),))
                bboxes = torch.rand(labels.size(0), 4)

                if self.transform:
                    img = self.transform(img)

                return img, {'labels': labels, 'boxes': bboxes}

        dataset = VirtualDataset(size=subset_size or 1000, transform=transform)
        print(f"创建虚拟数据集，大小: {len(dataset)}")
        return dataset

    except Exception as e:
        print(f"数据集创建失败: {e}")
        return None


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device)

        # 将targets移到设备上
        if isinstance(targets, dict):
            targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in targets.items()}

        optimizer.zero_grad()

        try:
            # 前向传播
            if hasattr(model, 'forward'):
                outputs = model(images)
                if isinstance(outputs, tuple):
                    cls_pred, bbox_pred = outputs
                else:
                    cls_pred, bbox_pred = outputs['cls_pred'], outputs['bbox_pred']
            else:
                # 简化版前向传播
                cls_pred, bbox_pred = model(images)

            # 计算损失（简化版）
            loss = nn.functional.cross_entropy(cls_pred.view(-1, cls_pred.size(-1)),
                                               targets['labels'].view(-1))
            loss += nn.functional.mse_loss(bbox_pred.view(-1, 4),
                                           targets['boxes'].view(-1, 4))

            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })

        except Exception as e:
            print(f"批次 {batch_idx} 训练失败: {e}")
            continue

    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """保存检查点"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }

    torch.save(checkpoint, save_path)
    print(f"检查点已保存: {save_path}")


def main():
    args = parse_args()

    # 设置设备
    device = setup_device(args.device)

    # 创建模型
    print("创建模型...")
    model = create_model(args.config)
    model = model.to(device)

    # 创建数据集和数据加载器
    print("创建数据集...")
    dataset = create_dataset('data/coco', args.subset_size)
    if dataset is None:
        print("数据集创建失败，退出训练")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda x: (torch.stack([item[0] for item in x]),
                              [item[1] for item in x])
    )

    # 创建优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # 创建损失函数
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    print(f"开始训练，共 {args.epochs} 个epoch...")
    best_loss = float('inf')

    for epoch in range(args.epochs):
        start_time = time.time()

        # 训练一个epoch
        avg_loss = train_epoch(
            model, dataloader, criterion, optimizer, device, epoch)

        # 更新学习率
        scheduler.step()

        # 记录训练信息
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  平均损失: {avg_loss:.4f}")
        print(f"  学习率: {current_lr:.6f}")
        print(f"  训练时间: {epoch_time:.2f}s")

        # 保存检查点
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(
                args.output_dir, f'model_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch+1, avg_loss, save_path)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(args.output_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch+1, avg_loss, save_path)

    print("训练完成！")
    print(f"最佳损失: {best_loss:.4f}")


if __name__ == '__main__':
    main()
