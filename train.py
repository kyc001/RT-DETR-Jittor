# train.py

import jittor as jt
import jittor.nn as nn
from tqdm import tqdm
import argparse
import os
from PIL import Image

# 导入项目中的核心模块
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'jittor_rt_detr'))

from src.data.coco.coco_dataset import COCODataset
from src.data.transforms import to_tensor, hflip
from src.nn.model import RTDETR
from src.nn.loss import DETRLoss

# 创建Compose类来处理变换序列
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        if target is None:
            # 只有图像的情况
            for t in self.transforms:
                if callable(t):
                    image = t(image)
            return image
        else:
            # 图像和目标都有的情况
            for t in self.transforms:
                if callable(t):
                    image, target = t(image, target)
            return image, target

# 添加resize变换
def resize_transform(image, target, size=(640, 640)):
    """将图像resize到指定尺寸并调整边界框"""
    original_size = image.size  # (width, height)
    resized_image = image.resize(size, Image.LANCZOS)

    # 调整边界框坐标
    if 'boxes' in target and target['boxes'].shape[0] > 0:
        scale_x = size[0] / original_size[0]
        scale_y = size[1] / original_size[1]

        boxes = target['boxes'].clone()
        boxes[:, [0, 2]] *= scale_x  # x坐标
        boxes[:, [1, 3]] *= scale_y  # y坐标
        target['boxes'] = boxes

    return resized_image, target

# 创建数据变换函数
def get_transforms(train=True):
    """创建数据变换管道"""
    transforms = []

    # 首先resize到固定尺寸
    transforms.append(lambda img, target: resize_transform(img, target, (640, 640)))

    if train:
        # 训练时添加数据增强（在转换为tensor之前）
        transforms.append(lambda img, target: hflip(img, target) if jt.rand(1).item() > 0.5 else (img, target))

    # 最后转换为tensor
    transforms.append(to_tensor)
    return Compose(transforms)

def main(args):
    """
    主训练函数
    """
    # 1. 设置和打印配置
    print("==================================================")
    print("===      RT-DETR Jittor - 训练脚本      ===")
    print("==================================================")
    for k, v in vars(args).items():
        print(f"{k:<20}: {v}")
    print("--------------------------------------------------")

    # 2. 准备数据集和数据加载器
    print(">>> 正在准备数据集...")
    train_transforms = get_transforms(train=True)

    # 处理指定特定图片的情况
    selected_image_ids = None
    if args.specific_image:
        print(f"🎯 指定图片模式: {args.specific_image}")
        # 从图片路径中提取图片ID
        import json
        with open(args.ann_file, 'r') as f:
            ann_data = json.load(f)

        # 从路径中提取文件名
        import os
        img_filename = os.path.basename(args.specific_image)

        # 查找对应的图片ID
        target_image_id = None
        for img in ann_data['images']:
            if img['file_name'] == img_filename:
                target_image_id = img['id']
                break

        if target_image_id is None:
            raise ValueError(f"找不到图片 {img_filename} 在标注文件中")

        selected_image_ids = [target_image_id]
        print(f"✅ 找到目标图片ID: {target_image_id}")

    train_dataset = COCODataset(
        img_dir=args.img_dir,
        ann_file=args.ann_file,
        transforms=train_transforms,
        subset_size=args.subset_size,
        selected_image_ids=selected_image_ids
    )

    # 创建数据加载器
    train_loader = jt.dataset.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_batch=COCODataset.collate_batch
    )
    print(f"✅ 数据集准备完成，共 {len(train_dataset)} 张图片。")

    # 3. 初始化模型、损失函数和优化器
    print(">>> 正在初始化模型...")
    # COCO 数据集有 80 个类别
    num_classes = 80 
    model = RTDETR(num_classes=num_classes)
    
    criterion = DETRLoss(
        num_classes=num_classes,
        lambda_cls=2.0,
        lambda_bbox=5.0,
        lambda_giou=2.0,
        eos_coef=0.1
    )
    
    # 使用 AdamW 优化器，这对于 Transformer 模型很有效
    optimizer = nn.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    lr_scheduler = jt.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop_epoch, gamma=0.1)
    
    print("✅ 模型、损失函数、优化器初始化完成。")

    # 4. 创建检查点保存目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"模型将保存至: {args.output_dir}")
    
    # 5. 开始训练循环
    print("==================================================")
    print(">>> 开始训练...")
    
    for epoch in range(args.epochs):
        model.train()
        criterion.train()
        
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        
        for batch_data in progress_bar:
            images, boxes_list, labels_list = batch_data

            # 准备目标格式 - boxes_list已经是cxcywh格式且归一化的
            targets = []
            for i in range(len(boxes_list)):
                targets.append({
                    'boxes': boxes_list[i],  # 已经是cxcywh格式
                    'labels': labels_list[i]
                })

            # 前向传播
            all_logits, all_boxes, enc_logits, enc_boxes = model(images)

            # 计算损失
            loss_dict = criterion(all_logits, all_boxes, targets, enc_logits, enc_boxes)
            total_loss = loss_dict['total_loss']
            
            # 反向传播和优化
            optimizer.step(total_loss)

            # 更新进度条显示
            epoch_loss += total_loss.item()
            progress_bar.set_postfix(loss=f"{total_loss.item():.4f}")

        # 更新学习率
        lr_scheduler.step()
        
        # 打印周期信息
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"--- Epoch {epoch + 1} 完成 ---")
        print(f"平均损失 (Avg Loss): {avg_epoch_loss:.4f}")
        print(f"当前学习率 (LR): {optimizer.lr:.6f}")
        print("--------------------------------------------------")

        # 保存模型检查点
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f'model_epoch_{epoch + 1}.pkl')
            jt.save(model.state_dict(), save_path)
            print(f"✅ 模型检查点已保存: {save_path}")

    print("🎉 训练完成！")
    final_save_path = os.path.join(args.output_dir, 'model_final.pkl')
    jt.save(model.state_dict(), final_save_path)
    print(f"✅ 最终模型已保存: {final_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RT-DETR Jittor Training Script')
    
    # 数据相关参数
    parser.add_argument('--img_dir', type=str, default='data/coco2017_50/train2017', help='训练集图片目录')
    parser.add_argument('--ann_file', type=str, default='data/coco2017_50/annotations/instances_train2017.json', help='训练集标注文件')
    
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='初始学习率')
    parser.add_argument('--lr_drop_epoch', type=int, default=40, help='学习率衰减的轮数')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    
    # 其他参数
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='模型保存目录')
    parser.add_argument('--save_interval', type=int, default=10, help='模型保存间隔（每N个epoch）')
    parser.add_argument('--subset_size', type=int, default=None, help='使用数据子集大小（用于快速调试）')
    parser.add_argument('--specific_image', type=str, default=None, help='指定特定图片进行训练（用于流程自检）')
    
    args = parser.parse_args()
    main(args)