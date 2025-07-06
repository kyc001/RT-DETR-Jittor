# train.py (最终正式版)

import jittor as jt
import jittor.nn as nn
from jittor import transform as T
from jittor.optim import AdamW
from jittor import lr_scheduler
import os
import numpy as np
import argparse
import math
from tqdm import tqdm

from model import RTDETR
from dataset import COCODataset
from loss import DETRLoss


def main():
    # 考虑到您的环境兼容性，我们不开启FP16
    jt.flags.use_cuda = 1

    parser = argparse.ArgumentParser(description="RT-DETR Training in Jittor")

    # 训练参数
    parser.add_argument('--img_dir', type=str, default='data/coco/val2017',
                        help='Path to the training image directory.')
    parser.add_argument('--ann_file', type=str,
                        default='data/coco/annotations/instances_val2017.json',
                        help='Path to the training annotation file.')
    parser.add_argument('--num_classes', type=int, default=80)
    parser.add_argument('--num_queries', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=2)  # 建议使用4或8
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr_drop_epoch', type=int, default=40)
    parser.add_argument('--clip_max_norm', type=float,
                        default=0.1, help='Gradient clipping max norm.')
    parser.add_argument('--subset_size', type=int, default=None,
                        help='Use only a subset of N images for quick experiments.')
    args = parser.parse_args()

    # 数据预处理
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        T.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 数据集和加载器
    dataset = COCODataset(args.img_dir, args.ann_file,
                          transforms=transform, subset_size=args.subset_size)

    # ## <<< 关键修复：固定为单进程加载以保证在您环境下的稳定性 >>>
    loader = jt.dataset.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,  # 正式训练必须打乱
        drop_last=True,
        num_workers=0
    )

    # 模型、损失函数、优化器和学习率衰减器
    model = RTDETR(num_classes=args.num_classes +
                   1, num_queries=args.num_queries)
    loss_fn = DETRLoss(num_classes=args.num_classes)
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_drop_epoch)

    print("Start training...")
    print(f"Dataset size: {len(dataset)} images")
    print(f"Total batches per epoch: {len(loader)}")

    loss_log = []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0

        progress_bar = tqdm(
            loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        for i, batch in enumerate(progress_bar):
            img, boxes, labels = batch

            targets = []
            for j in range(len(boxes)):
                targets.append({'boxes': boxes[j], 'labels': labels[j]})

            outputs_class, outputs_coord = model(img)
            loss_dict = loss_fn(outputs_class, outputs_coord, targets)
            loss = loss_dict['total_loss']

            if not math.isfinite(loss.item()):
                print(
                    f"\nWARNING: Non-finite loss, skipping batch {i}. Loss: {loss.item()}")
                continue

            optimizer.backward(loss)
            optimizer.clip_grad_norm(max_norm=args.clip_max_norm)
            optimizer.step()

            current_loss = loss.item()
            epoch_loss += current_loss
            progress_bar.set_postfix(loss=f'{current_loss:.4f}')

        scheduler.step()

        # 使用 len(loader) 来计算平均损失，这是正确的
        avg_loss = epoch_loss / len(loader) if len(loader) > 0 else 0

        print(
            f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f} | LR: {optimizer.lr:.1e}\n")
        loss_log.append(avg_loss)

        output_dir = "checkpoints"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model.save(os.path.join(output_dir, f'model_epoch_{epoch+1}.pkl'))
        np.save(os.path.join(output_dir, 'loss_curve.npy'), np.array(loss_log))

    print("Training finished!")


if __name__ == '__main__':
    main()
