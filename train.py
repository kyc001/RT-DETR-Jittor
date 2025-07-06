# train.py (诊断版本)

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
import traceback  # 导入用于打印详细错误的模块

from model import RTDETR
from dataset import COCODataset
from loss import DETRLoss


def main():
    # jt.flags.use_cuda_fp16 = 1
    jt.flags.use_cuda = 1
    parser = argparse.ArgumentParser(description="RT-DETR Training in Jittor")

    # 训练参数
    parser.add_argument('--img_dir', type=str, default='data/coco/val2017')
    parser.add_argument('--ann_file', type=str,
                        default='data/coco/annotations/instances_val2017.json')
    parser.add_argument('--num_classes', type=int, default=80)
    parser.add_argument('--num_queries', type=int, default=300)
    parser.add_argument('--batch_size', type=int,
                        default=4)  # 强制batch_size为1，逐个排查
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)  # 调试时只跑2个epoch
    parser.add_argument('--lr_drop_epoch', type=int, default=40)
    parser.add_argument('--clip_max_norm', type=float,
                        default=0.1, help='Gradient clipping max norm.')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--subset_size', type=int, default=None,
                        help='Use only a subset of N images for training')
    args = parser.parse_args()

    # 数据预处理
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        T.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 数据集和加载器
    print("Initializing dataset...")
    dataset = COCODataset(args.img_dir, args.ann_file,
                          transforms=transform, subset_size=args.subset_size)
    print(f"Dataset initialized. Number of items: {len(dataset)}")

    # ## <<< 关键诊断：num_workers=0 强制单进程加载，使错误暴露 >>>
    loader = jt.dataset.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,  # 调试时不打乱，确保顺序
        drop_last=True,
        num_workers=4  # 强制使用主进程加载数据
    )
    print(f"DataLoader initialized. Number of batches: {len(loader)}")

    # 模型、损失函数、优化器和学习率衰减器
    model = RTDETR(num_classes=args.num_classes +
                   1, num_queries=args.num_queries)
    loss_fn = DETRLoss(num_classes=args.num_classes)
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_drop_epoch)

    print("Start training...")
    loss_log = []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0

        # ## <<< 关键诊断：使用 try-except 捕捉循环中的任何错误 >>>
        try:
            progress_bar = tqdm(
                loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
            for i, batch in enumerate(progress_bar):
                print(f"--> Processing batch {i}")
                img, boxes, labels = batch

                targets = []
                for j in range(len(boxes)):
                    targets.append({'boxes': boxes[j], 'labels': labels[j]})

                print(f"--> Batch {i}: Performing model forward pass...")
                outputs_class, outputs_coord = model(img)

                print(f"--> Batch {i}: Calculating loss...")
                loss_dict = loss_fn(outputs_class, outputs_coord, targets)
                loss = loss_dict['total_loss']

                print(f"--> Batch {i}: Loss calculated: {loss.item()}")

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
        except Exception as e:
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!            捕获到致命错误             !!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {e}")
            print("\n详细追溯信息 (Traceback):")
            traceback.print_exc()
            print("\n程序已终止。请将以上完整错误信息发送给我进行分析。")
            break  # 发生错误后跳出epoch循环

        scheduler.step()
        avg_loss = epoch_loss / len(loader) if len(loader) > 0 else 0

        print(
            f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f} | LR: {optimizer.lr:.1e}\n")
        loss_log.append(avg_loss)

    print("Training finished!")


if __name__ == '__main__':
    main()
