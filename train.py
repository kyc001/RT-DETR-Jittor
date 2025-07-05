import jittor as jt
import jittor.nn as nn
from jittor import transform as T
import os
import numpy as np
import argparse

# 导入你的自定义模块
from model import RTDETR
from dataset import COCODataset
from loss import DETRLoss


def main():
    # 设置Jittor使用GPU
    jt.flags.use_cuda = 1

    parser = argparse.ArgumentParser(description="RT-DETR Training in Jittor")
    # 将常用训练参数设为默认值
    parser.add_argument('--img_dir', type=str,
                        default='data/coco/val2017')
    parser.add_argument('--ann_file', type=str,
                        default='data/coco/annotations/instances_val2017.json')
    parser.add_argument('--num_classes', type=int, default=80,
                        help='Number of object classes, excluding the "no-object" background class.')
    # RT-DETR uses 300 queries
    parser.add_argument('--num_queries', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=2,
                        help="Batch size.")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to the checkpoint to resume training from.")
    args = parser.parse_args()

    # 定义数据预处理/增强流程
    transform = T.Compose([
        T.Resize((640, 640)),  # RT-DETR uses 640x640
        T.ToTensor(),
        T.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 初始化数据集和数据加载器
    # 它会自动使用在 COCODataset 类中定义的 collate_batch 方法
    dataset = COCODataset(args.img_dir, args.ann_file, transforms=transform)
    loader = jt.dataset.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    # 初始化模型、损失函数和优化器
    # 注意：模型的输出维度是 num_classes + 1，为背景类留出空间
    model = RTDETR(num_classes=args.num_classes +
                   1, num_queries=args.num_queries)

    if args.resume is not None:
        print(f'加载断点模型: {args.resume}')
        model.load_parameters(jt.load(args.resume))

    loss_fn = DETRLoss(num_classes=args.num_classes)
    optimizer = nn.Adam(model.parameters(), lr=args.lr)

    # ====== 训练主循环 ======
    print("开始训练...")
    loss_curve = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(loader):
            # 自定义的 collate_batch 返回 (stacked_imgs, list_of_boxes, list_of_labels)
            img, boxes, labels = batch

            # 准备 targets 列表以匹配 loss_fn 的输入格式
            targets = []
            # len(boxes) 现在就是你的 batch_size
            for i in range(len(boxes)):
                targets.append({
                    'boxes': boxes[i],
                    'labels': labels[i]
                })

            # 模型前向传播
            pred_logits, pred_boxes = model(img)

            # 调用损失函数
            loss = loss_fn(pred_logits, pred_boxes, targets)

            # 优化器步骤
            optimizer.step(loss)

            current_loss = loss.item()
            total_loss += current_loss

            # 打印训练日志
            if i % 50 == 0:
                print(
                    f"Epoch: {epoch+1}/{args.epochs}, Batch: {i}/{len(loader)}, Loss: {current_loss:.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} 完成, 平均 Loss: {avg_loss:.4f}")
        loss_curve.append(avg_loss)

        # 保存模型
        model.save(f'model_epoch_{epoch+1}.pkl')
        np.save('loss_curve.npy', np.array(loss_curve))

    print("训练完成！")


if __name__ == '__main__':
    main()
