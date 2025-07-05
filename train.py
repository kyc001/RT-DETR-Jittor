import jittor as jt
import jittor.nn as nn
from jittor import transform as T
from jittor.optim import AdamW
from jittor import lr_scheduler
import os
import numpy as np
import argparse
from tqdm import tqdm

# ## <<< 关键修正：添加了对自定义模块的导入 >>>
from model import RTDETR
from dataset import COCODataset
from loss import DETRLoss


def main():
    jt.flags.use_cuda = 1
    parser = argparse.ArgumentParser(description="RT-DETR Training in Jittor")

    parser.add_argument('--img_dir', type=str, default='data/coco/val2017')
    parser.add_argument('--ann_file', type=str,
                        default='data/coco/annotations/instances_val2017.json')
    parser.add_argument('--num_classes', type=int, default=80)
    parser.add_argument('--num_queries', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr_drop_epoch', type=int, default=40)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        T.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = COCODataset(args.img_dir, args.ann_file, transforms=transform)
    loader = jt.dataset.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    model = RTDETR(num_classes=args.num_classes +
                   1, num_queries=args.num_queries)
    if args.resume is not None:
        print(f'Loading checkpoint: {args.resume}')
        model.load_parameters(jt.load(args.resume))

    loss_fn = DETRLoss(num_classes=args.num_classes)
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_drop_epoch)

    print("Start training...")
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

            pred_logits = outputs_class[-1]
            pred_boxes = outputs_coord[-1]

            loss_dict = loss_fn(pred_logits, pred_boxes, targets)
            loss = loss_dict['total_loss']

            optimizer.step(loss)

            current_loss = loss.item()
            epoch_loss += current_loss

            progress_bar.set_postfix(loss=f'{current_loss:.4f}')

        scheduler.step()

        avg_loss = epoch_loss / len(loader)
        progress_bar.set_postfix(avg_loss=f'{avg_loss:.4f}')
        progress_bar.close()

        print(
            f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f} | LR: {optimizer.lr:.1e}\n")
        loss_log.append(avg_loss)

        model.save(f'model_epoch_{epoch+1}.pkl')
        np.save('loss_curve.npy', np.array(loss_log))

    print("Training finished!")


if __name__ == '__main__':
    main()
