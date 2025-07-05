# test.py

import jittor as jt
import jittor.nn as nn
from jittor import transform as T
import argparse
import numpy as np
from tqdm import tqdm
import json

from model import RTDETR
from dataset import COCODataset, xyxy_to_cxcywh


def main():
    parser = argparse.ArgumentParser(description="RT-DETR Testing")
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--img_dir', type=str, default='data/coco/val2017')
    parser.add_argument('--ann_file', type=str,
                        default='data/coco/annotations/instances_val2017.json')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    jt.flags.use_cuda = 1

    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        T.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = COCODataset(args.img_dir, args.ann_file,
                          transforms=transform, is_train=False)
    # Note: for evaluation, shuffle should be False
    loader = jt.dataset.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False)

    # 80 classes + 1 background
    model = RTDETR(num_classes=81, num_queries=300)
    model.load_parameters(jt.load(args.weights))
    model.eval()

    results = []

    print("Running inference on validation set...")
    for img, _, _ in tqdm(loader):
        pred_logits, pred_boxes = model(img)

        # Post-process results for one image at a time
        # (num_queries, num_classes + 1)
        prob = nn.softmax(pred_logits, dim=-1)[0]
        scores, labels = prob[:, :-1].max(-1)  # Exclude background class

        boxes = pred_boxes[0]  # (num_queries, 4) in cxcywh format

        # Here you would typically use NMS and convert boxes back to original image size
        # For simplicity, we'll just save the raw predictions for now.
        # This part needs to be expanded for a full mAP calculation.

        # A proper evaluation would require mapping back to original image IDs
        # and sizes, then using a COCO evaluation script.
        # This is a placeholder to show the inference loop.

    print(f"Inference finished. {len(loader)} images processed.")
    # Here you would call a function to calculate mAP based on `results`
    # e.g., pycocotools.COCOeval


if __name__ == '__main__':
    main()
