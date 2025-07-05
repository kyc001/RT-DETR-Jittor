# vis.py

import jittor as jt
import jittor.nn as nn
from jittor import transform as T
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

from model import RTDETR  # 确保 model.py 在同一目录下


def main():
    parser = argparse.ArgumentParser(
        description="RT-DETR Visualization")
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained model weights (.pkl file)')
    parser.add_argument('--img_path', type=str, required=True,
                        help='Path to the image for visualization')
    parser.add_argument('--num_classes', type=int, default=80,
                        help="Number of object categories, excluding background.")
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help="Confidence threshold for filtering predictions.")
    args = parser.parse_args()

    jt.flags.use_cuda = 1

    # 初始化模型
    model = RTDETR(num_classes=args.num_classes + 1)

    # ## <<< 关键修正：先使用 jt.load() 加载文件，再送入 load_parameters >>>
    print(f"Loading weights from {args.weights}...")
    model.load_parameters(jt.load(args.weights))
    model.eval()

    # 准备图片
    original_img = Image.open(args.img_path).convert('RGB')
    w, h = original_img.size

    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        T.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = jt.array(transform(original_img)).unsqueeze(0)

    # 模型推理
    with jt.no_grad():
        outputs_class, outputs_coord = model(img_tensor)

    # 后处理 (与 test.py 保持一致)
    pred_logits = outputs_class[-1]
    pred_boxes = outputs_coord[-1]

    prob = nn.softmax(pred_logits, dim=-1)[0, :, :-1]
    scores = prob.max(dim=-1)
    labels = prob.argmax(dim=-1)

    keep_mask = scores > args.conf_threshold
    keep_indices = jt.nonzero(keep_mask).squeeze(1)

    if keep_indices.shape[0] == 0:
        print(f"No objects detected with confidence > {args.conf_threshold}")
        result_path = "vis_result_no_detection.jpg"
        original_img.save(result_path)
        print(f"Original image saved to {result_path}")
        return

    final_boxes_cxcywh = pred_boxes[0][keep_indices]
    final_scores = scores[keep_indices]
    final_labels = labels[keep_indices]

    boxes_xyxy = jt.zeros_like(final_boxes_cxcywh)
    boxes_xyxy[:, 0] = (final_boxes_cxcywh[:, 0] -
                        final_boxes_cxcywh[:, 2] / 2) * w
    boxes_xyxy[:, 1] = (final_boxes_cxcywh[:, 1] -
                        final_boxes_cxcywh[:, 3] / 2) * h
    boxes_xyxy[:, 2] = (final_boxes_cxcywh[:, 0] +
                        final_boxes_cxcywh[:, 2] / 2) * w
    boxes_xyxy[:, 3] = (final_boxes_cxcywh[:, 1] +
                        final_boxes_cxcywh[:, 3] / 2) * h

    # 可视化
    draw = ImageDraw.Draw(original_img)
    print(f"Detected {boxes_xyxy.shape[0]} objects. Visualizing...")
    for i in range(boxes_xyxy.shape[0]):
        box = boxes_xyxy[i].numpy()
        label = final_labels[i].item()
        score = final_scores[i].item()

        draw.rectangle(list(box), outline="red", width=3)
        text = f"Class {label}: {score:.2f}"

        try:
            font = ImageFont.truetype("arial.ttf", 20)
            draw.text((box[0], box[1]), text, fill="red", font=font)
        except IOError:
            draw.text((box[0], box[1]), text, fill="red")

    result_path = "vis_result.jpg"
    original_img.save(result_path)
    print(f"Visualization saved to {result_path}")


if __name__ == '__main__':
    main()
