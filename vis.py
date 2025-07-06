# vis.py (最终修复版 - 使用 jt.topk)

import jittor as jt
import jittor.nn as nn
from jittor import transform as T
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

from model import RTDETR

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

    model = RTDETR(num_classes=args.num_classes + 1)
    print(f"Loading weights from {args.weights}...")
    model.load_parameters(jt.load(args.weights))
    model.eval()

    original_img = Image.open(args.img_path).convert('RGB')
    w, h = original_img.size

    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        T.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = jt.array(transform(original_img)).unsqueeze(0)

    with jt.no_grad():
        outputs_class, outputs_coord = model(img_tensor)

    pred_logits = outputs_class[-1]
    pred_boxes = outputs_coord[-1]
    prob = nn.softmax(pred_logits, dim=-1)[0, :, :-1]

    # ## <<< 关键修复：使用 jt.topk(k=1) 代替 .max()，以稳定地获取值和索引 >>>
    scores, labels = jt.topk(prob, k=1, dim=-1)
    # .topk() 返回的张量会多一个维度，需要用 .squeeze() 去掉
    scores = scores.squeeze(-1)
    labels = labels.squeeze(-1)

    # 后续代码现在可以正确工作
    keep_mask = scores > args.conf_threshold
    
    final_boxes_cxcywh = pred_boxes[0][keep_mask]
    final_scores = scores[keep_mask]
    final_labels = labels[keep_mask]

    if final_labels.shape[0] == 0:
        print(f"检测完成：在此图片中没有发现置信度高于 {args.conf_threshold} 的目标。")
        result_path = "vis_result_no_detection.jpg"
        original_img.save(result_path)
        print(f"已将原图保存至: {result_path}")
        return

    boxes_xyxy = jt.zeros_like(final_boxes_cxcywh)
    boxes_xyxy[:, 0] = (final_boxes_cxcywh[:, 0] - final_boxes_cxcywh[:, 2] / 2) * w
    boxes_xyxy[:, 1] = (final_boxes_cxcywh[:, 1] - final_boxes_cxcywh[:, 3] / 2) * h
    boxes_xyxy[:, 2] = (final_boxes_cxcywh[:, 0] + final_boxes_cxcywh[:, 2] / 2) * w
    boxes_xyxy[:, 3] = (final_boxes_cxcywh[:, 1] + final_boxes_cxcywh[:, 3] / 2) * h

    draw = ImageDraw.Draw(original_img)
    print(f"检测到 {boxes_xyxy.shape[0]} 个目标，正在进行可视化...")
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