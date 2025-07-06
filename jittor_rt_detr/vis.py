#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RT-DETR 可视化脚本 (修复版)
解决Jittor框架兼容性问题
"""

import jittor as jt
import jittor.nn as nn
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

from model import RTDETR


def preprocess(img_path):
    """预处理图片"""
    img = Image.open(img_path).convert('RGB').resize((640, 640))
    arr = np.array(img).astype(np.float32) / 255.
    arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    arr = arr.transpose(2, 0, 1)
    return jt.array(arr).reshape((1, 3, 640, 640))


def main():
    parser = argparse.ArgumentParser(
        description="RT-DETR Visualization (Fixed)")
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained model weights (.pkl file)')
    parser.add_argument('--img_path', type=str, required=True,
                        help='Path to the image for visualization')
    parser.add_argument('--num_classes', type=int, default=80,
                        help="Number of object categories, excluding background.")
    parser.add_argument('--conf_threshold', type=float, default=0.2,
                        help="Confidence threshold for filtering predictions.")
    args = parser.parse_args()

    # 切换到CPU模式避免CUDA dtype冲突
    jt.flags.use_cuda = 0

    model = RTDETR(num_classes=args.num_classes + 1)
    print(f"Loading weights from {args.weights}...")
    model.load_parameters(jt.load(args.weights))
    model.eval()

    img_path = args.img_path
    img_tensor = preprocess(img_path)

    with jt.no_grad():
        outputs_class, outputs_coord = model(img_tensor)

    pred_logits = outputs_class[-1]
    pred_boxes = outputs_coord[-1]
    prob = nn.softmax(pred_logits, dim=-1)[0, :, :-1]

    # 使用topk获取最高置信度的预测
    scores, labels = jt.topk(prob, k=1, dim=-1)
    scores = scores.squeeze(-1)
    labels = labels.squeeze(-1)

    # 修复：确保类型和梯度
    scores = scores.float32().stop_grad()
    labels = labels.int32().stop_grad()
    boxes = pred_boxes[0].float32().stop_grad()

    # 获取图片宽高并转为float32
    original_img = Image.open(img_path).convert('RGB').resize((640, 640))
    w, h = original_img.size
    w = np.float32(w)
    h = np.float32(h)

    # 置信度筛选 - 使用numpy进行后处理
    keep_mask = (scores > args.conf_threshold)
    keep_indices = np.where(keep_mask.numpy())[0]

    if len(keep_indices) == 0:
        print(f"检测完成：在此图片中没有发现置信度高于 {args.conf_threshold} 的目标。")
        result_path = "vis_result_no_detection.jpg"
        original_img.save(result_path)
        print(f"已将原图保存至: {result_path}")
        return

    final_boxes_cxcywh = boxes.numpy()[keep_indices].astype(np.float32)
    final_scores = scores.numpy()[keep_indices].astype(np.float32)
    final_labels = labels.numpy()[keep_indices].astype(np.int32)

    # 转换为像素坐标，所有变量都用float32
    cx = final_boxes_cxcywh[:, 0].astype(np.float32)
    cy = final_boxes_cxcywh[:, 1].astype(np.float32)
    bw = final_boxes_cxcywh[:, 2].astype(np.float32)
    bh = final_boxes_cxcywh[:, 3].astype(np.float32)

    boxes_xyxy = np.zeros_like(final_boxes_cxcywh, dtype=np.float32)
    boxes_xyxy[:, 0] = (cx - bw / 2) * w
    boxes_xyxy[:, 1] = (cy - bh / 2) * h
    boxes_xyxy[:, 2] = (cx + bw / 2) * w
    boxes_xyxy[:, 3] = (cy + bh / 2) * h

    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w - 1)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h - 1)

    draw = ImageDraw.Draw(original_img)
    print(f"检测到 {boxes_xyxy.shape[0]} 个目标，正在进行可视化...")

    for i in range(boxes_xyxy.shape[0]):
        box = boxes_xyxy[i]
        label = final_labels[i]
        score = final_scores[i]

        draw.rectangle(list(box), outline="red", width=3)
        text = f"Class {label}: {score:.2f}"
        try:
            font = ImageFont.truetype("arial.ttf", 20)
            draw.text((box[0], box[1]), text, fill="red", font=font)
        except IOError:
            draw.text((box[0], box[1]), text, fill="red")

    result_path = "vis_result.jpg"
    original_img.save(result_path)
    print(f"可视化结果已保存至: {result_path}")


if __name__ == '__main__':
    main()
