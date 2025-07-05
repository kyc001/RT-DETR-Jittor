# test.py

import jittor as jt
import jittor.nn as nn
from jittor import transform as T
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from model import RTDETR
# 注意：请确保您 dataset.py 中的函数名与此处导入的一致
# from dataset import xyxy_to_cxcywh_and_normalize


def main():
    parser = argparse.ArgumentParser(
        description="RT-DETR Testing and Visualization")
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--img_path', type=str, required=True,
                        help='Path to the image for visualization')
    parser.add_argument('--num_classes', type=int, default=80)
    args = parser.parse_args()

    jt.flags.use_cuda = 1

    # 初始化模型并加载权重
    model = RTDETR(num_classes=args.num_classes + 1)
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

    numpy_img = transform(original_img)
    img_tensor = jt.array(numpy_img)

    # 模型推理
    with jt.no_grad():
        outputs_class, outputs_coord = model(img_tensor.unsqueeze(0))

    # 后处理
    pred_logits = outputs_class[-1]
    pred_boxes = outputs_coord[-1]

    prob = nn.softmax(pred_logits, dim=-1)[0]

    # ## <<< 关键修正：分两步获取分数和标签 >>>
    scores = prob[:, :-1].max(dim=-1)
    labels = prob[:, :-1].argmax(dim=-1)

    # ## <<< 关键修正：使用 jt.nonzero 获取索引 >>>
    keep_mask = scores > 0.5
    keep_indices = jt.nonzero(keep_mask).squeeze(1)

    boxes_cxcywh = pred_boxes[0][keep_indices]
    scores = scores[keep_indices]
    labels = labels[keep_indices]

    # 将坐标从归一化的 cxcywh 转换回原始图像尺寸的 xyxy
    boxes_xyxy = jt.zeros_like(boxes_cxcywh)
    boxes_xyxy[:, 0] = (boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2) * w
    boxes_xyxy[:, 1] = (boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2) * h
    boxes_xyxy[:, 2] = (boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2) * w
    boxes_xyxy[:, 3] = (boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2) * h

    # 可视化
    draw = ImageDraw.Draw(original_img)
    for i in range(boxes_xyxy.shape[0]):
        box = boxes_xyxy[i].numpy()
        label = labels[i].item()
        score = scores[i].item()

        draw.rectangle(list(box), outline="red", width=3)
        text = f"Class {label}: {score:.2f}"
        draw.text((box[0], box[1]), text, fill="red")

    result_path = "vis_result.jpg"
    original_img.save(result_path)
    print(f"Visualization saved to {result_path}")


if __name__ == '__main__':
    main()
