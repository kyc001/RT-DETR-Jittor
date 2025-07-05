# test.py

import jittor as jt
import jittor.nn as nn
from jittor import transform as T
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 请确保 model.py 与您修改后的一致
from model import RTDETR


def main():
    parser = argparse.ArgumentParser(
        description="RT-DETR Testing and Visualization")
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--img_path', type=str, required=True,
                        help='Path to the image for visualization')
    parser.add_argument('--num_classes', type=int, default=80,
                        help="Number of object categories, excluding background.")
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help="Confidence threshold for filtering predictions.")

    args = parser.parse_args()

    jt.flags.use_cuda = 1

    # 初始化模型并加载权重
    # 注意：这里的 num_classes 需要是 类别数 + 1 (为了匹配背景类)
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

    # (1, num_queries, num_classes + 1) -> (1, num_queries, num_classes)
    # 我们只关心前景类别，所以排除最后一个背景类
    prob = nn.softmax(pred_logits, dim=-1)[0, :, :-1]

    # 获取每个预测框最高的分数和对应的类别
    scores = prob.max(dim=-1)     # shape: (num_queries,)
    labels = prob.argmax(dim=-1)  # shape: (num_queries,)

    # 根据置信度阈值进行过滤
    keep_mask = scores > args.conf_threshold
    keep_indices = jt.nonzero(keep_mask).squeeze(1)

    # =================================================================
    # ## <<< 核心修复：检查是否有任何物体被检测到 >>>
    # =================================================================
    if keep_indices.shape[0] == 0:
        print(f"检测完成：在此图片中没有发现置信度高于 {args.conf_threshold} 的目标。")
        result_path = "vis_result_no_detection.jpg"
        original_img.save(result_path)
        print(f"已将原图保存至: {result_path}")
        return  # 提前退出，不再执行后续代码，从而避免报错
    # =================================================================

    # 如果有检测结果，则继续后续处理
    final_boxes_cxcywh = pred_boxes[0][keep_indices]
    final_scores = scores[keep_indices]
    final_labels = labels[keep_indices]

    # 将坐标从归一化的 cxcywh 转换回原始图像尺寸的 xyxy
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
    print(f"检测到 {boxes_xyxy.shape[0]} 个目标，正在进行可视化...")
    for i in range(boxes_xyxy.shape[0]):
        box = boxes_xyxy[i].numpy()
        label = final_labels[i].item()
        score = final_scores[i].item()

        draw.rectangle(list(box), outline="red", width=3)
        text = f"Class {label}: {score:.2f}"

        try:
            # 尝试使用更美观的字体
            font = ImageFont.truetype("arial.ttf", 20)
            draw.text((box[0], box[1]), text, fill="red", font=font)
        except IOError:
            # 如果字体文件不存在，则使用默认字体
            draw.text((box[0], box[1]), text, fill="red")

    result_path = "vis_result.jpg"
    original_img.save(result_path)
    print(f"可视化结果已保存至: {result_path}")


if __name__ == '__main__':
    main()
