#!/usr/bin/env python3
"""
RT-DETR PyTorch 演示脚本
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2
import time

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# COCO类别名称
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def create_demo_model():
    """创建演示用的简化模型"""
    class DemoRTDETR(torch.nn.Module):
        def __init__(self, num_classes=80):
            super().__init__()
            # 简化的backbone
            self.backbone = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 7, stride=2, padding=3),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(3, stride=2, padding=1),
                torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(128, 256, 3, stride=2, padding=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(inplace=True),
            )

            # 简化的transformer
            self.transformer = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(
                    d_model=256, nhead=8, dim_feedforward=1024),
                num_layers=6
            )

            # 分类头和回归头
            self.class_head = torch.nn.Linear(256, num_classes)
            self.bbox_head = torch.nn.Linear(256, 4)

        def forward(self, x):
            batch_size = x.size(0)
            x = self.backbone(x)  # [B, 256, H/32, W/32]
            x = x.flatten(2).transpose(1, 2)  # [B, N, 256]
            x = self.transformer(x)

            # 预测
            cls_pred = self.class_head(x)  # [B, N, num_classes]
            bbox_pred = self.bbox_head(x)  # [B, N, 4]

            return cls_pred, bbox_pred

    return DemoRTDETR()


def preprocess_image(image_path, target_size=(640, 640)):
    """预处理图片"""
    # 读取图片
    image = Image.open(image_path).convert('RGB')

    # 获取原始尺寸
    orig_w, orig_h = image.size

    # 调整尺寸
    image = image.resize(target_size, Image.Resampling.LANCZOS)

    # 转换为tensor
    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    # 标准化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std

    return image_tensor, (orig_w, orig_h)


def postprocess_predictions(cls_pred, bbox_pred, conf_threshold=0.3, orig_size=None):
    """后处理预测结果"""
    # 获取置信度和类别
    cls_scores, cls_ids = torch.max(cls_pred, dim=-1)

    # 过滤低置信度预测
    mask = cls_scores > conf_threshold
    filtered_scores = cls_scores[mask]
    filtered_ids = cls_ids[mask]
    filtered_boxes = bbox_pred[mask]

    # 转换为numpy
    scores = filtered_scores.cpu().numpy()
    class_ids = filtered_ids.cpu().numpy()
    boxes = filtered_boxes.cpu().numpy()

    # 如果提供了原始尺寸，调整边界框
    if orig_size:
        orig_w, orig_h = orig_size
        boxes[:, [0, 2]] *= orig_w / 640  # 调整x坐标
        boxes[:, [1, 3]] *= orig_h / 640  # 调整y坐标

    # 限制边界框在图片范围内
    boxes[:, [0, 2]] = np.clip(
        boxes[:, [0, 2]], 0, orig_w if orig_size else 640)
    boxes[:, [1, 3]] = np.clip(
        boxes[:, [1, 3]], 0, orig_h if orig_size else 640)

    return boxes, scores, class_ids


def visualize_detections(image_path, boxes, scores, class_ids, output_path=None):
    """可视化检测结果"""
    # 读取原始图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return

    # 绘制检测框
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int)

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制标签
        class_name = COCO_CLASSES[class_id] if class_id < len(
            COCO_CLASSES) else f'class_{class_id}'
        label = f'{class_name}: {score:.2f}'

        # 计算标签位置
        label_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(
            image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # 保存结果
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)
        print(f"结果已保存: {output_path}")
    else:
        # 显示结果
        cv2.imshow('RT-DETR Demo', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def create_demo_image():
    """创建演示图片"""
    # 创建一个简单的演示图片
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255

    # 绘制一些简单的图形
    cv2.rectangle(image, (100, 100), (200, 200), (0, 0, 255), -1)  # 红色矩形
    cv2.circle(image, (400, 150), 50, (255, 0, 0), -1)  # 蓝色圆形
    cv2.rectangle(image, (300, 300), (500, 400), (0, 255, 0), -1)  # 绿色矩形

    # 保存演示图片
    demo_path = 'demo_image.jpg'
    cv2.imwrite(demo_path, image)
    print(f"创建演示图片: {demo_path}")

    return demo_path


def run_demo():
    """运行演示"""
    print("=== RT-DETR PyTorch 演示 ===")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型
    print("创建演示模型...")
    model = create_demo_model().to(device)
    model.eval()

    # 创建演示图片
    print("创建演示图片...")
    demo_image_path = create_demo_image()

    # 预处理图片
    print("预处理图片...")
    image_tensor, orig_size = preprocess_image(demo_image_path)
    image_tensor = image_tensor.to(device)

    # 推理
    print("开始推理...")
    start_time = time.time()

    with torch.no_grad():
        cls_pred, bbox_pred = model(image_tensor)

    inference_time = time.time() - start_time
    print(f"推理时间: {inference_time:.3f}s")

    # 后处理
    print("后处理预测结果...")
    boxes, scores, class_ids = postprocess_predictions(
        cls_pred, bbox_pred, conf_threshold=0.3, orig_size=orig_size
    )

    # 打印结果
    print(f"检测到 {len(boxes)} 个目标")
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        class_name = COCO_CLASSES[class_id] if class_id < len(
            COCO_CLASSES) else f'class_{class_id}'
        print(f"  {i+1}. {class_name}: {score:.3f} at {box}")

    # 可视化结果
    print("可视化结果...")
    output_path = 'demo_result.jpg'
    visualize_detections(demo_image_path, boxes,
                         scores, class_ids, output_path)

    print("演示完成！")
    print(f"原始图片: {demo_image_path}")
    print(f"结果图片: {output_path}")


def main():
    """主函数"""
    try:
        run_demo()
    except Exception as e:
        print(f"演示运行失败: {e}")
        print("请检查PyTorch是否正确安装")


if __name__ == '__main__':
    main()
