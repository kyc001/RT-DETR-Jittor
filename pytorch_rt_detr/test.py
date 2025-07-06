#!/usr/bin/env python3
"""
RT-DETR PyTorch 测试脚本
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import time
from tqdm import tqdm

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


def parse_args():
    parser = argparse.ArgumentParser(description='RT-DETR Testing')
    parser.add_argument('--weights', type=str, required=True, help='模型权重路径')
    parser.add_argument('--img_path', type=str, help='测试图片路径')
    parser.add_argument('--img_dir', type=str, help='测试图片目录')
    parser.add_argument('--conf_threshold', type=float,
                        default=0.3, help='置信度阈值')
    parser.add_argument('--device', type=str, default='auto', help='测试设备')
    parser.add_argument('--output_dir', type=str,
                        default='results', help='输出目录')
    return parser.parse_args()


def setup_device(device_arg):
    """设置测试设备"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    print(f"使用设备: {device}")
    return device


def load_model(weights_path, device):
    """加载模型"""
    try:
        # 尝试加载检查点
        checkpoint = torch.load(weights_path, map_location=device)

        if 'model_state_dict' in checkpoint:
            # 从检查点加载
            from src.zoo.rtdetr import RTDETR
            model = RTDETR('configs/rtdetr_r50vd_6x_coco.yml')
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"从检查点加载模型: {weights_path}")
        else:
            # 直接加载模型权重
            model = torch.load(weights_path, map_location=device)
            print(f"直接加载模型: {weights_path}")

        model = model.to(device)
        model.eval()
        return model

    except Exception as e:
        print(f"模型加载失败: {e}")
        print("使用简化版模型进行测试")
        return create_simple_model(device)


def create_simple_model(device):
    """创建简化版模型用于测试"""
    class SimpleRTDETR(nn.Module):
        def __init__(self, num_classes=80):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )

            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=256, nhead=8, dim_feedforward=1024),
                num_layers=6
            )

            self.class_head = nn.Linear(256, num_classes)
            self.bbox_head = nn.Linear(256, 4)

        def forward(self, x):
            batch_size = x.size(0)
            x = self.backbone(x)
            x = x.flatten(2).transpose(1, 2)
            x = self.transformer(x)

            cls_pred = self.class_head(x)
            bbox_pred = self.bbox_head(x)

            return cls_pred, bbox_pred

    model = SimpleRTDETR().to(device)
    model.eval()
    return model


def preprocess_image(image_path, target_size=(640, 640)):
    """预处理图片"""
    # 读取图片
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path

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
        cv2.imshow('RT-DETR Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def test_single_image(model, image_path, device, conf_threshold=0.3, output_dir='results'):
    """测试单张图片"""
    print(f"测试图片: {image_path}")

    # 预处理
    image_tensor, orig_size = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

    # 推理
    start_time = time.time()
    with torch.no_grad():
        try:
            outputs = model(image_tensor)
            if isinstance(outputs, tuple):
                cls_pred, bbox_pred = outputs
            else:
                cls_pred, bbox_pred = outputs['cls_pred'], outputs['bbox_pred']
        except Exception as e:
            print(f"推理失败: {e}")
            return

    inference_time = time.time() - start_time

    # 后处理
    boxes, scores, class_ids = postprocess_predictions(
        cls_pred, bbox_pred, conf_threshold, orig_size
    )

    # 打印结果
    print(f"推理时间: {inference_time:.3f}s")
    print(f"检测到 {len(boxes)} 个目标")

    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        class_name = COCO_CLASSES[class_id] if class_id < len(
            COCO_CLASSES) else f'class_{class_id}'
        print(f"  {i+1}. {class_name}: {score:.3f} at {box}")

    # 可视化
    output_path = os.path.join(
        output_dir, f'detection_{os.path.basename(image_path)}')
    visualize_detections(image_path, boxes, scores, class_ids, output_path)

    return boxes, scores, class_ids


def test_directory(model, img_dir, device, conf_threshold=0.3, output_dir='results'):
    """测试目录中的所有图片"""
    if not os.path.exists(img_dir):
        print(f"目录不存在: {img_dir}")
        return

    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []

    for file in os.listdir(img_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(img_dir, file))

    print(f"找到 {len(image_files)} 张图片")

    # 测试每张图片
    for image_path in tqdm(image_files, desc="测试图片"):
        try:
            test_single_image(model, image_path, device,
                              conf_threshold, output_dir)
        except Exception as e:
            print(f"测试图片 {image_path} 失败: {e}")


def main():
    args = parse_args()

    # 设置设备
    device = setup_device(args.device)

    # 加载模型
    print("加载模型...")
    model = load_model(args.weights, device)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 测试
    if args.img_path:
        # 测试单张图片
        test_single_image(model, args.img_path, device,
                          args.conf_threshold, args.output_dir)
    elif args.img_dir:
        # 测试目录
        test_directory(model, args.img_dir, device,
                       args.conf_threshold, args.output_dir)
    else:
        print("请指定 --img_path 或 --img_dir 参数")
        return

    print("测试完成！")


if __name__ == '__main__':
    main()
