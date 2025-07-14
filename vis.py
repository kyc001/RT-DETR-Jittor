# vis.py

import jittor as jt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse
import os
import sys

# 导入项目中的核心模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'jittor_rt_detr'))

from src.nn.model import RTDETR

# 创建推理用的数据变换函数
def resize_and_normalize(image, target_size=(640, 640)):
    """将图像resize并归一化"""
    # Resize图像
    resized_image = image.resize(target_size, Image.LANCZOS)

    # 转换为numpy数组并归一化
    img_array = np.array(resized_image, dtype=np.float32) / 255.0

    # ImageNet标准化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std

    # 转换为CHW格式的tensor
    img_tensor = jt.array(img_array.transpose(2, 0, 1))

    return img_tensor

# COCO 类别名称 (前80个)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def preprocess_image(image_path):
    """加载并预处理单张图片"""
    img = Image.open(image_path).convert('RGB')
    original_size = img.size

    # 使用我们自定义的resize和归一化函数
    img_tensor = resize_and_normalize(img, target_size=(640, 640))

    # 添加 batch 维度
    return img_tensor.unsqueeze(0), original_size

def postprocess(outputs, original_size, conf_threshold=0.5):
    """
    对模型输出进行后处理，转换为最终的检测框。
    """
    # 解包模型输出
    logits, boxes, _, _ = outputs  # 忽略编码器输出

    # 我们只关心最后一层解码器的输出
    pred_logits = logits[-1]  # (1, num_queries, num_classes)
    pred_boxes = boxes[-1]    # (1, num_queries, 4)

    # 计算分数 - 使用sigmoid而不是softmax，因为这是多类别检测
    scores = jt.sigmoid(pred_logits)[0]  # (num_queries, num_classes)

    # 找到每个query最高分数的类别和值
    scores_max = jt.max(scores, dim=-1)[0]  # (num_queries,)
    labels = jt.argmax(scores, dim=-1)      # (num_queries,)

    # 过滤掉置信度低的预测 - 使用numpy方式处理
    keep_mask_np = (scores_max > conf_threshold).numpy()
    keep_indices_np = np.where(keep_mask_np)[0]

    if len(keep_indices_np) == 0:
        # 没有检测到任何目标
        return jt.zeros((0, 4)), jt.zeros((0,)), jt.zeros((0,), dtype='int64')

    # 转换回jittor索引
    keep_indices = jt.array(keep_indices_np, dtype='int64')

    final_boxes = pred_boxes[keep_indices, :]     # (num_keep, 4)
    final_scores = scores_max[keep_indices]       # (num_keep,)
    final_labels = labels[keep_indices]           # (num_keep,)

    # 将归一化的 (cx, cy, w, h) 格式的框转换为像素坐标 (x1, y1, x2, y2)
    if final_boxes.shape[0] > 0:
        orig_w, orig_h = original_size

        # 手动解包坐标
        cx = final_boxes[:, 0]
        cy = final_boxes[:, 1]
        w = final_boxes[:, 2]
        h = final_boxes[:, 3]

        x1 = (cx - 0.5 * w) * orig_w
        y1 = (cy - 0.5 * h) * orig_h
        x2 = (cx + 0.5 * w) * orig_w
        y2 = (cy + 0.5 * h) * orig_h

        final_boxes = jt.stack([x1, y1, x2, y2], dim=1)

    return final_boxes, final_scores, final_labels

def visualize(image_path, boxes, scores, labels, output_path="vis_result.jpg"):
    """将检测结果绘制在图片上"""
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    # 尝试加载字体
    try:
        # 尝试多种字体路径
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Arial.ttf",
            "arial.ttf"
        ]
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, 20)
                break
            except:
                continue
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    print(f"--- 可视化结果 (发现 {boxes.shape[0]} 个目标) ---")

    # 定义颜色列表
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]

    for i in range(boxes.shape[0]):
        box = boxes[i].numpy().tolist()
        score = scores[i].item()
        label_idx = labels[i].item()

        # 确保label_idx在有效范围内
        if label_idx >= len(COCO_CLASSES):
            class_name = f"class_{label_idx}"
        else:
            class_name = COCO_CLASSES[label_idx]

        # 选择颜色
        color = colors[i % len(colors)]

        # 确保坐标在合理范围内
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img.width, x2), min(img.height, y2)

        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 绘制标签和置信度
        text = f"{class_name}: {score:.2f}"

        # 计算文本背景框
        try:
            text_bbox = draw.textbbox((x1, y1-25), text, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1-25), text, fill="white", font=font)
        except:
            # 如果textbbox不可用，使用简单的文本绘制
            draw.text((x1, y1-25), text, fill=color, font=font)

        print(f"  - 目标: {class_name}, 置信度: {score:.2f}, 位置: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

    img.save(output_path)
    print(f"✅ 可视化结果已保存至: {output_path}")

def main(args):
    # 设置Jittor优化选项
    jt.flags.use_cuda = 1

    # 1. 初始化模型
    print(">>> 正在加载模型...")
    # 确保 num_classes 与训练时一致
    model = RTDETR(num_classes=80)

    # 加载权重
    state_dict = jt.load(args.weights)
    model.load_state_dict(state_dict)
    model.eval() # 切换到评估模式
    print(f"✅ 模型加载成功: {args.weights}")

    # 2. 预处理图片
    print(f">>> 正在处理图片: {args.img_path}")
    img_tensor, original_size = preprocess_image(args.img_path)
    print("✅ 图片预处理完成。")

    # 3. 执行推理（添加预热和优化）
    print(">>> 正在执行推理...")

    # 预热模型（第一次推理会比较慢，因为需要编译）
    if not hasattr(main, '_warmed_up'):
        print("  正在预热模型...")
        with jt.no_grad():
            _ = model(img_tensor)
        main._warmed_up = True
        print("  模型预热完成")

    # 实际推理
    with jt.no_grad():
        # 模型输出: (logits, boxes, enc_logits, enc_boxes)
        outputs = model(img_tensor)
    print("✅ 推理完成。")

    # 4. 后处理
    print(">>> 正在后处理...")
    boxes, scores, labels = postprocess(outputs, original_size, conf_threshold=args.conf_threshold)
    print("✅ 后处理完成。")

    # 5. 可视化
    visualize(args.img_path, boxes, scores, labels, args.output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RT-DETR Jittor Visualization Script')
    
    parser.add_argument('--weights', type=str, required=True, help='训练好的模型权重文件路径 (.pkl)')
    parser.add_argument('--img_path', type=str, required=True, help='需要进行检测的图片路径')
    parser.add_argument('--output_path', type=str, default='vis_result.jpg', help='可视化结果的保存路径')
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='置信度阈值，低于此值的检测将被忽略')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"权重文件未找到: {args.weights}")
    if not os.path.exists(args.img_path):
        raise FileNotFoundError(f"图片文件未找到: {args.img_path}")
        
    main(args)