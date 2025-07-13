#!/usr/bin/env python3
"""
多目标检测推理脚本
专用于测试训练好的RT-DETR模型在000000000785.jpg上检测人和滑雪板
"""

from src.nn.model import RTDETR
import jittor as jt
import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR/jittor_rt_detr')


# 简单的数据变换

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return image.resize(self.size, Image.LANCZOS)


class ToTensor:
    def __call__(self, image):
        image_array = np.array(image, dtype=np.float32) / 255.0
        return jt.array(image_array.transpose(2, 0, 1))


class Normalize:
    def __init__(self, mean, std):
        self.mean = jt.array(mean).view(-1, 1, 1)
        self.std = jt.array(std).view(-1, 1, 1)

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


def preprocess_image(image_path):
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    print(f"✅ Image loaded: {original_size}")
    transform = Compose([
        Resize((640, 640)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    print(
        f"✅ Image preprocessed: {original_size} -> {list(image_tensor.shape)}")
    return image_tensor, original_size


def load_class_mapping(cat_id_to_idx_path, ann_file):
    with open(cat_id_to_idx_path, 'r') as f:
        cat_id_to_idx = json.load(f)
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    cat_id_to_name = {cat['id']: cat['name']
                      for cat in coco_data['categories']}
    idx_to_name = {int(idx): cat_id_to_name[int(cat_id)]
                   for cat_id, idx in cat_id_to_idx.items()}
    return idx_to_name


def postprocess_predictions(logits, boxes, idx_to_name, original_size, conf_threshold=0.5):
    pred_logits = logits[-1][0]  # (num_queries, num_classes)
    pred_boxes = boxes[-1][0]    # (num_queries, 4)
    pred_scores = jt.sigmoid(pred_logits)
    pred_scores_np = pred_scores.numpy()
    pred_boxes_np = pred_boxes.numpy()
    detections = []
    for query_idx in range(pred_logits.shape[0]):
        scores = pred_scores_np[query_idx]
        box = pred_boxes_np[query_idx]
        max_class_idx = np.argmax(scores)
        max_score = scores[max_class_idx]
        if max_score > conf_threshold:
            cx, cy, w, h = box
            x1 = (cx - w/2) * original_size[0]
            y1 = (cy - h/2) * original_size[1]
            x2 = (cx + w/2) * original_size[0]
            y2 = (cy + h/2) * original_size[1]
            x1 = max(0, min(x1, original_size[0]))
            y1 = max(0, min(y1, original_size[1]))
            x2 = max(0, min(x2, original_size[0]))
            y2 = max(0, min(y2, original_size[1]))
            class_name = idx_to_name.get(
                max_class_idx, f"class_{max_class_idx}")
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'class_name': class_name,
                'confidence': max_score,
                'class_idx': max_class_idx
            })
    return detections


def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def apply_nms_per_class(detections, iou_threshold=0.3):
    if not detections:
        return []
    results = []
    detections_by_class = {}
    for det in detections:
        cls = det['class_name']
        detections_by_class.setdefault(cls, []).append(det)
    for cls, dets in detections_by_class.items():
        dets = sorted(dets, key=lambda x: x['confidence'], reverse=True)
        keep = []
        while dets:
            current = dets.pop(0)
            keep.append(current)
            dets = [d for d in dets if calculate_iou(
                current['bbox'], d['bbox']) < iou_threshold]
        results.extend(keep)
    return results


def visualize_results(image_path, detections, output_path):
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    color_map = {
        'person': (255, 0, 0),
        'skis': (0, 255, 0),
    }
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class_name']
        confidence = det['confidence']
        color = color_map.get(class_name, (0, 0, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{class_name}: {confidence:.3f}"
        draw.text((x1, y1-25), label, fill=color, font=font)
        print(
            f"  {i+1}. {class_name}: 置信度={confidence:.3f}, 边界框=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    image.save(output_path)
    print(f"✅ 检测结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="多目标推理 - 000000000785.jpg")
    parser.add_argument('--model_path', type=str,
                        default='multi_target_training/checkpoints/multi_target_model_final.pkl', help='模型路径')
    parser.add_argument('--image_path', type=str,
                        default='data/coco/val2017/000000000785.jpg', help='图片路径')
    parser.add_argument('--cat_id_to_idx_path', type=str,
                        default='multi_target_training/checkpoints/cat_id_to_idx.json', help='类别映射路径')
    parser.add_argument('--ann_file', type=str,
                        default='data/coco/annotations/instances_val2017.json', help='标注文件')
    parser.add_argument('--output_path', type=str,
                        default='multi_target_training/detection_result_785.jpg', help='输出路径')
    parser.add_argument('--conf_threshold', type=float,
                        default=0.5, help='置信度阈值')
    parser.add_argument('--nms_threshold', type=float,
                        default=0.3, help='NMS阈值')
    args = parser.parse_args()
    print("=== 🎯 多目标推理测试 ===")
    print(f"模型路径: {args.model_path}")
    print(f"图片路径: {args.image_path}")
    print(f"类别映射路径: {args.cat_id_to_idx_path}")
    print(f"置信度阈值: {args.conf_threshold}")
    print(f"NMS阈值: {args.nms_threshold}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"图片文件不存在: {args.image_path}")
    if not os.path.exists(args.cat_id_to_idx_path):
        raise FileNotFoundError(f"类别映射文件不存在: {args.cat_id_to_idx_path}")
    try:
        print("加载类别映射...")
        idx_to_name = load_class_mapping(
            args.cat_id_to_idx_path, args.ann_file)
        print(f"✅ 类别映射: {idx_to_name}")
        print("加载模型...")
        num_classes = len(idx_to_name)
        model = RTDETR(num_classes=num_classes)
        state_dict = jt.load(args.model_path)
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ 模型加载成功!")
        image_tensor, original_size = preprocess_image(args.image_path)
        print("执行推理...")
        with jt.no_grad():
            logits, boxes, enc_logits, enc_boxes = model(image_tensor)
            print("✅ 推理完成!")
        detections = postprocess_predictions(
            logits, boxes, idx_to_name, original_size, args.conf_threshold)
        print(f"📊 初步检测到 {len(detections)} 个目标")
        final_detections = apply_nms_per_class(detections, args.nms_threshold)
        print(f"📊 NMS后保留 {len(final_detections)} 个目标")
        visualize_results(args.image_path, final_detections, args.output_path)
        print("🎉 多目标推理完成!")
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
