#!/usr/bin/env python3
"""
熊模型推理脚本 - 测试专门训练的熊检测模型
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR/jittor_rt_detr')

import jittor as jt
from src.nn.model import RTDETR

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
    """预处理图片"""
    print(f"Processing image: {image_path}")
    
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    print(f"✅ Image loaded: {original_size}")
    
    # 数据变换
    transform = Compose([
        Resize((640, 640)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 应用变换并添加batch维度
    image_tensor = transform(image).unsqueeze(0)
    print(f"✅ Image preprocessed: {original_size} -> {list(image_tensor.shape)}")
    
    return image_tensor, original_size

def convert_cxcywh_to_xyxy(boxes):
    """将中心点格式转换为左上右下格式"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)

def nms(boxes, scores, iou_threshold=0.1):
    """改进的非极大值抑制 - 专门处理RT-DETR的中心点格式"""
    if len(boxes) == 0:
        return []
    
    print(f"NMS输入: {len(boxes)} 个框, IoU阈值: {iou_threshold}")
    
    # 转换为xyxy格式
    boxes_xyxy = convert_cxcywh_to_xyxy(boxes)
    
    # 计算面积
    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    # 按分数排序
    order = scores.argsort()[::-1]
    print(f"按置信度排序的前5个: {scores[order[:5]]}")
    
    keep = []
    iteration = 0
    while len(order) > 0:
        iteration += 1
        i = order[0]
        keep.append(i)
        print(f"NMS迭代 {iteration}: 保留框 {i}, 置信度 {scores[i]:.3f}")
        
        if len(order) == 1:
            break
            
        # 计算IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / (union + 1e-6)  # 避免除零
        
        # 打印IoU信息
        high_iou_count = np.sum(iou > iou_threshold)
        print(f"  与其他 {len(iou)} 个框的IoU: 最大 {np.max(iou):.3f}, 超过阈值的: {high_iou_count}")
        
        # 保留IoU小于阈值的框
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
        
        # 安全检查：如果保留的框太多，强制停止
        if len(keep) >= 3:
            print(f"  强制停止NMS，已保留 {len(keep)} 个框")
            break
    
    print(f"NMS完成: 保留 {len(keep)} 个框")
    return keep

def bear_postprocess(logits, boxes, conf_threshold=0.3, nms_threshold=0.1):
    """熊模型的后处理"""
    print("=== 熊模型后处理分析 ===")
    
    # 获取最后一层输出
    final_logits = logits[-1][0]  # (num_queries, num_classes)
    final_boxes = boxes[-1][0]    # (num_queries, 4)
    
    # 转换为numpy
    logits_np = final_logits.stop_grad().numpy()
    boxes_np = final_boxes.stop_grad().numpy()
    
    print(f"Logits shape: {logits_np.shape}")
    print(f"Boxes shape: {boxes_np.shape}")
    
    # 对于单类别模型，直接使用logits作为置信度
    if logits_np.shape[1] == 1:  # 只有一个类别（熊）
        print("🐻 单类别模型：只检测熊")
        confidences = 1.0 / (1.0 + np.exp(-logits_np[:, 0]))  # sigmoid
        
        print(f"置信度范围: [{np.min(confidences):.3f}, {np.max(confidences):.3f}]")
        
        # 过滤低置信度检测
        valid_mask = confidences > conf_threshold
        valid_confidences = confidences[valid_mask]
        valid_boxes = boxes_np[valid_mask]
        
        print(f"通过置信度筛选的检测: {len(valid_confidences)}")
        
        if len(valid_confidences) == 0:
            return [], [], []
        
        # 应用更严格的NMS
        keep_indices = nms(valid_boxes, valid_confidences, nms_threshold)
        print(f"NMS后保留的检测: {len(keep_indices)}")
        
        if len(keep_indices) == 0:
            return [], [], []
        
        final_boxes = valid_boxes[keep_indices]
        final_scores = valid_confidences[keep_indices]
        final_labels = np.zeros(len(final_scores), dtype=np.int64)  # 都是熊（类别0）
        
        # 按置信度排序
        sorted_indices = np.argsort(final_scores)[::-1]
        final_boxes = final_boxes[sorted_indices]
        final_scores = final_scores[sorted_indices]
        final_labels = final_labels[sorted_indices]
        
        # 只保留前1个最高置信度的检测（因为只有1只熊）
        max_detections = 1
        if len(final_boxes) > max_detections:
            final_boxes = final_boxes[:max_detections]
            final_scores = final_scores[:max_detections]
            final_labels = final_labels[:max_detections]
            print(f"限制检测数量为: {max_detections}")
        
        return final_boxes, final_scores, final_labels
    
    else:
        print("❌ 多类别模型，这不是我们期望的")
        return [], [], []

def visualize_bear_results(image_path, boxes, scores, labels, output_path):
    """可视化熊检测结果"""
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    print("Creating visualization...")
    print(f"原图尺寸: {image.size}")
    
    if len(boxes) == 0:
        print("❌ 没有检测到任何熊")
    else:
        print(f"检测到 {len(boxes)} 个熊:")
        
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            # 注意：RT-DETR的boxes格式是 [cx, cy, w, h] (中心点坐标和宽高)
            # 需要转换为 [x1, y1, x2, y2] 格式
            cx, cy, w, h = box
            
            # 转换为左上角和右下角坐标（归一化坐标 0-1）
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            
            # 转换到原图尺寸
            orig_w, orig_h = image.size
            x1_pixel = x1 * orig_w
            y1_pixel = y1 * orig_h
            x2_pixel = x2 * orig_w
            y2_pixel = y2 * orig_h
            
            # 确保坐标在图像范围内
            x1_pixel = max(0, min(x1_pixel, orig_w))
            y1_pixel = max(0, min(y1_pixel, orig_h))
            x2_pixel = max(0, min(x2_pixel, orig_w))
            y2_pixel = max(0, min(y2_pixel, orig_h))
            
            # 绘制边界框
            color = "red"
            draw.rectangle([x1_pixel, y1_pixel, x2_pixel, y2_pixel], outline=color, width=3)
            
            # 添加标签
            text = f"🐻 bear: {score:.3f}"
            draw.text((x1_pixel, max(0, y1_pixel-25)), text, fill=color, font=font)
            
            print(f"  - 🐻 bear: {score:.3f}")
            print(f"    归一化坐标: cx={cx:.3f}, cy={cy:.3f}, w={w:.3f}, h={h:.3f}")
            print(f"    像素坐标: [{x1_pixel:.1f}, {y1_pixel:.1f}, {x2_pixel:.1f}, {y2_pixel:.1f}]")
    
    image.save(output_path)
    print(f"结果保存至: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="熊模型推理")
    parser.add_argument('--model_path', type=str, default='single_bear_training/checkpoints/bear_model_final.pkl', help='模型路径')
    parser.add_argument('--image_path', type=str, default='data/coco/val2017/000000000285.jpg', help='图片路径')
    parser.add_argument('--output_path', type=str, default='single_bear_training/bear_detection_result.jpg', help='输出路径')
    parser.add_argument('--conf_threshold', type=float, default=0.05, help='置信度阈值')
    
    args = parser.parse_args()
    
    print("=== 🐻 熊模型推理测试 ===")
    print(f"模型路径: {args.model_path}")
    print(f"图片路径: {args.image_path}")
    print(f"置信度阈值: {args.conf_threshold}")
    
    # 检查文件
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"图片文件不存在: {args.image_path}")
    
    try:
        # 1. 创建模型
        print("Loading bear model...")
        model = RTDETR(num_classes=1)  # 只有一个类别：熊
        
        # 2. 加载权重
        state_dict = jt.load(args.model_path)
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ Bear model loaded successfully!")
        
        # 3. 预处理图片
        image_tensor, original_size = preprocess_image(args.image_path)
        
        # 4. 推理
        print("Running bear inference...")
        with jt.no_grad():
            logits, boxes, enc_logits, enc_boxes = model(image_tensor)
            print("✅ Bear inference completed successfully!")
        
        # 5. 后处理
        print("Bear post-processing...")
        det_boxes, det_scores, det_labels = bear_postprocess(logits, boxes, conf_threshold=args.conf_threshold)
        
        # 6. 可视化结果
        visualize_bear_results(args.image_path, det_boxes, det_scores, det_labels, args.output_path)
        
        # 7. 总结
        print("\n=== 🐻 熊检测总结 ===")
        if len(det_boxes) > 0:
            print(f"✅ 检测到 {len(det_boxes)} 只熊")
            print(f"最高置信度: {max(det_scores):.3f}")
            print("🎉 熊模型工作正常！")
        else:
            print("❌ 没有检测到熊")
            print("可能原因：")
            print("1. 置信度阈值太高")
            print("2. 模型训练不充分")
            print("3. 图片中确实没有熊")
        
        print(f"✅ 结果已保存: {args.output_path}")
        
    except Exception as e:
        print(f"❌ 推理失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
