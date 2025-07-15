#!/usr/bin/env python3
"""
RT-DETR Jittor版本模型测试脚本
测试训练好的模型在验证数据上的表现
"""

import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt
from jittor_rt_detr.src.nn.rtdetr_complete_pytorch_aligned import build_rtdetr_complete

# 设置Jittor
jt.flags.use_cuda = 1
jt.flags.auto_mixed_precision_level = 0

def safe_float32(tensor):
    """安全地将tensor转换为float32"""
    if isinstance(tensor, jt.Var):
        return tensor.float32()
    elif isinstance(tensor, np.ndarray):
        return jt.array(tensor.astype(np.float32))
    else:
        return jt.array(tensor, dtype=jt.float32)

def load_model(model_path, num_classes=80):
    """加载训练好的模型"""
    print(f">>> 加载模型: {model_path}")
    
    # 创建模型
    model = build_rtdetr_complete(num_classes=num_classes, hidden_dim=256, num_queries=300)
    
    # 加载权重
    if os.path.exists(model_path):
        state_dict = jt.load(model_path)
        model.load_state_dict(state_dict)
        print("✅ 模型权重加载成功")
    else:
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    model.eval()
    return model

def load_coco_categories():
    """加载COCO类别信息"""
    ann_file = "data/coco2017_50/annotations/instances_train2017.json"
    
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # 创建类别映射
    categories = coco_data['categories']
    id_to_name = {cat['id']: cat['name'] for cat in categories}
    idx_to_name = {idx: cat['name'] for idx, cat in enumerate(categories)}
    
    return idx_to_name, id_to_name

def preprocess_image(image_path, target_size=640):
    """预处理单张图片"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Resize
    image_resized = image.resize((target_size, target_size))
    
    # 转换为numpy数组并归一化
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    
    # 转换为Jittor tensor
    img_tensor = safe_float32(img_array).unsqueeze(0)  # 添加batch维度
    
    return img_tensor, image, original_size

def postprocess_predictions(outputs, confidence_threshold=0.5, original_size=(640, 640), target_size=640):
    """后处理预测结果"""
    pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
    pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4]
    
    # 计算置信度
    pred_scores = jt.sigmoid(pred_logits).max(dim=-1)[0]  # [num_queries]
    pred_labels = jt.sigmoid(pred_logits).argmax(dim=-1)  # [num_queries]
    
    # 过滤低置信度预测
    keep = pred_scores > confidence_threshold
    
    if keep.sum() == 0:
        return [], [], []
    
    filtered_scores = pred_scores[keep]
    filtered_labels = pred_labels[keep]
    filtered_boxes = pred_boxes[keep]
    
    # 转换边界框格式：从中心点格式到左上右下格式
    boxes_xyxy = []
    for box in filtered_boxes:
        cx, cy, w, h = box.numpy()
        
        # 转换到原始图片尺寸
        cx *= original_size[0]
        cy *= original_size[1]
        w *= original_size[0]
        h *= original_size[1]
        
        # 转换为左上右下格式
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        boxes_xyxy.append([x1, y1, x2, y2])
    
    return filtered_scores.numpy(), filtered_labels.numpy(), boxes_xyxy

def visualize_predictions(image, scores, labels, boxes, idx_to_name, save_path=None):
    """可视化预测结果"""
    # 创建绘图对象
    draw = ImageDraw.Draw(image)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # 颜色列表
    colors = [
        '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
        '#800000', '#008000', '#000080', '#808000', '#800080', '#008080'
    ]
    
    for i, (score, label, box) in enumerate(zip(scores, labels, boxes)):
        x1, y1, x2, y2 = box
        color = colors[i % len(colors)]
        
        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # 绘制标签和置信度
        class_name = idx_to_name.get(label, f'class_{label}')
        text = f'{class_name}: {score:.2f}'
        
        # 计算文本位置
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 绘制文本背景
        draw.rectangle([x1, y1-text_height-5, x1+text_width+10, y1], fill=color)
        
        # 绘制文本
        draw.text((x1+5, y1-text_height-2), text, fill='white', font=font)
    
    if save_path:
        image.save(save_path)
        print(f"✅ 可视化结果已保存到: {save_path}")
    
    return image

def test_single_image(model, image_path, idx_to_name, confidence_threshold=0.5, save_dir="results/test_results"):
    """测试单张图片"""
    print(f"\n>>> 测试图片: {os.path.basename(image_path)}")
    
    # 预处理
    img_tensor, original_image, original_size = preprocess_image(image_path)
    
    # 推理
    with jt.no_grad():
        outputs = model(img_tensor)
    
    # 后处理
    scores, labels, boxes = postprocess_predictions(
        outputs, confidence_threshold, original_size
    )
    
    print(f"检测到 {len(scores)} 个目标:")
    for i, (score, label, box) in enumerate(zip(scores, labels, boxes)):
        class_name = idx_to_name.get(label, f'class_{label}')
        print(f"  {i+1}. {class_name}: {score:.3f} at {box}")
    
    # 可视化
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"result_{os.path.basename(image_path)}")
    
    result_image = visualize_predictions(
        original_image.copy(), scores, labels, boxes, idx_to_name, save_path
    )
    
    return len(scores), scores, labels, boxes

def test_multiple_images(model, test_dir, idx_to_name, confidence_threshold=0.5, max_images=10):
    """测试多张图片"""
    print(f"\n>>> 批量测试图片 (最多{max_images}张)")
    
    # 获取测试图片列表
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for file in os.listdir(test_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(test_dir, file))
    
    image_files = image_files[:max_images]
    
    if not image_files:
        print(f"❌ 在 {test_dir} 中没有找到图片文件")
        return
    
    # 测试统计
    total_detections = 0
    detection_stats = {}
    
    for image_path in tqdm(image_files, desc="测试图片"):
        try:
            num_detections, scores, labels, boxes = test_single_image(
                model, image_path, idx_to_name, confidence_threshold
            )
            total_detections += num_detections
            
            # 统计类别
            for label in labels:
                class_name = idx_to_name.get(label, f'class_{label}')
                detection_stats[class_name] = detection_stats.get(class_name, 0) + 1
                
        except Exception as e:
            print(f"❌ 测试 {image_path} 失败: {e}")
    
    # 打印统计结果
    print(f"\n=== 测试统计 ===")
    print(f"测试图片数: {len(image_files)}")
    print(f"总检测数: {total_detections}")
    print(f"平均每张图片检测数: {total_detections/len(image_files):.1f}")
    
    if detection_stats:
        print(f"\n类别检测统计:")
        for class_name, count in sorted(detection_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count}")

def create_test_report(model_path, test_results):
    """创建测试报告"""
    report_path = "results/test_report.txt"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("RT-DETR Jittor版本模型测试报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"测试时间: {jt.misc.get_time()}\n\n")
        
        # 这里可以添加更多测试结果
        f.write("测试完成\n")
    
    print(f"✅ 测试报告已保存到: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='测试RT-DETR Jittor模型')
    parser.add_argument('--model_path', default='checkpoints/rtdetr_jittor.pkl', 
                       help='模型文件路径')
    parser.add_argument('--test_image', default=None, 
                       help='单张测试图片路径')
    parser.add_argument('--test_dir', default='data/coco2017_50/train2017', 
                       help='测试图片目录')
    parser.add_argument('--confidence', type=float, default=0.3, 
                       help='置信度阈值')
    parser.add_argument('--max_images', type=int, default=10, 
                       help='最大测试图片数')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("===        RT-DETR Jittor模型测试        ===")
    print("=" * 60)
    
    try:
        # 加载模型
        model = load_model(args.model_path)
        if model is None:
            return
        
        # 加载类别信息
        idx_to_name, id_to_name = load_coco_categories()
        print(f"✅ 加载了 {len(idx_to_name)} 个类别")
        
        # 测试
        if args.test_image:
            # 测试单张图片
            test_single_image(model, args.test_image, idx_to_name, args.confidence)
        else:
            # 测试多张图片
            test_multiple_images(model, args.test_dir, idx_to_name, args.confidence, args.max_images)
        
        # 创建测试报告
        create_test_report(args.model_path, {})
        
        print(f"\n🎯 模型测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
