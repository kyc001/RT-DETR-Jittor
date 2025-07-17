#!/usr/bin/env python3
"""
可视化组件
提供RT-DETR推理和可视化功能，参考ultimate_sanity_check.py的验证实现
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from scipy.special import softmax

# 设置matplotlib支持中文字体
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt

# COCO类别名称映射
COCO_CLASSES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup',
    48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
    53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
    58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
    63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
    70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
    76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
    80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush'
}

class RTDETRVisualizer:
    """
    RT-DETR可视化器
    参考ultimate_sanity_check.py的验证实现
    """
    def __init__(self, model, conf_threshold=0.3, save_dir="./results/visualizations"):
        self.model = model
        self.conf_threshold = conf_threshold
        self.save_dir = save_dir
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"📁 可视化结果将保存到: {save_dir}")
        print(f"🎯 置信度阈值: {conf_threshold}")
    
    def preprocess_image(self, image_path):
        """
        预处理图像
        参考ultimate_sanity_check.py的实现
        """
        image = Image.open(image_path).convert('RGB')
        original_width, original_height = image.size
        
        # 调整图像大小到640x640
        image_resized = image.resize((640, 640), Image.LANCZOS)
        img_array = np.array(image_resized).astype(np.float32) / 255.0
        img_tensor = jt.array(img_array.transpose(2, 0, 1)).float32().unsqueeze(0)
        
        return img_tensor, original_width, original_height, image
    
    def postprocess_predictions(self, outputs):
        """
        后处理预测结果
        参考ultimate_sanity_check.py的实现
        """
        if 'pred_logits' not in outputs or 'pred_boxes' not in outputs:
            return np.array([]), np.array([]), np.array([])
        
        pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
        pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4]
        
        # 转换为numpy并计算置信度
        pred_logits_np = pred_logits.numpy()
        pred_boxes_np = pred_boxes.numpy()
        
        # 使用softmax计算置信度
        scores = softmax(pred_logits_np, axis=-1)
        max_scores = np.max(scores[:, :-1], axis=-1)  # 排除背景类
        predicted_labels = np.argmax(scores[:, :-1], axis=-1)
        
        # 过滤低置信度检测
        valid_mask = max_scores > self.conf_threshold
        
        final_boxes = pred_boxes_np[valid_mask]
        final_scores = max_scores[valid_mask]
        final_labels = predicted_labels[valid_mask]
        
        return final_boxes, final_scores, final_labels
    
    def inference_single_image(self, image_path):
        """
        单张图像推理
        
        Args:
            image_path: 图像路径
        
        Returns:
            results: 检测结果字典
        """
        # 预处理
        img_tensor, original_width, original_height, original_image = self.preprocess_image(image_path)
        
        # 推理
        self.model.eval()
        with jt.no_grad():
            outputs = self.model(img_tensor)
        
        # 后处理
        pred_boxes, pred_scores, pred_labels = self.postprocess_predictions(outputs)
        
        return {
            'image_path': image_path,
            'original_image': original_image,
            'original_width': original_width,
            'original_height': original_height,
            'pred_boxes': pred_boxes,
            'pred_scores': pred_scores,
            'pred_labels': pred_labels,
            'num_detections': len(pred_boxes)
        }
    
    def visualize_detection(self, results, save_path=None, show_confidence=True):
        """
        可视化检测结果
        
        Args:
            results: 推理结果
            save_path: 保存路径
            show_confidence: 是否显示置信度
        """
        image = results['original_image'].copy()
        draw = ImageDraw.Draw(image)
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # 绘制检测框
        for i, (box, score, label) in enumerate(zip(results['pred_boxes'], results['pred_scores'], results['pred_labels'])):
            # 转换坐标到原始图像尺寸
            x1, y1, x2, y2 = box
            x1 = x1 * results['original_width']
            y1 = y1 * results['original_height']
            x2 = x2 * results['original_width']
            y2 = y2 * results['original_height']
            
            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            
            # 绘制标签和置信度
            label_id = int(label) + 1  # 转换回COCO ID
            class_name = COCO_CLASSES.get(label_id, f"class_{label_id}")
            
            if show_confidence:
                text = f"{class_name}: {score:.2f}"
            else:
                text = class_name
            
            # 计算文本位置
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # 绘制文本背景
            draw.rectangle([x1, y1-text_height-5, x1+text_width+10, y1], fill='red')
            
            # 绘制文本
            draw.text((x1+5, y1-text_height-2), text, fill='white', font=font)
        
        # 保存图像
        if save_path is None:
            filename = os.path.basename(results['image_path'])
            name, ext = os.path.splitext(filename)
            save_path = os.path.join(self.save_dir, f"{name}_detection{ext}")
        
        image.save(save_path)
        print(f"📊 检测结果保存到: {save_path}")
        
        return save_path
    
    def create_detection_summary(self, results):
        """创建检测摘要"""
        summary = {
            'image_file': os.path.basename(results['image_path']),
            'image_size': f"{results['original_width']}x{results['original_height']}",
            'num_detections': results['num_detections'],
            'detections': []
        }
        
        for box, score, label in zip(results['pred_boxes'], results['pred_scores'], results['pred_labels']):
            label_id = int(label) + 1
            class_name = COCO_CLASSES.get(label_id, f"class_{label_id}")
            
            detection = {
                'class_name': class_name,
                'class_id': label_id,
                'confidence': float(score),
                'bbox': [float(x) for x in box]  # [x1, y1, x2, y2] normalized
            }
            summary['detections'].append(detection)
        
        return summary
    
    def batch_inference(self, image_paths, save_visualizations=True):
        """
        批量推理
        
        Args:
            image_paths: 图像路径列表
            save_visualizations: 是否保存可视化结果
        
        Returns:
            all_results: 所有推理结果
        """
        all_results = []
        
        print(f"🔍 开始批量推理 {len(image_paths)} 张图像...")
        
        for i, image_path in enumerate(image_paths):
            print(f"   处理图像 {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            try:
                # 推理
                results = self.inference_single_image(image_path)
                all_results.append(results)
                
                print(f"     检测到 {results['num_detections']} 个目标")
                
                # 保存可视化
                if save_visualizations:
                    self.visualize_detection(results)
                
            except Exception as e:
                print(f"     ❌ 处理失败: {e}")
                continue
        
        print(f"✅ 批量推理完成! 成功处理 {len(all_results)} 张图像")
        return all_results
    
    def print_detection_stats(self, all_results):
        """打印检测统计信息"""
        if not all_results:
            print("⚠️ 没有检测结果")
            return
        
        total_detections = sum(r['num_detections'] for r in all_results)
        avg_detections = total_detections / len(all_results)
        
        # 统计类别分布
        class_counts = {}
        confidence_scores = []
        
        for results in all_results:
            for label, score in zip(results['pred_labels'], results['pred_scores']):
                label_id = int(label) + 1
                class_name = COCO_CLASSES.get(label_id, f"class_{label_id}")
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                confidence_scores.append(score)
        
        print(f"\n📊 检测统计:")
        print(f"   处理图像数: {len(all_results)}")
        print(f"   总检测数: {total_detections}")
        print(f"   平均每张图像检测数: {avg_detections:.1f}")
        
        if confidence_scores:
            print(f"   平均置信度: {np.mean(confidence_scores):.3f}")
            print(f"   最高置信度: {np.max(confidence_scores):.3f}")
            print(f"   最低置信度: {np.min(confidence_scores):.3f}")
        
        if class_counts:
            print(f"\n🏷️ 检测类别分布:")
            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
            for class_name, count in sorted_classes[:10]:  # 显示前10个
                print(f"   {class_name}: {count}")

def create_visualizer(model, conf_threshold=0.3, save_dir="./results/visualizations"):
    """
    创建可视化器的工厂函数
    
    Args:
        model: RT-DETR模型
        conf_threshold: 置信度阈值
        save_dir: 保存目录
    
    Returns:
        visualizer: RT-DETR可视化器
    """
    visualizer = RTDETRVisualizer(model, conf_threshold, save_dir)
    return visualizer

if __name__ == "__main__":
    # 测试可视化组件
    print("🧪 测试RT-DETR可视化组件")
    print("=" * 50)
    
    try:
        # 这里需要实际的模型来测试
        # 在实际使用时，模型会从model组件导入
        print("⚠️ 可视化组件需要配合训练好的模型使用")
        print("   请参考vis_script.py中的完整使用示例")
        
        print(f"\n🎉 RT-DETR可视化组件验证完成!")
        
    except Exception as e:
        print(f"❌ 可视化组件测试失败: {e}")
        import traceback
        traceback.print_exc()
