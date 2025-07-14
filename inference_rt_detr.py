#!/usr/bin/env python3
"""
RT-DETR 推理测试脚本
用于测试训练好的模型
"""

import os
import sys
import json
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'jittor_rt_detr'))

import jittor as jt
from src.nn.model import RTDETR

# 设置Jittor
jt.flags.use_cuda = 1

class RTDETRInference:
    """RT-DETR推理器"""
    
    def __init__(self, model_path, num_classes=80):
        self.model_path = model_path
        self.num_classes = num_classes
        self.load_model()
        self.load_categories()
    
    def load_model(self):
        """加载模型"""
        self.model = RTDETR(num_classes=self.num_classes)
        self.model = self.model.float32()
        
        if os.path.exists(self.model_path):
            state_dict = jt.load(self.model_path)
            self.model.load_state_dict(state_dict)
            print(f"✅ 模型加载成功: {self.model_path}")
        else:
            print(f"❌ 模型文件不存在: {self.model_path}")
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        self.model.eval()
    
    def load_categories(self):
        """加载COCO类别信息"""
        ann_file = "data/coco2017_50/annotations/instances_train2017.json"
        
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # 创建类别映射
        self.coco_categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        coco_ids = sorted(self.coco_categories.keys())
        self.model_idx_to_coco_id = {i: coco_id for i, coco_id in enumerate(coco_ids)}
        
        print(f"✅ 类别信息加载完成 ({len(self.coco_categories)} 个类别)")
    
    def preprocess_image(self, image_path):
        """预处理图片"""
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # 调整大小
        resized_image = image.resize((640, 640), Image.LANCZOS)
        img_array = np.array(resized_image, dtype=np.float32) / 255.0
        
        # 归一化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        img_tensor = jt.array(img_array.transpose(2, 0, 1), dtype='float32').unsqueeze(0)
        
        return img_tensor, image, original_size
    
    def postprocess_outputs(self, outputs, original_size, confidence_threshold=0.1):
        """后处理模型输出"""
        logits, boxes, _, _ = outputs
        
        # 使用最后一层的输出
        pred_logits = logits[-1][0]  # [num_queries, num_classes]
        pred_boxes = boxes[-1][0]    # [num_queries, 4]
        
        # 转换为numpy
        logits_np = pred_logits.stop_grad().numpy()
        boxes_np = pred_boxes.stop_grad().numpy()
        
        # 计算置信度
        exp_logits = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
        scores = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        max_scores = np.max(scores, axis=1)
        max_classes = np.argmax(scores, axis=1)
        
        # 筛选高置信度的检测
        valid_mask = max_scores > confidence_threshold
        valid_indices = np.where(valid_mask)[0]
        
        detections = []
        for idx in valid_indices:
            box = boxes_np[idx]
            score = max_scores[idx]
            class_idx = max_classes[idx]
            
            # 转换类别
            if class_idx < len(self.model_idx_to_coco_id):
                coco_id = self.model_idx_to_coco_id[class_idx]
                class_name = self.coco_categories[coco_id]
            else:
                continue
            
            # 转换坐标
            cx, cy, w, h = box
            x1 = (cx - w/2) * original_size[0]
            y1 = (cy - h/2) * original_size[1]
            x2 = (cx + w/2) * original_size[0]
            y2 = (cy + h/2) * original_size[1]
            
            # 边界检查
            x1 = max(0, min(x1, original_size[0]))
            y1 = max(0, min(y1, original_size[1]))
            x2 = max(0, min(x2, original_size[0]))
            y2 = max(0, min(y2, original_size[1]))
            
            if x2 > x1 and y2 > y1:
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(score),
                    'class_name': class_name,
                    'area': (x2-x1) * (y2-y1)
                })
        
        # 按置信度排序
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections
    
    def run_inference(self, image_path, confidence_threshold=0.1):
        """运行推理"""
        print(f"🔍 推理图片: {image_path}")
        
        # 预处理
        img_tensor, image, original_size = self.preprocess_image(image_path)
        
        # 推理
        with jt.no_grad():
            outputs = self.model(img_tensor)
        
        # 后处理
        detections = self.postprocess_outputs(outputs, original_size, confidence_threshold)
        
        print(f"📊 检测结果: {len(detections)} 个目标")
        for i, det in enumerate(detections[:10]):  # 只显示前10个
            print(f"  {i+1}. {det['class_name']}: {det['confidence']:.3f}")
        
        return image, detections
    
    def visualize_results(self, image, detections, save_path=None):
        """可视化结果"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        ax.imshow(image)
        ax.set_title(f"RT-DETR检测结果 ({len(detections)} 个目标)", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # 绘制检测框
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            color = colors[i % len(colors)]
            
            # 绘制边界框
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # 添加标签
            ax.text(x1, y1-5, f"{det['class_name']} ({det['confidence']:.2f})", 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                   fontsize=10, color='white', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ 结果已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()

def get_random_test_image():
    """获取随机测试图片"""
    data_dir = "data/coco2017_50/train2017"
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    selected_image = random.choice(image_files)
    return os.path.join(data_dir, selected_image)

def main():
    parser = argparse.ArgumentParser(description='RT-DETR推理脚本')
    parser.add_argument('--model_path', default='checkpoints/balanced_rt_detr_best_model.pkl', help='模型路径')
    parser.add_argument('--image_path', default=None, help='图片路径 (默认随机选择)')
    parser.add_argument('--confidence_threshold', type=float, default=0.1, help='置信度阈值')
    parser.add_argument('--save_path', default='inference_result.jpg', help='结果保存路径')
    parser.add_argument('--num_classes', type=int, default=80, help='类别数量')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("===      RT-DETR 推理测试      ===")
    print("=" * 60)
    
    # 选择测试图片
    if args.image_path is None:
        args.image_path = get_random_test_image()
        print(f"🎲 随机选择图片: {args.image_path}")
    
    # 创建推理器
    inference = RTDETRInference(args.model_path, args.num_classes)
    
    # 运行推理
    image, detections = inference.run_inference(args.image_path, args.confidence_threshold)
    
    # 可视化结果
    inference.visualize_results(image, detections, args.save_path)
    
    print("=" * 60)
    print("🎉 推理完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
