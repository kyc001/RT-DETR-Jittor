#!/usr/bin/env python3
"""
推理结果与真实标注对比可视化脚本
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

class ComparisonVisualizer:
    """推理结果与真实标注对比可视化器"""
    
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
            self.coco_data = json.load(f)
        
        # 创建类别映射
        self.coco_categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        coco_ids = sorted(self.coco_categories.keys())
        self.model_idx_to_coco_id = {i: coco_id for i, coco_id in enumerate(coco_ids)}
        
        print(f"✅ 类别信息加载完成 ({len(self.coco_categories)} 个类别)")
    
    def get_random_image_with_annotations(self):
        """获取随机图片及其标注"""
        # 收集有标注的图片
        images_dict = {img['id']: img for img in self.coco_data['images']}
        
        # 按图片分组标注
        annotations_by_image = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # 筛选有标注的图片
        valid_images = []
        for img_id, anns in annotations_by_image.items():
            if len(anns) > 0 and len(anns) <= 8 and img_id in images_dict:  # 限制标注数量
                valid_images.append(img_id)
        
        # 随机选择一张图片
        selected_img_id = random.choice(valid_images)
        img_info = images_dict[selected_img_id]
        annotations = annotations_by_image[selected_img_id]
        
        print(f"🎲 随机选择图片: {img_info['file_name']} (标注数: {len(annotations)})")
        
        return img_info, annotations
    
    def load_ground_truth(self, img_info, annotations):
        """加载真实标注"""
        gt_data = []
        
        print(f"\n📋 真实标注:")
        for i, ann in enumerate(annotations):
            x, y, w, h = ann['bbox']
            category_name = self.coco_categories[ann['category_id']]
            area = w * h
            
            gt_data.append({
                'bbox_coco': [x, y, w, h],
                'bbox_xyxy': [x, y, x+w, y+h],
                'category_name': category_name,
                'category_id': ann['category_id'],
                'area': area
            })
            
            print(f"  {i+1}. {category_name}: [{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}] (面积: {area:.1f})")
        
        return gt_data
    
    def run_inference(self, img_path, confidence_threshold=0.05):
        """运行推理"""
        print(f"\n🔍 推理图片: {img_path}")
        
        # 预处理
        image = Image.open(img_path).convert('RGB')
        original_size = image.size
        
        resized_image = image.resize((640, 640), Image.LANCZOS)
        img_array = np.array(resized_image, dtype=np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        img_tensor = jt.array(img_array.transpose(2, 0, 1), dtype='float32').unsqueeze(0)
        
        # 推理
        with jt.no_grad():
            outputs = self.model(img_tensor)
        
        logits, boxes, _, _ = outputs
        pred_logits = logits[-1][0]
        pred_boxes = boxes[-1][0]
        
        # 转换为numpy
        logits_np = pred_logits.stop_grad().numpy()
        boxes_np = pred_boxes.stop_grad().numpy()
        
        # 计算置信度
        exp_logits = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
        scores = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        max_scores = np.max(scores, axis=1)
        max_classes = np.argmax(scores, axis=1)
        
        print(f"📊 模型输出分析:")
        print(f"  - 最高置信度: {np.max(max_scores):.3f}")
        print(f"  - 平均置信度: {np.mean(max_scores):.3f}")
        print(f"  - >0.05置信度的数量: {np.sum(max_scores > 0.05)}")
        
        # 筛选检测结果
        valid_mask = max_scores > confidence_threshold
        valid_indices = np.where(valid_mask)[0]
        
        detections = []
        for idx in valid_indices:
            box = boxes_np[idx]
            score = max_scores[idx]
            class_idx = max_classes[idx]
            
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
        
        # 按置信度排序并限制数量
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        detections = detections[:15]  # 只保留前15个
        
        print(f"📊 检测结果: {len(detections)} 个目标")
        for i, det in enumerate(detections):
            print(f"  {i+1}. {det['class_name']}: {det['confidence']:.3f}")
        
        return image, detections
    
    def create_comparison_visualization(self, image, gt_data, detections, img_name, save_path):
        """创建对比可视化"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # 左图：真实标注
        ax1.imshow(image)
        ax1.set_title(f"Ground Truth\n{img_name}\n{len(gt_data)} objects", 
                     fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        # 绘制真实标注
        colors_gt = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        for i, gt in enumerate(gt_data):
            x1, y1, x2, y2 = gt['bbox_xyxy']
            color = colors_gt[i % len(colors_gt)]
            
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=3, edgecolor=color, facecolor='none')
            ax1.add_patch(rect)
            
            ax1.text(x1, y1-5, f"GT{i+1}: {gt['category_name']}", 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                    fontsize=12, color='white', fontweight='bold')
        
        # 右图：检测结果
        ax2.imshow(image)
        ax2.set_title(f"Model Predictions\n{len(detections)} detections", 
                     fontsize=16, fontweight='bold')
        ax2.axis('off')
        
        # 绘制检测结果
        colors_pred = ['cyan', 'magenta', 'yellow', 'lime', 'orange', 'pink', 'lightblue', 'lightgreen']
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            color = colors_pred[i % len(colors_pred)]
            
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor=color, facecolor='none')
            ax2.add_patch(rect)
            
            ax2.text(x1, y1-5, f"P{i+1}: {det['class_name']} ({det['confidence']:.2f})", 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                    fontsize=10, color='black', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 对比可视化已保存: {save_path}")
        return save_path
    
    def analyze_results(self, gt_data, detections):
        """分析检测结果"""
        print(f"\n📊 检测结果分析:")
        
        # 统计真实标注
        gt_counts = {}
        for gt in gt_data:
            class_name = gt['category_name']
            gt_counts[class_name] = gt_counts.get(class_name, 0) + 1
        
        # 统计检测结果
        pred_counts = {}
        for det in detections:
            class_name = det['class_name']
            pred_counts[class_name] = pred_counts.get(class_name, 0) + 1
        
        print(f"类别检测对比:")
        all_classes = set(list(gt_counts.keys()) + list(pred_counts.keys()))
        
        correct_classes = 0
        total_gt_classes = len(gt_counts)
        
        for class_name in sorted(all_classes):
            gt_count = gt_counts.get(class_name, 0)
            pred_count = pred_counts.get(class_name, 0)
            
            if gt_count > 0 and pred_count > 0:
                status = "✅ 检测到"
                correct_classes += 1
            elif gt_count > 0:
                status = "❌ 缺失"
            else:
                status = "⚠️ 误检"
                
            print(f"  {class_name}: 真实={gt_count}, 预测={pred_count} {status}")
        
        # 计算召回率
        if total_gt_classes > 0:
            recall = correct_classes / total_gt_classes
            print(f"\n📈 类别召回率: {recall:.2%} ({correct_classes}/{total_gt_classes})")
        
        return gt_counts, pred_counts
    
    def run_comparison(self, confidence_threshold=0.05):
        """运行完整的对比分析"""
        print("=" * 60)
        print("===      推理结果与真实标注对比      ===")
        print("=" * 60)
        
        # 1. 获取随机图片和标注
        img_info, annotations = self.get_random_image_with_annotations()
        
        # 2. 加载真实标注
        gt_data = self.load_ground_truth(img_info, annotations)
        
        # 3. 运行推理
        img_path = f"data/coco2017_50/train2017/{img_info['file_name']}"
        image, detections = self.run_inference(img_path, confidence_threshold)
        
        # 4. 分析结果
        gt_counts, pred_counts = self.analyze_results(gt_data, detections)
        
        # 5. 创建对比可视化
        save_path = f"comparison_{img_info['file_name'].replace('.jpg', '')}.jpg"
        self.create_comparison_visualization(image, gt_data, detections, 
                                           img_info['file_name'], save_path)
        
        print("=" * 60)
        print("🎯 对比分析完成！")
        print("=" * 60)
        
        return save_path

def main():
    parser = argparse.ArgumentParser(description='推理结果与真实标注对比')
    parser.add_argument('--model_path', default='checkpoints/balanced_rt_detr_best_model.pkl', help='模型路径')
    parser.add_argument('--confidence_threshold', type=float, default=0.05, help='置信度阈值')
    parser.add_argument('--num_classes', type=int, default=80, help='类别数量')
    
    args = parser.parse_args()
    
    # 创建对比可视化器
    visualizer = ComparisonVisualizer(args.model_path, args.num_classes)
    
    # 运行对比分析
    result_path = visualizer.run_comparison(args.confidence_threshold)
    
    print(f"对比结果已保存: {result_path}")

if __name__ == "__main__":
    main()
