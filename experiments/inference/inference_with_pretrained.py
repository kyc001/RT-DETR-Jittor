#!/usr/bin/env python3
"""
使用预训练权重直接进行推理的RT-DETR脚本
"""

import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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

def safe_int64(tensor):
    """安全地将tensor转换为int64"""
    if isinstance(tensor, jt.Var):
        return tensor.int64()
    elif isinstance(tensor, np.ndarray):
        return jt.array(tensor.astype(np.int64))
    else:
        return jt.array(tensor, dtype=jt.int64)

class RTDETRInference:
    """RT-DETR推理类"""
    
    def __init__(self, pretrained_path, num_classes=80, confidence_threshold=0.3):
        self.pretrained_path = pretrained_path
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.idx_to_name = None
        
        self._build_model()
        self._load_categories()
    
    def _build_model(self):
        """创建模型并加载预训练权重"""
        print(f">>> 创建RT-DETR模型")

        # 创建模型
        self.model = build_rtdetr_complete(
            num_classes=self.num_classes,
            hidden_dim=256,
            num_queries=300
        )

        # 加载预训练权重
        print(f">>> 使用预训练权重: {self.pretrained_path}")

        if os.path.exists(self.pretrained_path):
            # 加载预训练权重
            weights = jt.load(self.pretrained_path)

            if 'ema' in weights and 'module' in weights['ema']:
                pretrained_dict = weights['ema']['module']
                print(f"✅ 从ema.module提取到 {len(pretrained_dict)} 个参数")

                # 获取模型参数
                model_dict = self.model.state_dict()

                # 统计匹配情况
                matched_keys = []
                missing_keys = []
                shape_mismatch_keys = []

                for key in model_dict.keys():
                    if key in pretrained_dict:
                        if model_dict[key].shape == pretrained_dict[key].shape:
                            matched_keys.append(key)
                            # 复制参数
                            model_dict[key] = pretrained_dict[key]
                        else:
                            shape_mismatch_keys.append(key)
                    else:
                        missing_keys.append(key)

                # 加载权重
                self.model.load_state_dict(model_dict)

                print(f"✅ 预训练权重加载完成")
                print(f"  匹配参数: {len(matched_keys)}/{len(model_dict)}")
                print(f"  缺失参数: {len(missing_keys)}")
                print(f"  形状不匹配: {len(shape_mismatch_keys)}")

                if len(matched_keys) < 10:
                    print("⚠️ 匹配参数太少，可能无法正常工作")
            else:
                raise ValueError("预训练权重结构不符合预期")

            # 设置模型为评估模式
            self.model.eval()
            print("✅ 模型创建成功")
        else:
            raise FileNotFoundError(f"预训练权重文件不存在: {self.pretrained_path}")
    
    def _load_categories(self):
        """加载COCO类别信息"""
        ann_file = "data/coco2017_50/annotations/instances_train2017.json"
        
        if os.path.exists(ann_file):
            with open(ann_file, 'r') as f:
                coco_data = json.load(f)
            
            self.idx_to_name = {idx: cat['name'] for idx, cat in enumerate(coco_data['categories'])}
            print(f"✅ 加载了 {len(self.idx_to_name)} 个类别")
        else:
            # 使用默认COCO类别
            self.idx_to_name = {i: f'class_{i}' for i in range(self.num_classes)}
            print("⚠️ 使用默认类别名称")
    
    def preprocess_image(self, image_path, target_size=640):
        """预处理图片"""
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Resize保持比例
        image_resized = image.resize((target_size, target_size))
        
        # 转换为numpy数组并归一化
        img_array = np.array(image_resized, dtype=np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
        
        # 转换为Jittor tensor并添加batch维度
        img_tensor = safe_float32(img_array).unsqueeze(0)
        
        return img_tensor, image, original_size
    
    def inference(self, img_tensor):
        """使用模型直接进行推理"""
        with jt.no_grad():
            # 直接使用模型的forward方法
            outputs = self.model(img_tensor)
            return outputs
    
    def postprocess_predictions(self, outputs, original_size=(640, 640)):
        """后处理预测结果"""
        pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
        pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4]
        
        # 计算置信度和标签
        pred_probs = jt.sigmoid(pred_logits)
        pred_scores = pred_probs.max(dim=-1)[0]  # 最大类别概率作为置信度
        pred_labels = pred_probs.argmax(dim=-1)  # 最大概率对应的类别
        
        # 过滤低置信度预测
        keep = pred_scores > self.confidence_threshold
        
        if keep.sum() == 0:
            return [], [], []
        
        filtered_scores = pred_scores[keep].numpy()
        filtered_labels = pred_labels[keep].numpy()
        filtered_boxes = pred_boxes[keep].numpy()
        
        # 转换边界框格式：从中心点格式(cx, cy, w, h)到左上右下格式(x1, y1, x2, y2)
        boxes_xyxy = []
        for box in filtered_boxes:
            cx, cy, w, h = box
            
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
        
        return filtered_scores, filtered_labels, boxes_xyxy
    
    def visualize_results(self, image, scores, labels, boxes, save_path=None):
        """可视化检测结果"""
        draw = ImageDraw.Draw(image)
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # 颜色列表
        colors = [
            'red', 'green', 'blue', 'yellow', 'magenta', 'cyan',
            'orange', 'purple', 'brown', 'pink', 'gray', 'olive'
        ]
        
        for i, (score, label, box) in enumerate(zip(scores, labels, boxes)):
            x1, y1, x2, y2 = box
            color = colors[i % len(colors)]
            
            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # 绘制标签和置信度
            class_name = self.idx_to_name.get(label, f'class_{label}')
            text = f'{class_name}: {score:.2f}'
            
            # 绘制文本背景
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            draw.rectangle([x1, y1-text_height-4, x1+text_width+8, y1], fill=color)
            draw.text((x1+4, y1-text_height-2), text, fill='white', font=font)
        
        if save_path:
            image.save(save_path)
            print(f"✅ 可视化结果保存到: {save_path}")
        
        return image
    
    def infer_image(self, image_path, save_result=True):
        """对单张图片进行推理"""
        print(f"\n>>> 推理图片: {os.path.basename(image_path)}")
        
        # 预处理
        img_tensor, original_image, original_size = self.preprocess_image(image_path)
        
        # 推理
        outputs = self.inference(img_tensor)
        
        # 后处理
        scores, labels, boxes = self.postprocess_predictions(
            outputs, original_size
        )
        
        # 打印结果
        print(f"检测到 {len(scores)} 个目标:")
        for i, (score, label, box) in enumerate(zip(scores, labels, boxes)):
            class_name = self.idx_to_name.get(label, f'class_{label}')
            x1, y1, x2, y2 = box
            print(f"  {i+1}. {class_name}: {score:.3f} at ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
        
        # 可视化并保存
        if save_result and len(scores) > 0:
            os.makedirs("results/pretrained_inference", exist_ok=True)
            save_path = f"results/pretrained_inference/result_{os.path.basename(image_path)}"
            self.visualize_results(original_image.copy(), scores, labels, boxes, save_path)
        
        return scores, labels, boxes
    
    def infer_batch(self, image_dir, max_images=5):
        """批量推理图片"""
        print(f"\n>>> 批量推理 (最多{max_images}张)")
        
        # 获取图片文件列表
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for file in os.listdir(image_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(image_dir, file))
        
        image_files = image_files[:max_images]
        
        if not image_files:
            print(f"❌ 在 {image_dir} 中没有找到图片文件")
            return
        
        # 批量推理
        total_detections = 0
        detection_stats = {}
        
        for image_path in image_files:
            try:
                scores, labels, boxes = self.infer_image(image_path)
                total_detections += len(scores)
                
                # 统计类别
                for label in labels:
                    class_name = self.idx_to_name.get(label, f'class_{label}')
                    detection_stats[class_name] = detection_stats.get(class_name, 0) + 1
                    
            except Exception as e:
                print(f"❌ 推理 {image_path} 失败: {e}")
        
        # 打印统计结果
        print(f"\n=== 批量推理统计 ===")
        print(f"处理图片数: {len(image_files)}")
        print(f"总检测数: {total_detections}")
        print(f"平均每张图片检测数: {total_detections/len(image_files):.1f}")
        
        if detection_stats:
            print(f"\n类别检测统计:")
            for class_name, count in sorted(detection_stats.items(), key=lambda x: x[1], reverse=True):
                print(f"  {class_name}: {count}")

def main():
    parser = argparse.ArgumentParser(description='使用预训练权重进行RT-DETR推理')
    parser.add_argument('--pretrained', type=str, default='rtdetr_r18vd_dec3_6x_coco_from_paddle.pth',
                       help='预训练权重文件路径')
    parser.add_argument('--image', type=str, default=None,
                       help='单张图片路径')
    parser.add_argument('--image_dir', type=str, default='data/coco2017_50/train2017',
                       help='图片目录路径')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='置信度阈值')
    parser.add_argument('--max_images', type=int, default=5,
                       help='批量推理最大图片数')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("===    使用预训练权重进行RT-DETR推理    ===")
    print("=" * 60)
    
    try:
        # 创建推理器
        inferencer = RTDETRInference(
            args.pretrained, 
            confidence_threshold=args.confidence
        )
        
        if args.image:
            # 单张图片推理
            inferencer.infer_image(args.image)
        else:
            # 批量推理
            inferencer.infer_batch(args.image_dir, args.max_images)
        
        print(f"\n🎯 推理完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
