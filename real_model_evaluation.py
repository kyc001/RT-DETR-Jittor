#!/usr/bin/env python3
"""
真实模型推理和评估脚本
加载训练好的模型，在验证集上进行真实推理，计算实际的检测性能
"""

import os
import sys
import json
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

def load_jittor_model_real():
    """加载真实的Jittor模型"""
    try:
        import jittor as jt
        from jittor_rt_detr.src.nn.backbone.resnet import ResNet50
        from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
        import jittor.nn as nn
        
        jt.flags.use_cuda = 1
        
        class RTDETRModel(nn.Module):
            def __init__(self, num_classes=80):
                super().__init__()
                self.backbone = ResNet50(pretrained=False)
                self.transformer = RTDETRTransformer(
                    num_classes=num_classes,
                    hidden_dim=256,
                    num_queries=300,
                    feat_channels=[256, 512, 1024, 2048]
                )
            
            def execute(self, x, targets=None):
                features = self.backbone(x)
                return self.transformer(features, targets)
        
        model = RTDETRModel(num_classes=80)
        
        # 加载训练好的权重
        model_path = "/home/kyc/project/RT-DETR/results/jittor_finetune/rtdetr_jittor_finetune_50img_50epoch.pkl"
        if os.path.exists(model_path):
            model.load(model_path)
            print(f"✅ 成功加载Jittor模型: {model_path}")
            model.eval()
            return model, jt
        else:
            print(f"❌ 模型文件不存在: {model_path}")
            return None, None
        
    except Exception as e:
        print(f"❌ 加载Jittor模型失败: {e}")
        return None, None

def load_validation_images():
    """加载验证图像和标注"""
    val_img_dir = "/home/kyc/project/RT-DETR/data/coco2017_50/val2017"
    val_ann_file = "/home/kyc/project/RT-DETR/data/coco2017_50/annotations/instances_val2017.json"
    
    if not os.path.exists(val_img_dir) or not os.path.exists(val_ann_file):
        print(f"❌ 验证数据不存在")
        return None, None
    
    with open(val_ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # 创建图像ID到文件名的映射
    img_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # 创建图像ID到标注的映射
    img_id_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
    
    print(f"✅ 加载验证数据: {len(coco_data['images'])}张图片")
    return val_img_dir, img_id_to_filename, img_id_to_anns, coco_data

def preprocess_image(image_path):
    """预处理图像"""
    image = Image.open(image_path).convert('RGB')
    original_width, original_height = image.size
    
    # 调整图像大小到640x640
    image_resized = image.resize((640, 640), Image.LANCZOS)
    img_array = np.array(image_resized).astype(np.float32) / 255.0
    
    return img_array, original_width, original_height

def postprocess_predictions(outputs, conf_threshold=0.3):
    """后处理预测结果"""
    if 'pred_logits' not in outputs or 'pred_boxes' not in outputs:
        return np.array([]), np.array([]), np.array([])

    pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
    pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4]

    # 计算置信度并立即转换为numpy
    try:
        import jittor as jt

        # 先转换为numpy再处理，避免Jittor API问题
        pred_logits_np = pred_logits.numpy()
        pred_boxes_np = pred_boxes.numpy()

        # 使用numpy计算softmax和最大值
        from scipy.special import softmax
        scores = softmax(pred_logits_np, axis=-1)
        max_scores = np.max(scores[:, :-1], axis=-1)  # 排除背景类
        predicted_labels = np.argmax(scores[:, :-1], axis=-1)

        # 已经是numpy数组
        max_scores = max_scores
        predicted_labels = predicted_labels
        pred_boxes = pred_boxes_np

    except Exception as e:
        print(f"⚠️ Jittor处理失败，尝试numpy: {e}")
        # 如果是numpy数组
        try:
            from scipy.special import softmax
            if hasattr(pred_logits, 'numpy'):
                pred_logits = pred_logits.numpy()
            if hasattr(pred_boxes, 'numpy'):
                pred_boxes = pred_boxes.numpy()

            scores = softmax(pred_logits, axis=-1)
            max_scores = np.max(scores[:, :-1], axis=-1)
            predicted_labels = np.argmax(scores[:, :-1], axis=-1)
        except:
            return np.array([]), np.array([]), np.array([])

    # 过滤低置信度检测（现在都是numpy数组）
    valid_mask = max_scores > conf_threshold

    final_boxes = pred_boxes[valid_mask]
    final_scores = max_scores[valid_mask]
    final_labels = predicted_labels[valid_mask]

    return final_boxes, final_scores, final_labels

def evaluate_single_image(model, jt, image_path, ground_truth_anns, original_width, original_height):
    """评估单张图像"""
    # 预处理
    img_array, _, _ = preprocess_image(image_path)
    img_tensor = jt.array(img_array.transpose(2, 0, 1)).float32().unsqueeze(0)
    
    # 推理
    start_time = time.time()
    with jt.no_grad():
        outputs = model(img_tensor)
    inference_time = time.time() - start_time
    
    # 后处理
    pred_boxes, pred_scores, pred_labels = postprocess_predictions(outputs, conf_threshold=0.3)
    
    # 处理真实标注
    gt_boxes = []
    gt_labels = []
    
    for ann in ground_truth_anns:
        x, y, w, h = ann['bbox']
        category_id = ann['category_id']
        
        # 归一化坐标
        x1, y1 = x / original_width, y / original_height
        x2, y2 = (x + w) / original_width, (y + h) / original_height
        
        if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= 1 and y2 <= 1:
            gt_boxes.append([x1, y1, x2, y2])
            
            # COCO类别ID映射
            if category_id == 1:  # person
                mapped_label = 0
            elif category_id == 3:  # car
                mapped_label = 2
            elif category_id == 27:  # backpack
                mapped_label = 26
            elif category_id == 33:  # suitcase
                mapped_label = 32
            elif category_id == 84:  # book
                mapped_label = 83
            else:
                mapped_label = category_id - 1
            
            gt_labels.append(mapped_label)
    
    return {
        'pred_boxes': pred_boxes,
        'pred_scores': pred_scores,
        'pred_labels': pred_labels,
        'gt_boxes': np.array(gt_boxes) if gt_boxes else np.array([]).reshape(0, 4),
        'gt_labels': np.array(gt_labels) if gt_labels else np.array([]),
        'inference_time': inference_time,
        'num_predictions': len(pred_boxes),
        'num_ground_truths': len(gt_boxes)
    }

def calculate_iou(box1, box2):
    """计算IoU"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 计算交集
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

def calculate_precision_recall(all_results, iou_threshold=0.5):
    """计算精确率和召回率"""
    total_predictions = 0
    total_ground_truths = 0
    true_positives = 0
    
    for result in all_results:
        pred_boxes = result['pred_boxes']
        pred_labels = result['pred_labels']
        gt_boxes = result['gt_boxes']
        gt_labels = result['gt_labels']
        
        total_predictions += len(pred_boxes)
        total_ground_truths += len(gt_boxes)
        
        # 简化的匹配：对每个预测，找最佳匹配的GT
        for i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
            best_iou = 0
            best_match = False
            
            for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if pred_label == gt_label:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = iou >= iou_threshold
            
            if best_match:
                true_positives += 1
    
    precision = true_positives / total_predictions if total_predictions > 0 else 0
    recall = true_positives / total_ground_truths if total_ground_truths > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score, true_positives, total_predictions, total_ground_truths

def visualize_detection_result(image_path, result, save_path):
    """可视化检测结果"""
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    # 绘制预测框（红色）
    for box, score, label in zip(result['pred_boxes'], result['pred_scores'], result['pred_labels']):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height
        
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        draw.text((x1, y1-15), f'Pred: {int(label)} ({score:.2f})', fill='red')
    
    # 绘制真实框（绿色）
    for box, label in zip(result['gt_boxes'], result['gt_labels']):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height
        
        draw.rectangle([x1, y1, x2, y2], outline='green', width=2)
        draw.text((x1, y1+15), f'GT: {int(label)}', fill='green')
    
    image.save(save_path)
    print(f"📊 可视化结果保存到: {save_path}")

def main():
    print("🧪 真实模型推理评估")
    print("=" * 60)
    
    # 加载模型
    model, jt = load_jittor_model_real()
    if model is None:
        print("❌ 无法加载模型，退出评估")
        return
    
    # 加载验证数据
    val_img_dir, img_id_to_filename, img_id_to_anns, coco_data = load_validation_images()
    if val_img_dir is None:
        print("❌ 无法加载验证数据，退出评估")
        return
    
    # 评估所有图像
    all_results = []
    total_inference_time = 0
    
    print(f"\n🔍 开始真实推理评估...")
    
    # 只评估前10张图像以节省时间
    img_ids = list(img_id_to_filename.keys())[:10]
    
    for i, img_id in enumerate(img_ids):
        filename = img_id_to_filename[img_id]
        image_path = os.path.join(val_img_dir, filename)
        
        if not os.path.exists(image_path):
            continue
        
        # 获取真实标注
        ground_truth_anns = img_id_to_anns.get(img_id, [])
        
        # 获取原始图像尺寸
        image = Image.open(image_path)
        original_width, original_height = image.size
        
        print(f"   评估图像 {i+1}/{len(img_ids)}: {filename}")
        
        # 评估单张图像
        result = evaluate_single_image(model, jt, image_path, ground_truth_anns, original_width, original_height)
        all_results.append(result)
        total_inference_time += result['inference_time']
        
        print(f"     预测数: {result['num_predictions']}, GT数: {result['num_ground_truths']}, 推理时间: {result['inference_time']:.3f}s")
        
        # 可视化第一张图像的结果
        if i == 0:
            save_dir = "/home/kyc/project/RT-DETR/results/real_evaluation"
            os.makedirs(save_dir, exist_ok=True)
            vis_path = os.path.join(save_dir, f"detection_result_{filename}")
            visualize_detection_result(image_path, result, vis_path)
    
    # 计算总体性能
    print(f"\n📊 真实评估结果:")
    print("-" * 40)
    
    precision, recall, f1_score, tp, total_pred, total_gt = calculate_precision_recall(all_results)
    avg_inference_time = total_inference_time / len(all_results)
    fps = 1.0 / avg_inference_time
    
    print(f"总预测数: {total_pred}")
    print(f"总真实目标数: {total_gt}")
    print(f"正确检测数: {tp}")
    print(f"精确率: {precision:.3f}")
    print(f"召回率: {recall:.3f}")
    print(f"F1分数: {f1_score:.3f}")
    print(f"平均推理时间: {avg_inference_time:.3f}秒")
    print(f"FPS: {fps:.1f}")
    
    # 保存详细结果
    save_dir = "/home/kyc/project/RT-DETR/results/real_evaluation"
    os.makedirs(save_dir, exist_ok=True)
    
    evaluation_results = {
        'total_predictions': total_pred,
        'total_ground_truths': total_gt,
        'true_positives': tp,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_inference_time': avg_inference_time,
        'fps': fps,
        'detailed_results': all_results
    }
    
    with open(os.path.join(save_dir, "real_evaluation_results.json"), 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    print(f"\n💾 详细结果保存到: {save_dir}/real_evaluation_results.json")
    
    # 判断模型是否失败
    if precision < 0.1 or recall < 0.1 or f1_score < 0.1:
        print(f"\n❌ 模型性能极差！")
        print(f"   精确率 {precision:.3f} < 0.1 或 召回率 {recall:.3f} < 0.1")
        print(f"   需要检查代码并重新训练！")
        return False
    else:
        print(f"\n✅ 模型基本可用，但可能需要改进")
        return True

if __name__ == "__main__":
    main()
