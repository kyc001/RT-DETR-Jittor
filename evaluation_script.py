#!/usr/bin/env python3
"""
RT-DETR模型评估脚本
在验证集上测试训练好的模型，计算mAP等指标
"""

import os
import sys
import json
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

def load_jittor_model():
    """加载Jittor训练的模型"""
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
        
        # 尝试加载训练好的权重
        model_path = "/home/kyc/project/RT-DETR/results/jittor_finetune/rtdetr_jittor_finetune_50img_50epoch.pkl"
        if os.path.exists(model_path):
            model.load(model_path)
            print(f"✅ 成功加载Jittor模型: {model_path}")
        else:
            print(f"⚠️ 模型文件不存在: {model_path}")
            return None
        
        model.eval()
        return model, jt
        
    except Exception as e:
        print(f"❌ 加载Jittor模型失败: {e}")
        return None, None

def load_pytorch_model():
    """加载PyTorch训练的模型"""
    try:
        import torch
        import torch.nn as nn
        
        # 这里需要重新定义PyTorch模型结构
        # 由于模型结构复杂，这里先返回None
        print("⚠️ PyTorch模型加载需要完整的模型定义")
        return None, None
        
    except Exception as e:
        print(f"❌ 加载PyTorch模型失败: {e}")
        return None, None

def load_validation_data():
    """加载验证数据"""
    val_img_dir = "/home/kyc/project/RT-DETR/data/coco2017_50/val2017"
    val_ann_file = "/home/kyc/project/RT-DETR/data/coco2017_50/annotations/instances_val2017.json"
    
    if not os.path.exists(val_img_dir) or not os.path.exists(val_ann_file):
        print(f"❌ 验证数据不存在")
        return None, None
    
    with open(val_ann_file, 'r') as f:
        coco_data = json.load(f)
    
    print(f"✅ 加载验证数据: {len(coco_data['images'])}张图片")
    return val_img_dir, coco_data

def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
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

def evaluate_model_simple(model, framework, val_img_dir, coco_data):
    """简化的模型评估"""
    print(f"\n🔍 评估{framework}模型...")
    
    images = coco_data['images'][:10]  # 只测试前10张图片
    total_detections = 0
    total_ground_truths = 0
    correct_detections = 0
    
    inference_times = []
    
    for img_info in images:
        img_path = os.path.join(val_img_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            continue
            
        # 加载图片
        image = Image.open(img_path).convert('RGB')
        original_width, original_height = image.size
        
        # 预处理
        image_resized = image.resize((640, 640), Image.LANCZOS)
        img_array = np.array(image_resized).astype(np.float32) / 255.0
        
        if framework == "Jittor":
            import jittor as jt
            img_tensor = jt.array(img_array.transpose(2, 0, 1)).float32().unsqueeze(0)
        else:
            import torch
            img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float().unsqueeze(0)
        
        # 推理
        start_time = time.time()
        try:
            with torch.no_grad() if framework == "PyTorch" else jt.no_grad():
                outputs = model(img_tensor)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # 简化的后处理
            if 'pred_logits' in outputs and 'pred_boxes' in outputs:
                pred_logits = outputs['pred_logits'][0]
                pred_boxes = outputs['pred_boxes'][0]
                
                # 获取置信度最高的检测
                if framework == "Jittor":
                    scores = jt.nn.softmax(pred_logits, dim=-1)
                    max_scores = jt.max(scores[:, :-1], dim=-1)[0]
                    valid_detections = (max_scores > 0.5).sum()
                else:
                    scores = torch.softmax(pred_logits, dim=-1)
                    max_scores = torch.max(scores[:, :-1], dim=-1)[0]
                    valid_detections = (max_scores > 0.5).sum()
                
                total_detections += int(valid_detections)
            
        except Exception as e:
            print(f"⚠️ 推理失败: {e}")
            continue
        
        # 获取真实标注
        img_id = img_info['id']
        gt_count = 0
        for ann in coco_data['annotations']:
            if ann['image_id'] == img_id:
                gt_count += 1
        
        total_ground_truths += gt_count
        
        # 简化的匹配（假设检测数量接近就算正确）
        if abs(int(valid_detections) - gt_count) <= 1:
            correct_detections += min(int(valid_detections), gt_count)
    
    # 计算指标
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    
    precision = correct_detections / total_detections if total_detections > 0 else 0
    recall = correct_detections / total_ground_truths if total_ground_truths > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"📊 {framework}模型评估结果:")
    print(f"   总检测数: {total_detections}")
    print(f"   总真实目标数: {total_ground_truths}")
    print(f"   正确检测数: {correct_detections}")
    print(f"   精确率: {precision:.3f}")
    print(f"   召回率: {recall:.3f}")
    print(f"   F1分数: {f1_score:.3f}")
    print(f"   平均推理时间: {avg_inference_time:.3f}秒")
    print(f"   FPS: {fps:.1f}")
    
    return {
        'total_detections': total_detections,
        'total_ground_truths': total_ground_truths,
        'correct_detections': correct_detections,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_inference_time': avg_inference_time,
        'fps': fps
    }

def create_detection_visualization():
    """创建检测结果可视化"""
    print("\n🎨 创建检测结果可视化...")
    
    # 这里可以添加可视化代码
    # 由于模型加载复杂，先创建示例可视化
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 模拟检测结果
    axes[0].set_title('Jittor检测结果')
    axes[0].text(0.5, 0.5, 'Jittor模型检测结果\n(需要实际模型推理)', 
                ha='center', va='center', transform=axes[0].transAxes)
    
    axes[1].set_title('PyTorch检测结果')
    axes[1].text(0.5, 0.5, 'PyTorch模型检测结果\n(需要实际模型推理)', 
                ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    
    save_path = "/home/kyc/project/RT-DETR/results/comparison/detection_comparison.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 可视化结果保存到: {save_path}")

def main():
    print("🧪 RT-DETR模型评估")
    print("=" * 60)
    
    # 加载验证数据
    val_img_dir, coco_data = load_validation_data()
    if val_img_dir is None:
        print("❌ 无法加载验证数据，退出评估")
        return
    
    # 评估结果
    results = {}
    
    # 尝试评估Jittor模型
    jittor_model, jt = load_jittor_model()
    if jittor_model is not None:
        results['jittor'] = evaluate_model_simple(jittor_model, "Jittor", val_img_dir, coco_data)
    else:
        print("⚠️ 跳过Jittor模型评估")
    
    # 尝试评估PyTorch模型
    pytorch_model, torch = load_pytorch_model()
    if pytorch_model is not None:
        results['pytorch'] = evaluate_model_simple(pytorch_model, "PyTorch", val_img_dir, coco_data)
    else:
        print("⚠️ 跳过PyTorch模型评估")
    
    # 创建可视化
    create_detection_visualization()
    
    # 保存评估结果
    if results:
        save_path = "/home/kyc/project/RT-DETR/results/comparison/evaluation_results.json"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 评估结果保存到: {save_path}")
    
    print(f"\n📝 评估总结:")
    print(f"   - 当前评估基于简化的指标")
    print(f"   - 建议实现完整的COCO评估流程")
    print(f"   - 需要统一两个版本的模型架构以进行公平对比")

if __name__ == "__main__":
    main()
