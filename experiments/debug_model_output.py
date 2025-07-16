#!/usr/bin/env python3
"""
调试模型输出，查看实际预测的类别分布
"""

import os
import sys
import numpy as np

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt

# 设置Jittor
jt.flags.use_cuda = 1

def load_trained_model():
    """加载训练好的模型"""
    print("🔄 加载训练好的模型...")
    
    try:
        # 导入模型组件
        from jittor_rt_detr.src.nn.backbone.resnet import ResNet50
        from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
        
        # 创建模型架构
        backbone = ResNet50(pretrained=False)
        transformer = RTDETRTransformer(
            num_classes=80,
            hidden_dim=256,
            num_queries=300,
            feat_channels=[256, 512, 1024, 2048]
        )
        
        # 创建完整模型
        class RTDETRModel(jt.nn.Module):
            def __init__(self, backbone, transformer):
                super().__init__()
                self.backbone = backbone
                self.transformer = transformer
                
            def execute(self, x):
                features = self.backbone(x)
                outputs = self.transformer(features)
                return outputs
        
        model = RTDETRModel(backbone, transformer)
        
        # 加载训练好的权重
        model_path = '/home/kyc/project/RT-DETR/results/full_training/rtdetr_trained.pkl'
        checkpoint = jt.load(model_path)
        model.backbone.load_state_dict(checkpoint['backbone_state_dict'])
        model.transformer.load_state_dict(checkpoint['transformer_state_dict'])
        
        # 设置为评估模式
        model.eval()
        
        print(f"✅ 模型加载成功")
        return model
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

def load_real_image():
    """加载真实的训练集图像"""
    import json
    from PIL import Image
    import numpy as np

    # 数据路径
    data_dir = '/home/kyc/project/RT-DETR/data/coco2017_50'
    images_dir = os.path.join(data_dir, "train2017")
    annotations_file = os.path.join(data_dir, "annotations", "instances_train2017.json")

    # 加载标注
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)

    # 获取第一张图像
    first_image = coco_data['images'][0]
    image_path = os.path.join(images_dir, first_image['file_name'])

    print(f"📷 加载图像: {first_image['file_name']}")

    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    print(f"   原始尺寸: {original_size}")

    # 调整图像大小到640x640
    image_resized = image.resize((640, 640))
    image_array = np.array(image_resized).astype(np.float32) / 255.0
    image_tensor = jt.array(image_array.transpose(2, 0, 1)).unsqueeze(0)

    return image_tensor, first_image

def analyze_model_predictions():
    """分析模型预测的类别分布"""
    print("🔍 分析模型预测...")

    # 加载模型
    model = load_trained_model()
    if model is None:
        return

    # 加载真实图像
    real_input, image_info = load_real_image()
    
    with jt.no_grad():
        # 前向传播
        print(f"📊 使用真实图像: {image_info['file_name']}")
        features = model.backbone(real_input)
        outputs = model.transformer(features)
        
        # 获取预测结果
        pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
        pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4]
        
        print(f"📊 模型输出分析:")
        print(f"   pred_logits形状: {pred_logits.shape}")
        print(f"   pred_boxes形状: {pred_boxes.shape}")
        
        # 后处理
        pred_scores = jt.nn.softmax(pred_logits, dim=-1)
        pred_scores_no_bg = pred_scores[:, :-1]  # 排除背景类
        
        # 获取最高分数的类别
        max_result = jt.max(pred_scores_no_bg, dim=-1)
        if isinstance(max_result, tuple):
            max_scores = max_result[0]
        else:
            max_scores = max_result

        argmax_result = jt.argmax(pred_scores_no_bg, dim=-1)
        if isinstance(argmax_result, tuple):
            pred_classes = argmax_result[0]
        else:
            pred_classes = argmax_result
        
        # 转换为numpy
        scores_np = max_scores.numpy()
        classes_np = pred_classes.numpy()
        
        print(f"\n🎯 预测分析:")
        print(f"   分数范围: {scores_np.min():.4f} - {scores_np.max():.4f}")
        print(f"   类别索引范围: {classes_np.min()} - {classes_np.max()}")
        
        # 统计类别分布
        unique_classes, counts = np.unique(classes_np, return_counts=True)
        print(f"\n📈 类别分布 (前10个):")
        sorted_indices = np.argsort(counts)[::-1][:10]
        for i, idx in enumerate(sorted_indices):
            class_idx = unique_classes[idx]
            count = counts[idx]
            print(f"   {i+1}: 类别{class_idx} -> {count}次")
        
        # 显示最高分数的预测
        top_indices = np.argsort(scores_np)[::-1][:10]
        print(f"\n🏆 前10个最高分数预测:")
        for i, idx in enumerate(top_indices):
            class_idx = classes_np[idx]
            score = scores_np[idx]
            
            # 转换为COCO类别
            if class_idx == 0:
                coco_id = 1  # person
                class_name = "person"
            elif class_idx == 2:
                coco_id = 3  # car
                class_name = "car"
            elif class_idx == 26:
                coco_id = 27  # backpack
                class_name = "backpack"
            elif class_idx == 32:
                coco_id = 33  # suitcase
                class_name = "suitcase"
            elif class_idx == 83:
                coco_id = 84  # book
                class_name = "book"
            else:
                coco_id = class_idx + 1
                # COCO类别映射
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
                class_name = COCO_CLASSES.get(coco_id, f'class_{coco_id}')
            
            print(f"   {i+1}: 类别{class_idx} -> COCO{coco_id}({class_name}), 分数{score:.4f}")

def main():
    print("🔍 RT-DETR模型输出调试")
    print("=" * 60)
    
    analyze_model_predictions()

if __name__ == "__main__":
    main()
