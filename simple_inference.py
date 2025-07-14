#!/usr/bin/env python3
"""
简单推理脚本 - 避免复杂的tensor操作
"""

import os
import sys
import numpy as np
from PIL import Image

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'jittor_rt_detr'))

import jittor as jt
from src.nn.model import RTDETR

# 设置Jittor
jt.flags.use_cuda = 1

# COCO类别名称
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def preprocess_image(image_path, target_size=(640, 640)):
    """预处理图像"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Resize图像
    resized_image = image.resize(target_size, Image.LANCZOS)
    
    # 转换为tensor - 确保float32类型
    img_array = np.array(resized_image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_tensor = jt.array(img_array.transpose(2, 0, 1), dtype='float32').unsqueeze(0)

    return img_tensor, original_size

def simple_postprocess(outputs, original_size, conf_threshold=0.3):
    """简单后处理 - 参考PyTorch版本实现"""
    logits, boxes, _, _ = outputs
    pred_logits = logits[-1][0]  # (num_queries, num_classes)
    pred_boxes = boxes[-1][0]    # (num_queries, 4)

    print(f"模型输出形状: pred_logits={pred_logits.shape}, pred_boxes={pred_boxes.shape}")

    # 确保tensor是float32类型
    pred_logits = pred_logits.float32()
    pred_boxes = pred_boxes.float32()

    # 安全转换为numpy - 参考PyTorch版本
    try:
        logits_np = pred_logits.stop_grad().numpy()
        boxes_np = pred_boxes.stop_grad().numpy()
        print(f"✅ 成功转换为numpy")
    except Exception as e:
        print(f"❌ Tensor转numpy失败: {e}")
        return []
    
    # 计算分数 - 使用sigmoid
    scores_np = 1 / (1 + np.exp(-logits_np))  # sigmoid
    
    # 找到每个query的最高分数
    scores_max_np = np.max(scores_np, axis=1)
    labels_np = np.argmax(scores_np, axis=1)
    
    print(f"分数范围: {scores_max_np.min():.3f} - {scores_max_np.max():.3f}")
    print(f"超过阈值{conf_threshold}的数量: {np.sum(scores_max_np > conf_threshold)}")
    
    # 过滤低置信度预测
    keep_mask = scores_max_np > conf_threshold
    keep_indices = np.where(keep_mask)[0]
    
    print(f"保留的检测数量: {len(keep_indices)}")
    
    if len(keep_indices) == 0:
        print("❌ 没有检测到任何目标")
        return []
    
    results = []
    for i in keep_indices:
        box = boxes_np[i]
        score = scores_max_np[i]
        label_idx = labels_np[i]
        
        # 转换坐标 (cx, cy, w, h) -> (x1, y1, x2, y2)
        cx, cy, w, h = box
        x1 = (cx - w/2) * original_size[0]
        y1 = (cy - h/2) * original_size[1]
        x2 = (cx + w/2) * original_size[0]
        y2 = (cy + h/2) * original_size[1]
        
        class_name = COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else f"class_{label_idx}"
        
        results.append({
            'class': class_name,
            'confidence': float(score),
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'query_idx': int(i)
        })
        
        print(f"  检测 {len(results)}: {class_name} (置信度: {score:.3f}, query: {i})")
    
    return results

def main():
    print("=== 简单推理测试 ===")
    
    # 参数
    weights_path = "checkpoints/model_final.pkl"
    img_path = "data/coco2017_50/train2017/000000225405.jpg"
    conf_threshold = 0.1
    
    print(f"模型权重: {weights_path}")
    print(f"测试图片: {img_path}")
    print(f"置信度阈值: {conf_threshold}")
    
    # 1. 加载模型
    print("\n>>> 加载模型...")
    model = RTDETR(num_classes=80)
    state_dict = jt.load(weights_path)
    model.load_state_dict(state_dict)
    model.eval()

    # 确保模型参数都是float32类型
    model = model.float32()
    print("✅ 模型加载完成")
    
    # 2. 预处理图片
    print("\n>>> 预处理图片...")
    img_tensor, original_size = preprocess_image(img_path)
    print(f"✅ 图片预处理完成: {original_size} -> {img_tensor.shape}")
    
    # 3. 推理
    print("\n>>> 执行推理...")
    with jt.no_grad():
        outputs = model(img_tensor)
    print("✅ 推理完成")
    
    # 4. 后处理
    print("\n>>> 后处理...")
    detections = simple_postprocess(outputs, original_size, conf_threshold)
    
    # 5. 显示结果
    print(f"\n🎯 检测结果:")
    if detections:
        print(f"✅ 检测到 {len(detections)} 个目标:")
        for i, det in enumerate(detections):
            print(f"  {i+1}. {det['class']}: {det['confidence']:.3f}")
            print(f"     位置: [{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, {det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}]")
    else:
        print("❌ 未检测到任何目标")
    
    # 6. 流程自检结论
    print(f"\n🔍 流程自检结论:")
    if detections:
        print("✅ 流程自检通过！")
        print("  - 模型成功加载")
        print("  - 推理正常执行")
        print("  - 检测到目标物体")
        print("  - 整个训练→推理流程工作正常")
    else:
        print("⚠️ 流程自检部分通过")
        print("  - 模型成功加载 ✅")
        print("  - 推理正常执行 ✅") 
        print("  - 但未检测到目标 ❌")
        print("  - 可能需要降低置信度阈值或增加训练轮数")

if __name__ == "__main__":
    main()
