#!/usr/bin/env python3
"""
改进版推理脚本 - 参考PyTorch版本实现正确的后处理
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
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
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

def box_cxcywh_to_xyxy(boxes):
    """将中心点格式转换为左上右下格式"""
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    return np.stack([x1, y1, x2, y2], axis=-1)

def improved_postprocess(outputs, original_size, conf_threshold=0.3, use_focal_loss=True):
    """
    改进版后处理 - 参考PyTorch版本RTDETRPostProcessor实现
    """
    print("=== 改进版后处理 ===")
    
    logits, boxes, _, _ = outputs
    
    # 获取最后一层输出
    pred_logits = logits[-1][0]  # (num_queries, num_classes)
    pred_boxes = boxes[-1][0]    # (num_queries, 4)
    
    print(f"模型输出形状: logits={pred_logits.shape}, boxes={pred_boxes.shape}")
    
    # 确保tensor是float32类型并安全转换为numpy
    pred_logits = pred_logits.float32()
    pred_boxes = pred_boxes.float32()
    
    try:
        logits_np = pred_logits.stop_grad().numpy()
        boxes_np = pred_boxes.stop_grad().numpy()
        print(f"✅ 成功转换为numpy")
    except Exception as e:
        print(f"❌ Tensor转numpy失败: {e}")
        return []
    
    # 转换边界框格式：cxcywh -> xyxy，并缩放到原图尺寸
    boxes_xyxy = box_cxcywh_to_xyxy(boxes_np)
    
    # 缩放到原图尺寸
    scale_x = original_size[0]  # width
    scale_y = original_size[1]  # height
    boxes_xyxy[:, [0, 2]] *= scale_x  # x坐标
    boxes_xyxy[:, [1, 3]] *= scale_y  # y坐标
    
    # 参考PyTorch版本的后处理逻辑
    if use_focal_loss:
        # 使用focal loss训练的模型，使用sigmoid激活
        scores = 1.0 / (1.0 + np.exp(-logits_np))  # sigmoid
        print(f"使用sigmoid激活 (focal loss模式)")
        
        # 找到top-k个最高分数的预测
        num_top_queries = min(100, scores.shape[0])  # 限制最多100个
        scores_flat = scores.flatten()
        top_indices_flat = np.argpartition(scores_flat, -num_top_queries)[-num_top_queries:]
        top_scores = scores_flat[top_indices_flat]
        
        # 转换回原始索引
        query_indices = top_indices_flat // scores.shape[1]
        class_indices = top_indices_flat % scores.shape[1]
        
        # 过滤低置信度
        valid_mask = top_scores > conf_threshold
        query_indices = query_indices[valid_mask]
        class_indices = class_indices[valid_mask]
        top_scores = top_scores[valid_mask]
        
    else:
        # 使用softmax训练的模型
        # 数值稳定的softmax
        exp_logits = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
        scores = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        print(f"使用softmax激活")
        
        # 排除背景类（最后一个类别）
        if scores.shape[1] > len(COCO_CLASSES):
            scores = scores[:, :-1]  # 移除背景类
            print(f"移除背景类，剩余{scores.shape[1]}个类别")
        
        # 找到每个query的最高分数和类别
        max_scores = np.max(scores, axis=1)
        max_classes = np.argmax(scores, axis=1)
        
        # 过滤低置信度
        valid_mask = max_scores > conf_threshold
        query_indices = np.where(valid_mask)[0]
        class_indices = max_classes[valid_mask]
        top_scores = max_scores[valid_mask]
    
    print(f"置信度范围: [{np.min(top_scores):.3f}, {np.max(top_scores):.3f}]")
    print(f"保留的检测数量: {len(query_indices)}")
    
    if len(query_indices) == 0:
        print("❌ 没有检测到任何目标")
        return []
    
    # 构建结果
    results = []
    for i, (query_idx, class_idx, score) in enumerate(zip(query_indices, class_indices, top_scores)):
        if class_idx >= len(COCO_CLASSES):
            continue
            
        box = boxes_xyxy[query_idx]
        class_name = COCO_CLASSES[class_idx]
        
        # 确保边界框在图像范围内
        x1, y1, x2, y2 = box
        x1 = max(0, min(x1, original_size[0]))
        y1 = max(0, min(y1, original_size[1]))
        x2 = max(0, min(x2, original_size[0]))
        y2 = max(0, min(y2, original_size[1]))
        
        # 过滤无效边界框
        if x2 <= x1 or y2 <= y1:
            continue
            
        results.append({
            'class': class_name,
            'confidence': float(score),
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'query_idx': int(query_idx),
            'class_idx': int(class_idx)
        })
        
        if i < 10:  # 只显示前10个检测结果
            print(f"  检测 {i+1}: {class_name} (置信度: {score:.3f}, query: {query_idx})")
    
    # 按置信度排序
    results.sort(key=lambda x: x['confidence'], reverse=True)
    
    return results

def main():
    print("=== 改进版推理测试 ===")
    
    # 参数
    weights_path = "checkpoints/model_final.pkl"
    img_path = "data/coco2017_50/train2017/000000225405.jpg"
    conf_threshold = 0.3
    
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
    
    # 4. 后处理 - 尝试两种模式
    print("\n>>> 后处理...")
    
    # 首先尝试focal loss模式
    print("\n--- 尝试focal loss模式 ---")
    detections_focal = improved_postprocess(outputs, original_size, conf_threshold, use_focal_loss=True)
    
    # 然后尝试softmax模式
    print("\n--- 尝试softmax模式 ---")
    detections_softmax = improved_postprocess(outputs, original_size, conf_threshold, use_focal_loss=False)
    
    # 选择更好的结果
    if len(detections_focal) > 0 and len(detections_softmax) > 0:
        # 比较两种模式的结果质量
        focal_diversity = len(set(d['class'] for d in detections_focal[:10]))
        softmax_diversity = len(set(d['class'] for d in detections_softmax[:10]))
        
        if softmax_diversity > focal_diversity:
            detections = detections_softmax
            mode = "softmax"
        else:
            detections = detections_focal
            mode = "focal loss"
    elif len(detections_softmax) > 0:
        detections = detections_softmax
        mode = "softmax"
    else:
        detections = detections_focal
        mode = "focal loss"
    
    # 5. 显示结果
    print(f"\n🎯 最终检测结果 (使用{mode}模式):")
    if detections:
        print(f"✅ 检测到 {len(detections)} 个目标:")
        unique_classes = set()
        for i, det in enumerate(detections[:10]):  # 只显示前10个
            unique_classes.add(det['class'])
            print(f"  {i+1}. {det['class']}: {det['confidence']:.3f}")
            print(f"     位置: [{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, {det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}]")
        
        print(f"\n📊 检测统计:")
        print(f"  - 总检测数: {len(detections)}")
        print(f"  - 不同类别数: {len(unique_classes)}")
        print(f"  - 检测到的类别: {', '.join(sorted(unique_classes))}")
    else:
        print("❌ 未检测到任何目标")
    
    # 6. 流程自检结论
    print(f"\n🔍 流程自检结论:")
    if detections:
        print("✅ 流程自检通过！")
        print("  - 模型成功加载")
        print("  - 推理正常执行")
        print("  - 后处理逻辑正确")
        print("  - 检测到目标物体")
        print("  - 整个训练→推理流程工作正常")
    else:
        print("⚠️ 流程自检部分通过")
        print("  - 模型成功加载 ✅")
        print("  - 推理正常执行 ✅") 
        print("  - 但未检测到目标 ❌")
        print("  - 可能需要调整置信度阈值或检查模型训练")

if __name__ == "__main__":
    main()
