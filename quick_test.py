#!/usr/bin/env python3
"""
快速推理测试脚本 - 批量测试多张图片
"""

import os
import sys
import time
from PIL import Image
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'jittor_rt_detr'))

import jittor as jt
from src.nn.model import RTDETR

# 设置Jittor优化
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
    """快速图像预处理"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Resize图像
    resized_image = image.resize(target_size, Image.LANCZOS)
    
    # 转换为tensor
    img_array = np.array(resized_image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    img_tensor = jt.array(img_array.transpose(2, 0, 1)).unsqueeze(0)
    
    return img_tensor, original_size

def postprocess(outputs, original_size, conf_threshold=0.3):
    """快速后处理"""
    logits, boxes, _, _ = outputs
    pred_logits = logits[-1][0]  # (num_queries, num_classes)
    pred_boxes = boxes[-1][0]    # (num_queries, 4)

    # 计算分数
    scores = jt.sigmoid(pred_logits)
    scores_max = jt.max(scores, dim=-1)[0]
    labels_pred = jt.argmax(scores, dim=-1)
    
    # 过滤低置信度预测
    keep_mask = scores_max > conf_threshold
    keep_indices = jt.where(keep_mask)[0]
    
    if len(keep_indices) == 0:
        return [], [], []
    
    final_boxes = pred_boxes[keep_indices]
    final_scores = scores_max[keep_indices]
    final_labels = labels_pred[keep_indices]
    
    # 转换为像素坐标
    results = []
    for i in range(len(final_boxes)):
        box = final_boxes[i].numpy()
        score = final_scores[i].item()
        label_idx = final_labels[i].item()
        
        # 转换坐标
        cx, cy, w, h = box
        x1 = (cx - w/2) * original_size[0]
        y1 = (cy - h/2) * original_size[1]
        x2 = (cx + w/2) * original_size[0]
        y2 = (cy + h/2) * original_size[1]
        
        class_name = COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else f"class_{label_idx}"
        
        results.append({
            'class': class_name,
            'confidence': score,
            'bbox': [x1, y1, x2, y2]
        })
    
    return results

def test_images(model, image_paths, conf_threshold=0.3):
    """批量测试图片"""
    print(f"🚀 开始批量测试 {len(image_paths)} 张图片...")
    
    results = {}
    total_time = 0
    
    for i, img_path in enumerate(image_paths):
        print(f"\n📸 测试图片 {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
        
        try:
            # 预处理
            start_time = time.time()
            img_tensor, original_size = preprocess_image(img_path)
            
            # 推理
            with jt.no_grad():
                outputs = model(img_tensor)
            
            # 后处理
            detections = postprocess(outputs, original_size, conf_threshold)
            
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # 显示结果
            if detections:
                print(f"  ✅ 检测到 {len(detections)} 个目标 (用时: {inference_time:.2f}s)")
                for det in detections:
                    print(f"    - {det['class']}: {det['confidence']:.3f}")
            else:
                print(f"  ❌ 未检测到目标 (用时: {inference_time:.2f}s)")
            
            results[img_path] = {
                'detections': detections,
                'time': inference_time
            }
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            results[img_path] = {'error': str(e)}
    
    avg_time = total_time / len(image_paths)
    print(f"\n📊 批量测试完成!")
    print(f"  总用时: {total_time:.2f}s")
    print(f"  平均每张: {avg_time:.2f}s")
    print(f"  推理速度: {1/avg_time:.1f} FPS")
    
    return results

def main():
    print("=== RT-DETR 快速推理测试 ===")
    
    # 1. 加载模型
    print(">>> 加载模型...")
    model = RTDETR(num_classes=80)
    state_dict = jt.load("checkpoints/model_epoch_50.pkl")
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ 模型加载完成")
    
    # 2. 预热模型
    print(">>> 预热模型...")
    dummy_input = jt.randn(1, 3, 640, 640)
    with jt.no_grad():
        _ = model(dummy_input)
    print("✅ 模型预热完成")
    
    # 3. 选择测试图片
    test_images_list = [
        "data/coco2017_50/val2017/000000007991.jpg",    # 牙刷
        "data/coco2017_50/val2017/000000359855.jpg",    # 行李箱
        "data/coco2017_50/val2017/000000009769.jpg",    # 随机图片1
        "data/coco2017_50/val2017/000000015597.jpg",    # 随机图片2
        "data/coco2017_50/val2017/000000029187.jpg",    # 随机图片3
    ]
    
    # 过滤存在的图片
    existing_images = [img for img in test_images_list if os.path.exists(img)]
    print(f"📋 找到 {len(existing_images)} 张测试图片")
    
    # 4. 批量测试
    results = test_images(model, existing_images, conf_threshold=0.3)
    
    # 5. 总结
    print(f"\n🎯 测试总结:")
    detected_count = sum(1 for r in results.values() if 'detections' in r and r['detections'])
    total_detections = sum(len(r['detections']) for r in results.values() if 'detections' in r)
    
    print(f"  有检测结果的图片: {detected_count}/{len(existing_images)}")
    print(f"  总检测目标数: {total_detections}")
    
    # 显示每张图片的详细结果
    print(f"\n📋 详细结果:")
    for img_path, result in results.items():
        img_name = os.path.basename(img_path)
        if 'detections' in result:
            if result['detections']:
                classes = [det['class'] for det in result['detections']]
                print(f"  {img_name}: {classes}")
            else:
                print(f"  {img_name}: 无检测结果")
        else:
            print(f"  {img_name}: 处理失败")

if __name__ == "__main__":
    main()
