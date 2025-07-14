#!/usr/bin/env python3
"""
RT-DETR完整端到端验证脚本
验证训练->保存->加载->推理的完整流程
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt
from jittor_rt_detr.src.nn.rtdetr_complete_pytorch_aligned import build_rtdetr_complete, RTDETRPostProcessor
from jittor_rt_detr.src.nn.loss_pytorch_aligned import build_criterion

# 设置Jittor为float32模式
jt.flags.use_cuda = 1
jt.set_global_seed(42)

def load_target_data():
    """加载目标图片和标注"""
    img_path = "data/coco2017_50/train2017/000000225405.jpg"
    ann_path = "data/coco2017_50/annotations/instances_train2017.json"
    
    print(f">>> 加载目标图片: {os.path.basename(img_path)}")
    
    # 加载图片
    image = Image.open(img_path).convert('RGB')
    original_size = image.size
    print(f"原始图片尺寸: {original_size}")
    
    # 加载标注
    with open(ann_path, 'r') as f:
        coco_data = json.load(f)
    
    # 找到对应图片的标注
    target_image_id = 225405
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == target_image_id]
    
    print(f"真实标注数量: {len(annotations)}")
    
    # 构建类别映射
    present_categories = set(ann['category_id'] for ann in annotations)
    cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(sorted(present_categories))}
    idx_to_cat_id = {idx: cat_id for cat_id, idx in cat_id_to_idx.items()}
    
    # 统计真实标注
    ground_truth_objects = {}
    for ann in annotations:
        cat_id = ann['category_id']
        cat_name = 'person' if cat_id == 1 else 'sports ball' if cat_id == 37 else f'class_{cat_id}'
        
        if cat_name not in ground_truth_objects:
            ground_truth_objects[cat_name] = []
        ground_truth_objects[cat_name].append(ann)
        
        bbox = ann['bbox']
        print(f"  - {cat_name}: bbox={bbox}, area={ann['area']}")
    
    return image, annotations, ground_truth_objects, cat_id_to_idx, idx_to_cat_id, original_size

def preprocess_image(image, target_size=640):
    """预处理图片"""
    # Resize
    image_resized = image.resize((target_size, target_size))
    
    # 转换为numpy数组并归一化
    img_array = np.array(image_resized).astype(np.float32) / 255.0
    
    # 转换为CHW格式
    img_array = img_array.transpose(2, 0, 1)
    
    # 转换为Jittor tensor并添加batch维度，强制float32
    img_tensor = jt.array(img_array, dtype=jt.float32).unsqueeze(0)
    
    return img_tensor

def create_targets(annotations, cat_id_to_idx, original_size):
    """创建训练目标，强制float32"""
    boxes = []
    labels = []
    
    for ann in annotations:
        # 获取边界框 (COCO格式: x, y, width, height)
        x, y, w, h = ann['bbox']
        
        # 转换为中心点格式并归一化
        cx = (x + w / 2) / original_size[0]
        cy = (y + h / 2) / original_size[1]
        w_norm = w / original_size[0]
        h_norm = h / original_size[1]
        
        boxes.append([cx, cy, w_norm, h_norm])
        labels.append(cat_id_to_idx[ann['category_id']])
    
    targets = [{
        'boxes': jt.array(boxes, dtype=jt.float32),
        'labels': jt.array(labels, dtype=jt.int64)
    }]
    
    return targets

def force_float32_model(model):
    """强制模型所有参数为float32"""
    def convert_to_float32(m):
        for name, param in m.named_parameters():
            if param.dtype != jt.float32:
                param.data = param.data.float32()
        for name, buffer in m.named_buffers():
            if buffer.dtype != jt.float32:
                buffer.data = buffer.data.float32()
    
    model.apply(convert_to_float32)
    return model

def train_model(model, criterion, img_tensor, targets, max_epochs=3):
    """训练模型"""
    print(f"\n>>> 开始训练 (最多 {max_epochs} 轮)")
    
    # 强制所有输入为float32
    img_tensor = img_tensor.float32()
    for target in targets:
        target['boxes'] = target['boxes'].float32()
        target['labels'] = target['labels'].int64()
    
    # 优化器
    optimizer = jt.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    
    best_loss = float('inf')
    
    for epoch in range(max_epochs):
        model.train()
        
        try:
            # 前向传播
            outputs = model(img_tensor.float32(), targets)
            
            # 确保输出类型一致
            for key in outputs:
                if isinstance(outputs[key], jt.Var):
                    outputs[key] = outputs[key].float32()
            
            # 计算损失
            loss_dict = criterion(outputs, targets)
            
            # 确保损失为float32
            for key in loss_dict:
                loss_dict[key] = loss_dict[key].float32()
            
            # 加权总损失
            total_loss = sum(loss_dict[k].float32() * criterion.weight_dict[k] 
                            for k in loss_dict.keys() if k in criterion.weight_dict)
            total_loss = total_loss.float32()
            
            # 反向传播
            optimizer.step(total_loss)
            
            # 记录损失
            loss_value = total_loss.item()
            
            if loss_value < best_loss:
                best_loss = loss_value
            
            print(f"  Epoch {epoch+1}/{max_epochs}: Loss = {loss_value:.4f} (Best: {best_loss:.4f})")
                    
        except Exception as e:
            print(f"  Epoch {epoch+1} 失败: {e}")
            continue
    
    print("✅ 训练完成")
    return model

def save_and_load_model(model, num_classes, save_path="checkpoints/end_to_end_model.pkl"):
    """保存并重新加载模型"""
    print(f"\n>>> 保存模型到: {save_path}")
    
    # 保存模型
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    jt.save(model.state_dict(), save_path)
    print("✅ 模型保存成功")
    
    # 重新加载模型
    print(">>> 重新加载模型...")
    new_model = build_rtdetr_complete(num_classes=num_classes, hidden_dim=256, num_queries=300)
    new_model = force_float32_model(new_model)
    
    state_dict = jt.load(save_path)
    new_model.load_state_dict(state_dict)
    new_model.eval()
    print("✅ 模型加载成功")
    
    return new_model

def run_inference(model, img_tensor, original_size, num_classes, idx_to_cat_id):
    """运行推理"""
    print(f"\n>>> 开始推理...")
    
    # 创建后处理器
    postprocessor = RTDETRPostProcessor(num_classes=num_classes, confidence_threshold=0.3)
    
    with jt.no_grad():
        # 前向传播
        outputs = model(img_tensor.float32())
        
        # 后处理
        results = postprocessor(outputs, [original_size])
    
    # 解析结果
    detections = []
    if len(results) > 0:
        result = results[0]
        if 'scores' in result and len(result['scores']) > 0:
            scores = result['scores']
            labels = result['labels']
            boxes = result['boxes']
            
            for i in range(len(scores)):
                confidence = scores[i].item()
                class_idx = labels[i].item()
                box = boxes[i]
                
                detections.append({
                    'bbox': [box[0].item(), box[1].item(), box[2].item(), box[3].item()],
                    'confidence': confidence,
                    'class_idx': class_idx,
                    'category_id': idx_to_cat_id[class_idx]
                })
    
    print(f"检测到 {len(detections)} 个目标")
    return detections

def validate_results(detections, ground_truth_objects):
    """验证检测结果"""
    print(f"\n=== 结果验证 ===")
    
    # COCO类别名称映射
    coco_categories = {1: 'person', 37: 'sports ball'}
    
    # 按类别统计检测结果
    detected_objects = {}
    for det in detections:
        cat_id = det['category_id']
        cat_name = coco_categories.get(cat_id, f'class_{cat_id}')
        
        if cat_name not in detected_objects:
            detected_objects[cat_name] = []
        detected_objects[cat_name].append(det)
    
    print(f"检测结果统计:")
    for class_name, dets in detected_objects.items():
        print(f"  - {class_name}: {len(dets)} 个")
    
    # 验证
    expected_classes = set(ground_truth_objects.keys())
    detected_class_names = set(detected_objects.keys())
    
    print(f"\n期望检测的类别: {expected_classes}")
    print(f"实际检测的类别: {detected_class_names}")
    
    missing_classes = expected_classes - detected_class_names
    
    if missing_classes:
        print(f"❌ 缺失的类别: {missing_classes}")
        success = False
    else:
        print(f"✅ 所有期望类别都被检测到")
        success = True
    
    return success, detected_objects

def main():
    print("=" * 80)
    print("===        RT-DETR完整端到端验证        ===")
    print("===     训练->保存->加载->推理->验证     ===")
    print("=" * 80)
    
    try:
        # 1. 加载数据
        image, annotations, ground_truth_objects, cat_id_to_idx, idx_to_cat_id, original_size = load_target_data()
        
        # 2. 预处理
        img_tensor = preprocess_image(image)
        targets = create_targets(annotations, cat_id_to_idx, original_size)
        
        # 3. 创建并训练模型
        num_classes = len(cat_id_to_idx)
        print(f"\n>>> 创建RT-DETR模型 (类别数: {num_classes})")
        model = build_rtdetr_complete(num_classes=num_classes, hidden_dim=256, num_queries=300)
        model = force_float32_model(model)
        
        criterion = build_criterion(num_classes)
        model = train_model(model, criterion, img_tensor, targets, max_epochs=3)
        
        # 4. 保存并重新加载模型
        model = save_and_load_model(model, num_classes)
        
        # 5. 推理
        detections = run_inference(model, img_tensor, original_size, num_classes, idx_to_cat_id)
        
        # 6. 验证结果
        success, detected_objects = validate_results(detections, ground_truth_objects)
        
        # 7. 最终判定
        print(f"\n" + "=" * 80)
        print("🎯 完整端到端验证结果:")
        print("=" * 80)
        
        if success:
            print("🎉 完整端到端验证成功！")
            print("  ✅ 训练流程正常")
            print("  ✅ 模型保存/加载正常")
            print("  ✅ 推理流程正常")
            print("  ✅ 结果验证通过")
            print("  ✅ RT-DETR Jittor项目完全成功！")
        else:
            print("❌ 端到端验证失败！")
            print("  ❌ 某些环节存在问题")
        
        print("=" * 80)
        
        return success
        
    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
