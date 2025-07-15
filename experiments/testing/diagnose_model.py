#!/usr/bin/env python3
"""
RT-DETR模型诊断脚本
详细分析模型输出，找出检测不到目标的原因
"""

import os
import sys
import json
import numpy as np
from PIL import Image

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt
from jittor_rt_detr.src.nn.rtdetr_complete_pytorch_aligned import build_rtdetr_complete

# 设置Jittor
jt.flags.use_cuda = 1
jt.flags.auto_mixed_precision_level = 0

def safe_float32(tensor):
    if isinstance(tensor, jt.Var):
        return tensor.float32()
    elif isinstance(tensor, np.ndarray):
        return jt.array(tensor.astype(np.float32))
    else:
        return jt.array(tensor, dtype=jt.float32)

def diagnose_model():
    """诊断模型问题"""
    print("=" * 60)
    print("RT-DETR模型详细诊断")
    print("=" * 60)
    
    # 1. 加载模型
    model_path = "checkpoints/rtdetr_jittor.pkl"
    model = build_rtdetr_complete(num_classes=80, hidden_dim=256, num_queries=300)
    state_dict = jt.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 2. 加载测试图片
    test_image_path = "data/coco2017_50/train2017"
    image_files = [f for f in os.listdir(test_image_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("❌ 没有找到测试图片")
        return
    
    img_path = os.path.join(test_image_path, image_files[0])
    print(f">>> 诊断图片: {image_files[0]}")
    
    # 3. 预处理图片
    image = Image.open(img_path).convert('RGB')
    image_resized = image.resize((640, 640))
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    img_tensor = safe_float32(img_array).unsqueeze(0)
    
    print(f"输入图片形状: {img_tensor.shape}")
    print(f"输入数据范围: [{img_tensor.min().item():.3f}, {img_tensor.max().item():.3f}]")
    
    # 4. 前向传播
    with jt.no_grad():
        outputs = model(img_tensor)
    
    pred_logits = outputs['pred_logits'][0]  # [300, 80]
    pred_boxes = outputs['pred_boxes'][0]    # [300, 4]
    
    print(f"\n=== 模型输出分析 ===")
    print(f"pred_logits形状: {pred_logits.shape}")
    print(f"pred_boxes形状: {pred_boxes.shape}")
    
    # 5. 分析logits
    print(f"\n=== Logits分析 ===")
    logits_np = pred_logits.numpy()
    print(f"logits范围: [{logits_np.min():.3f}, {logits_np.max():.3f}]")
    print(f"logits均值: {logits_np.mean():.3f}")
    print(f"logits标准差: {logits_np.std():.3f}")
    
    # 6. 分析sigmoid后的概率
    print(f"\n=== Sigmoid概率分析 ===")
    pred_probs = jt.sigmoid(pred_logits)
    probs_np = pred_probs.numpy()
    print(f"概率范围: [{probs_np.min():.3f}, {probs_np.max():.3f}]")
    print(f"概率均值: {probs_np.mean():.3f}")
    print(f"概率标准差: {probs_np.std():.3f}")
    
    # 7. 分析最大概率
    max_probs = pred_probs.max(dim=-1)[0]
    max_probs_np = max_probs.numpy()
    print(f"每个查询的最大概率范围: [{max_probs_np.min():.3f}, {max_probs_np.max():.3f}]")
    print(f"最大概率均值: {max_probs_np.mean():.3f}")
    
    # 8. 统计不同阈值下的检测数
    print(f"\n=== 不同置信度阈值下的检测数 ===")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        count = (max_probs_np > threshold).sum()
        print(f"阈值 {threshold}: {count} 个检测")
    
    # 9. 分析top检测
    print(f"\n=== Top 10 检测分析 ===")
    top_indices = np.argsort(max_probs_np)[::-1][:10]
    
    for i, idx in enumerate(top_indices):
        prob = max_probs_np[idx]
        label = pred_probs[idx].argmax(dim=-1)[0].item()
        box = pred_boxes[idx].numpy()
        print(f"Top {i+1}: 查询{idx}, 概率={prob:.4f}, 类别={label}, 框={box}")
    
    # 10. 分析边界框
    print(f"\n=== 边界框分析 ===")
    boxes_np = pred_boxes.numpy()
    print(f"边界框范围:")
    print(f"  cx: [{boxes_np[:, 0].min():.3f}, {boxes_np[:, 0].max():.3f}]")
    print(f"  cy: [{boxes_np[:, 1].min():.3f}, {boxes_np[:, 1].max():.3f}]")
    print(f"  w:  [{boxes_np[:, 2].min():.3f}, {boxes_np[:, 2].max():.3f}]")
    print(f"  h:  [{boxes_np[:, 3].min():.3f}, {boxes_np[:, 3].max():.3f}]")
    
    # 11. 检查是否有合理的边界框
    print(f"\n=== 合理边界框检查 ===")
    # 检查边界框是否在合理范围内 (0-1之间，且宽高>0.01)
    valid_boxes = (
        (boxes_np[:, 0] >= 0) & (boxes_np[:, 0] <= 1) &  # cx
        (boxes_np[:, 1] >= 0) & (boxes_np[:, 1] <= 1) &  # cy
        (boxes_np[:, 2] > 0.01) & (boxes_np[:, 2] <= 1) &  # w
        (boxes_np[:, 3] > 0.01) & (boxes_np[:, 3] <= 1)    # h
    )
    valid_count = valid_boxes.sum()
    print(f"合理边界框数量: {valid_count}/300")
    
    # 12. 综合分析
    print(f"\n=== 诊断结论 ===")
    
    if max_probs_np.max() < 0.1:
        print("❌ 问题：所有预测概率都很低 (<0.1)")
        print("   可能原因：模型训练不充分，需要更多训练轮次")
    elif max_probs_np.max() < 0.3:
        print("⚠️ 问题：最高预测概率较低 (<0.3)")
        print("   建议：降低置信度阈值到0.1-0.2进行测试")
    else:
        print("✅ 预测概率正常，可能是阈值设置问题")
    
    if valid_count < 50:
        print("❌ 问题：大部分边界框不合理")
        print("   可能原因：边界框回归训练不充分")
    else:
        print("✅ 边界框预测基本正常")
    
    # 13. 建议的测试
    print(f"\n=== 建议的下一步测试 ===")
    best_threshold = 0.1
    for threshold in thresholds:
        count = (max_probs_np > threshold).sum()
        if count > 0:
            best_threshold = threshold
            break
    
    print(f"1. 使用较低阈值 {best_threshold} 进行测试")
    print(f"2. 检查训练是否需要继续更多轮次")
    print(f"3. 验证训练数据和标注是否正确")
    
    return best_threshold

def test_with_low_threshold(threshold=0.1):
    """使用低阈值进行测试"""
    print(f"\n>>> 使用低阈值 {threshold} 进行测试")
    
    # 加载模型
    model_path = "checkpoints/rtdetr_jittor.pkl"
    model = build_rtdetr_complete(num_classes=80, hidden_dim=256, num_queries=300)
    state_dict = jt.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 加载类别
    ann_file = "data/coco2017_50/annotations/instances_train2017.json"
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    idx_to_name = {idx: cat['name'] for idx, cat in enumerate(coco_data['categories'])}
    
    # 测试图片
    test_image_path = "data/coco2017_50/train2017"
    image_files = [f for f in os.listdir(test_image_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    img_path = os.path.join(test_image_path, image_files[0])
    
    # 预处理
    image = Image.open(img_path).convert('RGB')
    image_resized = image.resize((640, 640))
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    img_tensor = safe_float32(img_array).unsqueeze(0)
    
    # 推理
    with jt.no_grad():
        outputs = model(img_tensor)
    
    pred_logits = outputs['pred_logits'][0]
    pred_boxes = outputs['pred_boxes'][0]
    
    # 后处理
    pred_probs = jt.sigmoid(pred_logits)
    pred_scores = pred_probs.max(dim=-1)[0]
    pred_labels = pred_probs.argmax(dim=-1)
    
    # 过滤
    keep = pred_scores > threshold
    
    if keep.sum() > 0:
        filtered_scores = pred_scores[keep]
        filtered_labels = pred_labels[keep]
        filtered_boxes = pred_boxes[keep]
        
        print(f"✅ 检测到 {len(filtered_scores)} 个目标:")
        for i, (score, label, box) in enumerate(zip(filtered_scores, filtered_labels, filtered_boxes)):
            class_name = idx_to_name.get(label.item(), f'class_{label.item()}')
            cx, cy, w, h = box.numpy()
            print(f"  {i+1}. {class_name}: {score.item():.3f} at ({cx:.3f}, {cy:.3f}, {w:.3f}, {h:.3f})")
    else:
        print(f"❌ 即使使用阈值 {threshold} 也没有检测到目标")

def main():
    # 1. 详细诊断
    best_threshold = diagnose_model()
    
    # 2. 使用建议阈值测试
    test_with_low_threshold(best_threshold)

if __name__ == "__main__":
    main()
