#!/usr/bin/env python3
"""
深度诊断RT-DETR模型学习情况
分析模型是否真正学习到了特征
"""

import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw

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

def load_ground_truth():
    """加载真实标注数据"""
    ann_file = "data/coco2017_50/annotations/instances_train2017.json"
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # 创建图片ID到标注的映射
    image_id_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_id_to_anns:
            image_id_to_anns[img_id] = []
        image_id_to_anns[img_id].append(ann)
    
    # 创建类别映射
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    idx_to_name = {idx: cat['name'] for idx, cat in enumerate(coco_data['categories'])}
    
    return coco_data, image_id_to_anns, cat_id_to_name, cat_id_to_idx, idx_to_name

def analyze_single_image_deep(image_path, model, ground_truth_data):
    """深度分析单张图片"""
    coco_data, image_id_to_anns, cat_id_to_name, cat_id_to_idx, idx_to_name = ground_truth_data
    
    # 获取图片ID
    image_name = os.path.basename(image_path)
    image_id = None
    for img_info in coco_data['images']:
        if img_info['file_name'] == image_name:
            image_id = img_info['id']
            break
    
    if image_id is None:
        print(f"❌ 找不到图片 {image_name} 的标注信息")
        return
    
    print(f"\n" + "="*60)
    print(f"深度分析图片: {image_name} (ID: {image_id})")
    print("="*60)
    
    # 1. 分析真实标注
    print(f"\n=== 真实标注分析 ===")
    if image_id in image_id_to_anns:
        gt_anns = image_id_to_anns[image_id]
        print(f"真实目标数量: {len(gt_anns)}")
        for i, ann in enumerate(gt_anns):
            cat_name = cat_id_to_name[ann['category_id']]
            cat_idx = cat_id_to_idx[ann['category_id']]
            bbox = ann['bbox']  # [x, y, w, h]
            print(f"  {i+1}. {cat_name} (类别ID:{ann['category_id']}, 索引:{cat_idx}) at {bbox}")
    else:
        print("该图片没有标注")
        return
    
    # 2. 预处理图片
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    image_resized = image.resize((640, 640))
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    img_tensor = safe_float32(img_array).unsqueeze(0)
    
    # 3. 模型推理
    with jt.no_grad():
        outputs = model(img_tensor)
    
    pred_logits = outputs['pred_logits'][0]  # [300, 80]
    pred_boxes = outputs['pred_boxes'][0]    # [300, 4]
    
    print(f"\n=== 模型输出原始分析 ===")
    print(f"pred_logits形状: {pred_logits.shape}")
    print(f"pred_boxes形状: {pred_boxes.shape}")
    
    # 4. 分析logits分布
    logits_np = pred_logits.numpy()
    print(f"\nLogits统计:")
    print(f"  范围: [{logits_np.min():.3f}, {logits_np.max():.3f}]")
    print(f"  均值: {logits_np.mean():.3f}")
    print(f"  标准差: {logits_np.std():.3f}")
    
    # 5. 分析每个真实类别的logits
    print(f"\n=== 真实类别的logits分析 ===")
    for ann in gt_anns:
        cat_idx = cat_id_to_idx[ann['category_id']]
        cat_name = cat_id_to_name[ann['category_id']]
        
        # 该类别在所有查询中的logits
        class_logits = logits_np[:, cat_idx]
        print(f"{cat_name} (索引{cat_idx}):")
        print(f"  logits范围: [{class_logits.min():.3f}, {class_logits.max():.3f}]")
        print(f"  logits均值: {class_logits.mean():.3f}")
        print(f"  最大logit查询: {class_logits.argmax()} (值: {class_logits.max():.3f})")
    
    # 6. 分析sigmoid后的概率
    pred_probs = jt.sigmoid(pred_logits).numpy()
    print(f"\n=== Sigmoid概率分析 ===")
    print(f"概率范围: [{pred_probs.min():.3f}, {pred_probs.max():.3f}]")
    print(f"概率均值: {pred_probs.mean():.3f}")
    
    # 7. 分析每个查询的最大概率类别
    max_probs = pred_probs.max(axis=1)
    max_classes = pred_probs.argmax(axis=1)
    
    print(f"\n=== Top 10 查询分析 ===")
    top_indices = np.argsort(max_probs)[::-1][:10]
    
    for i, idx in enumerate(top_indices):
        prob = max_probs[idx]
        class_idx = max_classes[idx]
        class_name = idx_to_name.get(class_idx, f'unknown_{class_idx}')
        box = pred_boxes[idx].numpy()
        
        print(f"查询{idx}: 概率={prob:.4f}, 预测类别={class_name}({class_idx}), 框={box}")
    
    # 8. 检查是否预测了正确的类别
    print(f"\n=== 正确类别预测检查 ===")
    gt_class_indices = [cat_id_to_idx[ann['category_id']] for ann in gt_anns]
    
    for gt_idx in gt_class_indices:
        gt_name = idx_to_name[gt_idx]
        # 找到预测该类别概率最高的查询
        class_probs = pred_probs[:, gt_idx]
        best_query = class_probs.argmax()
        best_prob = class_probs[best_query]
        
        print(f"真实类别 {gt_name}({gt_idx}):")
        print(f"  最佳预测查询: {best_query}")
        print(f"  该类别概率: {best_prob:.4f}")
        print(f"  该查询最大概率类别: {idx_to_name[max_classes[best_query]]}({max_classes[best_query]})")
        print(f"  该查询最大概率值: {max_probs[best_query]:.4f}")
        
        if max_classes[best_query] == gt_idx:
            print(f"  ✅ 该查询正确预测了类别!")
        else:
            print(f"  ❌ 该查询预测错误")
    
    # 9. 边界框分析
    print(f"\n=== 边界框预测分析 ===")
    boxes_np = pred_boxes.numpy()
    
    # 检查边界框是否合理
    valid_boxes = (
        (boxes_np[:, 0] >= 0) & (boxes_np[:, 0] <= 1) &  # cx
        (boxes_np[:, 1] >= 0) & (boxes_np[:, 1] <= 1) &  # cy
        (boxes_np[:, 2] > 0.01) & (boxes_np[:, 2] <= 1) &  # w
        (boxes_np[:, 3] > 0.01) & (boxes_np[:, 3] <= 1)    # h
    )
    
    print(f"合理边界框数量: {valid_boxes.sum()}/300")
    print(f"边界框统计:")
    print(f"  cx范围: [{boxes_np[:, 0].min():.3f}, {boxes_np[:, 0].max():.3f}]")
    print(f"  cy范围: [{boxes_np[:, 1].min():.3f}, {boxes_np[:, 1].max():.3f}]")
    print(f"  w范围: [{boxes_np[:, 2].min():.3f}, {boxes_np[:, 2].max():.3f}]")
    print(f"  h范围: [{boxes_np[:, 3].min():.3f}, {boxes_np[:, 3].max():.3f}]")
    
    # 10. 可视化对比
    print(f"\n=== 创建对比可视化 ===")
    
    # 创建对比图
    fig_width = original_size[0] * 2
    fig_height = original_size[1]
    comparison_img = Image.new('RGB', (fig_width, fig_height), 'white')
    
    # 左侧：真实标注
    gt_img = image.copy()
    gt_draw = ImageDraw.Draw(gt_img)
    
    for ann in gt_anns:
        x, y, w, h = ann['bbox']
        gt_draw.rectangle([x, y, x+w, y+h], outline='green', width=3)
        cat_name = cat_id_to_name[ann['category_id']]
        gt_draw.text((x, y-20), f'GT: {cat_name}', fill='green')
    
    # 右侧：模型预测（top 3）
    pred_img = image.copy()
    pred_draw = ImageDraw.Draw(pred_img)
    
    for i, idx in enumerate(top_indices[:3]):
        if max_probs[idx] > 0.1:  # 只显示概率>0.1的
            box = pred_boxes[idx].numpy()
            cx, cy, w, h = box
            
            # 转换到原始图片坐标
            cx *= original_size[0]
            cy *= original_size[1]
            w *= original_size[0]
            h *= original_size[1]
            
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            
            color = ['red', 'blue', 'orange'][i]
            pred_draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            class_name = idx_to_name[max_classes[idx]]
            pred_draw.text((x1, y1-20), f'Pred: {class_name} ({max_probs[idx]:.2f})', fill=color)
    
    # 合并图片
    comparison_img.paste(gt_img, (0, 0))
    comparison_img.paste(pred_img, (original_size[0], 0))
    
    # 保存对比图
    os.makedirs("results/deep_analysis", exist_ok=True)
    save_path = f"results/deep_analysis/comparison_{image_name}"
    comparison_img.save(save_path)
    print(f"✅ 对比图保存到: {save_path}")
    
    # 11. 总结诊断
    print(f"\n=== 诊断总结 ===")
    
    # 检查是否学习到特征
    max_gt_prob = 0
    correct_predictions = 0
    
    for gt_idx in gt_class_indices:
        class_probs = pred_probs[:, gt_idx]
        best_prob = class_probs.max()
        max_gt_prob = max(max_gt_prob, best_prob)
        
        best_query = class_probs.argmax()
        if max_classes[best_query] == gt_idx and max_probs[best_query] > 0.2:
            correct_predictions += 1
    
    print(f"真实类别最高概率: {max_gt_prob:.4f}")
    print(f"正确预测数量: {correct_predictions}/{len(gt_class_indices)}")
    
    if max_gt_prob < 0.1:
        print("❌ 严重问题：模型完全没有学习到真实类别的特征")
        print("   建议：检查训练数据、损失函数、学习率")
    elif max_gt_prob < 0.3:
        print("⚠️ 问题：模型对真实类别的学习很弱")
        print("   建议：增加训练轮次、调整学习率、检查数据质量")
    elif correct_predictions == 0:
        print("⚠️ 问题：虽然有一定概率，但预测类别错误")
        print("   建议：检查类别映射、增加训练轮次")
    else:
        print("✅ 模型有一定学习效果，但需要继续训练")

def main():
    print("RT-DETR模型深度学习诊断")
    print("="*60)
    
    # 加载模型
    model_path = "checkpoints/rtdetr_jittor.pkl"
    model = build_rtdetr_complete(num_classes=80, hidden_dim=256, num_queries=300)
    state_dict = jt.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 加载真实标注
    ground_truth_data = load_ground_truth()
    
    # 分析几张图片
    test_image_dir = "data/coco2017_50/train2017"
    image_files = [f for f in os.listdir(test_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for i, image_file in enumerate(image_files[:3]):  # 分析前3张图片
        image_path = os.path.join(test_image_dir, image_file)
        analyze_single_image_deep(image_path, model, ground_truth_data)
        
        if i < 2:
            input("\n按Enter继续分析下一张图片...")

if __name__ == "__main__":
    main()
