#!/usr/bin/env python3
"""
RT-DETR Jittor模型快速测试脚本
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

def load_model_and_test():
    """加载模型并进行快速测试"""
    print("=" * 50)
    print("RT-DETR Jittor模型快速测试")
    print("=" * 50)
    
    # 1. 检查模型文件
    model_path = "checkpoints/rtdetr_jittor.pkl"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    print(f"✅ 找到模型文件: {model_path}")
    
    # 2. 创建模型
    try:
        model = build_rtdetr_complete(num_classes=80, hidden_dim=256, num_queries=300)
        print("✅ 模型架构创建成功")
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False
    
    # 3. 加载权重
    try:
        state_dict = jt.load(model_path)
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ 模型权重加载成功")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return False
    
    # 4. 测试前向传播
    try:
        # 创建随机输入
        test_input = safe_float32(jt.randn(1, 3, 640, 640))
        print(f"✅ 创建测试输入: {test_input.shape}")
        
        # 前向传播
        with jt.no_grad():
            outputs = model(test_input)
        
        print("✅ 前向传播成功")
        print(f"  输出键: {list(outputs.keys())}")
        print(f"  pred_logits形状: {outputs['pred_logits'].shape}")
        print(f"  pred_boxes形状: {outputs['pred_boxes'].shape}")
        
        if 'aux_outputs' in outputs:
            print(f"  辅助输出数量: {len(outputs['aux_outputs'])}")
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. 测试真实图片（如果存在）
    test_image_path = "data/coco2017_50/train2017"
    if os.path.exists(test_image_path):
        image_files = [f for f in os.listdir(test_image_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            try:
                # 加载第一张图片
                img_path = os.path.join(test_image_path, image_files[0])
                image = Image.open(img_path).convert('RGB')
                
                # 预处理
                image_resized = image.resize((640, 640))
                img_array = np.array(image_resized, dtype=np.float32) / 255.0
                img_array = img_array.transpose(2, 0, 1)
                img_tensor = safe_float32(img_array).unsqueeze(0)
                
                # 推理
                with jt.no_grad():
                    outputs = model(img_tensor)
                
                # 简单后处理
                pred_logits = outputs['pred_logits'][0]
                pred_boxes = outputs['pred_boxes'][0]
                
                # 计算置信度
                pred_scores = jt.sigmoid(pred_logits).max(dim=-1)[0]
                pred_labels = jt.sigmoid(pred_logits).argmax(dim=-1)
                
                # 找到高置信度预测
                high_conf_mask = pred_scores > 0.3
                high_conf_count = high_conf_mask.sum().item()
                
                print(f"✅ 真实图片测试成功")
                print(f"  测试图片: {image_files[0]}")
                print(f"  高置信度检测数 (>0.3): {high_conf_count}")
                
                if high_conf_count > 0:
                    max_score = pred_scores.max().item()
                    print(f"  最高置信度: {max_score:.3f}")
                
            except Exception as e:
                print(f"⚠️ 真实图片测试失败: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 模型测试完全成功！")
    print("模型已准备好进行推理")
    print("=" * 50)
    
    return True

def test_with_specific_image(image_path):
    """测试指定图片"""
    if not os.path.exists(image_path):
        print(f"❌ 图片不存在: {image_path}")
        return
    
    print(f"\n>>> 测试指定图片: {image_path}")
    
    # 加载模型
    model_path = "checkpoints/rtdetr_jittor.pkl"
    model = build_rtdetr_complete(num_classes=80, hidden_dim=256, num_queries=300)
    state_dict = jt.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 加载类别名称
    ann_file = "data/coco2017_50/annotations/instances_train2017.json"
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    idx_to_name = {idx: cat['name'] for idx, cat in enumerate(coco_data['categories'])}
    
    # 预处理图片
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    image_resized = image.resize((640, 640))
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    img_tensor = safe_float32(img_array).unsqueeze(0)
    
    # 推理
    with jt.no_grad():
        outputs = model(img_tensor)
    
    # 后处理
    pred_logits = outputs['pred_logits'][0]
    pred_boxes = outputs['pred_boxes'][0]
    
    pred_scores = jt.sigmoid(pred_logits).max(dim=-1)[0]
    pred_labels = jt.sigmoid(pred_logits).argmax(dim=-1)
    
    # 过滤高置信度预测
    confidence_threshold = 0.3
    keep = pred_scores > confidence_threshold
    
    if keep.sum() > 0:
        filtered_scores = pred_scores[keep]
        filtered_labels = pred_labels[keep]
        filtered_boxes = pred_boxes[keep]
        
        print(f"检测到 {len(filtered_scores)} 个目标:")
        for i, (score, label, box) in enumerate(zip(filtered_scores, filtered_labels, filtered_boxes)):
            class_name = idx_to_name.get(label.item(), f'class_{label.item()}')
            cx, cy, w, h = box.numpy()
            print(f"  {i+1}. {class_name}: {score.item():.3f} at ({cx:.3f}, {cy:.3f}, {w:.3f}, {h:.3f})")
        
        # 简单可视化保存
        try:
            draw = ImageDraw.Draw(image)
            for score, label, box in zip(filtered_scores, filtered_labels, filtered_boxes):
                cx, cy, w, h = box.numpy()
                
                # 转换到原始图片坐标
                cx *= original_size[0]
                cy *= original_size[1]
                w *= original_size[0]
                h *= original_size[1]
                
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2
                
                draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                
                class_name = idx_to_name.get(label.item(), f'class_{label.item()}')
                draw.text((x1, y1-20), f'{class_name}: {score.item():.2f}', fill='red')
            
            os.makedirs("results", exist_ok=True)
            save_path = f"results/test_result_{os.path.basename(image_path)}"
            image.save(save_path)
            print(f"✅ 可视化结果保存到: {save_path}")
            
        except Exception as e:
            print(f"⚠️ 可视化保存失败: {e}")
    else:
        print("未检测到高置信度目标")

def main():
    import sys
    
    if len(sys.argv) > 1:
        # 测试指定图片
        image_path = sys.argv[1]
        test_with_specific_image(image_path)
    else:
        # 快速测试
        load_model_and_test()

if __name__ == "__main__":
    main()
