#!/usr/bin/env python3
"""
简化版单张图片过拟合测试
专注于验证模型是否能够学习，避免复杂的数据类型问题
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
from jittor_rt_detr.src.nn.loss_pytorch_aligned import build_criterion

# 设置Jittor
jt.flags.use_cuda = 1
jt.set_global_seed(42)
jt.flags.auto_mixed_precision_level = 0

def safe_float32(tensor):
    """确保张量为float32类型"""
    if isinstance(tensor, jt.Var):
        return tensor.float32()
    elif isinstance(tensor, np.ndarray):
        return jt.array(tensor.astype(np.float32))
    else:
        return jt.array(tensor, dtype=jt.float32)

def safe_int64(tensor):
    """确保张量为int64类型"""
    if isinstance(tensor, jt.Var):
        return tensor.int64()
    elif isinstance(tensor, np.ndarray):
        return jt.array(tensor.astype(np.int64))
    else:
        return jt.array(np.array(tensor, dtype=np.int64))

def load_single_image_data():
    """加载单张图片的数据"""
    # 加载COCO数据
    ann_file = "data/coco2017_50/annotations/instances_train2017.json"
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # 选择第一张有标注的图片
    image_name = "000000343218.jpg"
    
    # 找到指定图片
    image_info = None
    for img in coco_data['images']:
        if img['file_name'] == image_name:
            image_info = img
            break
    
    # 获取该图片的标注
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_info['id']]
    
    print(f"选择图片: {image_name}")
    print(f"图片尺寸: {image_info['width']} x {image_info['height']}")
    print(f"标注数量: {len(annotations)}")
    
    return image_info, annotations, coco_data

def prepare_training_data(image_info, annotations, coco_data):
    """准备训练数据"""
    image_path = f"data/coco2017_50/train2017/{image_info['file_name']}"
    
    # 加载和预处理图片
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    image_resized = image.resize((640, 640))
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    img_tensor = safe_float32(img_array).unsqueeze(0)  # batch_size=1
    
    # 创建目标 - 只取前5个目标，避免过于复杂
    cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    boxes, labels = [], []
    
    for ann in annotations[:5]:  # 只取前5个目标
        x, y, w, h = ann['bbox']
        cx = np.float32(x + w / 2) / np.float32(original_size[0])
        cy = np.float32(y + h / 2) / np.float32(original_size[1])
        w_norm = np.float32(w) / np.float32(original_size[0])
        h_norm = np.float32(h) / np.float32(original_size[1])
        
        boxes.append([cx, cy, w_norm, h_norm])
        labels.append(cat_id_to_idx[ann['category_id']])
    
    targets = [{
        'boxes': safe_float32(np.array(boxes, dtype=np.float32)),
        'labels': safe_int64(np.array(labels, dtype=np.int64))
    }]
    
    print(f"训练目标数量: {len(boxes)}")
    for i, (box, label) in enumerate(zip(boxes, labels)):
        cat_name = coco_data['categories'][label]['name']
        print(f"  {i+1}. {cat_name}: [{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f}]")
    
    return img_tensor, targets

def simple_inference_test(model, img_tensor, epoch):
    """简化的推理测试"""
    model.eval()
    
    with jt.no_grad():
        try:
            outputs = model(img_tensor)
            
            # 简单检查输出形状
            pred_logits = outputs['pred_logits']
            pred_boxes = outputs['pred_boxes']
            
            print(f"Epoch {epoch}: 推理成功")
            print(f"  pred_logits shape: {pred_logits.shape}")
            print(f"  pred_boxes shape: {pred_boxes.shape}")
            
            # 计算最大置信度
            pred_probs = jt.sigmoid(pred_logits[0])
            max_scores = pred_probs.max(dim=-1)[0]
            max_score = max_scores.max().item()
            
            print(f"  最大置信度: {max_score:.4f}")
            
            return True, max_score
            
        except Exception as e:
            print(f"Epoch {epoch}: 推理失败 - {e}")
            return False, 0.0

def main():
    print("=" * 60)
    print("===        简化版单张图片过拟合测试        ===")
    print("=" * 60)
    
    try:
        # 加载数据
        image_info, annotations, coco_data = load_single_image_data()
        img_tensor, targets = prepare_training_data(image_info, annotations, coco_data)
        
        # 创建模型
        num_classes = len(coco_data['categories'])
        model = build_rtdetr_complete(num_classes=num_classes, hidden_dim=256, num_queries=300)
        print(f"✅ 创建模型，类别数: {num_classes}")
        
        # 创建损失函数
        criterion = build_criterion(num_classes)
        print("✅ 创建损失函数")
        
        # 优化器
        optimizer = jt.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
        print("✅ 创建优化器")
        
        # 初始测试
        print("\n=== 训练前测试 ===")
        success, initial_score = simple_inference_test(model, img_tensor, 0)
        
        if not success:
            print("❌ 初始推理失败，无法继续训练")
            return
        
        # 训练历史
        training_history = {'losses': [], 'scores': [initial_score]}
        
        # 开始训练
        print(f"\n开始训练 - 50轮")
        model.train()
        
        for epoch in range(1, 51):
            try:
                # 前向传播
                outputs = model(img_tensor)
                
                # 计算损失
                loss_dict = criterion(outputs, targets)
                total_loss = sum(loss_dict.values())
                
                # 反向传播
                optimizer.step(total_loss)
                
                training_history['losses'].append(total_loss.item())
                
                # 定期测试
                if epoch % 10 == 0 or epoch == 1:
                    print(f"\n=== Epoch {epoch} ===")
                    print(f"Loss: {total_loss.item():.4f}")
                    
                    success, score = simple_inference_test(model, img_tensor, epoch)
                    if success:
                        training_history['scores'].append(score)
                    else:
                        print("⚠️ 推理测试失败")
                        
                elif epoch % 5 == 0:
                    print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")
                    
            except Exception as e:
                print(f"❌ Epoch {epoch} 训练失败: {e}")
                break
        
        # 最终测试
        print(f"\n=== 最终测试 ===")
        success, final_score = simple_inference_test(model, img_tensor, 50)
        
        # 总结
        print(f"\n" + "=" * 60)
        print("🎯 训练总结:")
        print("=" * 60)
        print(f"初始最大置信度: {initial_score:.4f}")
        if success:
            print(f"最终最大置信度: {final_score:.4f}")
            improvement = final_score - initial_score
            print(f"置信度提升: {improvement:.4f}")
            
            if improvement > 0.1:
                print("✅ 训练成功！模型学习能力正常")
            elif improvement > 0.01:
                print("⚠️ 部分成功：模型有轻微改善")
            else:
                print("❌ 训练效果不明显")
        else:
            print("❌ 最终推理失败")
        
        if len(training_history['losses']) > 0:
            print(f"最终损失: {training_history['losses'][-1]:.4f}")
            print(f"损失变化: {training_history['losses'][0]:.4f} → {training_history['losses'][-1]:.4f}")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
