#!/usr/bin/env python3
"""
RT-DETR优化版50张照片训练脚本
专门解决数据类型不一致问题，确保训练稳定性
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt
from jittor_rt_detr.src.nn.rtdetr_complete_pytorch_aligned import build_rtdetr_complete
from jittor_rt_detr.src.nn.dtype_safe_loss import build_dtype_safe_criterion

# 设置Jittor为float32模式
jt.flags.use_cuda = 1
jt.set_global_seed(42)

def force_float32_consistency():
    """强制Jittor使用float32一致性"""
    # 设置默认数据类型
    jt.flags.auto_mixed_precision_level = 0
    print("✅ 强制float32一致性设置完成")

def load_coco_data():
    """加载COCO数据集"""
    data_dir = "data/coco2017_50/train2017"
    ann_file = "data/coco2017_50/annotations/instances_train2017.json"
    
    print(f">>> 加载COCO数据集")
    print(f"数据目录: {data_dir}")
    print(f"标注文件: {ann_file}")
    
    # 加载标注
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # 统计信息
    print(f"图片数量: {len(coco_data['images'])}")
    print(f"标注数量: {len(coco_data['annotations'])}")
    print(f"类别数量: {len(coco_data['categories'])}")
    
    return coco_data, data_dir

def preprocess_image(image_path, target_size=640):
    """预处理单张图片"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Resize
    image_resized = image.resize((target_size, target_size))
    
    # 转换为numpy数组并归一化
    img_array = np.array(image_resized).astype(np.float32) / 255.0
    
    # 转换为CHW格式
    img_array = img_array.transpose(2, 0, 1)
    
    # 转换为Jittor tensor，强制float32
    img_tensor = jt.array(img_array, dtype=jt.float32)
    
    return img_tensor, original_size

def create_targets_for_image(image_id, annotations, categories, original_size):
    """为单张图片创建训练目标"""
    # 找到该图片的所有标注
    image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]
    
    if not image_annotations:
        return None
    
    # 创建类别映射
    cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(categories)}
    
    boxes = []
    labels = []
    
    for ann in image_annotations:
        # 获取边界框 (COCO格式: x, y, width, height)
        x, y, w, h = ann['bbox']
        
        # 转换为中心点格式并归一化
        cx = (x + w / 2) / original_size[0]
        cy = (y + h / 2) / original_size[1]
        w_norm = w / original_size[0]
        h_norm = h / original_size[1]
        
        boxes.append([cx, cy, w_norm, h_norm])
        labels.append(cat_id_to_idx[ann['category_id']])
    
    targets = {
        'boxes': jt.array(boxes, dtype=jt.float32),
        'labels': jt.array(labels, dtype=jt.int64)
    }
    
    return targets

def create_dataloader(coco_data, data_dir, batch_size=2, max_images=50):
    """创建数据加载器"""
    images = coco_data['images'][:max_images]
    annotations = coco_data['annotations']
    categories = coco_data['categories']
    
    print(f">>> 创建数据加载器")
    print(f"批次大小: {batch_size}")
    print(f"最大图片数: {max_images}")
    print(f"实际图片数: {len(images)}")
    
    # 预处理所有数据
    processed_data = []
    
    for img_info in tqdm(images, desc="预处理图片"):
        image_path = os.path.join(data_dir, img_info['file_name'])
        
        if not os.path.exists(image_path):
            continue
        
        # 预处理图片
        img_tensor, original_size = preprocess_image(image_path)
        
        # 创建目标
        targets = create_targets_for_image(img_info['id'], annotations, categories, original_size)
        
        if targets is not None:
            processed_data.append((img_tensor, targets))
    
    print(f"成功预处理: {len(processed_data)} 张图片")
    
    # 创建批次
    batches = []
    for i in range(0, len(processed_data), batch_size):
        batch_data = processed_data[i:i+batch_size]
        
        # 组装批次
        batch_images = jt.stack([item[0] for item in batch_data])
        batch_targets = [item[1] for item in batch_data]
        
        batches.append((batch_images, batch_targets))
    
    print(f"创建批次数: {len(batches)}")
    return batches, len(categories)

def optimized_training(model, criterion, dataloader, num_epochs=30, learning_rate=1e-4):
    """优化的训练流程"""
    print(f"\n>>> 开始优化训练 (轮数: {num_epochs})")
    
    # 优化器
    optimizer = jt.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = jt.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    training_history = {
        'epoch_losses': [],
        'best_loss': float('inf'),
        'successful_epochs': 0
    }
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        try:
            for batch_idx, (images, targets) in enumerate(dataloader):
                # 确保数据类型一致
                images = images.float32()
                for target in targets:
                    target['boxes'] = target['boxes'].float32()
                    target['labels'] = target['labels'].int64()
                
                # 前向传播
                outputs = model(images, targets)
                
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
                epoch_losses.append(loss_value)
                
                if batch_idx % 5 == 0:
                    print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}: Loss = {loss_value:.4f}")
            
            # 计算平均损失
            avg_loss = np.mean(epoch_losses)
            training_history['epoch_losses'].append(avg_loss)
            
            if avg_loss < training_history['best_loss']:
                training_history['best_loss'] = avg_loss
            
            training_history['successful_epochs'] += 1
            
            print(f"✅ Epoch {epoch+1}/{num_epochs}: Avg Loss = {avg_loss:.4f} (Best: {training_history['best_loss']:.4f})")
            
            # 更新学习率
            scheduler.step()
            
        except Exception as e:
            print(f"❌ Epoch {epoch+1} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n=== 训练完成 ===")
    print(f"成功轮数: {training_history['successful_epochs']}/{num_epochs}")
    print(f"最佳损失: {training_history['best_loss']:.4f}")
    
    return model, training_history

def save_model(model, save_path, training_history):
    """安全保存模型"""
    print(f"\n>>> 保存模型到: {save_path}")
    
    try:
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存模型状态
        model_state = {
            'model_state_dict': model.state_dict(),
            'training_history': training_history,
            'model_config': {
                'num_classes': model.num_classes,
                'hidden_dim': model.hidden_dim,
                'num_queries': model.num_queries
            }
        }
        
        # 使用Jittor的保存方法
        jt.save(model_state, save_path)
        print("✅ 模型保存成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型保存失败: {e}")
        return False

def plot_training_history(training_history, save_path="results/training_history.png"):
    """绘制训练历史"""
    if not training_history['epoch_losses']:
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(training_history['epoch_losses'], 'b-', linewidth=2, label='Training Loss')
    plt.axhline(y=training_history['best_loss'], color='r', linestyle='--', label=f'Best Loss: {training_history["best_loss"]:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('RT-DETR Training History (50 Images)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 训练历史图保存到: {save_path}")

def main():
    print("=" * 80)
    print("===        RT-DETR优化版50张照片训练        ===")
    print("===      解决数据类型不一致问题，确保稳定性      ===")
    print("=" * 80)
    
    try:
        # 1. 强制float32一致性
        force_float32_consistency()
        
        # 2. 加载数据
        coco_data, data_dir = load_coco_data()
        
        # 3. 创建数据加载器
        dataloader, num_classes = create_dataloader(coco_data, data_dir, batch_size=2, max_images=50)
        
        # 4. 创建模型
        print(f"\n>>> 创建RT-DETR模型 (类别数: {num_classes})")
        model = build_rtdetr_complete(num_classes=num_classes, hidden_dim=256, num_queries=300)
        print("✅ 模型创建成功")
        
        # 5. 创建原始损失函数（已修复数据类型问题）
        from jittor_rt_detr.src.nn.loss_pytorch_aligned import build_criterion
        criterion = build_criterion(num_classes)
        print("✅ 修复后的损失函数创建成功")
        
        # 6. 开始训练
        model, training_history = optimized_training(
            model, criterion, dataloader,
            num_epochs=5, learning_rate=1e-4
        )
        
        # 7. 保存模型
        save_path = "checkpoints/optimized_50_images_model.pkl"
        save_success = save_model(model, save_path, training_history)
        
        # 8. 绘制训练历史
        plot_training_history(training_history)
        
        # 9. 最终总结
        print(f"\n" + "=" * 80)
        print("🎯 优化版50张照片训练结果:")
        print("=" * 80)
        
        if training_history['successful_epochs'] >= 20:
            print("🎉 训练完全成功！")
            print(f"  ✅ 成功完成 {training_history['successful_epochs']} 轮训练")
            print(f"  ✅ 最佳损失: {training_history['best_loss']:.4f}")
            print(f"  ✅ 模型保存: {'成功' if save_success else '失败'}")
            print("  ✅ RT-DETR Jittor版本训练稳定，可用于生产！")
        else:
            print("⚠️ 训练部分成功")
            print(f"  ⚠️ 完成 {training_history['successful_epochs']} 轮训练")
            print(f"  ⚠️ 最佳损失: {training_history['best_loss']:.4f}")
            print("  ⚠️ 建议调整超参数或检查数据")
        
        print("=" * 80)
        
        return training_history['successful_epochs'] >= 20
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
