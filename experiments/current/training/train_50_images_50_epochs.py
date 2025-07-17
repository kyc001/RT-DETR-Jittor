#!/usr/bin/env python3
"""
基于ultimate_sanity_check.py的方法，训练50张图像50次
"""

import os
import sys
import json
import numpy as np
from PIL import Image
from datetime import datetime

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt
from jittor import nn

# 设置Jittor
jt.flags.use_cuda = 1

def load_train_data():
    """加载训练数据（50张图像）"""
    print("🔄 加载训练数据...")
    
    # 数据路径
    data_dir = '/home/kyc/project/RT-DETR/data/coco2017_50'
    images_dir = os.path.join(data_dir, "train2017")
    annotations_file = os.path.join(data_dir, "annotations", "instances_train2017.json")
    
    # 加载标注
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # 构建图像ID到标注的映射
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # 准备训练数据
    train_data = []
    
    for img_info in coco_data['images']:
        image_id = img_info['id']
        image_path = os.path.join(images_dir, img_info['file_name'])
        
        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            
            # 调整图像大小到640x640
            image_resized = image.resize((640, 640))
            image_array = np.array(image_resized).astype(np.float32) / 255.0
            image_tensor = jt.array(image_array.transpose(2, 0, 1)).unsqueeze(0)
            
            # 获取标注
            annotations = image_annotations.get(image_id, [])
            if not annotations:
                continue
            
            # 处理标注
            boxes = []
            labels = []
            
            for ann in annotations:
                # COCO格式: [x, y, width, height] -> 归一化的中心点格式
                x, y, w, h = ann['bbox']
                
                # 转换为归一化坐标
                cx = (x + w/2) / original_size[0]
                cy = (y + h/2) / original_size[1]
                nw = w / original_size[0]
                nh = h / original_size[1]
                
                boxes.append([cx, cy, nw, nh])
                
                # 类别映射（COCO ID -> 0-based索引）
                category_id = ann['category_id']
                if category_id == 1:
                    mapped_label = 0  # person
                elif category_id == 3:
                    mapped_label = 2  # car
                elif category_id == 27:
                    mapped_label = 26  # backpack
                elif category_id == 33:
                    mapped_label = 32  # suitcase
                elif category_id == 84:
                    mapped_label = 83  # book
                else:
                    mapped_label = category_id - 1  # 其他类别减1
                
                labels.append(mapped_label)
            
            if boxes:
                target = {
                    'boxes': jt.array(boxes),
                    'labels': jt.array(labels)
                }
                
                train_data.append({
                    'image_tensor': image_tensor,
                    'target': target,
                    'image_id': image_id,
                    'file_name': img_info['file_name']
                })
                
        except Exception as e:
            print(f"⚠️ 跳过图像 {img_info['file_name']}: {e}")
            continue
    
    print(f"✅ 加载完成: {len(train_data)}张训练图像")
    return train_data

def create_model():
    """创建模型（完全按照ultimate_sanity_check.py）"""
    print("🔄 创建模型...")
    
    try:
        from jittor_rt_detr.src.nn.backbone.resnet import ResNet50
        from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
        from jittor_rt_detr.src.nn.criterion.rtdetr_criterion import build_criterion
        
        # 创建模型
        backbone = ResNet50(pretrained=False)
        transformer = RTDETRTransformer(
            num_classes=80,
            hidden_dim=256,
            num_queries=300,
            feat_channels=[256, 512, 1024, 2048]
        )
        criterion = build_criterion(num_classes=80)
        
        print(f"✅ 模型创建成功")
        
        # 检查参数数量
        backbone_params = sum(p.numel() for p in backbone.parameters())
        transformer_params = sum(p.numel() for p in transformer.parameters())
        
        print(f"📊 模型参数:")
        print(f"   Backbone参数: {backbone_params:,}")
        print(f"   Transformer参数: {transformer_params:,}")
        print(f"   总参数: {backbone_params + transformer_params:,}")
        
        return backbone, transformer, criterion
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return None, None, None

def fix_batchnorm(module):
    """修复BatchNorm（完全按照ultimate_sanity_check.py）"""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()
            # 确保BatchNorm参数可训练
            if hasattr(m, 'weight') and m.weight is not None:
                m.weight.requires_grad = True
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = True

def train_model(backbone, transformer, criterion, train_data, num_epochs=50):
    """训练模型（基于ultimate_sanity_check.py的方法）"""
    print(f"\n🚀 开始训练 {len(train_data)} 张图像，{num_epochs} 轮")
    print("=" * 60)
    
    try:
        # 设置模型为训练模式
        backbone.train()
        transformer.train()
        
        # 修复BatchNorm
        fix_batchnorm(backbone)
        fix_batchnorm(transformer)
        
        # 收集所有需要梯度的参数
        all_params = []
        for module in [backbone, transformer]:
            for param in module.parameters():
                if param.requires_grad:
                    all_params.append(param)
        
        # 创建优化器
        optimizer = jt.optim.Adam(all_params, lr=1e-4, weight_decay=0)
        
        # 计算可训练参数的总元素数
        trainable_elements = sum(p.numel() for p in all_params)

        print(f"📊 训练配置:")
        print(f"   可训练参数张量数: {len(all_params)}")
        print(f"   可训练参数元素数: {trainable_elements:,}")
        print(f"   学习率: 1e-4")
        print(f"   优化器: Adam")
        
        # 训练循环
        all_losses = []
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # 确保模型在训练模式
            backbone.train()
            transformer.train()
            
            for i, data in enumerate(train_data):
                img_tensor = data['image_tensor']
                target = data['target']
                
                # 前向传播
                feats = backbone(img_tensor)
                outputs = transformer(feats)
                
                # 损失计算
                loss_dict = criterion(outputs, [target])
                total_loss = sum(loss_dict.values())
                
                # 反向传播和参数更新
                optimizer.step(total_loss)
                
                epoch_losses.append(total_loss.numpy().item())
            
            # 记录平均损失
            avg_loss = np.mean(epoch_losses)
            all_losses.append(avg_loss)
            
            # 打印进度
            if epoch % 10 == 0 or epoch < 5 or epoch >= num_epochs - 5:
                print(f"   Epoch {epoch+1:2d}/{num_epochs}: 平均损失 = {avg_loss:.4f}")
        
        print(f"\n✅ 训练完成!")
        print(f"   初始损失: {all_losses[0]:.4f}")
        print(f"   最终损失: {all_losses[-1]:.4f}")
        print(f"   损失下降: {all_losses[0] - all_losses[-1]:.4f}")
        
        return backbone, transformer, all_losses
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def save_model(backbone, transformer, losses, save_path):
    """保存训练好的模型"""
    print(f"💾 保存模型到: {save_path}")
    
    try:
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 准备检查点
        checkpoint = {
            'backbone_state_dict': backbone.state_dict(),
            'transformer_state_dict': transformer.state_dict(),
            'epoch': len(losses),
            'loss': losses[-1] if losses else 0.0,
            'timestamp': datetime.now().isoformat(),
            'training_losses': losses,
            'additional_info': {
                'training_method': 'ultimate_sanity_check_style',
                'num_images': 50,
                'num_epochs': len(losses),
                'learning_rate': 1e-4,
                'optimizer': 'Adam'
            }
        }
        
        # 保存
        jt.save(checkpoint, save_path)
        
        print(f"✅ 模型保存成功")
        print(f"   训练轮数: {len(losses)}")
        print(f"   最终损失: {losses[-1]:.4f}")
        
    except Exception as e:
        print(f"❌ 模型保存失败: {e}")

def main():
    print("🧪 RT-DETR 50张图像50轮训练")
    print("基于ultimate_sanity_check.py的方法")
    print("=" * 60)
    
    # 1. 加载训练数据
    train_data = load_train_data()
    if not train_data:
        print("❌ 无训练数据，退出")
        return
    
    # 2. 创建模型
    backbone, transformer, criterion = create_model()
    if backbone is None:
        print("❌ 模型创建失败，退出")
        return
    
    # 3. 训练模型
    trained_backbone, trained_transformer, losses = train_model(
        backbone, transformer, criterion, train_data, num_epochs=50
    )
    
    if trained_backbone is None:
        print("❌ 训练失败，退出")
        return
    
    # 4. 保存模型
    save_path = '/home/kyc/project/RT-DETR/results/sanity_check_training/rtdetr_50img_50epoch.pkl'
    save_model(trained_backbone, trained_transformer, losses, save_path)
    
    print("\n🎉 训练完成！")
    print(f"模型已保存到: {save_path}")

if __name__ == "__main__":
    main()
