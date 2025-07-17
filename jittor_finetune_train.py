#!/usr/bin/env python3
"""
Jittor版本RT-DETR微调训练脚本
使用预训练ResNet50权重进行微调
"""

import os
import sys
import json
import time
import numpy as np
from PIL import Image
import jittor as jt
import jittor.nn as nn

# Jittor可以直接加载.pth文件，不需要torch

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

# 导入Jittor版本的模型
from jittor_rt_detr.src.nn.backbone.resnet import ResNet50
from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
from jittor_rt_detr.src.nn.criterion.rtdetr_criterion import build_criterion

# Jittor设置
jt.flags.use_cuda = 1
jt.set_global_seed(42)
jt.flags.auto_mixed_precision_level = 0

print(f"🔥 Jittor版本: {jt.__version__}")
print(f"🔥 CUDA可用: {jt.flags.use_cuda}")

class COCODataset:
    """COCO数据集加载器"""
    def __init__(self, img_dir, ann_file):
        self.img_dir = img_dir
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.img_id_to_filename = {img['id']: img['file_name'] for img in self.coco_data['images']}
        self.img_ids = list(self.img_id_to_filename.keys())
        print(f"📊 加载了 {len(self.img_ids)} 张图像")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        filename = self.img_id_to_filename[img_id]
        
        # 加载图像
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert('RGB')
        original_width, original_height = image.size
        
        # 调整图像大小到640x640
        image_resized = image.resize((640, 640), Image.LANCZOS)
        img_array = np.array(image_resized).astype(np.float32) / 255.0
        img_tensor = jt.array(img_array.transpose(2, 0, 1)).float32()
        
        # 获取标注
        annotations = []
        labels = []
        
        for ann in self.coco_data['annotations']:
            if ann['image_id'] == img_id:
                x, y, w, h = ann['bbox']
                category_id = ann['category_id']
                
                # 归一化坐标
                x1, y1 = x / original_width, y / original_height
                x2, y2 = (x + w) / original_width, (y + h) / original_height
                
                if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= 1 and y2 <= 1:
                    annotations.append([x1, y1, x2, y2])
                    
                    # COCO类别ID转换
                    if category_id == 1:  # person
                        mapped_label = 0
                    elif category_id == 3:  # car
                        mapped_label = 2
                    elif category_id == 27:  # backpack
                        mapped_label = 26
                    elif category_id == 33:  # suitcase
                        mapped_label = 32
                    elif category_id == 84:  # book
                        mapped_label = 83
                    else:
                        mapped_label = category_id - 1
                    
                    labels.append(mapped_label)
        
        # 创建目标
        if annotations:
            target = {
                'boxes': jt.array(annotations).float32(),
                'labels': jt.array(labels).int64()
            }
        else:
            target = {
                'boxes': jt.zeros((0, 4)).float32(),
                'labels': jt.zeros((0,)).int64()
            }
        
        return img_tensor, target

class RTDETRModel(nn.Module):
    """RT-DETR模型 - 支持预训练权重加载"""
    def __init__(self, num_classes=80, pretrained_path=None):
        super().__init__()
        # 使用Jittor的预训练权重，更简单可靠
        self.backbone = ResNet50(pretrained=True)  # 使用Jittor的预训练权重
        self.transformer = RTDETRTransformer(
            num_classes=num_classes,
            hidden_dim=256,
            num_queries=300,
            feat_channels=[256, 512, 1024, 2048]  # 修复通道数匹配问题
        )

        print("✅ 使用Jittor内置预训练权重")
    

    
    def execute(self, x, targets=None):
        features = self.backbone(x)
        return self.transformer(features, targets)

def train_jittor_finetune():
    """Jittor微调训练"""
    print("🧪 Jittor版本RT-DETR微调训练 - 50张图像50轮")
    print("使用预训练ResNet50权重")
    print("=" * 60)
    
    # 数据路径
    img_dir = "/home/kyc/project/RT-DETR/data/coco2017_50/train2017"
    ann_file = "/home/kyc/project/RT-DETR/data/coco2017_50/annotations/instances_train2017.json"
    
    # 创建数据集
    print("🔄 加载训练数据...")
    dataset = COCODataset(img_dir, ann_file)
    print(f"✅ 数据加载完成: {len(dataset)}张训练图像")
    
    # 创建模型
    print("🔄 创建模型...")
    model = RTDETRModel(num_classes=80)
    criterion = build_criterion(num_classes=80)
    print("✅ 模型创建成功")
    
    # 统计参数
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    transformer_params = sum(p.numel() for p in model.transformer.parameters())
    total_params = backbone_params + transformer_params
    
    print("📊 模型参数:")
    print(f"   Backbone参数: {backbone_params:,}")
    print(f"   Transformer参数: {transformer_params:,}")
    print(f"   总参数: {total_params:,}")
    
    # 创建优化器 - 使用较小的学习率进行微调
    optimizer = jt.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    
    # 训练配置
    num_epochs = 50
    losses = []
    
    print(f"\n🚀 开始微调训练 {len(dataset)} 张图像，{num_epochs} 轮")
    print("=" * 60)
    
    # 开始训练
    model.train()
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for idx in range(len(dataset)):
            images, targets = dataset[idx]
            
            # 添加batch维度
            images = images.unsqueeze(0)
            targets = [targets]
            
            # 前向传播
            outputs = model(images, targets)
            
            # 损失计算
            loss_dict = criterion(outputs, targets)
            total_loss = sum(loss_dict.values())
            
            # 反向传播 - 使用Jittor的优化器API
            optimizer.step(total_loss)
            
            epoch_losses.append(float(total_loss.data))
        
        # 计算平均损失
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        # 每10轮或最后几轮打印进度
        if (epoch + 1) % 10 == 0 or epoch < 5 or epoch >= num_epochs - 5:
            print(f"   Epoch {epoch + 1:2d}/{num_epochs}: 平均损失 = {avg_loss:.4f}")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n✅ 微调训练完成!")
    print(f"   初始损失: {losses[0]:.4f}")
    print(f"   最终损失: {losses[-1]:.4f}")
    print(f"   损失下降: {losses[0] - losses[-1]:.4f}")
    print(f"   训练时间: {training_time:.1f}秒")
    
    # 保存模型
    save_dir = "/home/kyc/project/RT-DETR/results/jittor_finetune"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "rtdetr_jittor_finetune_50img_50epoch.pkl")
    
    model.save(save_path)
    
    # 保存训练结果
    results = {
        'losses': losses,
        'training_time': training_time,
        'num_epochs': num_epochs,
        'final_loss': losses[-1],
        'loss_reduction': losses[0] - losses[-1]
    }
    
    import pickle
    with open(os.path.join(save_dir, "training_results.pkl"), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"💾 模型保存到: {save_path}")
    print(f"📊 训练结果保存到: {os.path.join(save_dir, 'training_results.pkl')}")
    
    print(f"\n🎉 Jittor微调训练完成！")
    return losses, training_time

if __name__ == "__main__":
    train_jittor_finetune()
