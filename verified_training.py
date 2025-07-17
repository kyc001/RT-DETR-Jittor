#!/usr/bin/env python3
"""
基于验证过的ultimate_sanity_check.py的正确训练脚本
使用经过验证的损失函数和训练方法
"""

import os
import sys
import json
import time
import numpy as np
from PIL import Image

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt
import jittor.nn as nn

# 设置Jittor
jt.flags.use_cuda = 1
jt.set_global_seed(42)
jt.flags.auto_mixed_precision_level = 0

class VerifiedCOCODataset:
    """基于验证过的代码的数据集加载器"""
    def __init__(self, img_dir, ann_file):
        self.img_dir = img_dir
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.img_id_to_filename = {img['id']: img['file_name'] for img in self.coco_data['images']}
        self.img_ids = list(self.img_id_to_filename.keys())
        
        # 数据增强：重复数据
        self.img_ids = self.img_ids * 3  # 3倍数据增强
        
        print(f"📊 验证数据集大小: {len(self.img_ids)} 张图像")

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
        
        # 获取标注 - 使用验证过的方法
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
                    
                    # 使用验证过的类别映射
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
                'boxes': jt.array(annotations, dtype=jt.float32),
                'labels': jt.array(labels, dtype=jt.int64)
            }
        else:
            target = {
                'boxes': jt.zeros((0, 4), dtype=jt.float32),
                'labels': jt.zeros((0,), dtype=jt.int64)
            }
        
        return img_tensor, target

def verified_training():
    """使用验证过的方法进行训练"""
    print("🎯 基于验证代码的RT-DETR训练")
    print("=" * 60)
    
    try:
        # 使用验证过的导入
        from jittor_rt_detr.src.nn.backbone.resnet import ResNet50
        from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
        from jittor_rt_detr.src.nn.criterion.rtdetr_criterion import build_criterion
        
        # 数据路径
        img_dir = "/home/kyc/project/RT-DETR/data/coco2017_50/train2017"
        ann_file = "/home/kyc/project/RT-DETR/data/coco2017_50/annotations/instances_train2017.json"
        
        # 创建数据集
        print("🔄 加载验证训练数据...")
        dataset = VerifiedCOCODataset(img_dir, ann_file)
        print(f"✅ 验证数据加载完成: {len(dataset)}张训练图像")
        
        # 创建模型 - 使用验证过的方法
        print("🔄 创建验证模型...")
        backbone = ResNet50(pretrained=False)
        transformer = RTDETRTransformer(
            num_classes=80,
            hidden_dim=256,
            num_queries=300,
            feat_channels=[256, 512, 1024, 2048]
        )
        criterion = build_criterion(num_classes=80)  # 使用验证过的损失函数！
        print("✅ 验证模型创建成功")
        
        # 统计参数
        backbone_params = sum(p.numel() for p in backbone.parameters())
        transformer_params = sum(p.numel() for p in transformer.parameters())
        total_params = backbone_params + transformer_params
        print(f"📊 模型参数:")
        print(f"   Backbone参数: {backbone_params:,}")
        print(f"   Transformer参数: {transformer_params:,}")
        print(f"   总参数: {total_params:,}")
        
        # 修复BatchNorm问题 - 使用验证过的方法
        def fix_batchnorm(module):
            for m in module.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.train()
                    if hasattr(m, 'weight') and m.weight is not None:
                        m.weight.requires_grad = True
                    if hasattr(m, 'bias') and m.bias is not None:
                        m.bias.requires_grad = True
        
        fix_batchnorm(backbone)
        fix_batchnorm(transformer)
        
        # 收集所有需要梯度的参数
        all_params = []
        for module in [backbone, transformer]:
            for param in module.parameters():
                if param.requires_grad:
                    all_params.append(param)
        
        # 创建优化器 - 使用验证过的设置
        optimizer = jt.optim.Adam(all_params, lr=1e-4, weight_decay=0)
        
        # 训练配置
        num_epochs = 100  # 增加训练轮数
        losses = []
        
        print(f"\n🚀 开始验证训练 {len(dataset)} 张图像，{num_epochs} 轮")
        print("=" * 60)
        
        # 开始训练
        backbone.train()
        transformer.train()
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # 随机打乱数据
            indices = np.random.permutation(len(dataset))
            
            for i in range(len(dataset)):
                idx = indices[i]
                img_tensor, target = dataset[idx]
                
                # 添加batch维度
                img_tensor = img_tensor.unsqueeze(0)
                targets = [target]
                
                # 前向传播 - 使用验证过的方法
                feats = backbone(img_tensor)
                outputs = transformer(feats)
                
                # 损失计算 - 使用验证过的损失函数
                loss_dict = criterion(outputs, targets)
                total_loss = sum(loss_dict.values())
                
                # 反向传播 - 使用验证过的方法
                optimizer.step(total_loss)
                
                epoch_losses.append(total_loss.item())
            
            # 计算平均损失
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            # 每10轮或最后几轮打印进度
            if (epoch + 1) % 10 == 0 or epoch < 5 or epoch >= num_epochs - 5:
                print(f"   Epoch {epoch + 1:3d}/{num_epochs}: 平均损失 = {avg_loss:.4f}")
            
            # 学习率衰减
            if (epoch + 1) % 30 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                print(f"   学习率衰减到: {optimizer.param_groups[0]['lr']:.6f}")
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\n✅ 验证训练完成!")
        print(f"   初始损失: {losses[0]:.4f}")
        print(f"   最终损失: {losses[-1]:.4f}")
        print(f"   损失下降: {losses[0] - losses[-1]:.4f}")
        print(f"   训练时间: {training_time:.1f}秒")
        
        # 保存模型
        save_dir = "/home/kyc/project/RT-DETR/results/verified_training"
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存backbone和transformer
        backbone_path = os.path.join(save_dir, "verified_backbone.pkl")
        transformer_path = os.path.join(save_dir, "verified_transformer.pkl")
        
        backbone.save(backbone_path)
        transformer.save(transformer_path)
        
        # 保存训练结果
        training_results = {
            'losses': losses,
            'training_time': training_time,
            'num_epochs': num_epochs,
            'final_loss': losses[-1],
            'loss_reduction': losses[0] - losses[-1],
            'total_params': total_params,
            'dataset_size': len(dataset)
        }
        
        import pickle
        with open(os.path.join(save_dir, "verified_training_results.pkl"), 'wb') as f:
            pickle.dump(training_results, f)
        
        print(f"💾 验证模型保存到:")
        print(f"   Backbone: {backbone_path}")
        print(f"   Transformer: {transformer_path}")
        print(f"📊 训练结果保存到: {save_dir}/verified_training_results.pkl")
        print(f"\n🎉 验证训练完成！")
        
        return True, losses, training_time, total_params
        
    except Exception as e:
        print(f"❌ 验证训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False, [], 0, 0

if __name__ == "__main__":
    success, losses, training_time, total_params = verified_training()
    
    if success:
        print(f"\n🎯 训练成功总结:")
        print(f"   损失下降: {(losses[0] - losses[-1])/losses[0]*100:.1f}%")
        print(f"   训练效率: {(losses[0] - losses[-1])/training_time*60:.4f} 损失下降/分钟")
        print(f"   参数效率: {(losses[0] - losses[-1])/(total_params/1e6):.4f} 损失下降/百万参数")
    else:
        print(f"\n❌ 训练失败，需要检查问题")
