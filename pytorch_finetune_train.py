#!/usr/bin/env python3
"""
PyTorch版本RT-DETR微调训练脚本
使用预训练ResNet50权重进行微调
"""

import os
import sys
import json
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 PyTorch版本: {torch.__version__}")
print(f"🔥 使用设备: {device}")

class COCODataset(Dataset):
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
        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
        
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
                'boxes': torch.tensor(annotations, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.long)
            }
        else:
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.long)
            }
        
        return img_tensor, target

class PretrainedResNet50(nn.Module):
    """使用预训练权重的ResNet50骨干网络"""
    def __init__(self, pretrained_path=None):
        super().__init__()
        # 加载预训练的ResNet50
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"🔄 加载预训练权重: {pretrained_path}")
            self.resnet = models.resnet50(pretrained=False)
            self.resnet.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
            print("✅ 预训练权重加载成功")
        else:
            print("🔄 使用在线预训练权重")
            self.resnet = models.resnet50(pretrained=True)
        
        # 移除最后的分类层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # 添加特征提取层
        self.layer1 = nn.Sequential(*list(self.resnet.children())[:5])  # 到layer1
        self.layer2 = nn.Sequential(*list(self.resnet.children())[5:6])  # layer2
        self.layer3 = nn.Sequential(*list(self.resnet.children())[6:7])  # layer3
        self.layer4 = nn.Sequential(*list(self.resnet.children())[7:8])  # layer4
        
    def forward(self, x):
        # 提取多尺度特征
        x = self.layer1(x)  # 1/4
        feat1 = self.layer2(x)  # 1/8, 512通道
        feat2 = self.layer3(feat1)  # 1/16, 1024通道
        feat3 = self.layer4(feat2)  # 1/32, 2048通道
        
        return [feat1, feat2, feat3]

class SimpleEncoder(nn.Module):
    """简化的编码器"""
    def __init__(self):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(1024, 256, 1),
            nn.Conv2d(2048, 256, 1)
        ])
        
    def forward(self, feats):
        return [proj(feat) for proj, feat in zip(self.proj, feats)]

class SimpleDecoder(nn.Module):
    """简化的解码器"""
    def __init__(self, num_classes=80, num_queries=100):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = 256
        self.num_classes = num_classes
        
        # 查询嵌入
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim)
        
        # 简化的transformer层
        self.self_attn = nn.MultiheadAttention(self.hidden_dim, 8, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(self.hidden_dim, 8, batch_first=True)
        
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.hidden_dim)
        )
        
        # 输出头
        self.class_embed = nn.Linear(self.hidden_dim, num_classes)
        self.bbox_embed = nn.Linear(self.hidden_dim, 4)
        
    def forward(self, feats, targets=None):
        batch_size = feats[0].shape[0]
        
        # 处理特征
        feat_flat = []
        for feat in feats:
            b, c, h, w = feat.shape
            feat_flat.append(feat.flatten(2).transpose(1, 2))  # [B, HW, C]
        
        memory = torch.cat(feat_flat, dim=1)  # [B, total_HW, C]
        
        # 查询嵌入
        queries = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 简化的transformer解码
        # Self-attention
        queries2, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + queries2)
        
        # Cross-attention
        queries2, _ = self.cross_attn(queries, memory, memory)
        queries = self.norm2(queries + queries2)
        
        # FFN
        queries2 = self.ffn(queries)
        queries = self.norm3(queries + queries2)
        
        # 输出
        outputs_class = self.class_embed(queries)
        outputs_coord = self.bbox_embed(queries).sigmoid()
        
        return {
            'pred_logits': outputs_class,
            'pred_boxes': outputs_coord
        }

class RTDETRModel(nn.Module):
    """完整的RT-DETR模型"""
    def __init__(self, num_classes=80, pretrained_path=None):
        super().__init__()
        self.backbone = PretrainedResNet50(pretrained_path)
        self.encoder = SimpleEncoder()
        self.decoder = SimpleDecoder(num_classes)
        
    def forward(self, x, targets=None):
        feats = self.backbone(x)
        feats = self.encoder(feats)
        return self.decoder(feats, targets)

class SimpleCriterion(nn.Module):
    """简化的损失函数"""
    def __init__(self, num_classes=80, alpha=0.25, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        
    def focal_loss(self, pred_logits, targets):
        """简化的Focal Loss计算"""
        # 简化：直接计算所有查询的交叉熵损失
        batch_size, num_queries, num_classes = pred_logits.shape

        # 创建全背景类的目标
        target_classes = torch.full((batch_size, num_queries), num_classes - 1,
                                  dtype=torch.long, device=pred_logits.device)

        # 计算交叉熵损失
        ce_loss = F.cross_entropy(pred_logits.view(-1, num_classes),
                                target_classes.view(-1), reduction='mean')

        return ce_loss
    
    def bbox_loss(self, pred_boxes, targets):
        """边界框损失"""
        if len(targets) == 0 or len(targets[0]['boxes']) == 0:
            return torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
        
        num_targets = min(len(targets[0]['boxes']), pred_boxes.shape[1])
        pred_subset = pred_boxes[0, :num_targets]
        target_subset = targets[0]['boxes'][:num_targets]
        
        return F.l1_loss(pred_subset, target_subset)
    
    def forward(self, outputs, targets):
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        loss_focal = self.focal_loss(pred_logits, targets)
        loss_bbox = self.bbox_loss(pred_boxes, targets)
        
        return {
            'loss_focal': loss_focal,
            'loss_bbox': loss_bbox * 5.0,
            'loss_giou': torch.tensor(0.0, device=pred_logits.device, requires_grad=True)
        }

def train_pytorch_finetune():
    """PyTorch微调训练"""
    print("🧪 PyTorch版本RT-DETR微调训练 - 50张图像50轮")
    print("使用预训练ResNet50权重")
    print("=" * 60)
    
    # 数据路径
    img_dir = "/home/kyc/project/RT-DETR/data/coco2017_50/train2017"
    ann_file = "/home/kyc/project/RT-DETR/data/coco2017_50/annotations/instances_train2017.json"
    pretrained_path = "/home/kyc/project/RT-DETR/pretrained_weights/resnet50_pytorch.pth"
    
    # 创建数据集
    print("🔄 加载训练数据...")
    dataset = COCODataset(img_dir, ann_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    print(f"✅ 数据加载完成: {len(dataset)}张训练图像")
    
    # 创建模型
    print("🔄 创建模型...")
    model = RTDETRModel(num_classes=80, pretrained_path=pretrained_path).to(device)
    criterion = SimpleCriterion(num_classes=80).to(device)
    print("✅ 模型创建成功")
    
    # 统计参数
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    total_params = backbone_params + encoder_params + decoder_params
    
    print("📊 模型参数:")
    print(f"   Backbone参数: {backbone_params:,}")
    print(f"   Encoder参数: {encoder_params:,}")
    print(f"   Decoder参数: {decoder_params:,}")
    print(f"   总参数: {total_params:,}")
    
    # 创建优化器 - 使用较小的学习率进行微调
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    
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
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in targets.items()}]
            
            # 前向传播
            outputs = model(images, targets)
            
            # 损失计算
            loss_dict = criterion(outputs, targets)
            total_loss = sum(loss_dict.values())
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_losses.append(total_loss.item())
        
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
    save_dir = "/home/kyc/project/RT-DETR/results/pytorch_finetune"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "rtdetr_pytorch_finetune_50img_50epoch.pth")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'training_time': training_time,
        'num_epochs': num_epochs,
        'final_loss': losses[-1],
        'loss_reduction': losses[0] - losses[-1]
    }, save_path)
    
    print(f"💾 模型保存到: {save_path}")
    print(f"\n🎉 PyTorch微调训练完成！")
    
    return losses, training_time

if __name__ == "__main__":
    train_pytorch_finetune()
