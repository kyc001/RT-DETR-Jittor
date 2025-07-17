#!/usr/bin/env python3
"""
与Jittor版本完全对齐的PyTorch RT-DETR训练脚本
使用相同的模型架构、参数配置和训练流程
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
    """COCO数据集加载器 - 与Jittor版本完全一致"""
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
        
        # 获取标注 - 与Jittor版本完全一致
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
                    
                    # COCO类别ID转换 - 与Jittor版本完全一致
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

class AlignedResNet50(nn.Module):
    """与Jittor版本对齐的ResNet50骨干网络"""
    def __init__(self, pretrained=True):
        super().__init__()
        # 使用预训练的ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # 提取各层
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 输出: 256通道
        self.layer2 = resnet.layer2  # 输出: 512通道
        self.layer3 = resnet.layer3  # 输出: 1024通道
        self.layer4 = resnet.layer4  # 输出: 2048通道
        
        print("✅ 使用PyTorch预训练ResNet50权重")
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        c2 = self.layer1(x)   # 1/4, 256通道
        c3 = self.layer2(c2)  # 1/8, 512通道
        c4 = self.layer3(c3)  # 1/16, 1024通道
        c5 = self.layer4(c4)  # 1/32, 2048通道
        
        return [c2, c3, c4, c5]  # 返回多尺度特征，与Jittor版本对齐

class AlignedTransformer(nn.Module):
    """与Jittor版本对齐的Transformer（内存优化版）"""
    def __init__(self, num_classes=80, hidden_dim=256, num_queries=100):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        # 特征投影层 - 与Jittor版本对齐
        self.input_proj = nn.ModuleList([
            nn.Conv2d(256, hidden_dim, 1),   # c2
            nn.Conv2d(512, hidden_dim, 1),   # c3
            nn.Conv2d(1024, hidden_dim, 1),  # c4
            nn.Conv2d(2048, hidden_dim, 1),  # c5
        ])
        
        # 查询嵌入 - 与Jittor版本对齐
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Transformer层 - 简化以适应内存限制
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=512,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=512,
            dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        # 输出头 - 与Jittor版本对齐
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = nn.Linear(hidden_dim, 4)
        
        # 位置编码 - 动态生成
        self.register_buffer('pos_embed_cache', torch.zeros(1, 50000, hidden_dim))
        
    def forward(self, features, targets=None):
        batch_size = features[0].shape[0]
        
        # 特征投影和展平
        srcs = []
        for i, feat in enumerate(features):
            src = self.input_proj[i](feat)
            b, c, h, w = src.shape
            src = src.flatten(2).transpose(1, 2)  # [B, HW, C]
            srcs.append(src)
        
        # 拼接多尺度特征
        src = torch.cat(srcs, dim=1)  # [B, total_HW, C]

        # 简化：不使用位置编码，避免维度问题
        # seq_len = src.shape[1]
        # 如果需要位置编码，可以动态生成
        
        # 编码器
        memory = self.encoder(src)
        
        # 查询嵌入
        tgt = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 解码器
        hs = self.decoder(tgt, memory)
        
        # 输出头
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        return {
            'pred_logits': outputs_class,
            'pred_boxes': outputs_coord
        }

class AlignedRTDETRModel(nn.Module):
    """与Jittor版本完全对齐的RT-DETR模型"""
    def __init__(self, num_classes=80):
        super().__init__()
        self.backbone = AlignedResNet50(pretrained=True)
        self.transformer = AlignedTransformer(
            num_classes=num_classes,
            hidden_dim=256,
            num_queries=100  # 减少查询数量以适应内存
        )
        
    def forward(self, x, targets=None):
        features = self.backbone(x)
        return self.transformer(features, targets)

class AlignedCriterion(nn.Module):
    """与Jittor版本对齐的损失函数"""
    def __init__(self, num_classes=80, alpha=0.25, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        
    def focal_loss(self, pred_logits, targets):
        """Focal Loss计算"""
        batch_size, num_queries, num_classes = pred_logits.shape
        
        # 创建目标类别
        target_classes = torch.full((batch_size, num_queries), num_classes - 1, 
                                  dtype=torch.long, device=pred_logits.device)
        
        # 简化的目标分配
        if len(targets) > 0 and len(targets[0]['labels']) > 0:
            num_targets = min(len(targets[0]['labels']), num_queries // 4)
            if num_targets > 0:
                target_classes[0, :num_targets] = targets[0]['labels'][:num_targets]
        
        # 计算focal loss
        ce_loss = F.cross_entropy(pred_logits.view(-1, num_classes), 
                                target_classes.view(-1), reduction='none')
        ce_loss = ce_loss.view(batch_size, num_queries)
        
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        return focal_loss.mean()
    
    def bbox_loss(self, pred_boxes, targets):
        """边界框损失"""
        if len(targets) == 0 or len(targets[0]['boxes']) == 0:
            return torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
        
        num_targets = min(len(targets[0]['boxes']), pred_boxes.shape[1])
        if num_targets == 0:
            return torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
            
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

def train_aligned_pytorch():
    """与Jittor版本对齐的PyTorch训练"""
    print("🧪 与Jittor对齐的PyTorch版本RT-DETR训练")
    print("使用相同的模型架构和配置")
    print("=" * 60)
    
    # 数据路径
    img_dir = "/home/kyc/project/RT-DETR/data/coco2017_50/train2017"
    ann_file = "/home/kyc/project/RT-DETR/data/coco2017_50/annotations/instances_train2017.json"
    
    # 创建数据集
    print("🔄 加载训练数据...")
    dataset = COCODataset(img_dir, ann_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    print(f"✅ 数据加载完成: {len(dataset)}张训练图像")
    
    # 创建模型 - 与Jittor版本对齐
    print("🔄 创建模型...")
    model = AlignedRTDETRModel(num_classes=80).to(device)
    criterion = AlignedCriterion(num_classes=80).to(device)
    print("✅ 模型创建成功")
    
    # 统计参数
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    transformer_params = sum(p.numel() for p in model.transformer.parameters())
    total_params = backbone_params + transformer_params
    
    print("📊 模型参数:")
    print(f"   Backbone参数: {backbone_params:,}")
    print(f"   Transformer参数: {transformer_params:,}")
    print(f"   总参数: {total_params:,}")
    
    # 创建优化器 - 与Jittor版本对齐
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    
    # 训练配置
    num_epochs = 50
    losses = []
    
    print(f"\n🚀 开始对齐训练 {len(dataset)} 张图像，{num_epochs} 轮")
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
    
    print(f"\n✅ 对齐训练完成!")
    print(f"   初始损失: {losses[0]:.4f}")
    print(f"   最终损失: {losses[-1]:.4f}")
    print(f"   损失下降: {losses[0] - losses[-1]:.4f}")
    print(f"   训练时间: {training_time:.1f}秒")
    
    # 保存模型
    save_dir = "/home/kyc/project/RT-DETR/results/pytorch_aligned"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "rtdetr_pytorch_aligned_50img_50epoch.pth")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'training_time': training_time,
        'num_epochs': num_epochs,
        'final_loss': losses[-1],
        'loss_reduction': losses[0] - losses[-1],
        'total_params': total_params,
        'backbone_params': backbone_params,
        'transformer_params': transformer_params
    }, save_path)
    
    print(f"💾 模型保存到: {save_path}")
    print(f"\n🎉 PyTorch对齐训练完成！")
    
    return losses, training_time, total_params

if __name__ == "__main__":
    train_aligned_pytorch()
