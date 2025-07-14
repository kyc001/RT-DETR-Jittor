#!/usr/bin/env python3
"""
快速流程自检脚本 - 验证RT-DETR Jittor实现的核心功能
"""

import os
import sys
import json
import numpy as np
from PIL import Image

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'jittor_rt_detr'))

import jittor as jt
from src.nn.model import RTDETR
from src.nn.loss import DETRLoss

# 设置Jittor
jt.flags.use_cuda = 1

def load_test_data():
    """加载测试数据"""
    img_path = "data/coco2017_50/train2017/000000225405.jpg"
    ann_file = "data/coco2017_50/annotations/instances_train2017.json"
    
    # 加载图片
    image = Image.open(img_path).convert('RGB')
    original_size = image.size
    
    # 简单预处理
    resized_image = image.resize((640, 640), Image.LANCZOS)
    img_array = np.array(resized_image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    img_tensor = jt.array(img_array.transpose(2, 0, 1)).unsqueeze(0)
    
    # 加载标注
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # 找到目标图片
    target_image = None
    for img in coco_data['images']:
        if img['file_name'] == "000000225405.jpg":
            target_image = img
            break
    
    # 找到标注
    annotations = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == target_image['id']:
            annotations.append(ann)
    
    # 简化：只使用前2个标注来避免复杂性
    annotations = annotations[:2]
    
    # 处理标注
    boxes = []
    labels = []
    for ann in annotations:
        x, y, w, h = ann['bbox']
        # 转换为归一化的cxcywh格式
        cx = (x + w/2) / original_size[0]
        cy = (y + h/2) / original_size[1]
        w_norm = w / original_size[0]
        h_norm = h / original_size[1]
        boxes.append([cx, cy, w_norm, h_norm])
        labels.append(0 if ann['category_id'] == 37 else 1)  # sports ball=0, person=1
    
    boxes_tensor = jt.array(boxes, dtype='float32')
    labels_tensor = jt.array(labels, dtype='int64')
    
    targets = [{
        'boxes': boxes_tensor,
        'labels': labels_tensor
    }]
    
    return img_tensor, targets, len(annotations)

def test_model_creation():
    """测试模型创建"""
    print(">>> 测试模型创建...")
    try:
        model = RTDETR(num_classes=2)
        print("✅ 模型创建成功")
        return model
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return None

def test_forward_pass(model, img_tensor):
    """测试前向传播"""
    print(">>> 测试前向传播...")
    try:
        model.eval()
        with jt.no_grad():
            outputs = model(img_tensor)
        logits, boxes, enc_logits, enc_boxes = outputs
        print(f"✅ 前向传播成功")
        print(f"  - logits.shape: {logits.shape}")
        print(f"  - boxes.shape: {boxes.shape}")
        return outputs
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return None

def test_loss_computation(model, img_tensor, targets):
    """测试损失计算"""
    print(">>> 测试损失计算...")
    try:
        model.train()
        criterion = DETRLoss(num_classes=2)

        # 确保输入tensor是float32类型
        img_tensor = img_tensor.float32()

        outputs = model(img_tensor)
        logits, boxes, enc_logits, enc_boxes = outputs

        # 确保所有输出tensor都是float32类型
        logits = logits.float32()
        boxes = boxes.float32()
        enc_logits = enc_logits.float32()
        enc_boxes = enc_boxes.float32()

        loss_dict = criterion(logits, boxes, targets, enc_logits, enc_boxes)
        total_loss = sum(loss_dict.values())

        print(f"✅ 损失计算成功")
        print(f"  - total_loss: {float(total_loss.data):.4f}")
        for key, value in loss_dict.items():
            print(f"  - {key}: {float(value.data):.4f}")
        return True
    except Exception as e:
        print(f"❌ 损失计算失败: {e}")
        return False

def test_training_step(model, img_tensor, targets):
    """测试训练步骤"""
    print(">>> 测试训练步骤...")
    try:
        model.train()
        criterion = DETRLoss(num_classes=2)
        optimizer = jt.optim.Adam(model.parameters(), lr=1e-4)

        # 确保输入tensor是float32类型
        img_tensor = img_tensor.float32()

        # 前向传播
        outputs = model(img_tensor)
        logits, boxes, enc_logits, enc_boxes = outputs

        # 确保所有输出tensor都是float32类型
        logits = logits.float32()
        boxes = boxes.float32()
        enc_logits = enc_logits.float32()
        enc_boxes = enc_boxes.float32()

        # 计算损失
        loss_dict = criterion(logits, boxes, targets, enc_logits, enc_boxes)
        total_loss = sum(loss_dict.values())

        # 反向传播
        optimizer.zero_grad()
        optimizer.backward(total_loss)
        optimizer.step()

        print(f"✅ 训练步骤成功")
        print(f"  - 损失值: {float(total_loss.data):.4f}")
        return True
    except Exception as e:
        print(f"❌ 训练步骤失败: {e}")
        return False

def main():
    print("=" * 60)
    print("===      RT-DETR Jittor 快速流程自检      ===")
    print("=" * 60)
    
    # 1. 加载测试数据
    print(">>> 加载测试数据...")
    try:
        img_tensor, targets, num_annotations = load_test_data()
        print(f"✅ 测试数据加载成功")
        print(f"  - 图片形状: {img_tensor.shape}")
        print(f"  - 标注数量: {num_annotations}")
    except Exception as e:
        print(f"❌ 测试数据加载失败: {e}")
        return
    
    # 2. 测试模型创建
    model = test_model_creation()
    if model is None:
        return
    
    # 3. 测试前向传播
    outputs = test_forward_pass(model, img_tensor)
    if outputs is None:
        return
    
    # 4. 测试损失计算
    loss_success = test_loss_computation(model, img_tensor, targets)
    if not loss_success:
        return
    
    # 5. 测试训练步骤
    training_success = test_training_step(model, img_tensor, targets)
    if not training_success:
        return
    
    # 6. 总结
    print("\n" + "=" * 60)
    print("🎉 快速流程自检总结:")
    print("  ✅ 数据加载: 成功")
    print("  ✅ 模型创建: 成功")
    print("  ✅ 前向传播: 成功")
    print("  ✅ 损失计算: 成功")
    print("  ✅ 训练步骤: 成功")
    print("  ✅ 核心流程: 完全正常")
    print("\n💡 结论: RT-DETR Jittor实现的核心训练流程工作正常！")
    print("   可以进行完整的训练和推理。")
    print("=" * 60)

if __name__ == "__main__":
    main()
