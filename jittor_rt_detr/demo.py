#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RT-DETR Jittor 演示脚本
展示模型的基本功能和使用方法
"""

import jittor as jt
import jittor.nn as nn
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

from model import RTDETR
from dataset import COCODataset
from loss import DETRLoss


def demo_model_creation():
    """演示模型创建"""
    print("=== 模型创建演示 ===")

    try:
        # 创建模型
        model = RTDETR(num_classes=81, num_queries=300, embed_dim=256)
        print("✅ RT-DETR 模型创建成功")

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   模型参数数量: {total_params:,}")
        print(f"   查询数量: {model.num_queries}")
        print(f"   嵌入维度: {256}")

        return model

    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return None


def demo_forward_pass(model):
    """演示前向传播"""
    print("\n=== 前向传播演示 ===")

    try:
        # 创建示例输入
        batch_size = 2
        input_shape = (batch_size, 3, 640, 640)
        dummy_input = jt.randn(input_shape)

        print(f"✅ 创建示例输入: {input_shape}")

        # 前向传播
        with jt.no_grad():
            outputs_class, outputs_coord = model(dummy_input)

        print("✅ 前向传播成功")
        print(f"   分类输出形状: {outputs_class[-1].shape}")
        print(f"   坐标输出形状: {outputs_coord[-1].shape}")

        # 分析输出
        num_queries = outputs_class[-1].shape[1]
        num_classes = outputs_class[-1].shape[2]

        print(f"   查询数量: {num_queries}")
        print(f"   类别数量: {num_classes}")

        return outputs_class, outputs_coord

    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return None, None


def demo_loss_function():
    """演示损失函数"""
    print("\n=== 损失函数演示 ===")

    try:
        # 创建损失函数
        loss_fn = DETRLoss(num_classes=80)
        print("✅ DETR 损失函数创建成功")

        # 创建示例预测和目标
        batch_size = 2
        num_queries = 300
        num_classes = 81

        pred_logits = jt.randn(batch_size, num_queries, num_classes)
        pred_boxes = jt.randn(batch_size, num_queries, 4)

        # 创建示例目标
        targets = []
        for i in range(batch_size):
            num_objects = np.random.randint(1, 5)
            boxes = jt.randn(num_objects, 4)
            labels = jt.randint(0, 80, (num_objects,))
            targets.append({'boxes': boxes, 'labels': labels})

        print(f"✅ 创建示例数据")
        print(f"   批次大小: {batch_size}")
        print(f"   预测查询数: {num_queries}")
        print(f"   目标对象数: {[len(t['boxes']) for t in targets]}")

        # 计算损失
        loss_dict = loss_fn(pred_logits.unsqueeze(
            0), pred_boxes.unsqueeze(0), targets)

        print("✅ 损失计算成功")
        for key, value in loss_dict.items():
            print(f"   {key}: {value.item():.4f}")

        return loss_fn

    except Exception as e:
        print(f"❌ 损失函数演示失败: {e}")
        return None


def demo_dataset_loading():
    """演示数据集加载"""
    print("\n=== 数据集加载演示 ===")

    try:
        # 检查数据目录
        data_dir = "data/coco/val2017"
        ann_file = "data/coco/annotations/instances_val2017.json"

        if not os.path.exists(data_dir):
            print(f"⚠️  数据目录不存在: {data_dir}")
            print("请先运行: python prepare_data.py --download")
            return None

        if not os.path.exists(ann_file):
            print(f"⚠️  标注文件不存在: {ann_file}")
            print("请先运行: python prepare_data.py --download")
            return None

        # 创建数据集
        from jittor import transform as T

        transform = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
            T.ImageNormalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])

        dataset = COCODataset(data_dir, ann_file,
                              transforms=transform, subset_size=10)
        print(f"✅ 数据集创建成功")
        print(f"   数据集大小: {len(dataset)} 张图片")

        # 加载一个样本
        if len(dataset) > 0:
            img, boxes, labels = dataset[0]
            print(f"   图片形状: {img.shape}")
            print(f"   边界框数量: {len(boxes)}")
            print(f"   标签数量: {len(labels)}")

        return dataset

    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return None


def demo_visualization():
    """演示可视化功能"""
    print("\n=== 可视化演示 ===")

    try:
        # 检查测试图片
        test_image_path = "test.png"
        if not os.path.exists(test_image_path):
            print(f"⚠️  测试图片不存在: {test_image_path}")
            print("请准备一张测试图片")
            return

        # 创建模型
        model = RTDETR(num_classes=81)

        # 加载图片
        img = Image.open(test_image_path).convert('RGB')
        print(f"✅ 加载测试图片: {img.size}")

        # 预处理
        from jittor import transform as T
        transform = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
            T.ImageNormalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])

        img_tensor = transform(img).unsqueeze(0)
        print(f"✅ 图片预处理完成: {img_tensor.shape}")

        # 模型推理
        with jt.no_grad():
            outputs_class, outputs_coord = model(img_tensor)

        print("✅ 模型推理完成")

        # 后处理
        pred_logits = outputs_class[-1]
        pred_boxes = outputs_coord[-1]

        # 获取最高置信度的预测
        prob = nn.softmax(pred_logits, dim=-1)[0, :, :-1]
        scores, labels = jt.topk(prob, k=1, dim=-1)
        scores = scores.squeeze(-1)
        labels = labels.squeeze(-1)

        # 筛选高置信度预测
        keep_mask = scores > 0.3
        keep_indices = jt.where(keep_mask)[0]

        if len(keep_indices) > 0:
            print(f"✅ 检测到 {len(keep_indices)} 个目标")

            # 可视化
            draw = ImageDraw.Draw(img)

            for i in range(min(len(keep_indices), 5)):  # 最多显示5个
                idx = keep_indices[i].item()
                score = scores[idx].item()
                label = labels[idx].item()
                box = pred_boxes[0, idx].numpy()

                # 转换为像素坐标
                w, h = img.size
                x1 = (box[0] - box[2]/2) * w
                y1 = (box[1] - box[3]/2) * h
                x2 = (box[0] + box[2]/2) * w
                y2 = (box[1] + box[3]/2) * h

                # 绘制边界框
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1, y1), f"Class {label}: {score:.2f}", fill="red")

            # 保存结果
            result_path = "demo_result.jpg"
            img.save(result_path)
            print(f"✅ 可视化结果已保存: {result_path}")
        else:
            print("⚠️  未检测到任何目标")

    except Exception as e:
        print(f"❌ 可视化演示失败: {e}")


def demo_performance():
    """演示性能测试"""
    print("\n=== 性能测试演示 ===")

    try:
        # 创建模型
        model = RTDETR(num_classes=81)

        # 创建测试输入
        input_shape = (1, 3, 640, 640)
        dummy_input = jt.randn(input_shape)

        # 预热
        print("预热模型...")
        for _ in range(3):
            with jt.no_grad():
                _ = model(dummy_input)

        # 性能测试
        print("开始性能测试...")
        import time

        times = []
        for _ in range(10):
            start_time = time.time()
            with jt.no_grad():
                _ = model(dummy_input)
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = np.mean(times)
        std_time = np.std(times)

        print(f"✅ 性能测试完成")
        print(f"   平均推理时间: {avg_time*1000:.2f} ms")
        print(f"   标准差: {std_time*1000:.2f} ms")
        print(f"   最大时间: {max(times)*1000:.2f} ms")
        print(f"   最小时间: {min(times)*1000:.2f} ms")

    except Exception as e:
        print(f"❌ 性能测试失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="RT-DETR Jittor 演示脚本")
    parser.add_argument('--model', action='store_true',
                        help='演示模型创建')
    parser.add_argument('--forward', action='store_true',
                        help='演示前向传播')
    parser.add_argument('--loss', action='store_true',
                        help='演示损失函数')
    parser.add_argument('--dataset', action='store_true',
                        help='演示数据集加载')
    parser.add_argument('--vis', action='store_true',
                        help='演示可视化')
    parser.add_argument('--perf', action='store_true',
                        help='演示性能测试')
    parser.add_argument('--all', action='store_true',
                        help='运行所有演示')

    args = parser.parse_args()

    print("🚀 RT-DETR Jittor 演示脚本")
    print("="*50)

    # 检查是否指定了特定演示
    if not any([args.model, args.forward, args.loss, args.dataset, args.vis, args.perf, args.all]):
        print("请指定要运行的演示，或使用 --all 运行所有演示")
        print("示例: python demo.py --all")
        return

    # 运行指定的演示
    if args.all or args.model:
        model = demo_model_creation()

    if args.all or args.forward:
        if 'model' not in locals():
            model = demo_model_creation()
        if model:
            demo_forward_pass(model)

    if args.all or args.loss:
        demo_loss_function()

    if args.all or args.dataset:
        demo_dataset_loading()

    if args.all or args.vis:
        demo_visualization()

    if args.all or args.perf:
        demo_performance()

    print("\n🎉 演示完成！")
    print("="*50)


if __name__ == '__main__':
    main()
