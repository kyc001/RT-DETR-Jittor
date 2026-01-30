# RT-DETR Jittor 实现

[English](#english) | [中文](#中文)

<a name="中文"></a>
## 中文文档

### 项目概述

本项目是 **RT-DETR（Real-Time Detection Transformer）** 从 PyTorch 框架到 **Jittor** 框架的完整迁移实现。RT-DETR 是一个高效的实时目标检测模型，结合了 Transformer 架构的优势和实时推理的需求。

**迁移完成时间**: 2026-01-29

### ✅ 迁移状态

| 模块 | 状态 | 参数量 | 备注 |
|------|------|--------|------|
| **ResNet Backbone** | ✅ 完成 | 23.58M | 支持 ResNet18/34/50/101，variant='d' 支持 |
| **HybridEncoder** | ✅ 完成 | 11.97M | 完整 FPN+PAN 结构 |
| **RTDETRTransformer** | ✅ 完成 | 7.47M | 含去噪训练支持 |
| **Criterion** | ✅ 完成 | - | VFL/Focal/BCE 损失 |
| **完整模型** | ✅ 完成 | **43.02M** | 端到端前向传播 |
| **EMA 模块** | ✅ 完成 | - | 指数移动平均，含内存泄漏防护 |
| **权重转换工具** | ✅ 完成 | - | PyTorch ↔ Jittor 双向转换 |

### 🎯 测试结果

```
============================================================
RT-DETR Jittor 模块测试
============================================================

✓ Backbone 测试通过
  - 输入: [1,3,640,640]
  - 输出: [[1,512,80,80], [1,1024,40,40], [1,2048,20,20]]
  - 参数量: 23,580,512 (23.58M)

✓ Encoder 测试通过
  - 输入: 3个特征图
  - 输出: [[1,256,80,80], [1,256,40,40], [1,256,20,20]]
  - 参数量: 11,970,816 (11.97M)

✓ Decoder 测试通过
  - 输出: pred_logits [1,300,80], pred_boxes [1,300,4]
  - 参数量: 7,468,044 (7.47M)

✓ Criterion 测试通过
  - 损失类型: loss_vfl, loss_bbox, loss_giou

✓ 完整模型测试通过
  - 总参数量: 43,019,372 (43.02M)
============================================================
```

### 📁 项目结构

```
RT-DETR-Jittor/
├── rtdetr_jittor/                    # Jittor 实现
│   ├── configs/
│   │   └── rtdetr/
│   │       ├── rtdetr_base.yml
│   │       ├── rtdetr_r18vd_6x_coco.yml
│   │       └── rtdetr_r50vd_6x_coco.yml
│   ├── src/
│   │   ├── core/                     # 配置管理
│   │   ├── optim/
│   │   │   └── ema.py               # EMA 模块
│   │   ├── components/
│   │   │   ├── trainer.py           # 训练器（支持 EMA）
│   │   │   ├── dataset.py           # 数据集
│   │   │   └── visualizer.py        # 可视化
│   │   ├── nn/
│   │   │   └── backbone/
│   │   │       └── resnet.py        # ResNet 骨干网络
│   │   └── zoo/
│   │       └── rtdetr/
│   │           ├── rtdetr.py         # 主模型
│   │           ├── hybrid_encoder.py # 混合编码器
│   │           ├── rtdetr_decoder.py # Transformer 解码器
│   │           ├── rtdetr_criterion.py # 损失函数
│   │           ├── box_ops.py        # 边界框操作
│   │           ├── denoising.py      # 去噪模块
│   │           ├── matcher.py        # 匈牙利匹配器
│   │           └── utils.py          # 工具函数
│   ├── tools/
│   │   ├── train.py                 # 训练脚本
│   │   ├── eval.py                  # 评估脚本
│   │   └── convert_weights.py       # 权重转换工具
│   ├── test_modules.py              # 模块测试脚本
│   └── test_ema.py                  # EMA 测试脚本
│
├── rtdetr_pytorch/                   # PyTorch 原版（参考）
└── README.md
```

### 🚀 快速开始

#### 1. 环境配置

```bash
# 创建并激活 Jittor 环境
conda create -n jt python=3.8
conda activate jt

# 安装 Jittor
pip install jittor

# 安装依赖
pip install pycocotools PyYAML scipy pillow matplotlib numpy
```

#### 2. 运行测试

```bash
cd rtdetr_jittor
python test_modules.py
```

#### 3. 权重转换

```bash
# PyTorch 权重转换为 Jittor
python tools/convert_weights.py --pt2jt -i model.pth -o model.pkl

# Jittor 权重转换为 PyTorch
python tools/convert_weights.py --jt2pt -i model.pkl -o model.pth
```

#### 4. 训练

```bash
cd rtdetr_jittor
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

### 🔧 主要技术实现

#### 已完成的模块

1. **ResNet Backbone** (`src/nn/backbone/resnet.py`)
   - 支持 BasicBlock (ResNet18/34) 和 BottleNeck (ResNet50/101)
   - 实现 FrozenBatchNorm2d
   - 支持 freeze_at 和 freeze_norm 功能
   - 支持 variant='d' 变体

2. **HybridEncoder** (`src/zoo/rtdetr/hybrid_encoder.py`)
   - ConvNormLayer - 卷积+归一化+激活
   - RepVggBlock - 重参数化 VGG 块
   - CSPRepLayer - 跨阶段部分连接层
   - TransformerEncoderLayer - Transformer 编码器层
   - 完整的 FPN（自顶向下）+ PAN（自底向上）结构

3. **RTDETRTransformer** (`src/zoo/rtdetr/rtdetr_decoder.py`)
   - MSDeformableAttention - 多尺度可变形注意力
   - TransformerDecoder - 多层解码器
   - 去噪训练支持
   - 锚点生成和多层输出头

4. **Criterion** (`src/zoo/rtdetr/rtdetr_criterion.py`)
   - HungarianMatcher - 匈牙利匹配器
   - 多种损失类型: VFL, Focal, BCE, L1, GIoU
   - 辅助损失和 CDN 去噪损失支持

5. **EMA 模块** (`src/optim/ema.py`)
   - 指数移动平均实现
   - 内存泄漏防护（Jittor 惰性求值适配）

6. **权重转换工具** (`tools/convert_weights.py`)
   - PyTorch ↔ Jittor 双向转换

### ⚠️ Jittor 兼容性说明

以下 PyTorch API 在 Jittor 中需要特殊处理：

| PyTorch API | Jittor 替代方案 |
|-------------|--------------|
| `torch.tile()` | 自定义 `tile()` 函数 |
| `tensor.repeat()` | `jt.concat([tensor] * n)` |
| `nn.binary_cross_entropy_with_logits()` | 自定义实现 |
| `torch.topk(..., dim=1)` | 使用 `jt.argsort()` 逐 batch 处理 |
| `nn.ModuleList(generator)` | `nn.ModuleList([list])` |

### 🛡️ 内存泄漏防护

由于 Jittor 使用惰性求值机制，需要注意：

```python
# EMA 更新后同步
ema.update(model)
jt.sync_all()  # 强制执行计算图

# 训练循环中同步
optimizer.step(total_loss)
jt.sync_all()  # 防止计算图累积
```

### 📊 参数量对比

| 统计方式 | Backbone | Encoder | Decoder | 总计 |
|---------|----------|---------|---------|------|
| Jittor (含 running stats) | 23.58M | 11.97M | 7.47M | **43.02M** |
| PyTorch 方式 | 23.53M | 11.95M | 7.47M | **42.94M** |
| 官方 README | - | - | - | **42M** |

> **说明**: Jittor 将 BatchNorm 的 `running_mean` 和 `running_var` 计入 `parameters()`，而 PyTorch 将它们视为 buffers，导致约 0.07M 的差异。

---

<a name="english"></a>
## English Documentation

### Project Overview

This project is a complete migration of **RT-DETR (Real-Time Detection Transformer)** from PyTorch to **Jittor** framework. RT-DETR is an efficient real-time object detection model that combines the advantages of Transformer architecture with real-time inference requirements.

**Migration Completed**: 2026-01-29

### ✅ Migration Status

| Module | Status | Parameters | Notes |
|--------|--------|------------|-------|
| **ResNet Backbone** | ✅ Done | 23.58M | Supports ResNet18/34/50/101, variant='d' |
| **HybridEncoder** | ✅ Done | 11.97M | Complete FPN+PAN structure |
| **RTDETRTransformer** | ✅ Done | 7.47M | With denoising training support |
| **Criterion** | ✅ Done | - | VFL/Focal/BCE losses |
| **Full Model** | ✅ Done | **43.02M** | End-to-end forward pass |
| **EMA Module** | ✅ Done | - | With memory leak protection |
| **Weight Converter** | ✅ Done | - | PyTorch ↔ Jittor bidirectional |

### 🚀 Quick Start

#### 1. Environment Setup

```bash
# Create and activate Jittor environment
conda create -n jt python=3.8
conda activate jt

# Install Jittor
pip install jittor

# Install dependencies
pip install pycocotools PyYAML scipy pillow matplotlib numpy
```

#### 2. Run Tests

```bash
cd rtdetr_jittor
python test_modules.py
```

#### 3. Weight Conversion

```bash
# PyTorch to Jittor
python tools/convert_weights.py --pt2jt -i model.pth -o model.pkl

# Jittor to PyTorch
python tools/convert_weights.py --jt2pt -i model.pkl -o model.pth
```

#### 4. Training

```bash
cd rtdetr_jittor
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

### 📝 License

This project is for educational and research purposes.

### 🙏 Acknowledgments

- [RT-DETR](https://github.com/lyuwenyu/RT-DETR) - Original PyTorch implementation
- [Jittor](https://github.com/Jittor/jittor) - Deep learning framework

### 📚 References

- [RT-DETR Paper](https://arxiv.org/abs/2304.08069)
- [Jittor Documentation](https://cg.cs.tsinghua.edu.cn/jittor/)
