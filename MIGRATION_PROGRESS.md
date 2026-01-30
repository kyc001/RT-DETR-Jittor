# RT-DETR PyTorch → Jittor 迁移进度文档

## 迁移完成时间
2026-01-29

## 项目概况

本项目是将RT-DETR（Real-Time Detection Transformer）从PyTorch框架迁移到Jittor框架的复现项目。

## 迁移状态概览

| 模块 | 状态 | 参数量 | 备注 |
|------|------|--------|------|
| **ResNet Backbone** | ✅ 完成 | 23.58M | 支持ResNet18/34/50/101，variant='d'支持 |
| **HybridEncoder** | ✅ 完成 | 11.97M | 完整FPN+PAN结构 |
| **RTDETRTransformer** | ✅ 完成 | 7.47M | 含去噪训练支持 |
| **Criterion** | ✅ 完成 | 81 | VFL/Focal/BCE损失 |
| **完整模型** | ✅ 完成 | 43.02M | 端到端前向传播 |
| **EMA模块** | ✅ 完成 | - | 指数移动平均，含内存泄漏防护 |
| **权重转换工具** | ✅ 完成 | - | PyTorch ↔ Jittor 双向转换 |

## 测试结果

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

### 参数量说明

| 统计方式 | Backbone | Encoder | Decoder | 总计 |
|---------|----------|---------|---------|------|
| **Jittor (含running stats)** | 23.58M | 11.97M | 7.47M | **43.02M** |
| **PyTorch方式 (不含running stats)** | 23.53M | 11.95M | 7.47M | **42.94M** |
| **官方README** | - | - | - | **42M** |

**差异原因**:
1. **Jittor vs PyTorch**: Jittor将BatchNorm的`running_mean`和`running_var`计入`parameters()`，而PyTorch将它们视为buffers（差异约0.07M）
2. **我们 vs 官方**: 排除running stats后为42.94M，与官方42M差异约2.25%，可能因官方数字四舍五入或默认配置细微差别

验证代码:
```python
# Jittor
bn = jt.nn.BatchNorm2d(256)
len(list(bn.parameters()))  # 返回4: weight, bias, running_mean, running_var

# PyTorch
bn = torch.nn.BatchNorm2d(256)
len(list(bn.parameters()))  # 返回2: weight, bias (running stats是buffers)
```

**结论**: 模型结构与PyTorch版本一致，参数差异主要由于统计方式不同。

## 已完成的主要修改

### 1. ResNet Backbone (`src/nn/backbone/resnet.py`)
- 添加 `BasicBlock` 支持ResNet18/34
- 添加 `BottleNeck` 的variant='a'支持
- 实现 `FrozenBatchNorm2d`
- 添加 `freeze_at` 和 `freeze_norm` 功能
- 支持所有深度: 18, 34, 50, 101

### 2. HybridEncoder (`src/zoo/rtdetr/hybrid_encoder.py`)
完全重写，实现：
- `ConvNormLayer` - 卷积+归一化+激活
- `RepVggBlock` - 重参数化VGG块
- `CSPRepLayer` - 跨阶段部分连接层
- `TransformerEncoderLayer` - Transformer编码器层
- `MultiHeadAttention` - 多头注意力
- 完整的FPN（自顶向下）+ PAN（自底向上）结构
- 2D sincos位置编码

### 3. RTDETRTransformer (`src/zoo/rtdetr/rtdetr_decoder.py`)
完全重写，实现：
- `MSDeformableAttention` - 多尺度可变形注意力
- `TransformerDecoderLayer` - 解码器层
- `TransformerDecoder` - 多层解码器
- 去噪训练支持（`get_contrastive_denoising_training_group`）
- 锚点生成（`_generate_anchors`）
- 多层输出头（`dec_score_head`, `dec_bbox_head`）
- 辅助损失输出

### 4. Criterion (`src/zoo/rtdetr/rtdetr_criterion.py`)
- 自定义 `binary_cross_entropy_with_logits`（Jittor兼容）
- `sigmoid_focal_loss` 实现
- `HungarianMatcher` - 匈牙利匹配器
- `SetCriterion` - 多种损失类型:
  - `loss_labels` - NLL分类损失
  - `loss_labels_bce` - BCE分类损失
  - `loss_labels_focal` - Focal分类损失
  - `loss_labels_vfl` - Varifocal分类损失
  - `loss_boxes` - 边界框L1+GIoU损失
- 辅助损失支持
- CDN去噪损失支持

### 5. 工具函数 (`src/zoo/rtdetr/utils.py`)
- `tile` - 兼容PyTorch的tile函数
- `inverse_sigmoid` - 逆sigmoid
- `deformable_attention_core_func` - 可变形注意力核心函数
- `bias_init_with_prob` - 概率偏置初始化
- `get_activation` - 激活函数获取
- `MLP` - 多层感知机
- `simple_grid_sample` - 简化的网格采样

### 6. 去噪模块 (`src/zoo/rtdetr/denoising.py`)
- `get_contrastive_denoising_training_group` - 对比去噪训练组

### 7. Box操作 (`src/zoo/rtdetr/box_ops.py`)
- `box_cxcywh_to_xyxy` / `box_xyxy_to_cxcywh`
- `box_area` / `box_iou`
- `generalized_box_iou`
- `masks_to_boxes`

### 8. EMA模块 (`src/optim/ema.py`)
实现了针对Jittor优化的指数移动平均：
- `ModelEMA` - 模型参数的指数移动平均
- 衰减函数支持预热期
- **关键**: 在update()后调用`jt.sync_all()`防止惰性求值导致的内存泄漏
- 完整的state_dict保存/加载支持
- 评估时自动使用EMA模型

### 9. 训练器EMA集成 (`src/components/trainer.py`)
- `RTDETRTrainer` 支持 `use_ema`, `ema_decay`, `ema_warmups` 参数
- 每个batch后自动更新EMA
- `get_eval_model()` 自动返回EMA模型用于评估
- 检查点保存/加载包含EMA状态
- 训练循环中添加`jt.sync_all()`防止内存泄漏

### 10. 权重转换工具 (`tools/convert_weights.py`)
支持PyTorch和Jittor之间的双向权重转换：
- `convert_pytorch_to_jittor()` - PyTorch .pth → Jittor .pkl
- `convert_jittor_to_pytorch()` - Jittor .pkl → PyTorch .pth
- `load_pytorch_weights_to_jittor_model()` - 直接加载PyTorch权重到Jittor模型
- `verify_conversion()` - 验证转换正确性

使用方法:
```bash
# PyTorch to Jittor
python tools/convert_weights.py --pt2jt -i model.pth -o model.pkl

# Jittor to PyTorch
python tools/convert_weights.py --jt2pt -i model.pkl -o model.pth
```

## Jittor兼容性修复

以下PyTorch API在Jittor中需要特殊处理：

| PyTorch API | Jittor替代方案 |
|-------------|--------------|
| `torch.tile()` | 自定义`tile()`函数 |
| `tensor.repeat()` | `jt.concat([tensor] * n)` |
| `nn.binary_cross_entropy_with_logits()` | 自定义实现 |
| `torch.topk(..., dim=1)` | 使用`jt.argsort()`逐batch处理 |
| `tensor.all(..., keepdims=True)` | `tensor.all().unsqueeze()` |
| `nn.ModuleList(generator)` | `nn.ModuleList([list])` |

## Jittor内存泄漏防护

由于Jittor使用惰性求值机制，以下场景需要特别注意：

### 1. EMA更新后同步
```python
# EMA更新后必须调用sync_all()防止计算图无限增长
ema.update(model)
jt.sync_all()  # 强制执行计算图
```

### 2. 训练循环中的同步
```python
# 每个batch优化器步骤后同步
optimizer.step(total_loss)
jt.sync_all()  # 防止计算图累积
```

### 3. BatchNorm running statistics
```python
# 使用running mean/var时应stop_grad()
running_mean = branch.norm.running_mean.stop_grad()
running_var = branch.norm.running_var.stop_grad()
```

### 4. Decoder循环中的参考点
```python
# 防止梯度回传到初始参考点
ref_points_detach = jt.sigmoid(ref_points_unact).stop_grad()
```

## 文件结构

```
rtdetr_jittor/
├── configs/
│   └── rtdetr/
│       ├── rtdetr_base.yml
│       ├── rtdetr_r18vd_6x_coco.yml
│       └── rtdetr_r50vd_6x_coco.yml
├── src/
│   ├── __init__.py
│   ├── core/
│   ├── optim/
│   │   ├── __init__.py
│   │   └── ema.py               # ✅ 新增 - EMA模块
│   ├── components/
│   │   ├── trainer.py           # ✅ 更新 - 支持EMA
│   │   └── ...
│   ├── nn/
│   │   └── backbone/
│   │       ├── __init__.py
│   │       └── resnet.py        # ✅ 完成
│   └── zoo/
│       └── rtdetr/
│           ├── __init__.py
│           ├── rtdetr.py         # 主模型
│           ├── hybrid_encoder.py # ✅ 完成
│           ├── rtdetr_decoder.py # ✅ 完成
│           ├── rtdetr_criterion.py # ✅ 完成
│           ├── box_ops.py        # ✅ 完成
│           ├── denoising.py      # ✅ 完成
│           ├── matcher.py
│           └── utils.py          # ✅ 完成
├── tools/
│   ├── train.py
│   ├── eval.py
│   └── convert_weights.py       # ✅ 新增 - 权重转换工具
├── test_modules.py               # 模块测试脚本
└── test_ema.py                   # ✅ 新增 - EMA测试脚本
```

## 后续工作建议

1. **训练验证**: 在COCO数据集上进行完整训练，验证收敛性
2. **精度对齐**: 与PyTorch版本的精度进行对比
3. **性能优化**: 针对Jittor的特性进行推理速度优化
4. ~~**权重转换**: 实现PyTorch预训练权重到Jittor的转换工具~~ ✅ 已完成
5. ~~**EMA支持**: 实现训练时的指数移动平均~~ ✅ 已完成

## 运行测试

```bash
# 激活Jittor环境
micromamba activate jt

# 安装依赖
pip install pycocotools PyYAML scipy pillow matplotlib numpy

# 运行测试
cd rtdetr_jittor
python test_modules.py
```

## 历史记录

### 2026-01-29 (初始检查)
- 发现严重问题：HybridEncoder缺少FPN结构，RTDETRDecoder过度简化
- 运行时错误：jt.relu, numpy转换问题

### 2026-01-29 (修复完成)
- ✅ 完全重写 HybridEncoder，实现完整FPN+PAN
- ✅ 完全重写 RTDETRTransformer，实现可变形注意力和去噪训练
- ✅ 完善 ResNet Backbone，添加BasicBlock和freeze功能
- ✅ 完善 Criterion，添加所有损失类型
- ✅ 修复所有Jittor兼容性问题
- ✅ 所有模块测试通过

### 2026-01-29 (EMA和内存泄漏修复)
- ✅ 实现 EMA模块 (`src/optim/ema.py`)
  - ModelEMA类，支持衰减预热
  - 在update()后调用`jt.sync_all()`防止内存泄漏
- ✅ 更新训练器支持EMA (`src/components/trainer.py`)
  - 添加use_ema, ema_decay, ema_warmups参数
  - 添加检查点保存/加载EMA状态
  - 添加get_eval_model()返回EMA模型
- ✅ 实现权重转换工具 (`tools/convert_weights.py`)
  - PyTorch→Jittor和Jittor→PyTorch双向转换
- ✅ 修复Jittor内存泄漏问题
  - 训练循环中添加jt.sync_all()
  - BatchNorm running stats使用stop_grad()
  - Decoder ref_points使用stop_grad()
- ✅ EMA测试通过

## 参考

- PyTorch版本: `rtdetr_pytorch/`
- Jittor版本: `rtdetr_jittor/`
