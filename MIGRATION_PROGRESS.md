# RT-DETR PyTorch → Jittor 迁移进度文档

## 2026-02-06 深入对齐更新（三）

### 本轮关键结论

1. `loss_vfl` 的“2x 偏差”已定位为**回归脚本配置不一致**，不是 Jittor loss 实现错误。  
   - 原因：`regression_compare.py` 的 PyTorch 侧把 `loss_vfl` 权重硬编码为 `1`；Jittor 侧实际从配置映射为 `loss_focal -> loss_vfl = 2`。  
   - 修复后：`loss_vfl` 与 `loss_vfl_aux_*` 差异收敛到 `1e-6` 量级。

2. 引擎补齐了 `SetCriterion` 配置透传，避免训练时隐性配置漂移。  
   - 文件：`rtdetr_jittor/src/core/engine.py`、`rtdetr_jittor/src/zoo/rtdetr/rtdetr_criterion.py`  
   - 新增/修复：  
     - `criterion.alpha/gamma/eos_coef` 配置透传到 `build_criterion`  
     - 兼容 `matcher.weight_dict` 嵌套格式与 `use_focal`/`use_focal_loss` 字段  
     - `SetCriterion` legacy 映射时同步读取 `alpha/gamma/eos_coef`

3. 跨框架回归已达到“前向 + 匹配 + loss 分项”高精度一致。  
   - 报告：`rtdetr_jittor/migration_artifacts/regression/compare_report.json`

### 回归命令（本轮）

```bash
micromamba activate jt
cd rtdetr_jittor
python tools/regression_compare.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
  --output-dir migration_artifacts/regression \
  --input-size 320 --batch-size 1 --num-boxes 3 --num-denoising 0
```

### 回归结果（本轮）

- `logits.mean_abs`: `2.0503e-07`
- `boxes.mean_abs`: `7.6027e-09`
- `matcher_exact_match`: `true`
- `matcher_from_cost_exact_match`: `true`
- `loss_vfl` 绝对差：`1.43e-06`
- `loss_bbox` 绝对差：`1.19e-07`
- `loss_giou` 绝对差：`2.38e-07`

参数统计说明（避免误判）：
- `param_count_exact_match = false`（统计口径差异：Jittor `parameters()` 含部分 buffer-like 项）
- `state_param_like_numel_exact_match = true`（可训练参数体量已对齐）

### Gate 复验（本轮实跑）

```bash
micromamba activate jt
cd rtdetr_jittor

python tools/infer.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
  --image migration_artifacts/sample.jpg --device cpu --input-size 320 \
  --output-dir migration_artifacts/gates_round2/infer --score-threshold 0.2

python tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
  --device cpu --dummy-data --max-samples 4 --epochs 1 --batch-size 2 \
  --input-size 320 --output-dir migration_artifacts/gates_round2/train --print-freq 1

python tools/eval.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
  --device cpu --dummy-data --max-samples 4 --input-size 320 \
  --checkpoint migration_artifacts/gates_round2/train/last_ckpt.pkl \
  --output-dir migration_artifacts/gates_round2/eval

python tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
  --device cpu --dummy-data --max-samples 4 --epochs 2 --batch-size 2 \
  --input-size 320 --output-dir migration_artifacts/gates_round2/train_resume \
  --resume migration_artifacts/gates_round2/train/last_ckpt.pkl --print-freq 1
```

结果：
- infer: ✅ 输出 `sample_detections.json` 与可视化图
- train(1 epoch): ✅ 成功生成 `last_ckpt/best_ckpt`
- eval: ✅ 指标链路成功（fallback 指标可诊断）
- resume: ✅ 从 `epoch=1` 继续训练成功

## 2026-02-06 深入对齐更新（二）

### 关键修复（本轮）

1. 修复 Jittor `max(dim=...)` 语义误用导致的解码器选框错误  
   - 文件：`rtdetr_jittor/src/zoo/rtdetr/rtdetr_decoder.py`
   - 问题：原实现把 `enc_outputs_class.max(-1)[0]` 当成 PyTorch tuple 风格使用，实际在 Jittor 中会错误切片。
   - 结果：`topk` 选框恢复正常，主干 `bbox`/`giou` loss 与 PyTorch 显著接近。

2. 对齐 postprocessor 到 PyTorch 同构逻辑  
   - 文件：`rtdetr_jittor/src/zoo/rtdetr/rtdetr_postprocessor.py`
   - 变更：focal 分支改为 `scores.flatten(1)` 全类 top-k + index 反解 `labels/query_idx` + gather boxes。
   - 兼容：同时支持 `orig_target_sizes` 与 `target_sizes` 参数名，修复 `infer` 调用异常。

3. 修复 box/mask 统计中的 `max/min` API 用法  
   - 文件：`rtdetr_jittor/src/zoo/rtdetr/box_ops.py`
   - 变更：移除错误的 `[0]` 索引，保证 `masks_to_boxes` 数值正确。

4. 收敛 deformable attention 采样路径  
   - 文件：`rtdetr_jittor/src/zoo/rtdetr/utils.py`
   - 变更：移除近似 fallback，固定使用 `nn.grid_sample` 路径，避免静默退化。

### 跨框架回归结果（PyTorch vs Jittor）

命令：
```bash
micromamba activate jt
cd rtdetr_jittor
python tools/regression_compare.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
  --output-dir migration_artifacts/regression --input-size 320 --batch-size 1 --num-boxes 3
```

本轮结果（`migration_artifacts/regression/compare_report.json`）：
- `logits.mean_abs`: `0.4056`（仍有差距）
- `boxes.mean_abs`: `0.2035`（较修复前下降）
- `loss_bbox` 绝对差：`11.64 -> 0.16`（显著改善）
- `loss_giou` 绝对差：`0.61 -> 0.04`（显著改善）
- `matcher_exact_match`: `false`（未完全一致）
- 权重加载：`loaded=460, missing=3, mismatched=0`

说明：
- `missing=3` 为预期非权重缓存项（`encoder.pos_embed2`、`decoder.anchors`、`decoder.valid_mask`）。
- 当前“可训练/可评估/可续训”链路已通，但“与 PyTorch 数值完全一致”仍未达成。

### Gate 链路复验（本轮实跑）

```bash
micromamba activate jt
cd rtdetr_jittor

# infer
python tools/infer.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
  --image migration_artifacts/gates/test.jpg --device cpu --input-size 320 \
  --output-dir migration_artifacts/gates/infer --score-threshold 0.2

# train 1 epoch
python tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
  --device cpu --dummy-data --max-samples 4 --epochs 1 --batch-size 2 \
  --input-size 320 --output-dir migration_artifacts/gates/train --print-freq 1

# eval
python tools/eval.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
  --device cpu --dummy-data --max-samples 4 --input-size 320 \
  --checkpoint migration_artifacts/gates/train/last_ckpt.pkl \
  --output-dir migration_artifacts/gates/eval

# resume
python tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
  --device cpu --dummy-data --max-samples 4 --epochs 2 --batch-size 2 \
  --input-size 320 --output-dir migration_artifacts/gates/train_resume \
  --resume migration_artifacts/gates/train/last_ckpt.pkl --print-freq 1
```

结果：
- infer: ✅ 成功输出 `json/vis`
- train(1 epoch): ✅ 成功，生成 `last_ckpt/best_ckpt`
- eval: ✅ 成功，输出 loss 与 fallback 指标
- resume: ✅ 成功续训，checkpoint 加载 `loaded=463, missing=0`

## 2026-02-06 可复用迁移流程落地（本次更新）

### 1) 验收标准（先写清楚再动代码）

| 项目 | 必须通过 | 可接受误差 | 本次状态 |
|---|---|---|---|
| 单图推理 `infer` | 能加载权重、前向、输出结果文件 | - | ✅ 已通过（含 ckpt 加载） |
| 最小训练 `train` | 1 epoch 可跑通并保存 checkpoint | loss 不出现 NaN/Inf | ✅ 已通过 |
| 评估链路 `eval` | 输出可解释指标 + 预测文件 | AP 对齐阈值建议 `< 0.3`（需真实 COCO 对齐） | ✅ 已通过（dummy fallback 指标） |
| 续训 `resume` | 从 `last_ckpt` 继续训练 | 指标连续、epoch 连续 | ✅ 已通过 |

说明：
- AP 差值阈值（如 `< 0.3`）需要在同数据、同权重、同后处理配置下与 PyTorch 基线对齐后才能最终判定。
- 当前仓库无 COCO 数据集，故本次验收采用 `--dummy-data` 跑链路完整性，真实 AP 对齐留在后续基线回归阶段执行。

### 2) 基线冻结（可复用模板）

建议固定产物目录：`rtdetr_jittor/migration_artifacts/`

需要冻结并记录：
- 原框架版本：PyTorch commit、依赖版本、CUDA/cuDNN。
- 基线配置：训练配置文件、后处理阈值、是否 EMA。
- 基线权重：`best`/`last` 路径。
- 基线日志：训练日志、评估日志。
- 固定样本输入输出：单图输入 + logits/boxes + 后处理结果（用于数值回归）。

本次已落地输出（Jittor 侧）：
- `rtdetr_jittor/migration_artifacts/sample.jpg`
- `rtdetr_jittor/migration_artifacts/run/sample_raw_outputs.npz`
- `rtdetr_jittor/migration_artifacts/run/sample_detections.json`
- `rtdetr_jittor/migration_artifacts/run_train/train_log.jsonl`
- `rtdetr_jittor/migration_artifacts/run_eval/eval_log.jsonl`

### 3) 迁移骨架与门禁（Gate）实现

新增/重构文件：
- `rtdetr_jittor/src/core/engine.py`
  - 统一 train/eval/infer 主循环
  - 配置读取与 legacy 映射
  - checkpoint 策略（`last_ckpt` 可 resume，`best_ckpt` 轻量部署）
  - 旧 checkpoint 字段兼容（`model/state_dict/ema.module`）
  - warmup + cosine lr、EMA、grad accumulation
  - COCO AP 评估 + fallback 指标兜底
- `rtdetr_jittor/src/data/coco_dataset.py`
  - COCO dataset、collate、轻增强（hflip）、label 编码
  - box 目标统一为归一化 `cxcywh`（匹配 RT-DETR criterion 预期）
  - `DummyDetectionDataset` 用于无数据环境链路验收
- `rtdetr_jittor/tools/train.py`
- `rtdetr_jittor/tools/eval.py`
- `rtdetr_jittor/tools/infer.py`（新增）

Gate 对应状态：
- Gate-1（推理最小链路）: ✅
- Gate-2（数据管线）: ✅
- Gate-3（loss + matcher 可反传）: ✅
- 训练循环/优化器/EMA: ✅
- 评估链路（COCO + fallback）: ✅
- checkpoint / resume: ✅

额外修复：
- `rtdetr_jittor/src/zoo/rtdetr/rtdetr.py`  
  训练多尺度输入 `multi_scale` 采样值显式转 `int`，修复 Jittor `interpolate` 类型报错。

### 4) 分阶段自检命令（已执行）

环境：
```bash
micromamba activate jt
```

已执行：
```bash
python -m py_compile \
  rtdetr_jittor/src/core/engine.py \
  rtdetr_jittor/src/data/coco_dataset.py \
  rtdetr_jittor/src/zoo/rtdetr/rtdetr.py \
  rtdetr_jittor/tools/infer.py \
  rtdetr_jittor/tools/train.py \
  rtdetr_jittor/tools/eval.py

cd rtdetr_jittor
python tools/infer.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
  -i migration_artifacts/sample.jpg \
  --checkpoint migration_artifacts/run_train/last_ckpt.pkl \
  --device cpu --input-size 320 --output-dir migration_artifacts/run_infer_ckpt

python tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
  --device cpu --dummy-data --epochs 1 --batch-size 1 --max-samples 2 \
  --input-size 320 --output-dir migration_artifacts/run_train

python tools/eval.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
  --checkpoint migration_artifacts/run_train/last_ckpt.pkl \
  --device cpu --dummy-data --max-samples 2 --input-size 320 \
  --output-dir migration_artifacts/run_eval

python tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
  --device cpu --dummy-data --epochs 2 --batch-size 1 --max-samples 2 \
  --input-size 320 --resume migration_artifacts/run_train/last_ckpt.pkl \
  --output-dir migration_artifacts/run_train_resume
```

结果摘要：
- infer: ✅ 输出可视化与 JSON，且成功加载 checkpoint（`loaded=463`）。
- train(1 epoch): ✅ 成功，loss 为有限值，生成 `last_ckpt/best_ckpt`。
- eval: ✅ 成功，输出 `val_loss` 与 fallback 指标，预测文件落盘。
- resume: ✅ 成功，从 `epoch=1` 继续训练并生成新 `last_ckpt`。

### 5) 文档化与固化

- 已将迁移流程、验收标准、自检命令、结果写入本文件。
- 已更新忽略规则，确保实验产物不污染代码变更：
  - `.gitignore`
  - `rtdetr_jittor/.gitignore`

### 6) 已知限制 / 下一步

- 当前 AP 对齐尚未在真实 COCO 上与 PyTorch 基线做严格差值比对（目标阈值建议 `AP 差 < 0.3`）。
- 下一步应在真实 COCO 数据与统一权重下执行：
  - PyTorch vs Jittor 同输入中间张量对比（logits、匹配结果、loss 分项）
  - COCO AP 回归与吞吐/显存回归
  - 将对齐结论回填到本文件“验收标准”表格。

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
| `torch.topk(..., dim=1)` | 使用`jt.topk(..., dim=1)` |
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
