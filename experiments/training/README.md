# RT-DETR Jittor版本训练

本目录包含RT-DETR Jittor版本的训练代码，已完全解决数据类型不匹配问题和辅助损失问题。

## 文件说明

- `train_rtdetr.py` - 主训练脚本（推荐使用）
- `production_rtdetr_training.py` - 生产级训练脚本（功能完整）
- `config.py` - 训练配置文件
- `README.md` - 本说明文件

## 快速开始

### 1. 环境准备

确保已安装Jittor并激活环境：
```bash
conda activate jt
```

### 2. 数据准备

确保COCO数据集已准备好：
```
data/coco2017_50/
├── train2017/          # 图片文件
└── annotations/
    └── instances_train2017.json  # 标注文件
```

### 3. 开始训练

使用主训练脚本：
```bash
cd /home/kyc/project/RT-DETR
python experiments/training/train_rtdetr.py
```

或使用生产级脚本：
```bash
python experiments/training/production_rtdetr_training.py
```

## 配置说明

在 `config.py` 中可以修改以下配置：

### 数据配置
- `max_images`: 使用的图片数量（None表示全部）
- `batch_size`: 批次大小
- `target_size`: 图片缩放尺寸

### 训练配置
- `num_epochs`: 训练轮数
- `learning_rate`: 学习率
- `weight_decay`: 权重衰减

### 模型配置
- `hidden_dim`: 隐藏层维度
- `num_queries`: 查询数量
- `num_decoder_layers`: 解码器层数

## 输出文件

训练完成后会生成：
- `checkpoints/rtdetr_jittor.pkl` - 训练好的模型
- `results/training_history.png` - 训练损失曲线图

## 技术特点

### 已解决的问题

1. **数据类型不匹配问题**
   - 修复了Focal Loss中float64/float32混合的问题
   - 所有tensor操作强制使用float32

2. **辅助损失问题**
   - 实现了完整的深度监督机制
   - 所有6层解码器的辅助输出都参与训练

3. **梯度计算问题**
   - 确保所有参数都有梯度
   - 避免了参数被设置为零的警告

### 核心修复

1. **FixedFocalLoss类**
   ```python
   # 使用scatter_操作替代one_hot，避免float64
   target_onehot = jt.zeros(src_logits.shape, dtype=jt.float32)
   target_onehot.scatter_(-1, target_classes.unsqueeze(-1), safe_float32(1.0))
   ```

2. **完整的辅助损失**
   ```python
   # 处理所有辅助输出
   if 'aux_outputs' in outputs:
       for i, aux_outputs in enumerate(outputs['aux_outputs']):
           # 计算辅助损失
   ```

3. **数据类型安全函数**
   ```python
   def safe_float32(tensor):
       # 确保所有tensor都是float32
   ```

## 性能表现

- ✅ 训练稳定，无数据类型错误
- ✅ 损失正常下降
- ✅ 所有参数参与训练
- ✅ 深度监督机制正常工作

## 注意事项

1. **Jittor警告**
   ```
   The `Parameter` interface isn't needed in Jittor
   ```
   这个警告可以安全忽略，不影响训练效果。

2. **内存使用**
   - 建议使用GPU训练
   - 根据显存大小调整batch_size

3. **训练时间**
   - 50张图片约需要几分钟
   - 完整数据集需要更长时间

## 扩展使用

### 使用更多数据
修改 `config.py` 中的 `max_images` 为 `None` 或更大的数值。

### 调整模型大小
修改 `MODEL_CONFIG` 中的参数：
- 增大 `hidden_dim` 提升模型容量
- 增大 `num_queries` 处理更多目标

### 自定义损失权重
修改 `LOSS_WEIGHTS` 中的权重值。

## 故障排除

如果遇到问题：

1. **CUDA内存不足**
   - 减小 `batch_size`
   - 减小 `target_size`

2. **训练不收敛**
   - 调整学习率
   - 检查数据质量

3. **其他错误**
   - 检查数据路径是否正确
   - 确保Jittor环境正常

## 联系方式

如有问题，请检查：
1. Jittor版本是否最新
2. CUDA环境是否正确
3. 数据格式是否符合COCO标准
