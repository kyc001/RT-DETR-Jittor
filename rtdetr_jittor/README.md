# RT-DETR Jittor

RT-DETR (Real-Time Detection Transformer) 的 Jittor 框架实现，从 PyTorch 版本迁移。

## 模型参数

| 模块 | 参数量 |
|------|--------|
| ResNet50 Backbone | 23.58M |
| HybridEncoder | 11.97M |
| RTDETRTransformer | 7.47M |
| **总计** | **43.02M** |

> 注：Jittor 将 BatchNorm 的 running_mean/running_var 计入参数，PyTorch 方式统计约为 42.94M

## 环境配置

```bash
# 创建环境
micromamba create -n jt python=3.8
micromamba activate jt

# 安装 Jittor
pip install jittor

# 安装依赖
pip install pycocotools PyYAML scipy pillow matplotlib numpy
```

## 测试

```bash
# 模块测试
python test_modules.py

# EMA 测试
python test_ema.py
```

## 项目结构

```
rtdetr_jittor/
├── configs/                    # 配置文件
│   └── rtdetr/
├── src/
│   ├── core/                   # 配置系统
│   ├── optim/
│   │   └── ema.py              # EMA 模块
│   ├── components/
│   │   └── trainer.py          # 训练器（支持EMA）
│   ├── nn/
│   │   └── backbone/
│   │       └── resnet.py       # ResNet Backbone
│   └── zoo/rtdetr/
│       ├── hybrid_encoder.py   # HybridEncoder (FPN+PAN)
│       ├── rtdetr_decoder.py   # RTDETRTransformer
│       ├── rtdetr_criterion.py # 损失函数
│       ├── box_ops.py          # Box 操作
│       ├── denoising.py        # 去噪模块
│       └── utils.py            # 工具函数
├── tools/
│   ├── train.py                # 训练脚本
│   ├── eval.py                 # 评估脚本
│   └── convert_weights.py      # 权重转换工具
└── test_modules.py             # 测试脚本
```

## 权重转换

```bash
# PyTorch → Jittor
python tools/convert_weights.py --pt2jt -i model.pth -o model.pkl

# Jittor → PyTorch
python tools/convert_weights.py --jt2pt -i model.pkl -o model.pth
```

## 训练（带EMA）

```python
from src.components.trainer import RTDETRTrainer

trainer = RTDETRTrainer(
    model, criterion, optimizer,
    use_ema=True,
    ema_decay=0.9999,
    ema_warmups=2000
)

# 训练时自动更新EMA
trainer.train(dataset, num_epochs=72)

# 评估时使用EMA模型
eval_model = trainer.get_eval_model()
```

## Jittor 兼容性说明

### API 替代方案

| PyTorch | Jittor |
|---------|--------|
| `torch.tile()` | 自定义 `tile()` 函数 |
| `torch.topk(dim=1)` | 逐 batch 使用 `argsort` |
| `F.binary_cross_entropy_with_logits` | 自定义实现 |
| `nn.ModuleList(generator)` | `nn.ModuleList([list])` |

### 内存泄漏防护

由于 Jittor 使用惰性求值，需要在关键位置调用 `jt.sync_all()`:

```python
optimizer.step(loss)
jt.sync_all()  # 防止计算图累积

ema.update(model)  # EMA 内部已包含 sync_all()
```

## 参考

- [RT-DETR PyTorch](https://github.com/lyuwenyu/RT-DETR)
- [Jittor](https://github.com/Jittor/jittor)

## License

Apache License 2.0
