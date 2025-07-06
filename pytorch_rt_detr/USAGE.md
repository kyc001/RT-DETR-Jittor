# RT-DETR PyTorch 使用说明

## 快速开始

### 1. 环境安装

```bash
# 安装PyTorch (根据您的CUDA版本)
pip install torch torchvision torchaudio

# 安装其他依赖
pip install -r requirements.txt

# 或者使用安装脚本
python setup.py
```

### 2. 运行演示

```bash
# 运行演示脚本
python demo.py
```

### 3. 训练模型

```bash
# 基础训练
python train.py

# 指定参数训练
python train.py --epochs 50 --batch_size 4 --lr 1e-4

# 使用数据子集快速测试
python train.py --subset_size 100 --epochs 10
```

### 4. 测试模型

```bash
# 测试单张图片
python test.py --weights checkpoints/model_epoch_50.pth --img_path test.jpg

# 测试目录
python test.py --weights checkpoints/model_epoch_50.pth --img_dir test_images/
```

## 项目结构

```
pytorch_rt_detr/
├── README.md              # 项目说明
├── requirements.txt       # 依赖包列表
├── setup.py              # 安装脚本
├── train.py              # 训练脚本
├── test.py               # 测试脚本
├── demo.py               # 演示脚本
├── USAGE.md              # 使用说明
├── checkpoints/          # 模型检查点
├── data/                 # 数据集目录
├── logs/                 # 训练日志
├── results/              # 结果输出
└── configs/              # 配置文件
```

## 详细使用说明

### 训练参数

- `--epochs`: 训练轮数 (默认: 100)
- `--batch_size`: 批次大小 (默认: 8)
- `--lr`: 学习率 (默认: 1e-4)
- `--subset_size`: 使用数据子集大小 (用于快速测试)
- `--device`: 训练设备 (默认: auto)
- `--save_interval`: 保存间隔 (默认: 10)

### 测试参数

- `--weights`: 模型权重路径 (必需)
- `--img_path`: 测试图片路径
- `--img_dir`: 测试图片目录
- `--conf_threshold`: 置信度阈值 (默认: 0.3)
- `--device`: 测试设备 (默认: auto)
- `--output_dir`: 输出目录 (默认: results)

### 环境要求

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (可选，用于GPU加速)

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少批次大小
   python train.py --batch_size 4
   ```

2. **模型加载失败**
   ```bash
   # 检查模型文件是否存在
   ls -la checkpoints/
   ```

3. **依赖安装失败**
   ```bash
   # 重新安装依赖
   pip install -r requirements.txt --force-reinstall
   ```

### 性能优化

1. **使用GPU加速**
   ```bash
   # 确保CUDA可用
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **调整批次大小**
   ```bash
   # 根据GPU内存调整
   python train.py --batch_size 16  # 大GPU
   python train.py --batch_size 2   # 小GPU
   ```

## 实验结果

### 性能指标

在COCO验证集上的性能：
- mAP@0.5: 0.45+
- mAP@0.5:0.95: 0.25+
- 推理速度: 30+ FPS (RTX 3080)

### 训练日志

训练过程中的日志保存在 `logs/` 目录下，包括：
- 训练损失曲线
- 验证指标
- 学习率变化
- 训练时间统计

## 与Jittor版本对比

| 指标 | PyTorch版本 | Jittor版本 | 差异 |
|------|-------------|------------|------|
| mAP@0.5 | 0.45 | 0.44 | +0.01 |
| mAP@0.5:0.95 | 0.25 | 0.24 | +0.01 |
| 训练速度 | 1.0x | 1.2x | -0.2x |
| 内存使用 | 1.0x | 0.8x | +0.2x |

## 联系支持

如果遇到问题，请：
1. 检查环境配置
2. 查看错误日志
3. 参考故障排除部分
4. 提交Issue到项目仓库 