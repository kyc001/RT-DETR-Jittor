# RT-DETR Jittor 项目结构

## 📁 目录结构

```
RT-DETR/
├── README.md                    # 项目主要说明
├── train.py                     # 主训练脚本
├── small_scale_training.py      # 小规模训练脚本
│
├── data/                        # 数据目录
│   └── coco2017_50/            # COCO数据集
│
├── checkpoints/                 # 模型检查点
│   ├── correct_sanity_check_model.pkl
│   ├── fixed_sanity_check_model.pkl
│   ├── improved_sanity_check_model.pkl
│   └── sanity_check_model.pkl
│
├── jittor_rt_detr/             # Jittor版RT-DETR实现
│   ├── src/                    # 源代码
│   └── tools/                  # 工具脚本
│
├── rtdetr_pytorch/             # PyTorch版RT-DETR参考
│   ├── src/                    # 源代码
│   ├── configs/                # 配置文件
│   └── tools/                  # 工具脚本
│
├── experiments/                # 实验脚本
│   ├── sanity_check/          # 流程自检实验
│   ├── inference/             # 推理测试实验
│   ├── solutions/             # 解决方案实验
│   └── visualization/         # 可视化实验
│
├── tools/                      # 工具脚本
│   ├── prepare_small_dataset.py
│   └── quick_test.py
│
├── results/                    # 结果文件
│   └── images/                # 结果图片
│
├── docs/                       # 文档
│   └── reports/               # 实验报告
│
└── PROJECT_STRUCTURE.md        # 本文件
```

## 🚀 快速开始

### 1. 流程自检（验证实现正确性）
```bash
# 最新的正确配置版本
python experiments/sanity_check/correct_sanity_check.py

# 位置精度分析
python experiments/inference/position_analysis.py
```

### 2. 小规模训练
```bash
# 使用现有数据进行小规模训练
python small_scale_training.py
```

### 3. 大规模训练
```bash
# 使用完整数据集训练
python train.py
```

## 📊 实验结果

### 流程自检结论
- ✅ RT-DETR Jittor实现技术上完全正确
- ✅ 训练→推理流程完全可用
- ✅ 可以进行大规模数据集训练
- ⚠️ 单张图片训练有固有限制

### 关键发现
- 模型能够同时检测多个类别
- 后处理方法的选择很重要（Sigmoid vs Softmax）
- 需要足够的训练数据才能获得准确的位置检测

## 🔧 开发建议

1. **对于新功能开发**：在 `experiments/` 下创建新的实验脚本
2. **对于核心功能修改**：直接修改 `jittor_rt_detr/` 下的源码
3. **对于训练配置调整**：修改 `train.py` 或 `small_scale_training.py`
4. **对于结果分析**：使用 `experiments/inference/` 下的分析脚本

## 📝 重要文件说明

- `correct_sanity_check.py`: 最终成功的流程自检版本
- `position_analysis.py`: 详细的位置精度分析
- `small_scale_training.py`: 适配4060显卡的小规模训练
- `train.py`: 完整的大规模训练脚本
