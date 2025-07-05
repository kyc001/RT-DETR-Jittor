# RT-DETR in Jittor (Jittor 版 RT-DETR)

本项目是 **RT-DETR (Real-Time DEtection TRansformer)** 模型在 Jittor 深度学习框架下的实现。该项目作为“新芽计划”第二阶段的成果，旨在展示在模型复现、训练、性能对齐方面的工程能力。

![Jittor Logo](https://raw.githubusercontent.com/Jittor/jittor/master/assets/logo.png)

## 1. 项目特性
- **骨干网络 (Backbone)**: 使用标准的 ResNet-50 网络进行特征提取。
- **编码器/解码器**: 基于 Transformer 的编码器和解码器结构，与 DETR 系列论文保持一致。
- **损失函数**: 完整的损失函数实现，包括分类损失 (交叉熵)、边界框损失 (L1 Loss) 和 GIoU 损失。
- **训练流程**: 支持多批次 (Multi-Batch) 训练，并集成了学习率衰减策略。
- **评估脚本**: 包含一个独立的测试脚本，用于在验证集上进行推理和评估。

## 2. 环境配置

首先，请确保您已正确安装 Jittor。具体步骤请参考 [Jittor 官方安装文档](https://cg.cs.tsinghua.edu.cn/jittor/tutorial/install/)。

然后，安装项目所需的其他 Python 依赖包：
```bash
pip install -r requirements.txt
```
`requirements.txt` 文件应包含以下内容:
```txt
numpy
scipy
Pillow
tqdm
```

## 3. 数据准备

本项目使用 COCO 2017 数据集。

1.  **下载数据**: 建议从 [COCO 官网](https://cocodataset.org/#download) 下载 `train2017` 图片、`val2017` 图片以及对应的 `annotations` 标注文件。
2.  **组织数据**: 请将下载并解压后的文件，按照以下目录结构存放：
    ```
    RT-DETR-Jittor/
    └── data/
        └── coco/
            ├── annotations/
            │   ├── instances_train2017.json
            │   └── instances_val2017.json
            ├── train2017/
            │   └── 000000000009.jpg
            └── val2017/
                └── 000000000139.jpg
    ```

## 4. 模型训练

要使用默认参数在 `val2017` 数据集上开始训练（用于快速测试），请直接运行：
```bash
python train.py
```
若想在完整的 `train2017` 数据集上训练以获得更好性能，请修改 `train.py` 中的默认路径：
```python
# 在 train.py 中
parser.add_argument('--img_dir', type=str, default='data/coco/train2017')
parser.add_argument('--ann_file', type=str, default='data/coco/annotations/instances_train2017.json')
```

您也可以通过命令行参数自定义训练配置：
```bash
python train.py --lr 1e-5 --batch_size 8 --epochs 100
```

## 5. 测试与评估

使用 `test.py` 脚本来评估一个训练好的模型。您需要提供已保存的模型权重文件 (`.pkl`) 的路径。

```bash
python test.py --weights model_epoch_50.pkl
```
该脚本将在 `val2017` 验证集上运行推理。要计算完整的 mAP 指标，其输出结果需要经过格式化后，使用 `pycocotools` 等官方工具进行处理。

## 6. 实验日志与对齐情况

本部分记录了此 Jittor 实现在少量数据上与标准 PyTorch 版本的对齐情况。

### 训练过程日志

以下是训练过程中的日志样例，清晰地展示了学习率变化以及各项损失的分解情况：
```
# TODO: 请在此处粘贴您实际的训练日志
开始训练...
Epoch: 1/50, Batch: 0/1250, LR: 1.0e-04, Loss: 8.5123 (cls: 2.8910, bbox: 3.1234, giou: 2.5001)
Epoch: 1/50, Batch: 50/1250, LR: 1.0e-04, Loss: 5.4321 (cls: 1.9876, bbox: 2.0123, giou: 1.4322)
...
Epoch 1 完成, 平均 Loss: 5.8765
```

### 损失曲线 (Loss Curve)

训练损失曲线呈现出稳定的下降趋势，表明模型在进行有效学习。

*（要生成此图，请在训练后运行 `python loss_plot.py`，然后将生成的 `loss_curve.png` 图片嵌入到此处。）*

![训练损失曲线](loss_curve.png)

### 性能对齐 (vs. PyTorch)

| 指标 | PyTorch (参考) | Jittor (本项目) |
| :--- | :---: | :---: |
| **平均损失 (10 epochs)** | ~4.5 | ~4.8 |
| **mAP (在验证子集)** | TODO | TODO |
| **推理速度** | TODO | TODO |

*（请注意：以上为占位数据。您需要在一个相同的、小规模的数据子集上运行一个 PyTorch 版本，以获得用于公平比较的参考数值。）*