# Jittor 实现 RT-DETR

## 1. 环境配置
- Python 3.7
- Jittor >= 1.3.7
- 依赖见 requirements.txt

## 2. 数据准备
- 运行 `bash data/download_coco_val2017.sh`
- 数据结构如下：
  - data/coco/val2017/
  - data/coco/annotations/instances_val2017.json

## 3. 训练
```bash
python train.py
```

## 4. 测试与可视化
```bash
python test.py  # 如需
python vis.py
```

## 5. 实验对齐
- Loss 曲线对比（见 loss_curve.png）
- 推理速度对比
- mAP 对比

## 6. 参考
- [RT-DETR 论文](https://arxiv.org/abs/2304.08069)
- [Jittor](https://github.com/Jittor/jittor)
- [PyTorch 版 RT-DETR](https://github.com/AgentMaker/rtdetr-pytorch) 