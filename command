# 常用命令记录

# 基础训练和测试命令
python test.py --weights checkpoints/model_epoch_20.pkl --img_path 000000000139.jpg --conf_threshold 0.3

python vis.py --weights checkpoints/model_epoch_50.pkl --img_path test.png --conf_threshold 0.3

python train.py --epochs 2 --batch_size 4 --lr 1e-4 

python train.py --subset_size 40

python vis.py --weights checkpoints/model_epoch_20.pkl --img_path test.png

# 修复版测试脚本 (推荐使用)
python test_fixed.py --weights checkpoints/model_epoch_20.pkl --img_path test.png

python test_simple.py --weights checkpoints/model_epoch_20.pkl --img_path test.png

# 完整项目使用流程
# 1. 安装项目
python setup.py

# 2. 运行演示
python demo.py

# 3. 开始训练
python train.py --subset_size 100 --epochs 20

# 4. 测试模型 (使用修复版)
python test_fixed.py --weights checkpoints/model_epoch_20.pkl --img_path test.jpg

# 5. 可视化结果
python vis.py --weights checkpoints/model_epoch_20.pkl --img_path test.jpg
python vis.py --weights checkpoints/model_epoch_20.pkl --img_path test.png

# 6. 性能对比
python benchmark.py

# 7. 实验记录
python experiment_log.py

# 8. 数据准备
python prepare_data.py --download --extract

# 使用说明
# === 使用说明 ===
# 1. 运行演示: python demo.py
# 2. 开始训练: python train.py
# 3. 测试模型: python test_fixed.py --weights checkpoints/model_epoch_20.pkl --img_path test.png
# 4. 可视化结果: python vis.py --weights checkpoints/model_epoch_20.pkl --img_path test.png
# 5. 性能对比: python benchmark.py
# 6. 实验记录: python experiment_log.py

# 故障排除
# 如果遇到Jittor兼容性问题，请使用以下修复版脚本：
# - test_fixed.py: 最终修复版测试脚本
# - test_simple.py: 简化版测试脚本
# - vis.py: 修复版可视化脚本