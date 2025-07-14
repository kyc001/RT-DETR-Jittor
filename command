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




conda create -n py python=3.10 -y
conda activate py
pip install "numpy<2.0" torch==2.0.1 torchvision==0.15.2 onnx==1.14.0 onnxruntime==1.15.1 pycocotools PyYAML scipy transformers autopep8 -i https://pypi.tuna.tsinghua.edu.cn/simple


# 首先确保您已不在任何环境中
conda deactivate

# 然后彻底删除它 (请将jt替换成您要删除的实际环境名)
conda remove --name py --all



which g++ && g++ --version
conda create -n jt -c conda-forge python=3.7 gcc_linux-64 gxx_linux-64 -y
conda activate jt 
# 安装 Jittor
rm -rf ~/.cache/jittor/
python -m pip install jittor
rm -rf /home/kyc/.cache/jittor/cutlass
python3.7 -m jittor.test.test_example
python3.7 -m jittor.test.test_cudnn_op

from torchvision import datapoints
from torchvision import tv_tensors



python -c "from torchvision import tv_tensors; print(tv_tensors)"


conda activate py


python -m pip uninstall torch torchvision -y
rm -rf ~/.cache/pip

python -m pip install torch torchvision --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple 

# 首先，确保您位于正确的项目根目录
cd ~/project/RT-DETR/pytorch_rt_detr
python tools/train.py --config configs/rtdetr/rtdetr_r18vd_6x_coco.yml
python -m tools.train --config configs/rtdetr/rtdetr_r18vd_6x_coco.yml

pip install -r requirements.txt

现在我们关心jittor版模型运行以及推理情况，环境已经配置好了，conda activate jt即可激活
如你所见，思路是利用单张图片进行小规模训练，将得到的模型进行推理，验证训练逻辑，模型逻辑正确性。
single_bear_training中使用一张熊的照片完成了单目标检测，接下来要完成multi_target_training完成多目标检测，检测对象为人和滑雪板，请你帮我完善multi_target_training代码

主要的问题有：无法正确识别两种物体，只能同时识别出一种，同一物体多次计数，物体边界框没有正确框住。

你可以重新训练，但是为了节省效率，每次训练的epoches都不能超过30

评估模型推理结果的正确性你可以与原始标注的数据进行对照，找出问题

一张图片做多目标检测训练，目的是快速验证“模型结构、训练流程、推理流程”是否通畅、无bug，而不是追求检测精度或泛化能力。也就是说，这其实是一个“流程自检”或“最小可行性验证（sanity check）”，只要模型能学会这张图片上的目标，推理时能正确检测出所有标注物体，流程就算没问题