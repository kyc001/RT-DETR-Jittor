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

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 



该项目终极目标是利用jittor框架复现RT-DETR模型，实现论文中描述的实时目标检测任务。

在出现问题需要修复的时候，你可以参考rtdetr_pytorch内代码，将其使用jittor框架实现，翻译代码可以借助convert.py脚本

由于算力有限，在完成项目后，希望能够在大规模训练之前进行流程自检查，用一张照片进行训练不超过40次，再用训练出来的模型检测同一张照片，将可视化结果与真实标注做对比，data/coco2017_50/train2017/000000225405.jpg选择这张图片进行流程自检，推理所用照片也用这张


一切设计都应以rtdetr_pytorch内代码为准！

在出现问题需要修复的时候，你可以参考rtdetr_pytorch内代码，将其使用jittor框架实现，不兼容的函数如果可以单独实现就不要简化。

我们认为：只有训练后的模型能够正确检测出所有物体，才能算是项目成功，自检成功的标准是，训练后的模型能够成功识别物体个数并正确检测出物体的边界框，可以通过与数据原始标注进行对比实现。否则需要不断重复自检的过程检查错误，无法正确检测物体的模型直接删除，在检查的过程中始终维护核心代码。



在项目成功后，进行一次完整的训练，请将环境配置、数据准备脚本、训练脚本、测试脚本、与 PyTorch 实现对齐的实验 Log、性能 Log 都放在 README 中。如果计算资源有限，用少量数据的训练效果和 PyTorch 版本的结果对齐。请将训练过程 Log、Loss 曲线，结果、可视化等对齐情况进行记录。


可以参考官方文档https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/

The `Parameter` interface isn't needed in Jittor, this interface
does nothings and it is just used for compatible.
    
A Jittor Var is a Parameter
when it is a member of Module, if you don't want a Jittor
Var menber is treated as a Parameter, just name it startswith
underscore `_`.


注意到在可视化结果中，结果集中为person，且都分布在图片正中心，怀疑模型未成功学习到特征，检查一下标注数据中的坐标转换是否正确？


需要优化的是梯度传播问题


再次全面检查一遍代码功能，并将结构，文件名对齐，进一步修复，如果遇到问题可以参考https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html和from jittor.utils.pytorch_converter import convert 转换函数



w 0715 22:23:08.508958 08 grad.cc:81] grads[0] '' doesn't have gradient. It will be set to zero: Var(50:1:1:1:i1:o0:s0:n0:g1,float32,,0)[2,3,]


from jittor_rt_detr.src.nn.backbone.resnet import ResNet50
from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
from jittor_rt_detr.src.nn.criterion.rtdetr_criterion import build_criterion


对这张照片进行100次过拟合训练，/home/kyc/project/RT-DETR/data/coco2017_50/train2017/000000282037.jpg并用训练好的模型对这张照片进行推理，训练后的模型能够成功识别物体个数并正确检测出物体的边界框，可以通过与数据原始标注进行对比实现。否则需要不断重复自检的过程检查错误，无法正确检测物体的模型直接删除，在检查的过程中始终维护核心代码，不要重复创建新脚本


000000282037.jpg


因为现在已经通过了单元自检，我希望扩大训练的规模
希望使用train50作为训练集，训练50次，保存训练得到的模型，再使用val50进行测试，将测试结果数据进行记录，可视化
使用的脚本根据已有的单元自检的脚本进行迁移，尽量不要从头开始写，可能会出现别的问题

/home/kyc/project/RT-DETR/experiments/ultimate_sanity_check.py，参考rtdetr_pytorch内代码



戴老师您好,我选择用jittor框架实现的是RT-DETR模型,参数量为31M,我现在已经能确定复现的模型是可用的,
但是模型内有一个ResNet50骨干网络,一个混合编码器和一个transformer解码器,需要的数据量很大
由于计算资源有限,如果用少量数据的训练效果很差，学习不到特征。


首先，将pytorch版本与jittor版本对齐，然后应该添加的评估:
✅ COCO mAP@0.5 和 mAP@0.5:0.95
✅ 推理速度（FPS）测试
✅ 检测结果可视化对比
✅ 在验证集上的性能测试


我发现你修复有问题的脚本倾向于新建脚本而不是修改原有脚本，我不喜欢这样，这样容易造成版本混乱。
这版本的训练忘记导入预训练权重了，我们进行的是微调训练而不是从头训练

修改训练脚本:

实现加载预训练权重的逻辑。

实现“两阶段冻结训练”逻辑。

引入更强的数据增强（研究一下Albumentations）。

开启AMP (jt.flags.auto_mixed_precision_level = 2)。

开始训练:

第一阶段，用 1e-4 左右的学习率训练冻结的模型。

第二阶段，用 1e-6 左右的学习率对整个模型进行微调。

冻结训练法 (Freeze Training)
这是一个非常有效的微调技巧，可以极大地节省计算资源并防止模型在小数据上被“带偏”。

实施步骤（两阶段训练）：

第一阶段：冻结主干，只训练“头”

加载预训练权重后，将模型的主干网络（backbone, 即ResNet50部分）的所有参数设置为不可训练（param.requires_grad = False）。

只训练后面的Transformer解码器部分。因为主干网络已经学会了提取通用特征，我们先让模型专注于学习如何利用这些特征来定位和分类您的特定目标。

这个阶段可以用一个相对稍大的学习率（如1e-4）训练10-20个epochs。

第二阶段：解冻所有，整体微调

在第一阶段训练后，将整个模型的所有参数都设置为可训练（param.requires_grad = True）。

使用一个非常非常小的学习率（如1e-6）对整个模型进行“全局微调”，训练更多的epochs。这就像在已经画好的画上进行精修抛光。

