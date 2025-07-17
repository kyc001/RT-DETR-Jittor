# RT-DETR 项目文件结构

## 📁 目录说明

### current/ - 当前使用的文件
- **sanity_check/** - 过拟合验证相关
  - `ultimate_sanity_check.py` - 单张图像过拟合验证（✅ 成功版本）
  - `sanity_check_5_images.py` - 5张图像过拟合训练（✅ 修复坐标格式后）
  - `sanity_check_10_images_fixed.py` - 10张图像修复版本（🔧 解决模式崩塌）

- **training/** - 训练相关
  - `full_scale_training.py` - 完整训练脚本
  - `train_50_images_50_epochs.py` - 50张图像训练

- **inference/** - 推理相关
  - `test_trained_model.py` - 测试训练好的模型
  - `test_sanity_check_model.py` - 测试过拟合模型

- **visualization/** - 可视化相关
  - `visualize_detection.py` - 检测结果可视化

### archive/ - 存档文件
- **analysis/** - 分析和诊断工具
- **training_experiments/** - 过时的训练实验
- **debug/** - 调试相关文件

### utils/ - 工具脚本
- `organize_project_files.py` - 项目文件整理脚本

## 🎯 推荐使用流程

1. **单张图像验证**: `current/sanity_check/ultimate_sanity_check.py`
2. **5张图像训练**: `current/sanity_check/sanity_check_5_images.py`
3. **10张图像训练**: `current/sanity_check/sanity_check_10_images_fixed.py`
4. **完整训练**: `current/training/full_scale_training.py`
5. **模型测试**: `current/inference/test_trained_model.py`

## 📊 项目状态

- ✅ 单张图像过拟合: 完全成功
- ✅ 5张图像过拟合: 坐标格式修复后成功
- 🔧 10张图像过拟合: 修复模式崩塌问题中
- 🚀 完整训练: 准备就绪

## 🔧 已解决的问题

1. **坐标格式不一致**: 统一使用 [x1, y1, x2, y2] 格式
2. **梯度传播问题**: 修复MSDeformableAttention和编码器输出头
3. **Jittor API兼容性**: 完全对齐PyTorch版本
4. **数据类型问题**: 解决backward propagation数据类型错误

## 🎯 下一步计划

1. 完全解决10张图像的模式崩塌问题
2. 扩展到20-50张图像训练
3. 进行完整COCO数据集训练
4. 性能优化和部署准备
