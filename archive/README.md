# RT-DETR 项目代码归档

本目录包含RT-DETR项目开发过程中的历史版本代码，这些代码已被更新的版本替代，但保留以供参考。

## 归档结构

### main_directory_old_versions/
主目录中的旧版本文件：
- `compare_inference_gt.py` - 推理结果与真实标注对比的早期版本
- `comprehensive_test.py` - 综合测试脚本的早期版本
- `convert.py` - 权重转换脚本的早期版本
- `convert_msdeformable_attention.py` - MS-Deformable Attention转换的早期版本
- `convert_msdeformable_attention_complete.py` - 完整MS-Deformable Attention转换
- `convert_rtdetr_decoder.py` - RT-DETR解码器转换的早期版本
- `convert_utils.py` - 转换工具的早期版本
- `inference_rt_detr.py` - RT-DETR推理脚本的早期版本
- `train_rt_detr.py` - RT-DETR训练脚本的早期版本
- `train_rtdetr_professional.py` - 专业版训练脚本
- `train_rtdetr_simplified.py` - 简化版训练脚本

### experiments/inference/
推理相关的历史版本：
- `detailed_analysis.py` - 详细分析版本
- `final_verification.py` - 最终验证版本
- `fixed_inference.py` - 修复版推理
- `improved_inference.py` - 改进版推理
- `position_analysis.py` - 位置分析版本
- `safe_inference.py` - 安全推理版本
- `simple_inference.py` - 简单推理版本

**当前使用版本**：
- `rtdetr_inference.py` - 主要推理脚本
- `inference_with_pretrained.py` - 预训练权重推理脚本

### experiments/sanity_check/
流程自检相关的历史版本：
- `complete_end_to_end_validation.py` - 完整端到端验证
- `complete_pytorch_aligned_validation.py` - 完整PyTorch对齐验证
- `complete_sanity_check.py` - 完整流程自检
- `correct_sanity_check.py` - 修正版流程自检
- `dtype_fixed_validation.py` - 数据类型修复验证
- `final_complete_validation.py` - 最终完整验证
- `fixed_sanity_check.py` - 修复版流程自检
- `improved_sanity_check.py` - 改进版流程自检
- `minimal_sanity_check.py` - 最小化流程自检
- `pytorch_aligned_validation.py` - PyTorch对齐验证
- `quick_sanity_check.py` - 快速流程自检
- `sanity_check.py` - 基础流程自检
- `strict_validation_check.py` - 严格验证检查
- `systematic_pytorch_aligned_validation.py` - 系统化PyTorch对齐验证
- `target_image_sanity_check.py` - 目标图像流程自检

**当前使用版本**：
- `final_success_validation.py` - 最终成功验证版本

### experiments/training/
训练相关的历史版本：
- `finetune_direct.py` - 直接微调版本
- `finetune_with_pretrained.py` - 预训练权重微调版本
- `optimized_50_images_training.py` - 50张图片优化训练
- `overfit_single_image.py` - 单张图片过拟合训练

**当前使用版本**：
- `train_rtdetr.py` - 主要训练脚本
- `production_rtdetr_training.py` - 生产环境训练脚本

### experiments/testing/
测试相关的历史版本：
- `deep_diagnosis.py` - 深度诊断
- `diagnose_model.py` - 模型诊断
- `quick_test.py` - 快速测试
- `test_rtdetr_model.py` - RT-DETR模型测试
- `verify_model_learning.py` - 模型学习验证

**当前使用版本**：
- `check_coordinate_conversion.py` - 坐标转换检查

### experiments/visualization/
可视化相关的历史版本：
- `vis.py` - 可视化脚本早期版本
- `visualize_results.py` - 结果可视化早期版本

**当前使用版本**：
- `show_ground_truth.py` - 显示真实标注

### experiments/solutions/
解决方案相关的历史版本：
- `optimized_solution.py` - 优化解决方案

**当前使用版本**：
- `complete_solution.py` - 完整解决方案

### jittor_rt_detr/src/nn/
核心神经网络模块的历史版本：
- `loss.py` - 原始损失函数实现
- `dtype_safe_loss.py` - 数据类型安全损失函数
- `simple_loss.py` - 简化版损失函数
- `model.py` - 原始模型实现
- `rtdetr_complete.py` - 完整RT-DETR模型早期版本
- `rtdetr_pytorch_aligned.py` - PyTorch对齐模型早期版本
- `dtype_safe_rtdetr.py` - 数据类型安全RT-DETR模型
- `ms_deformable_attention.py` - MS-Deformable Attention原始实现

**当前使用版本**：
- `rtdetr_complete_pytorch_aligned.py` - 最终成功的完整RT-DETR模型
- `loss_pytorch_aligned.py` - 最终成功的损失函数
- `msdeformable_attention_pytorch_aligned.py` - 最终成功的注意力机制
- `utils_pytorch_aligned.py` - PyTorch对齐工具函数
- `rtdetr_decoder_pytorch_aligned.py` - PyTorch对齐解码器

## 版本演进说明

### 推理模块演进
1. `simple_inference.py` → `improved_inference.py` → `fixed_inference.py` → `rtdetr_inference.py`
2. 主要改进：错误处理、坐标转换修复、类别映射修复

### 训练模块演进
1. `train_rtdetr_simplified.py` → `train_rtdetr_professional.py` → `train_rtdetr.py`
2. 主要改进：损失函数优化、数据类型修复、训练稳定性提升

### 流程自检演进
1. `sanity_check.py` → `improved_sanity_check.py` → `final_success_validation.py`
2. 主要改进：PyTorch对齐、数据类型一致性、完整性验证

### 核心模块演进
1. **损失函数**：`loss.py` → `dtype_safe_loss.py` → `loss_pytorch_aligned.py`
2. **RT-DETR模型**：`model.py` → `rtdetr_complete.py` → `rtdetr_complete_pytorch_aligned.py`
3. **注意力机制**：`ms_deformable_attention.py` → `msdeformable_attention_pytorch_aligned.py`
4. 主要改进：数据类型安全、PyTorch完全对齐、训练稳定性

## 重要里程碑

1. **数据类型问题解决** - `dtype_fixed_validation.py`
2. **PyTorch对齐完成** - `pytorch_aligned_validation.py`
3. **训练流程稳定** - `production_rtdetr_training.py`
4. **推理功能完善** - `rtdetr_inference.py`
5. **项目最终成功** - `final_success_validation.py`

## 使用建议

- 如需参考历史实现，请查看对应的归档文件
- 新功能开发请基于当前使用版本进行
- 如发现当前版本问题，可参考历史版本的解决方案
- 建议定期清理不再需要的归档文件

---
*归档时间：2025-07-15*
*归档原因：项目整理，保持主目录整洁*
