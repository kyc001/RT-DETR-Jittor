#!/usr/bin/env python3
"""
整理项目文件，按功能分类
"""

import os
import shutil
from datetime import datetime

def create_directory_structure():
    """创建整理后的目录结构"""
    directories = {
        'archive': '存档的旧文件和实验',
        'archive/debug': '调试相关的文件',
        'archive/training_experiments': '各种训练实验',
        'archive/analysis': '数据分析和诊断文件',
        'current': '当前正在使用的文件',
        'current/sanity_check': '过拟合验证相关',
        'current/training': '训练相关',
        'current/inference': '推理相关',
        'current/visualization': '可视化相关',
        'results': '训练结果和模型',
        'utils': '工具脚本'
    }
    
    base_dir = '/home/kyc/project/RT-DETR/experiments'
    
    for dir_path, description in directories.items():
        full_path = os.path.join(base_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
        
        # 创建README文件说明目录用途
        readme_path = os.path.join(full_path, 'README.md')
        if not os.path.exists(readme_path):
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(f"# {dir_path}\n\n{description}\n\n创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return directories

def organize_files():
    """按功能整理文件"""
    base_dir = '/home/kyc/project/RT-DETR/experiments'
    
    # 文件分类规则
    file_categories = {
        # 当前正在使用的核心文件
        'current/sanity_check': [
            'ultimate_sanity_check.py',  # 单张图像过拟合验证（成功版本）
            'sanity_check_5_images.py',  # 5张图像过拟合训练（修复坐标格式后）
            'sanity_check_10_images_fixed.py',  # 10张图像修复版本
        ],
        
        'current/training': [
            'full_scale_training.py',  # 完整训练脚本
            'train_50_images_50_epochs.py',  # 50张图像训练
        ],
        
        'current/inference': [
            'test_trained_model.py',  # 测试训练好的模型
            'test_sanity_check_model.py',  # 测试过拟合模型
        ],
        
        'current/visualization': [
            'visualize_detection.py',  # 检测结果可视化
        ],
        
        # 分析和诊断工具
        'archive/analysis': [
            'analyze_10_images_problem.py',  # 分析10张图像问题
            'analyze_train_data.py',  # 训练数据分析
            'analyze_val_dataset.py',  # 验证数据分析
            'debug_coordinate_format.py',  # 坐标格式调试
            'debug_model_output.py',  # 模型输出调试
            'debug_model_weights.py',  # 模型权重调试
            'debug_parameters.py',  # 参数调试
            'debug_class_mapping.py',  # 类别映射调试
            'diagnose_same_output_problem.py',  # 相同输出问题诊断
        ],
        
        # 过时的训练实验
        'archive/training_experiments': [
            'sanity_check_10_images.py',  # 10张图像训练（有模式崩塌问题）
            'test_fixed_model.py',  # 测试修复模型
            'quick_training_test.py',  # 快速训练测试
            'start_full_training.py',  # 开始完整训练
            'test_full_training.py',  # 测试完整训练
            'test_with_train_images.py',  # 用训练图像测试
        ],
        
        # 工具脚本
        'utils': [
            'cleanup_debug_files.py',  # 清理调试文件
            'organize_project_files.py',  # 本文件
        ]
    }
    
    print("🗂️ 开始整理项目文件...")
    print("=" * 60)
    
    moved_files = 0
    skipped_files = 0
    
    for target_dir, files in file_categories.items():
        target_path = os.path.join(base_dir, target_dir)
        
        print(f"\n📁 整理到 {target_dir}/:")
        
        for filename in files:
            source_path = os.path.join(base_dir, filename)
            dest_path = os.path.join(target_path, filename)
            
            if os.path.exists(source_path):
                try:
                    shutil.move(source_path, dest_path)
                    print(f"   ✅ {filename}")
                    moved_files += 1
                except Exception as e:
                    print(f"   ❌ {filename}: {e}")
                    skipped_files += 1
            else:
                print(f"   ⚠️ {filename} (文件不存在)")
                skipped_files += 1
    
    # 移动已存在的目录
    existing_dirs = {
        'debug_archive': 'archive/debug',
        'inference': 'current/inference',
        'training': 'current/training',
        'visualization': 'current/visualization',
        'sanity_check': 'current/sanity_check',
        'testing': 'archive/testing',
        'solutions': 'archive/solutions',
        'pretrained': 'current/pretrained',
        'reports': 'archive/reports',
        'overfit_data': 'current/overfit_data'
    }
    
    print(f"\n📁 整理目录:")
    for source_dir, target_dir in existing_dirs.items():
        source_path = os.path.join(base_dir, source_dir)
        target_path = os.path.join(base_dir, target_dir)
        
        if os.path.exists(source_path) and source_path != target_path:
            try:
                # 如果目标目录已存在，合并内容
                if os.path.exists(target_path):
                    for item in os.listdir(source_path):
                        item_source = os.path.join(source_path, item)
                        item_target = os.path.join(target_path, item)
                        if not os.path.exists(item_target):
                            shutil.move(item_source, item_target)
                    os.rmdir(source_path)
                else:
                    shutil.move(source_path, target_path)
                
                print(f"   ✅ {source_dir} -> {target_dir}")
                moved_files += 1
            except Exception as e:
                print(f"   ❌ {source_dir}: {e}")
                skipped_files += 1
    
    print(f"\n📊 整理统计:")
    print(f"   成功移动: {moved_files} 个文件/目录")
    print(f"   跳过: {skipped_files} 个文件/目录")
    
    return moved_files, skipped_files

def create_project_overview():
    """创建项目概览文件"""
    overview_content = """# RT-DETR 项目文件结构

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
"""
    
    overview_path = '/home/kyc/project/RT-DETR/experiments/PROJECT_OVERVIEW.md'
    with open(overview_path, 'w', encoding='utf-8') as f:
        f.write(overview_content)
    
    print(f"📋 项目概览已创建: {overview_path}")

def main():
    print("🗂️ RT-DETR 项目文件整理")
    print("按功能分类整理实验文件")
    print("=" * 60)
    
    # 1. 创建目录结构
    print("1. 创建目录结构...")
    directories = create_directory_structure()
    print(f"✅ 创建了 {len(directories)} 个目录")
    
    # 2. 整理文件
    print("\n2. 整理文件...")
    moved_files, skipped_files = organize_files()
    
    # 3. 创建项目概览
    print("\n3. 创建项目概览...")
    create_project_overview()
    
    # 4. 总结
    print(f"\n🎉 整理完成!")
    print(f"   移动了 {moved_files} 个文件/目录")
    print(f"   跳过了 {skipped_files} 个文件/目录")
    print(f"\n📁 新的项目结构:")
    print(f"   current/ - 当前使用的核心文件")
    print(f"   archive/ - 存档的实验和调试文件")
    print(f"   utils/ - 工具脚本")
    print(f"\n📋 查看 PROJECT_OVERVIEW.md 了解详细说明")

if __name__ == "__main__":
    main()
