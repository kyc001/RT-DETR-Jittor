#!/usr/bin/env python3
"""
项目文件整理脚本
将项目根目录下的文件按功能分类整理到不同文件夹
"""

import os
import shutil
from pathlib import Path

def organize_project():
    """整理项目文件"""
    print("=" * 60)
    print("===      项目文件整理      ===")
    print("=" * 60)
    
    # 定义文件分类
    file_categories = {
        # 核心训练和推理脚本 - 保留在根目录
        'core_scripts': [
            'train.py',
            'small_scale_training.py'
        ],
        
        # 流程自检相关脚本 - 移动到 experiments/sanity_check/
        'sanity_check_scripts': [
            'complete_sanity_check.py',
            'correct_sanity_check.py',
            'fixed_sanity_check.py',
            'improved_sanity_check.py',
            'minimal_sanity_check.py',
            'quick_sanity_check.py',
            'sanity_check.py'
        ],
        
        # 推理和测试脚本 - 移动到 experiments/inference/
        'inference_scripts': [
            'detailed_analysis.py',
            'final_verification.py',
            'fixed_inference.py',
            'improved_inference.py',
            'position_analysis.py',
            'safe_inference.py',
            'simple_inference.py'
        ],
        
        # 解决方案和优化脚本 - 移动到 experiments/solutions/
        'solution_scripts': [
            'complete_solution.py',
            'optimized_solution.py'
        ],
        
        # 可视化脚本 - 移动到 experiments/visualization/
        'visualization_scripts': [
            'show_ground_truth.py',
            'vis.py',
            'visualize_results.py'
        ],
        
        # 工具脚本 - 移动到 tools/
        'tool_scripts': [
            'prepare_small_dataset.py',
            'quick_test.py',
            '1.py'
        ],
        
        # 结果图片 - 移动到 results/images/
        'result_images': [
            'complete_solution_result.jpg',
            'complete_visualization.jpg',
            'detailed_analysis_result.jpg',
            'fixed_inference_result.jpg',
            'ground_truth_visualization.jpg',
            'improved_sanity_check_result.jpg',
            'position_analysis_result.jpg',
            'sanity_check_result.jpg',
            'vis_epoch50_359855.jpg',
            'vis_epoch50_7991.jpg'
        ],
        
        # 报告文档 - 移动到 docs/reports/
        'reports': [
            'FINAL_PROBLEM_ANALYSIS.md',
            'FINAL_SANITY_CHECK_REPORT.md',
            'SANITY_CHECK_REPORT.md'
        ]
    }
    
    # 创建目标目录
    target_dirs = {
        'experiments/sanity_check': 'sanity_check_scripts',
        'experiments/inference': 'inference_scripts', 
        'experiments/solutions': 'solution_scripts',
        'experiments/visualization': 'visualization_scripts',
        'tools': 'tool_scripts',
        'results/images': 'result_images',
        'docs/reports': 'reports'
    }
    
    # 创建所有目标目录
    for target_dir in target_dirs.keys():
        os.makedirs(target_dir, exist_ok=True)
        print(f"✅ 创建目录: {target_dir}")
    
    # 移动文件
    moved_files = []
    
    for target_dir, category in target_dirs.items():
        files_to_move = file_categories[category]
        
        print(f"\n>>> 移动文件到 {target_dir}:")
        
        for filename in files_to_move:
            if os.path.exists(filename):
                target_path = os.path.join(target_dir, filename)
                shutil.move(filename, target_path)
                moved_files.append((filename, target_path))
                print(f"  ✅ {filename} -> {target_path}")
            else:
                print(f"  ⚠️ 文件不存在: {filename}")
    
    # 保留在根目录的重要文件
    keep_in_root = [
        'README.md',
        'train.py',
        'small_scale_training.py',
        'data/',
        'checkpoints/',
        'jittor_rt_detr/',
        'rtdetr_pytorch/',
        'jittor_convert/',
        'command'
    ]
    
    print(f"\n>>> 保留在根目录的文件:")
    for item in keep_in_root:
        if os.path.exists(item):
            print(f"  ✅ {item}")
        else:
            print(f"  ⚠️ 不存在: {item}")
    
    # 创建整理后的目录结构说明
    create_structure_readme()
    
    print(f"\n" + "=" * 60)
    print("🎉 项目整理完成！")
    print(f"总共移动了 {len(moved_files)} 个文件")
    print("=" * 60)
    
    return moved_files

def create_structure_readme():
    """创建项目结构说明文件"""
    structure_content = """# RT-DETR Jittor 项目结构

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
"""
    
    with open("PROJECT_STRUCTURE.md", "w", encoding='utf-8') as f:
        f.write(structure_content)
    
    print("✅ 创建项目结构说明: PROJECT_STRUCTURE.md")

def main():
    print("开始整理项目文件...")
    
    # 确认操作
    response = input("确定要整理项目文件吗？这将移动很多文件到子目录中。(y/N): ")
    
    if response.lower() in ['y', 'yes']:
        moved_files = organize_project()
        
        print(f"\n📋 移动的文件列表:")
        for old_path, new_path in moved_files:
            print(f"  {old_path} -> {new_path}")
        
        print(f"\n💡 如果需要撤销，可以手动将文件移回根目录")
        
    else:
        print("取消整理操作")

if __name__ == "__main__":
    main()
