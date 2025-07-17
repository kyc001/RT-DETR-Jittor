#!/usr/bin/env python3
"""
清理training目录中的冗余文件
"""

import os
import shutil

def cleanup_training_directory():
    """清理training目录，只保留核心文件"""
    training_dir = '/home/kyc/project/RT-DETR/experiments/current/training'
    archive_dir = '/home/kyc/project/RT-DETR/experiments/archive/training_experiments'
    
    # 需要保留的核心文件
    keep_files = {
        'full_scale_training.py',  # 完整训练脚本
        'train_50_images_50_epochs.py',  # 50张图像训练
        'production_rtdetr_training.py',  # 生产环境训练
        'config.py',  # 配置文件
        'README.md',  # 说明文件
    }
    
    # 需要移动到archive的文件
    archive_files = [
        'aligned_overfit_test.py',
        'complete_alignment_test.py', 
        'complete_functionality_test.py',
        'dtype_safe_overfit_test.py',
        'final_complete_test.py',
        'fixed_aligned_test.py',
        'fully_aligned_test.py',
        'overfit_single_image_v2.py',
        'simple_aligned_test.py',
        'simple_overfit_test.py',
        'train_rtdetr.py'
    ]
    
    print("🧹 清理training目录...")
    print("=" * 50)
    
    moved_count = 0
    
    for filename in archive_files:
        source_path = os.path.join(training_dir, filename)
        dest_path = os.path.join(archive_dir, filename)
        
        if os.path.exists(source_path):
            try:
                shutil.move(source_path, dest_path)
                print(f"   ✅ 移动: {filename}")
                moved_count += 1
            except Exception as e:
                print(f"   ❌ 失败: {filename} - {e}")
        else:
            print(f"   ⚠️ 不存在: {filename}")
    
    # 检查剩余文件
    remaining_files = []
    for item in os.listdir(training_dir):
        if os.path.isfile(os.path.join(training_dir, item)):
            remaining_files.append(item)
    
    print(f"\n📊 清理结果:")
    print(f"   移动文件: {moved_count} 个")
    print(f"   保留文件: {len(remaining_files)} 个")
    
    print(f"\n📁 保留的核心文件:")
    for filename in remaining_files:
        if filename in keep_files:
            print(f"   ✅ {filename}")
        else:
            print(f"   ⚠️ {filename} (未在保留列表中)")
    
    return moved_count

def main():
    print("🧹 RT-DETR Training目录清理")
    print("移除冗余的实验文件，只保留核心训练脚本")
    print("=" * 60)
    
    moved_count = cleanup_training_directory()
    
    print(f"\n🎉 清理完成!")
    print(f"   移动了 {moved_count} 个文件到 archive/training_experiments/")
    print(f"\n📁 current/training/ 现在只包含核心训练脚本")

if __name__ == "__main__":
    main()
