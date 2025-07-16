#!/usr/bin/env python3
"""
清理调试文件脚本
将重复的调试脚本移动到archive文件夹中，保持项目整洁
"""

import os
import shutil
from pathlib import Path

def cleanup_debug_files():
    """清理experiments目录中的重复调试文件"""
    
    # 当前目录
    experiments_dir = Path("/home/kyc/project/RT-DETR/experiments")
    
    # 创建archive目录
    archive_dir = experiments_dir / "debug_archive"
    archive_dir.mkdir(exist_ok=True)
    
    print("🧹 开始清理调试文件...")
    print(f"📁 Archive目录: {archive_dir}")
    
    # 定义要移动的文件列表
    debug_files_to_move = [
        # 测试脚本
        "test_gradient_fix.py",
        "test_jittor_api.py", 
        "test_mixed_precision_fix.py",
        "test_pytorch_converter.py",
        "simple_gradient_test.py",
        "correct_jittor_training_test.py",
        "final_gradient_test.py",
        "final_core_functionality_test.py",
        
        # 修复脚本
        "fix_gradient_issues.py",
        "fix_gradient_warnings.py",
        "fix_backward_dtype_issue.py",
        "fix_mixed_precision_issue.py",
        "fix_overfit_issues.py",
        "fix_remaining_issues.py",
        "fix_jittor_max_api.py",
        "fix_and_align_code.py",
        "deep_dtype_fix.py",
        "final_api_fix.py",
        "final_overfit_fix.py",
        "final_fixed_overfit.py",
        "apply_previous_dtype_fix.py",
        
        # 检查脚本
        "check_structure_alignment.py",
        "comprehensive_code_check.py",
        "comprehensive_code_check_final.py",
        "comprehensive_final_check.py",
        "comprehensive_code_alignment.py",
        "comprehensive_fix_and_align.py",
        "detailed_alignment_check.py",
        "final_code_check.py",
        "final_comprehensive_report.py",
        "align_file_structure.py",
        
        # 推理脚本
        "fixed_inference.py",
        "fixed_inference_correct_api.py",
        "direct_import_test.py",
        
        # 过拟合脚本
        "coco_single_image_overfit.py",
        "single_image_overfit_training.py",
        "specific_image_overfit.py",
        
        # 清理脚本
        "cleanup_duplicate_files.py",
        "final_cleanup.py",
        "final_verification.py",
        "verify_cleanup.py",
    ]
    
    # 保留的重要文件
    keep_files = [
        "ultimate_sanity_check.py",  # 最终完善的测试脚本
        "visualize_detection.py",    # 可视化工具
        "detection_visualization.png",  # 生成的图片
        "cleanup_debug_files.py",    # 本清理脚本
    ]
    
    moved_count = 0
    not_found_count = 0
    
    print("\n📦 移动调试文件到archive...")
    
    for filename in debug_files_to_move:
        source_path = experiments_dir / filename
        target_path = archive_dir / filename
        
        if source_path.exists():
            try:
                shutil.move(str(source_path), str(target_path))
                print(f"✅ 移动: {filename}")
                moved_count += 1
            except Exception as e:
                print(f"❌ 移动失败 {filename}: {e}")
        else:
            not_found_count += 1
    
    print(f"\n📊 清理统计:")
    print(f"   ✅ 成功移动: {moved_count} 个文件")
    print(f"   ⚠️ 未找到: {not_found_count} 个文件")
    
    # 显示保留的文件
    print(f"\n🎯 保留的重要文件:")
    for filename in keep_files:
        file_path = experiments_dir / filename
        if file_path.exists():
            print(f"   ✅ {filename}")
        else:
            print(f"   ❌ {filename} (不存在)")
    
    # 检查剩余的.py文件
    remaining_py_files = list(experiments_dir.glob("*.py"))
    print(f"\n📋 剩余的Python文件:")
    for py_file in remaining_py_files:
        print(f"   📄 {py_file.name}")
    
    print(f"\n🎉 清理完成！")
    print(f"   📁 调试文件已移动到: {archive_dir}")
    print(f"   🧹 experiments目录现在更整洁了！")

if __name__ == "__main__":
    cleanup_debug_files()
