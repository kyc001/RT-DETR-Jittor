#!/usr/bin/env python3
"""
清理jittor_rt_detr目录中的重复和多余文件
"""

import os
import shutil
import sys
from datetime import datetime

def create_archive_directory():
    """创建归档目录"""
    archive_dir = "archive/jittor_rt_detr_old_versions"
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir, exist_ok=True)
        print(f"✅ 创建归档目录: {archive_dir}")
    return archive_dir

def identify_duplicate_files():
    """识别重复和多余的文件"""
    print("=" * 60)
    print("===        识别重复和多余文件        ===")
    print("=" * 60)
    
    jittor_dir = "jittor_rt_detr"
    duplicate_files = []
    
    # 检查已知的重复文件
    known_duplicates = [
        "jittor_rt_detr/src/zoo/rtdetr/rtdetr_decoder_old.py",
        "jittor_rt_detr/src/zoo/rtdetr/rtdetr_decoder_old2.py",
        "jittor_rt_detr/src/zoo/rtdetr/msdeformable_attention_fixed.py",
        "jittor_rt_detr/src/zoo/rtdetr/msdeformable_attention_optimized.py",
    ]
    
    print("检查已知重复文件:")
    for file_path in known_duplicates:
        if os.path.exists(file_path):
            duplicate_files.append(file_path)
            print(f"  🔍 发现: {file_path}")
        else:
            print(f"  ✅ 不存在: {file_path}")
    
    # 检查__pycache__目录
    print("\n检查__pycache__目录:")
    for root, dirs, files in os.walk(jittor_dir):
        if "__pycache__" in dirs:
            pycache_path = os.path.join(root, "__pycache__")
            duplicate_files.append(pycache_path)
            print(f"  🔍 发现: {pycache_path}")
    
    # 检查其他可能的重复文件
    print("\n检查其他可能的重复文件:")
    for root, dirs, files in os.walk(jittor_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # 检查文件名模式
            if any(pattern in file.lower() for pattern in ['backup', 'old', 'temp', 'tmp', 'test', 'debug', 'copy', 'duplicate']):
                if file_path not in duplicate_files:
                    duplicate_files.append(file_path)
                    print(f"  🔍 发现: {file_path}")
            # 检查版本号模式
            if any(pattern in file for pattern in ['_v1', '_v2', '_v3', '_1', '_2', '_3']):
                if file_path not in duplicate_files:
                    duplicate_files.append(file_path)
                    print(f"  🔍 发现: {file_path}")
    
    print(f"\n总共发现 {len(duplicate_files)} 个重复/多余文件")
    return duplicate_files

def move_files_to_archive(duplicate_files, archive_dir):
    """将重复文件移动到归档目录"""
    print("\n" + "=" * 60)
    print("===        移动文件到归档目录        ===")
    print("=" * 60)
    
    moved_count = 0
    failed_count = 0
    
    for file_path in duplicate_files:
        try:
            if os.path.exists(file_path):
                # 创建相对路径结构
                rel_path = os.path.relpath(file_path, "jittor_rt_detr")
                archive_path = os.path.join(archive_dir, rel_path)
                
                # 创建目标目录
                archive_parent = os.path.dirname(archive_path)
                if not os.path.exists(archive_parent):
                    os.makedirs(archive_parent, exist_ok=True)
                
                # 移动文件或目录
                if os.path.isdir(file_path):
                    shutil.move(file_path, archive_path)
                    print(f"✅ 移动目录: {file_path} -> {archive_path}")
                else:
                    shutil.move(file_path, archive_path)
                    print(f"✅ 移动文件: {file_path} -> {archive_path}")
                
                moved_count += 1
            else:
                print(f"⚠️ 文件不存在: {file_path}")
        
        except Exception as e:
            print(f"❌ 移动失败: {file_path} - {e}")
            failed_count += 1
    
    print(f"\n移动结果:")
    print(f"  成功移动: {moved_count} 个")
    print(f"  移动失败: {failed_count} 个")
    
    return moved_count, failed_count

def verify_core_files():
    """验证核心文件是否完整"""
    print("\n" + "=" * 60)
    print("===        验证核心文件完整性        ===")
    print("=" * 60)
    
    core_files = [
        "jittor_rt_detr/src/nn/backbone/resnet.py",
        "jittor_rt_detr/src/zoo/rtdetr/rtdetr_decoder.py",
        "jittor_rt_detr/src/nn/criterion/rtdetr_criterion.py",
        "jittor_rt_detr/src/zoo/rtdetr/rtdetr.py",
        "jittor_rt_detr/src/zoo/rtdetr/hybrid_encoder.py",
        "jittor_rt_detr/src/zoo/rtdetr/matcher.py",
        "jittor_rt_detr/src/zoo/rtdetr/box_ops.py",
        "jittor_rt_detr/src/zoo/rtdetr/utils.py",
        "jittor_rt_detr/tools/train.py",
        "jittor_rt_detr/src/core/config.py",
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in core_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path}")
    
    print(f"\n核心文件检查结果:")
    print(f"  存在: {len(existing_files)}/{len(core_files)}")
    print(f"  缺失: {len(missing_files)}")
    
    if missing_files:
        print("⚠️ 缺失的核心文件:")
        for file_path in missing_files:
            print(f"    - {file_path}")
    
    return len(missing_files) == 0

def show_final_structure():
    """显示清理后的目录结构"""
    print("\n" + "=" * 60)
    print("===        清理后的目录结构        ===")
    print("=" * 60)
    
    jittor_dir = "jittor_rt_detr"
    
    # 统计文件数量
    total_files = 0
    total_dirs = 0
    
    for root, dirs, files in os.walk(jittor_dir):
        # 排除__pycache__目录
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        
        level = root.replace(jittor_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        total_dirs += 1
        
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if not file.endswith('.pyc'):
                print(f"{subindent}{file}")
                total_files += 1
    
    print(f"\n目录结构统计:")
    print(f"  目录数: {total_dirs}")
    print(f"  文件数: {total_files}")

def main():
    print("🧹 清理jittor_rt_detr目录中的重复和多余文件")
    print("=" * 80)
    
    # 检查当前目录
    if not os.path.exists("jittor_rt_detr"):
        print("❌ 错误: jittor_rt_detr目录不存在")
        return
    
    # 1. 创建归档目录
    archive_dir = create_archive_directory()
    
    # 2. 识别重复文件
    duplicate_files = identify_duplicate_files()
    
    if not duplicate_files:
        print("✅ 没有发现重复或多余的文件")
        return
    
    # 3. 确认操作
    print(f"\n准备移动 {len(duplicate_files)} 个文件到归档目录")
    print("这些文件将被移动到:", archive_dir)
    
    # 4. 移动文件
    moved_count, failed_count = move_files_to_archive(duplicate_files, archive_dir)
    
    # 5. 验证核心文件
    core_files_ok = verify_core_files()
    
    # 6. 显示最终结构
    show_final_structure()
    
    # 总结
    print("\n" + "=" * 80)
    print("🎯 清理总结:")
    print("=" * 80)
    
    if moved_count > 0:
        print(f"✅ 成功移动 {moved_count} 个重复/多余文件")
        print(f"✅ 文件已归档到: {archive_dir}")
    
    if failed_count > 0:
        print(f"⚠️ {failed_count} 个文件移动失败")
    
    if core_files_ok:
        print("✅ 所有核心文件完整")
    else:
        print("❌ 部分核心文件缺失")
    
    print("✅ jittor_rt_detr目录清理完成")
    print("=" * 80)

if __name__ == "__main__":
    main()
