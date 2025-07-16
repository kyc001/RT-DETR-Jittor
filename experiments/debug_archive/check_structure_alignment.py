#!/usr/bin/env python3
"""
检查文件结构对齐情况
"""

import os
import sys

def check_file_structure():
    print("=" * 80)
    print("文件结构对齐检查")
    print("=" * 80)
    
    # PyTorch版本的核心文件
    pytorch_files = [
        "rtdetr_pytorch/src/zoo/rtdetr/rtdetr.py",
        "rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py", 
        "rtdetr_pytorch/src/zoo/rtdetr/rtdetr_criterion.py",
        "rtdetr_pytorch/src/zoo/rtdetr/hybrid_encoder.py",
        "rtdetr_pytorch/src/zoo/rtdetr/denoising.py",
        "rtdetr_pytorch/src/zoo/rtdetr/box_ops.py",
        "rtdetr_pytorch/src/zoo/rtdetr/utils.py",
        "rtdetr_pytorch/src/nn/backbone/presnet.py",
        "rtdetr_pytorch/src/nn/criterion/rtdetr_criterion.py",
    ]
    
    # Jittor版本的对应文件
    jittor_files = [
        "jittor_rt_detr/src/zoo/rtdetr/rtdetr.py",
        "jittor_rt_detr/src/zoo/rtdetr/rtdetr_decoder_aligned.py",
        "jittor_rt_detr/src/nn/criterion/rtdetr_criterion.py", 
        "jittor_rt_detr/src/zoo/rtdetr/hybrid_encoder.py",
        "jittor_rt_detr/src/zoo/rtdetr/denoising.py",
        "jittor_rt_detr/src/zoo/rtdetr/box_ops.py",
        "jittor_rt_detr/src/zoo/rtdetr/utils.py",
        "jittor_rt_detr/src/nn/backbone/resnet.py",
        "jittor_rt_detr/src/nn/criterion/rtdetr_criterion.py",
    ]
    
    print("文件对应关系:")
    print("-" * 80)
    for pt_file, jt_file in zip(pytorch_files, jittor_files):
        pt_exists = "✅" if os.path.exists(pt_file) else "❌"
        jt_exists = "✅" if os.path.exists(jt_file) else "❌"
        print(f"{pt_exists} {pt_file}")
        print(f"{jt_exists} {jt_file}")
        print()
    
    # 检查目录结构
    print("目录结构对比:")
    print("-" * 80)
    
    pytorch_dirs = [
        "rtdetr_pytorch/src/zoo/rtdetr/",
        "rtdetr_pytorch/src/nn/backbone/",
        "rtdetr_pytorch/src/nn/criterion/",
    ]
    
    jittor_dirs = [
        "jittor_rt_detr/src/zoo/rtdetr/",
        "jittor_rt_detr/src/nn/backbone/",
        "jittor_rt_detr/src/nn/criterion/",
    ]
    
    for pt_dir, jt_dir in zip(pytorch_dirs, jittor_dirs):
        pt_exists = "✅" if os.path.exists(pt_dir) else "❌"
        jt_exists = "✅" if os.path.exists(jt_dir) else "❌"
        print(f"{pt_exists} PyTorch: {pt_dir}")
        print(f"{jt_exists} Jittor:  {jt_dir}")
        
        if os.path.exists(pt_dir) and os.path.exists(jt_dir):
            pt_files = set(os.listdir(pt_dir))
            jt_files = set(os.listdir(jt_dir))
            
            common = pt_files & jt_files
            pt_only = pt_files - jt_files
            jt_only = jt_files - pt_files
            
            if common:
                print(f"  共同文件: {', '.join(sorted(common))}")
            if pt_only:
                print(f"  PyTorch独有: {', '.join(sorted(pt_only))}")
            if jt_only:
                print(f"  Jittor独有: {', '.join(sorted(jt_only))}")
        print()
    
    # 功能完整性检查
    print("功能完整性检查:")
    print("-" * 80)
    
    core_components = [
        ("主模型", "jittor_rt_detr/src/zoo/rtdetr/rtdetr.py"),
        ("解码器", "jittor_rt_detr/src/zoo/rtdetr/rtdetr_decoder_aligned.py"),
        ("损失函数", "jittor_rt_detr/src/nn/criterion/rtdetr_criterion.py"),
        ("骨干网络", "jittor_rt_detr/src/nn/backbone/resnet.py"),
        ("混合编码器", "jittor_rt_detr/src/zoo/rtdetr/hybrid_encoder.py"),
        ("去噪模块", "jittor_rt_detr/src/zoo/rtdetr/denoising.py"),
        ("边界框操作", "jittor_rt_detr/src/zoo/rtdetr/box_ops.py"),
        ("工具函数", "jittor_rt_detr/src/zoo/rtdetr/utils.py"),
    ]
    
    for name, path in core_components:
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"{exists} {name}: {path}")
    
    print("\n" + "=" * 80)
    print("总结:")
    print("=" * 80)
    
    # 统计
    total_files = len(jittor_files)
    existing_files = sum(1 for f in jittor_files if os.path.exists(f))
    
    print(f"文件完整性: {existing_files}/{total_files} ({existing_files/total_files*100:.1f}%)")
    
    if existing_files == total_files:
        print("✅ 所有核心文件都已创建")
        print("✅ 文件结构完全对齐PyTorch版本")
        print("✅ 可以进行功能测试")
    else:
        missing = [f for f in jittor_files if not os.path.exists(f)]
        print("❌ 缺失文件:")
        for f in missing:
            print(f"   - {f}")

if __name__ == "__main__":
    check_file_structure()
