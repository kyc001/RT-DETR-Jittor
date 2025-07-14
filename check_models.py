#!/usr/bin/env python3
"""
检查现有模型的详细信息
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'jittor_rt_detr'))

import jittor as jt
from src.nn.model import RTDETR

# 设置Jittor
jt.flags.use_cuda = 1

def check_model_info(model_path, expected_classes):
    """检查模型信息"""
    print(f"\n>>> 检查模型: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在")
        return False
    
    try:
        # 尝试加载模型
        model = RTDETR(num_classes=expected_classes)
        model = model.float32()
        
        state_dict = jt.load(model_path)
        model.load_state_dict(state_dict)
        
        print(f"✅ 模型加载成功")
        print(f"  - 期望类别数: {expected_classes}")
        
        # 检查模型参数
        total_params = 0
        for name, param in model.named_parameters():
            total_params += param.numel()
        
        print(f"  - 总参数量: {total_params:,}")
        
        # 检查关键层的形状
        for name, param in model.named_parameters():
            if 'class_embed' in name:
                print(f"  - {name}: {param.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

def main():
    print("=" * 60)
    print("===      检查现有模型      ===")
    print("=" * 60)
    
    models_to_check = [
        ("checkpoints/small_scale_best_model.pkl", 80),
        ("checkpoints/small_scale_final_model.pkl", 80),
    ]
    
    for model_path, expected_classes in models_to_check:
        success = check_model_info(model_path, expected_classes)
        
        if not success:
            # 尝试其他可能的类别数
            print(f"  尝试其他类别数...")
            for num_classes in [2, 58, 80]:
                if num_classes != expected_classes:
                    print(f"    尝试 {num_classes} 个类别...")
                    try:
                        model = RTDETR(num_classes=num_classes)
                        model = model.float32()
                        state_dict = jt.load(model_path)
                        model.load_state_dict(state_dict)
                        print(f"    ✅ 成功！实际类别数: {num_classes}")
                        break
                    except:
                        print(f"    ❌ 失败")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
