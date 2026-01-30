"""
RT-DETR 权重转换工具

支持以下转换:
- PyTorch (.pth) → Jittor (.pkl)
- Jittor (.pkl) → PyTorch (.pth)

使用方法:
    # PyTorch to Jittor
    python convert_weights.py --pt2jt --input model.pth --output model.pkl

    # Jittor to PyTorch
    python convert_weights.py --jt2pt --input model.pkl --output model.pth
"""

import os
import sys
import argparse
import numpy as np


def convert_pytorch_to_jittor(pt_path, jt_path):
    """
    将PyTorch权重转换为Jittor格式

    Args:
        pt_path: PyTorch权重文件路径 (.pth)
        jt_path: Jittor权重输出路径 (.pkl)

    Note:
        由于两个框架可能没有同时安装，这里使用numpy作为中间格式
    """
    try:
        import torch
    except ImportError:
        print("Error: PyTorch not installed. Cannot load .pth file.")
        return False

    print(f"Loading PyTorch weights from: {pt_path}")

    # 加载PyTorch权重
    state_dict = torch.load(pt_path, map_location='cpu')

    # 处理不同格式的checkpoint
    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'ema' in state_dict and 'module' in state_dict['ema']:
        # 优先使用EMA权重
        print("Using EMA weights")
        state_dict = state_dict['ema']['module']

    # 转换为numpy格式
    numpy_dict = {}
    for key, value in state_dict.items():
        if hasattr(value, 'numpy'):
            numpy_dict[key] = value.cpu().numpy()
        else:
            numpy_dict[key] = value
        print(f"  Converted: {key} -> shape {numpy_dict[key].shape if hasattr(numpy_dict[key], 'shape') else type(numpy_dict[key])}")

    # 保存为Jittor格式
    import pickle
    with open(jt_path, 'wb') as f:
        pickle.dump(numpy_dict, f)

    print(f"Jittor weights saved to: {jt_path}")
    print(f"Total parameters: {len(numpy_dict)}")
    return True


def convert_jittor_to_pytorch(jt_path, pt_path):
    """
    将Jittor权重转换为PyTorch格式

    Args:
        jt_path: Jittor权重文件路径 (.pkl)
        pt_path: PyTorch权重输出路径 (.pth)
    """
    try:
        import torch
    except ImportError:
        print("Error: PyTorch not installed. Cannot save .pth file.")
        return False

    print(f"Loading Jittor weights from: {jt_path}")

    # 加载Jittor权重
    import pickle
    with open(jt_path, 'rb') as f:
        numpy_dict = pickle.load(f)

    # 转换为PyTorch tensor
    torch_dict = {}
    for key, value in numpy_dict.items():
        if isinstance(value, np.ndarray):
            torch_dict[key] = torch.from_numpy(value)
        else:
            torch_dict[key] = value
        print(f"  Converted: {key} -> shape {torch_dict[key].shape if hasattr(torch_dict[key], 'shape') else type(torch_dict[key])}")

    # 保存为PyTorch格式
    torch.save(torch_dict, pt_path)

    print(f"PyTorch weights saved to: {pt_path}")
    print(f"Total parameters: {len(torch_dict)}")
    return True


def load_pytorch_weights_to_jittor_model(model, pt_path, strict=True):
    """
    直接将PyTorch权重加载到Jittor模型

    Args:
        model: Jittor模型
        pt_path: PyTorch权重文件路径
        strict: 是否严格匹配键名

    Returns:
        model: 加载了权重的模型
        missing_keys: 缺失的键
        unexpected_keys: 意外的键
    """
    try:
        import torch
    except ImportError:
        print("Error: PyTorch not installed. Cannot load .pth file.")
        return model, [], []

    import jittor as jt

    print(f"Loading PyTorch weights from: {pt_path}")

    # 加载PyTorch权重
    state_dict = torch.load(pt_path, map_location='cpu')

    # 处理不同格式的checkpoint
    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'ema' in state_dict and 'module' in state_dict['ema']:
        print("Using EMA weights")
        state_dict = state_dict['ema']['module']

    # 获取模型的state_dict
    model_state = model.state_dict()
    missing_keys = []
    unexpected_keys = []
    matched_keys = []

    # 转换并加载权重
    new_state_dict = {}
    for key, value in state_dict.items():
        if key in model_state:
            if hasattr(value, 'numpy'):
                np_value = value.cpu().numpy()
            else:
                np_value = value

            # 检查形状是否匹配
            if model_state[key].shape == np_value.shape:
                new_state_dict[key] = jt.array(np_value)
                matched_keys.append(key)
            else:
                print(f"Shape mismatch for {key}: model {model_state[key].shape} vs loaded {np_value.shape}")
                if not strict:
                    missing_keys.append(key)
        else:
            unexpected_keys.append(key)

    # 检查缺失的键
    for key in model_state.keys():
        if key not in new_state_dict:
            missing_keys.append(key)

    # 加载权重
    model.load_state_dict(new_state_dict)

    print(f"Matched keys: {len(matched_keys)}")
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    return model, missing_keys, unexpected_keys


def verify_conversion(pt_path, jt_path, tolerance=1e-5):
    """
    验证转换是否正确

    Args:
        pt_path: PyTorch权重文件
        jt_path: Jittor权重文件
        tolerance: 数值容差

    Returns:
        bool: 是否转换正确
    """
    try:
        import torch
    except ImportError:
        print("Error: PyTorch not installed for verification.")
        return False

    import pickle

    # 加载两边的权重
    pt_state = torch.load(pt_path, map_location='cpu')
    if 'model' in pt_state:
        pt_state = pt_state['model']
    elif 'state_dict' in pt_state:
        pt_state = pt_state['state_dict']

    with open(jt_path, 'rb') as f:
        jt_state = pickle.load(f)

    # 比较
    all_close = True
    for key in pt_state.keys():
        if key in jt_state:
            pt_val = pt_state[key].cpu().numpy() if hasattr(pt_state[key], 'numpy') else pt_state[key]
            jt_val = jt_state[key]

            if isinstance(pt_val, np.ndarray) and isinstance(jt_val, np.ndarray):
                if not np.allclose(pt_val, jt_val, atol=tolerance):
                    print(f"Mismatch at {key}: max diff = {np.max(np.abs(pt_val - jt_val))}")
                    all_close = False

    if all_close:
        print("Verification passed: All weights match!")
    else:
        print("Verification failed: Some weights do not match!")

    return all_close


def main():
    parser = argparse.ArgumentParser(description='RT-DETR Weight Converter')
    parser.add_argument('--pt2jt', action='store_true', help='Convert PyTorch to Jittor')
    parser.add_argument('--jt2pt', action='store_true', help='Convert Jittor to PyTorch')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input weight file path')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output weight file path')
    parser.add_argument('--verify', action='store_true', help='Verify conversion')

    args = parser.parse_args()

    if args.pt2jt:
        success = convert_pytorch_to_jittor(args.input, args.output)
        if success and args.verify:
            verify_conversion(args.input, args.output)
    elif args.jt2pt:
        success = convert_jittor_to_pytorch(args.input, args.output)
        if success and args.verify:
            verify_conversion(args.output, args.input)
    else:
        print("Please specify conversion direction: --pt2jt or --jt2pt")
        parser.print_help()


if __name__ == '__main__':
    main()
