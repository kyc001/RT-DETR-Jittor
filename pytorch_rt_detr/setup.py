#!/usr/bin/env python3
"""
RT-DETR PyTorch 安装脚本
"""

import os
import sys
import subprocess
import platform


def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("错误: 需要Python 3.8或更高版本")
        return False
    print(f"Python版本: {sys.version}")
    return True


def install_pytorch():
    """安装PyTorch"""
    print("安装PyTorch...")

    # 检测CUDA版本
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"CUDA可用: {cuda_available}")
    except ImportError:
        pass

    # 根据系统选择安装命令
    system = platform.system().lower()

    if system == "linux":
        if cuda_available:
            cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        else:
            cmd = "pip install torch torchvision torchaudio"
    elif system == "windows":
        if cuda_available:
            cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        else:
            cmd = "pip install torch torchvision torchaudio"
    else:  # macOS
        cmd = "pip install torch torchvision torchaudio"

    try:
        subprocess.run(cmd.split(), check=True)
        print("PyTorch安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"PyTorch安装失败: {e}")
        return False


def install_requirements():
    """安装其他依赖"""
    print("安装其他依赖...")

    requirements = [
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
        "tqdm>=4.60.0",
        "matplotlib>=3.5.0",
        "pycocotools>=2.0.0",
        "tensorboard>=2.8.0",
        "timm>=0.6.0",
        "einops>=0.4.0"
    ]

    for req in requirements:
        try:
            subprocess.run(["pip", "install", req], check=True)
            print(f"安装成功: {req}")
        except subprocess.CalledProcessError as e:
            print(f"安装失败: {req} - {e}")
            return False

    return True


def create_directories():
    """创建必要的目录"""
    directories = [
        "checkpoints",
        "data",
        "logs",
        "results",
        "configs"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")


def test_installation():
    """测试安装"""
    print("测试安装...")

    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")

        import torchvision
        print(f"TorchVision版本: {torchvision.__version__}")

        import cv2
        print(f"OpenCV版本: {cv2.__version__}")

        import numpy as np
        print(f"NumPy版本: {np.__version__}")

        # 测试CUDA
        if torch.cuda.is_available():
            print(f"CUDA可用，设备数量: {torch.cuda.device_count()}")
            print(f"当前设备: {torch.cuda.current_device()}")
        else:
            print("CUDA不可用，将使用CPU")

        print("安装测试通过！")
        return True

    except ImportError as e:
        print(f"导入错误: {e}")
        return False


def main():
    """主安装函数"""
    print("=== RT-DETR PyTorch 安装脚本 ===")

    # 检查Python版本
    if not check_python_version():
        return

    # 安装PyTorch
    if not install_pytorch():
        print("PyTorch安装失败，退出安装")
        return

    # 安装其他依赖
    if not install_requirements():
        print("依赖安装失败，退出安装")
        return

    # 创建目录
    create_directories()

    # 测试安装
    if not test_installation():
        print("安装测试失败")
        return

    print("\n=== 安装完成 ===")
    print("现在可以运行以下命令:")
    print("1. 演示: python demo.py")
    print("2. 训练: python train.py")
    print("3. 测试: python test.py --weights checkpoints/model.pth --img_path test.jpg")


if __name__ == '__main__':
    main()
