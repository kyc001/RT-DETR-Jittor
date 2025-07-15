#!/usr/bin/env python3
"""
测试PyTorch转换工具
"""

import sys
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt
from jittor.utils.pytorch_converter import convert
import torch
import torch.nn as torch_nn

# 设置Jittor
jt.flags.use_cuda = 1

def test_pytorch_converter():
    print("测试PyTorch转换工具...")
    
    # 创建一个简单的PyTorch模型
    class SimplePyTorchModel(torch_nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch_nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = torch_nn.BatchNorm2d(64)
            self.relu = torch_nn.ReLU(inplace=True)
            self.maxpool = torch_nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            return x
    
    # 创建PyTorch模型
    pytorch_model = SimplePyTorchModel()
    print("✅ PyTorch模型创建成功")
    
    # 转换为Jittor模型
    try:
        jittor_model = convert(pytorch_model)
        print("✅ 转换为Jittor模型成功")
        
        # 测试前向传播
        x = jt.randn(1, 3, 224, 224)
        output = jittor_model(x)
        print(f"✅ Jittor模型前向传播成功: {output.shape}")
        
        # 查看转换后的模型结构
        print("\n转换后的模型结构:")
        print(jittor_model)
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pytorch_converter()
