#!/usr/bin/env python3
"""
测试Jittor的正确API用法
"""

import sys
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt
import jittor.nn as nn

# 设置Jittor
jt.flags.use_cuda = 1

def test_jittor_apis():
    print("测试Jittor API...")
    
    # 测试池化操作
    x = jt.randn(1, 64, 112, 112)
    print(f"输入形状: {x.shape}")
    
    # 测试不同的池化API
    try:
        # 方法1: nn.pool
        pool1 = nn.pool(x, kernel_size=3, stride=2, padding=1, op='maximum')
        print(f"✅ nn.pool: {pool1.shape}")
    except Exception as e:
        print(f"❌ nn.pool: {e}")
    
    try:
        # 方法2: nn.max_pool2d
        pool2 = nn.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        print(f"✅ nn.max_pool2d: {pool2.shape}")
    except Exception as e:
        print(f"❌ nn.max_pool2d: {e}")
    
    try:
        # 方法3: jt.nn.Pool
        pool_layer = nn.Pool(kernel_size=3, stride=2, padding=1, op='maximum')
        pool3 = pool_layer(x)
        print(f"✅ nn.Pool: {pool3.shape}")
    except Exception as e:
        print(f"❌ nn.Pool: {e}")
    
    # 测试激活函数
    try:
        relu = nn.ReLU()
        out = relu(x)
        print(f"✅ nn.ReLU: {out.shape}")
    except Exception as e:
        print(f"❌ nn.ReLU: {e}")
    
    # 测试softmax
    try:
        logits = jt.randn(1, 10)
        softmax1 = nn.softmax(logits, dim=-1)
        print(f"✅ nn.softmax: {softmax1.shape}")
    except Exception as e:
        print(f"❌ nn.softmax: {e}")
    
    try:
        softmax2 = jt.nn.softmax(logits, dim=-1)
        print(f"✅ jt.nn.softmax: {softmax2.shape}")
    except Exception as e:
        print(f"❌ jt.nn.softmax: {e}")
    
    # 测试sigmoid
    try:
        sigmoid1 = nn.sigmoid(logits)
        print(f"✅ nn.sigmoid: {sigmoid1.shape}")
    except Exception as e:
        print(f"❌ nn.sigmoid: {e}")
    
    try:
        sigmoid2 = jt.sigmoid(logits)
        print(f"✅ jt.sigmoid: {sigmoid2.shape}")
    except Exception as e:
        print(f"❌ jt.sigmoid: {e}")
    
    # 测试interpolate
    try:
        interp1 = nn.interpolate(x, size=[224, 224])
        print(f"✅ nn.interpolate: {interp1.shape}")
    except Exception as e:
        print(f"❌ nn.interpolate: {e}")
    
    try:
        interp2 = jt.nn.interpolate(x, size=[224, 224])
        print(f"✅ jt.nn.interpolate: {interp2.shape}")
    except Exception as e:
        print(f"❌ jt.nn.interpolate: {e}")

if __name__ == "__main__":
    test_jittor_apis()
