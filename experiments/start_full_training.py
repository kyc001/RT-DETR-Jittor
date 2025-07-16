#!/usr/bin/env python3
"""
启动完整训练的简化脚本
"""

import os
import sys
import subprocess
from datetime import datetime

def main():
    print("🚀 RT-DETR完整训练启动器")
    print("=" * 60)
    
    # 检查环境
    print("🔍 检查环境...")
    
    # 检查数据目录
    data_dir = '/home/kyc/project/RT-DETR/data/coco2017_50'
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    train_dir = os.path.join(data_dir, 'train2017')
    val_dir = os.path.join(data_dir, 'val2017')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"❌ 训练或验证目录不存在")
        return
    
    print(f"✅ 数据目录检查通过")
    
    # 检查结果目录
    results_dir = '/home/kyc/project/RT-DETR/results/full_training'
    os.makedirs(results_dir, exist_ok=True)
    print(f"✅ 结果目录准备完成: {results_dir}")
    
    # 显示训练配置
    print("\n📋 训练配置:")
    print(f"   数据集: train50 + val50")
    print(f"   训练轮数: 50")
    print(f"   学习率: 1e-4")
    print(f"   模型: RT-DETR (ResNet50 + Transformer)")
    print(f"   类别数: 80 (COCO)")
    print(f"   结果保存: {results_dir}")
    
    # 确认开始训练
    print("\n⚠️  注意: 完整训练可能需要数小时时间")
    response = input("🤔 确定开始训练吗? (y/N): ").strip().lower()
    
    if response not in ['y', 'yes']:
        print("❌ 训练已取消")
        return
    
    print(f"\n🎯 开始训练... {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 运行训练脚本
    try:
        # 切换到正确的conda环境并运行训练
        cmd = [
            'conda', 'run', '-n', 'jt',
            'python', 'full_scale_training.py'
        ]
        
        # 在experiments目录下运行
        os.chdir('/home/kyc/project/RT-DETR/experiments')
        
        # 启动训练进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时显示输出
        for line in process.stdout:
            print(line.rstrip())
        
        # 等待进程完成
        return_code = process.wait()
        
        if return_code == 0:
            print("\n🎉 训练完成！")
            print(f"📊 查看结果: {results_dir}")
        else:
            print(f"\n❌ 训练失败，返回码: {return_code}")
            
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        if 'process' in locals():
            process.terminate()
    except Exception as e:
        print(f"\n❌ 训练启动失败: {e}")

if __name__ == "__main__":
    main()
