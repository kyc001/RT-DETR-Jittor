#!/usr/bin/env python3
"""
Jittor vs PyTorch RT-DETR性能对比分析
基于真实训练结果的对比
"""

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

def analyze_training_results():
    """分析训练结果"""
    print("🔍 RT-DETR Jittor vs PyTorch 性能对比分析")
    print("=" * 60)
    
    # 从训练日志中提取的真实数据
    jittor_results = {
        'framework': 'Jittor',
        'version': '1.3.9.14',
        'model_params': 31139848,
        'backbone_params': 23580512,
        'transformer_params': 7559336,
        'training_time': 422.7,  # 秒
        'initial_loss': 6.2949,
        'final_loss': 1.9466,
        'loss_reduction': 4.3483,
        'epochs': 50,
        'dataset_size': 50,
        'batch_size': 1,
        'learning_rate': 5e-5,
        'pretrained': 'Jittor内置预训练权重',
        'losses': [6.2949, 4.8378, 4.2038, 3.7147, 3.4295, 3.0148, 2.5821, 2.3231, 2.1215, 2.0160, 1.9583, 1.9949, 1.9542, 1.9466]
    }
    
    pytorch_results = {
        'framework': 'PyTorch',
        'version': '2.0.1+cu117',
        'model_params': 25526932,
        'backbone_params': 23508032,
        'encoder_params': 918272,
        'decoder_params': 1100628,
        'training_time': 191.4,  # 秒
        'initial_loss': 2.7684,
        'final_loss': 0.7783,
        'loss_reduction': 1.9901,
        'epochs': 50,
        'dataset_size': 50,
        'batch_size': 1,
        'learning_rate': 5e-5,
        'pretrained': 'PyTorch预训练ResNet50权重',
        'losses': [2.7684, 1.1772, 1.1422, 1.1367, 1.1439, 1.0821, 0.8435, 0.8159, 0.7836, 0.7843, 0.7797, 0.7821, 0.7814, 0.7783]
    }
    
    print("📊 训练配置对比")
    print("-" * 40)
    print(f"{'指标':<20} {'Jittor':<15} {'PyTorch':<15} {'差异':<10}")
    print("-" * 40)
    print(f"{'框架版本':<20} {jittor_results['version']:<15} {pytorch_results['version']:<15}")
    print(f"{'总参数':<20} {jittor_results['model_params']:,} {pytorch_results['model_params']:,} {((jittor_results['model_params']-pytorch_results['model_params'])/pytorch_results['model_params']*100):+.1f}%")
    print(f"{'Backbone参数':<20} {jittor_results['backbone_params']:,} {pytorch_results['backbone_params']:,} {((jittor_results['backbone_params']-pytorch_results['backbone_params'])/pytorch_results['backbone_params']*100):+.1f}%")
    print(f"{'数据集大小':<20} {jittor_results['dataset_size']:<15} {pytorch_results['dataset_size']:<15}")
    print(f"{'训练轮数':<20} {jittor_results['epochs']:<15} {pytorch_results['epochs']:<15}")
    print(f"{'学习率':<20} {jittor_results['learning_rate']:<15} {pytorch_results['learning_rate']:<15}")
    
    print(f"\n📈 训练性能对比")
    print("-" * 40)
    print(f"{'指标':<20} {'Jittor':<15} {'PyTorch':<15} {'差异':<10}")
    print("-" * 40)
    print(f"{'训练时间(秒)':<20} {jittor_results['training_time']:<15.1f} {pytorch_results['training_time']:<15.1f} {((jittor_results['training_time']-pytorch_results['training_time'])/pytorch_results['training_time']*100):+.1f}%")
    print(f"{'每轮用时(秒)':<20} {jittor_results['training_time']/50:<15.1f} {pytorch_results['training_time']/50:<15.1f} {(((jittor_results['training_time']/50)-(pytorch_results['training_time']/50))/(pytorch_results['training_time']/50)*100):+.1f}%")
    print(f"{'初始损失':<20} {jittor_results['initial_loss']:<15.4f} {pytorch_results['initial_loss']:<15.4f} {((jittor_results['initial_loss']-pytorch_results['initial_loss'])/pytorch_results['initial_loss']*100):+.1f}%")
    print(f"{'最终损失':<20} {jittor_results['final_loss']:<15.4f} {pytorch_results['final_loss']:<15.4f} {((jittor_results['final_loss']-pytorch_results['final_loss'])/pytorch_results['final_loss']*100):+.1f}%")
    print(f"{'损失下降':<20} {jittor_results['loss_reduction']:<15.4f} {pytorch_results['loss_reduction']:<15.4f} {((jittor_results['loss_reduction']-pytorch_results['loss_reduction'])/pytorch_results['loss_reduction']*100):+.1f}%")
    
    print(f"\n🎯 收敛性分析")
    print("-" * 40)
    jittor_convergence_rate = (jittor_results['initial_loss'] - jittor_results['final_loss']) / jittor_results['training_time']
    pytorch_convergence_rate = (pytorch_results['initial_loss'] - pytorch_results['final_loss']) / pytorch_results['training_time']
    
    print(f"Jittor收敛速度: {jittor_convergence_rate:.6f} 损失/秒")
    print(f"PyTorch收敛速度: {pytorch_convergence_rate:.6f} 损失/秒")
    print(f"收敛速度差异: {((jittor_convergence_rate-pytorch_convergence_rate)/pytorch_convergence_rate*100):+.1f}%")
    
    # 绘制损失曲线对比图
    plt.figure(figsize=(12, 8))
    
    # 创建epoch数组
    jittor_epochs = [1, 2, 3, 4, 5, 10, 20, 30, 40, 46, 47, 48, 49, 50]
    pytorch_epochs = [1, 2, 3, 4, 5, 10, 20, 30, 40, 46, 47, 48, 49, 50]
    
    plt.subplot(2, 2, 1)
    plt.plot(jittor_epochs, jittor_results['losses'], 'b-o', label='Jittor', linewidth=2, markersize=6)
    plt.plot(pytorch_epochs, pytorch_results['losses'], 'r-s', label='PyTorch', linewidth=2, markersize=6)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 训练时间对比
    plt.subplot(2, 2, 2)
    frameworks = ['Jittor', 'PyTorch']
    times = [jittor_results['training_time'], pytorch_results['training_time']]
    colors = ['blue', 'red']
    bars = plt.bar(frameworks, times, color=colors, alpha=0.7)
    plt.ylabel('训练时间 (秒)')
    plt.title('训练时间对比')
    for bar, time in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # 参数数量对比
    plt.subplot(2, 2, 3)
    jittor_params = [jittor_results['backbone_params']/1e6, jittor_results['transformer_params']/1e6]
    pytorch_params = [pytorch_results['backbone_params']/1e6, (pytorch_results['encoder_params']+pytorch_results['decoder_params'])/1e6]
    
    x = np.arange(2)
    width = 0.35
    
    plt.bar(x - width/2, jittor_params, width, label='Jittor', color='blue', alpha=0.7)
    plt.bar(x + width/2, pytorch_params, width, label='PyTorch', color='red', alpha=0.7)
    
    plt.xlabel('模型组件')
    plt.ylabel('参数数量 (M)')
    plt.title('模型参数对比')
    plt.xticks(x, ['Backbone', 'Transformer'])
    plt.legend()
    
    # 收敛速度对比
    plt.subplot(2, 2, 4)
    convergence_rates = [jittor_convergence_rate, pytorch_convergence_rate]
    bars = plt.bar(frameworks, convergence_rates, color=colors, alpha=0.7)
    plt.ylabel('收敛速度 (损失/秒)')
    plt.title('收敛速度对比')
    for bar, rate in zip(bars, convergence_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + rate*0.01, 
                f'{rate:.6f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    save_dir = "/home/kyc/project/RT-DETR/results/comparison"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "jittor_vs_pytorch_comparison.png"), dpi=300, bbox_inches='tight')
    print(f"\n📊 对比图表已保存到: {save_dir}/jittor_vs_pytorch_comparison.png")
    
    # 保存详细对比结果
    comparison_results = {
        'jittor': jittor_results,
        'pytorch': pytorch_results,
        'analysis': {
            'jittor_convergence_rate': jittor_convergence_rate,
            'pytorch_convergence_rate': pytorch_convergence_rate,
            'training_time_diff_percent': ((jittor_results['training_time']-pytorch_results['training_time'])/pytorch_results['training_time']*100),
            'final_loss_diff_percent': ((jittor_results['final_loss']-pytorch_results['final_loss'])/pytorch_results['final_loss']*100),
            'param_count_diff_percent': ((jittor_results['model_params']-pytorch_results['model_params'])/pytorch_results['model_params']*100)
        }
    }
    
    with open(os.path.join(save_dir, "detailed_comparison.pkl"), 'wb') as f:
        pickle.dump(comparison_results, f)
    
    print(f"📋 详细对比数据已保存到: {save_dir}/detailed_comparison.pkl")
    
    print(f"\n🎉 性能对比分析完成！")
    print(f"\n📝 总结:")
    print(f"   • Jittor版本参数更多但训练时间更长")
    print(f"   • PyTorch版本训练速度更快，收敛更快")
    print(f"   • 两个版本都能成功训练并收敛")
    print(f"   • Jittor版本最终损失较高，可能需要调整超参数")
    
    return comparison_results

if __name__ == "__main__":
    analyze_training_results()
