#!/usr/bin/env python3
"""
Jittor vs PyTorch RT-DETR性能对比分析（简化版）
基于真实训练结果的对比，不依赖matplotlib
"""

import os
import json

def analyze_training_results():
    """分析训练结果"""
    print("🔍 RT-DETR Jittor vs PyTorch 性能对比分析")
    print("=" * 80)
    
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
        'losses_by_epoch': {
            1: 6.2949, 2: 4.8378, 3: 4.2038, 4: 3.7147, 5: 3.4295,
            10: 3.0148, 20: 2.5821, 30: 2.3231, 40: 2.1215,
            46: 2.0160, 47: 1.9583, 48: 1.9949, 49: 1.9542, 50: 1.9466
        }
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
        'losses_by_epoch': {
            1: 2.7684, 2: 1.1772, 3: 1.1422, 4: 1.1367, 5: 1.1439,
            10: 1.0821, 20: 0.8435, 30: 0.8159, 40: 0.7836,
            46: 0.7843, 47: 0.7797, 48: 0.7821, 49: 0.7814, 50: 0.7783
        }
    }
    
    print("📊 基本配置对比")
    print("-" * 80)
    print(f"{'指标':<25} {'Jittor':<20} {'PyTorch':<20} {'差异':<15}")
    print("-" * 80)
    print(f"{'框架版本':<25} {jittor_results['version']:<20} {pytorch_results['version']:<20}")
    print(f"{'总参数':<25} {jittor_results['model_params']:,} {pytorch_results['model_params']:,} {((jittor_results['model_params']-pytorch_results['model_params'])/pytorch_results['model_params']*100):+.1f}%")
    print(f"{'Backbone参数':<25} {jittor_results['backbone_params']:,} {pytorch_results['backbone_params']:,} {((jittor_results['backbone_params']-pytorch_results['backbone_params'])/pytorch_results['backbone_params']*100):+.1f}%")
    print(f"{'数据集大小':<25} {jittor_results['dataset_size']:<20} {pytorch_results['dataset_size']:<20}")
    print(f"{'训练轮数':<25} {jittor_results['epochs']:<20} {pytorch_results['epochs']:<20}")
    print(f"{'学习率':<25} {jittor_results['learning_rate']:<20} {pytorch_results['learning_rate']:<20}")
    print(f"{'预训练权重':<25} {'Jittor内置':<20} {'PyTorch ResNet50':<20}")
    
    print(f"\n📈 训练性能对比")
    print("-" * 80)
    print(f"{'指标':<25} {'Jittor':<20} {'PyTorch':<20} {'差异':<15}")
    print("-" * 80)
    print(f"{'训练时间(秒)':<25} {jittor_results['training_time']:<20.1f} {pytorch_results['training_time']:<20.1f} {((jittor_results['training_time']-pytorch_results['training_time'])/pytorch_results['training_time']*100):+.1f}%")
    print(f"{'每轮用时(秒)':<25} {jittor_results['training_time']/50:<20.1f} {pytorch_results['training_time']/50:<20.1f} {(((jittor_results['training_time']/50)-(pytorch_results['training_time']/50))/(pytorch_results['training_time']/50)*100):+.1f}%")
    print(f"{'初始损失':<25} {jittor_results['initial_loss']:<20.4f} {pytorch_results['initial_loss']:<20.4f} {((jittor_results['initial_loss']-pytorch_results['initial_loss'])/pytorch_results['initial_loss']*100):+.1f}%")
    print(f"{'最终损失':<25} {jittor_results['final_loss']:<20.4f} {pytorch_results['final_loss']:<20.4f} {((jittor_results['final_loss']-pytorch_results['final_loss'])/pytorch_results['final_loss']*100):+.1f}%")
    print(f"{'损失下降':<25} {jittor_results['loss_reduction']:<20.4f} {pytorch_results['loss_reduction']:<20.4f} {((jittor_results['loss_reduction']-pytorch_results['loss_reduction'])/pytorch_results['loss_reduction']*100):+.1f}%")
    
    print(f"\n🎯 收敛性分析")
    print("-" * 80)
    jittor_convergence_rate = (jittor_results['initial_loss'] - jittor_results['final_loss']) / jittor_results['training_time']
    pytorch_convergence_rate = (pytorch_results['initial_loss'] - pytorch_results['final_loss']) / pytorch_results['training_time']
    
    print(f"Jittor收敛速度:   {jittor_convergence_rate:.6f} 损失/秒")
    print(f"PyTorch收敛速度: {pytorch_convergence_rate:.6f} 损失/秒")
    print(f"收敛速度差异:     {((jittor_convergence_rate-pytorch_convergence_rate)/pytorch_convergence_rate*100):+.1f}%")
    
    print(f"\n📋 详细损失曲线对比")
    print("-" * 80)
    print(f"{'Epoch':<8} {'Jittor Loss':<15} {'PyTorch Loss':<15} {'差异':<15}")
    print("-" * 80)
    
    key_epochs = [1, 2, 3, 4, 5, 10, 20, 30, 40, 46, 47, 48, 49, 50]
    for epoch in key_epochs:
        jt_loss = jittor_results['losses_by_epoch'][epoch]
        pt_loss = pytorch_results['losses_by_epoch'][epoch]
        diff_percent = ((jt_loss - pt_loss) / pt_loss * 100)
        print(f"{epoch:<8} {jt_loss:<15.4f} {pt_loss:<15.4f} {diff_percent:+.1f}%")
    
    print(f"\n🏆 关键性能指标总结")
    print("-" * 80)
    
    # 计算关键指标
    jittor_efficiency = jittor_results['loss_reduction'] / (jittor_results['training_time'] / 60)  # 损失下降/分钟
    pytorch_efficiency = pytorch_results['loss_reduction'] / (pytorch_results['training_time'] / 60)
    
    jittor_param_efficiency = jittor_results['loss_reduction'] / (jittor_results['model_params'] / 1e6)  # 损失下降/百万参数
    pytorch_param_efficiency = pytorch_results['loss_reduction'] / (pytorch_results['model_params'] / 1e6)
    
    print(f"训练效率 (损失下降/分钟):")
    print(f"  Jittor:   {jittor_efficiency:.4f}")
    print(f"  PyTorch:  {pytorch_efficiency:.4f}")
    print(f"  差异:     {((jittor_efficiency-pytorch_efficiency)/pytorch_efficiency*100):+.1f}%")
    
    print(f"\n参数效率 (损失下降/百万参数):")
    print(f"  Jittor:   {jittor_param_efficiency:.4f}")
    print(f"  PyTorch:  {pytorch_param_efficiency:.4f}")
    print(f"  差异:     {((jittor_param_efficiency-pytorch_param_efficiency)/pytorch_param_efficiency*100):+.1f}%")
    
    print(f"\n内存和计算资源对比:")
    print(f"  Jittor模型大小:   {jittor_results['model_params']/1e6:.1f}M 参数")
    print(f"  PyTorch模型大小: {pytorch_results['model_params']/1e6:.1f}M 参数")
    print(f"  参数差异:         {((jittor_results['model_params']-pytorch_results['model_params'])/1e6):+.1f}M ({((jittor_results['model_params']-pytorch_results['model_params'])/pytorch_results['model_params']*100):+.1f}%)")
    
    # 保存详细对比结果
    comparison_results = {
        'jittor': jittor_results,
        'pytorch': pytorch_results,
        'analysis': {
            'jittor_convergence_rate': jittor_convergence_rate,
            'pytorch_convergence_rate': pytorch_convergence_rate,
            'jittor_efficiency': jittor_efficiency,
            'pytorch_efficiency': pytorch_efficiency,
            'jittor_param_efficiency': jittor_param_efficiency,
            'pytorch_param_efficiency': pytorch_param_efficiency,
            'training_time_diff_percent': ((jittor_results['training_time']-pytorch_results['training_time'])/pytorch_results['training_time']*100),
            'final_loss_diff_percent': ((jittor_results['final_loss']-pytorch_results['final_loss'])/pytorch_results['final_loss']*100),
            'param_count_diff_percent': ((jittor_results['model_params']-pytorch_results['model_params'])/pytorch_results['model_params']*100)
        }
    }
    
    # 保存结果
    save_dir = "/home/kyc/project/RT-DETR/results/comparison"
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "detailed_comparison.json"), 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细对比数据已保存到: {save_dir}/detailed_comparison.json")
    
    print(f"\n🎉 性能对比分析完成！")
    print(f"\n📝 主要结论:")
    print(f"   ✅ 两个版本都能成功训练并收敛")
    print(f"   ⚡ PyTorch版本训练速度更快 (快 {abs(((jittor_results['training_time']-pytorch_results['training_time'])/pytorch_results['training_time']*100)):.1f}%)")
    print(f"   🎯 PyTorch版本收敛效果更好 (最终损失低 {abs(((jittor_results['final_loss']-pytorch_results['final_loss'])/pytorch_results['final_loss']*100)):.1f}%)")
    print(f"   📊 Jittor版本参数更多 (多 {((jittor_results['model_params']-pytorch_results['model_params'])/pytorch_results['model_params']*100):.1f}%)")
    print(f"   🔧 两个版本都使用了预训练权重进行微调")
    print(f"   📈 在有限算力下，微调方案证明有效")
    
    return comparison_results

if __name__ == "__main__":
    analyze_training_results()
