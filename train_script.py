#!/usr/bin/env python3
"""
组件化的RT-DETR训练脚本
使用可复用的组件进行训练，参考ultimate_sanity_check.py的验证实现
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

# 导入组件
from components.model import create_rtdetr_model, create_optimizer
from components.dataset import create_coco_dataset
from components.trainer import quick_train

def main():
    """主训练函数"""
    print("🚀 RT-DETR组件化训练")
    print("=" * 60)
    
    # 配置参数
    config = {
        'num_classes': 80,
        'pretrained': True,
        'lr': 1e-4,
        'weight_decay': 0,
        'num_epochs': 50,
        'data_root': '/home/kyc/project/RT-DETR/data/coco2017_50',
        'augment_factor': 3,  # 数据增强倍数
        'save_dir': '/home/kyc/project/RT-DETR/results/component_training'
    }
    
    print("📋 训练配置:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    try:
        # 1. 创建模型和损失函数
        print(f"\n🔄 步骤1: 创建模型...")
        model, criterion = create_rtdetr_model(
            num_classes=config['num_classes'],
            pretrained=config['pretrained']
        )
        
        # 2. 创建优化器
        print(f"\n🔄 步骤2: 创建优化器...")
        optimizer = create_optimizer(
            model,
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        # 3. 创建数据集
        print(f"\n🔄 步骤3: 创建数据集...")
        dataset = create_coco_dataset(
            data_root=config['data_root'],
            split='train',
            augment_factor=config['augment_factor']
        )
        
        # 4. 开始训练
        print(f"\n🔄 步骤4: 开始训练...")
        trainer, results = quick_train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataset=dataset,
            num_epochs=config['num_epochs'],
            save_dir=config['save_dir']
        )
        
        # 5. 训练完成总结
        print(f"\n🎉 训练完成总结:")
        print(f"   最终损失: {results['final_loss']:.4f}")
        print(f"   损失下降: {results['loss_reduction']:.4f}")
        print(f"   训练时间: {results['total_time']:.1f}秒")
        print(f"   训练效率: {results['loss_reduction']/results['total_time']*60:.4f} 损失下降/分钟")
        
        # 6. 模型信息
        model_stats = model.get_param_stats()
        print(f"   参数效率: {results['loss_reduction']/(model_stats['total_params']/1e6):.4f} 损失下降/百万参数")
        
        print(f"\n💾 训练结果保存在: {config['save_dir']}")
        print(f"   模型文件: rtdetr_trained.pkl")
        print(f"   训练结果: training_results.pkl")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n✅ 组件化训练成功完成!")
        print(f"   可以使用vis_script.py进行推理可视化")
        print(f"   可以使用eval_script.py进行模型评估")
    else:
        print(f"\n❌ 组件化训练失败!")
        print(f"   请检查错误信息并修复问题")
