#!/usr/bin/env python3
"""
使用组件化模块的RT-DETR可视化脚本
基于jittor_rt_detr.src.components的可复用组件
"""

import os
import sys
import glob

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

# 导入组件化模块
from jittor_rt_detr.src.components.model import create_rtdetr_model
from jittor_rt_detr.src.components.visualizer import create_visualizer

def main():
    """主可视化函数"""
    print("🎨 RT-DETR组件化可视化 (集成版)")
    print("=" * 60)
    
    # 配置参数
    config = {
        'num_classes': 80,
        'pretrained': False,  # 加载训练好的模型，不需要预训练权重
        'model_path': '/home/kyc/project/RT-DETR/results/integrated_training/rtdetr_trained.pkl',
        'conf_threshold': 0.3,
        'test_images_dir': '/home/kyc/project/RT-DETR/data/coco2017_50/val2017',
        'save_dir': '/home/kyc/project/RT-DETR/results/integrated_visualizations'
    }
    
    print("📋 可视化配置:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    try:
        # 1. 创建模型
        print(f"\n🔄 步骤1: 创建模型...")
        model, _ = create_rtdetr_model(
            num_classes=config['num_classes'],
            pretrained=config['pretrained']
        )
        
        # 2. 加载训练好的权重
        print(f"\n🔄 步骤2: 加载训练好的模型...")
        if os.path.exists(config['model_path']):
            model.load(config['model_path'])
            print(f"✅ 模型加载成功: {config['model_path']}")
        else:
            print(f"❌ 模型文件不存在: {config['model_path']}")
            print(f"   请先运行train_with_components.py进行训练")
            return False
        
        # 3. 创建可视化器
        print(f"\n🔄 步骤3: 创建可视化器...")
        visualizer = create_visualizer(
            model=model,
            conf_threshold=config['conf_threshold'],
            save_dir=config['save_dir']
        )
        
        # 4. 获取测试图像
        print(f"\n🔄 步骤4: 获取测试图像...")
        if os.path.exists(config['test_images_dir']):
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(glob.glob(os.path.join(config['test_images_dir'], ext)))
            
            if image_paths:
                print(f"✅ 找到 {len(image_paths)} 张测试图像")
                # 限制测试图像数量
                image_paths = image_paths[:10]  # 只测试前10张
                print(f"   将测试前 {len(image_paths)} 张图像")
            else:
                print(f"❌ 在 {config['test_images_dir']} 中没有找到图像文件")
                return False
        else:
            print(f"❌ 测试图像目录不存在: {config['test_images_dir']}")
            return False
        
        # 5. 批量推理和可视化
        print(f"\n🔄 步骤5: 批量推理和可视化...")
        all_results = visualizer.batch_inference(
            image_paths=image_paths,
            save_visualizations=True
        )
        
        # 6. 打印检测统计
        print(f"\n🔄 步骤6: 检测统计...")
        visualizer.print_detection_stats(all_results)
        
        # 7. 单张图像详细分析
        if all_results:
            print(f"\n🔍 详细分析第一张图像:")
            first_result = all_results[0]
            summary = visualizer.create_detection_summary(first_result)
            
            print(f"   图像文件: {summary['image_file']}")
            print(f"   图像尺寸: {summary['image_size']}")
            print(f"   检测数量: {summary['num_detections']}")
            
            if summary['detections']:
                print(f"   检测详情:")
                for i, det in enumerate(summary['detections'][:5]):  # 显示前5个
                    print(f"     {i+1}. {det['class_name']}: {det['confidence']:.3f}")
        
        print(f"\n💾 可视化结果保存在: {config['save_dir']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def single_image_demo(image_path):
    """单张图像演示"""
    print(f"\n🖼️ 单张图像演示: {os.path.basename(image_path)}")
    
    # 配置
    config = {
        'num_classes': 80,
        'pretrained': False,
        'model_path': '/home/kyc/project/RT-DETR/results/integrated_training/rtdetr_trained.pkl',
        'conf_threshold': 0.3,
        'save_dir': '/home/kyc/project/RT-DETR/results/single_demo'
    }
    
    try:
        # 创建模型和可视化器
        model, _ = create_rtdetr_model(config['num_classes'], config['pretrained'])
        model.load(config['model_path'])
        visualizer = create_visualizer(model, config['conf_threshold'], config['save_dir'])
        
        # 推理
        results = visualizer.inference_single_image(image_path)
        
        # 可视化
        save_path = visualizer.visualize_detection(results)
        
        # 打印结果
        print(f"   检测到 {results['num_detections']} 个目标")
        print(f"   结果保存到: {save_path}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 单张图像演示失败: {e}")
        return False

if __name__ == "__main__":
    # 主要的批量可视化
    success = main()
    
    if success:
        print(f"\n✅ 组件化可视化成功完成!")
        print(f"   组件位置: jittor_rt_detr.src.components")
        
        # 可选：单张图像演示
        demo_image = "/home/kyc/project/RT-DETR/data/coco2017_50/val2017/000000369771.jpg"
        if os.path.exists(demo_image):
            single_image_demo(demo_image)
        
    else:
        print(f"\n❌ 组件化可视化失败!")
        print(f"   请检查错误信息并修复问题")
        print(f"   确保已经运行train_with_components.py完成训练")
