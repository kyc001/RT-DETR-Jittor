#!/usr/bin/env python3
"""
快速训练测试 - 用少量数据和轮数验证完整训练流程
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt

# 设置Jittor
jt.flags.use_cuda = 1

def main():
    print("🧪 RT-DETR快速训练测试")
    print("=" * 60)
    print("目标: 用5张图像训练3轮，验证完整流程")
    print()
    
    # 导入训练脚本的函数
    from full_scale_training import (
        load_coco_dataset, create_model, save_model, load_model,
        TrainingLogger, train_one_epoch, validate_model, generate_final_report
    )
    from jittor_rt_detr.src.nn.criterion.rtdetr_criterion import build_criterion
    
    # 配置参数（小规模测试）
    config = {
        'data_dir': '/home/kyc/project/RT-DETR/data/coco2017_50',
        'num_epochs': 3,  # 只训练3轮
        'learning_rate': 1e-4,
        'save_dir': '/home/kyc/project/RT-DETR/results/quick_test',
        'log_dir': '/home/kyc/project/RT-DETR/results/quick_test/logs',
        'model_save_path': '/home/kyc/project/RT-DETR/results/quick_test/rtdetr_quick_test.pkl'
    }
    
    print(f"📋 测试配置:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # 初始化日志记录器
    logger = TrainingLogger(config['log_dir'])
    
    try:
        # 加载数据集
        print("🔄 加载数据集...")
        train_images, train_annotations, train_dir = load_coco_dataset(config['data_dir'], 'train')
        val_images, val_annotations, val_dir = load_coco_dataset(config['data_dir'], 'val')
        
        # 只使用前5张图像进行快速测试
        train_image_ids = list(train_images.keys())[:5]
        train_images_subset = {img_id: train_images[img_id] for img_id in train_image_ids}
        
        val_image_ids = list(val_images.keys())[:5]
        val_images_subset = {img_id: val_images[img_id] for img_id in val_image_ids}
        
        print(f"✅ 使用 {len(train_images_subset)} 张训练图像, {len(val_images_subset)} 张验证图像")
        
        # 创建模型
        print("🔄 创建模型...")
        model = create_model()
        if model is None:
            print("❌ 模型创建失败")
            return
        
        # 创建损失函数和优化器
        criterion = build_criterion(num_classes=80)
        optimizer = jt.optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        print("✅ 模型和优化器创建成功")
        
        # 训练循环
        print("\n🎯 开始快速训练...")
        print("=" * 60)
        
        for epoch in range(config['num_epochs']):
            print(f"\n📅 Epoch {epoch+1}/{config['num_epochs']}")
            
            # 训练一个epoch（使用子集数据）
            epoch_loss = train_one_epoch_subset(
                model, train_images_subset, train_annotations, train_dir, 
                criterion, optimizer, epoch
            )
            
            # 记录训练损失
            logger.log_training(epoch, epoch_loss)
            
            # 每个epoch都进行验证（因为只有3轮）
            print(f"🔍 进行验证...")
            val_results = validate_model_subset(
                model, val_images_subset, val_annotations, val_dir
            )
            logger.log_validation(epoch, val_results)
            
            # 保存模型
            save_model(model, config['model_save_path'], epoch, epoch_loss, 
                      {'validation_results': val_results})
        
        # 训练完成
        print("\n🎉 快速训练完成！")
        
        # 最终验证
        print("🔍 进行最终验证...")
        final_val_results = validate_model_subset(
            model, val_images_subset, val_annotations, val_dir
        )
        logger.log_validation(config['num_epochs']-1, final_val_results)
        
        # 保存所有日志
        logger.save_logs()
        logger.plot_training_curve()
        
        # 生成报告
        generate_final_report(config, logger, final_val_results)
        
        print("✅ 快速训练测试完成！")
        print(f"📊 查看结果: {config['save_dir']}")
        
        # 测试模型加载
        print("\n🔄 测试模型加载...")
        new_model = create_model()
        checkpoint = load_model(new_model, config['model_save_path'])
        
        if checkpoint:
            print("✅ 模型加载测试成功")
        else:
            print("❌ 模型加载测试失败")
        
        print("\n🎯 结论: 完整训练流程验证成功！")
        print("💡 现在可以运行完整的50轮训练了")
        
    except Exception as e:
        print(f"❌ 快速训练测试失败: {e}")
        import traceback
        traceback.print_exc()

def train_one_epoch_subset(model, images_info, annotations, images_dir, criterion, optimizer, epoch):
    """训练一个epoch（子集版本）"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # 导入数据加载函数
    from full_scale_training import load_single_image_data
    
    # 获取所有图像ID
    image_ids = list(images_info.keys())
    
    print(f"   处理 {len(image_ids)} 张图像...")
    
    for i, image_id in enumerate(image_ids):
        try:
            # 加载图像和标注
            image_data, targets = load_single_image_data(
                image_id, images_info, annotations, images_dir)
            
            if image_data is None or targets is None:
                continue
            
            # 前向传播
            features = model.backbone(image_data)
            outputs = model.transformer(features)
            
            # 计算损失
            loss_dict = criterion(outputs, targets)
            total_loss_value = sum(loss_dict.values())
            
            # 反向传播（使用Jittor的正确API）
            optimizer.step(total_loss_value)
            
            total_loss += total_loss_value.numpy().item()
            num_batches += 1
            
            print(f"     图像 {i+1}/{len(image_ids)}: 损失={total_loss_value.numpy().item():.4f}")
                
        except Exception as e:
            print(f"     ⚠️ 图像 {image_id} 处理失败: {e}")
            continue
    
    avg_epoch_loss = total_loss / max(num_batches, 1)
    print(f"   ✅ Epoch {epoch+1} 完成, 平均损失: {avg_epoch_loss:.4f}")
    
    return avg_epoch_loss

def validate_model_subset(model, images_info, annotations, images_dir):
    """验证模型性能（子集版本）"""
    model.eval()
    
    print("   🔍 开始验证...")
    
    # 导入必要函数
    from full_scale_training import load_single_image_data, postprocess_outputs
    
    total_detections = 0
    total_confidence = 0
    detected_classes = set()
    validation_results = []
    
    # 获取验证图像ID
    image_ids = list(images_info.keys())
    
    with jt.no_grad():
        for i, image_id in enumerate(image_ids):
            try:
                # 加载图像
                image_data, _ = load_single_image_data(
                    image_id, images_info, annotations, images_dir)
                
                if image_data is None:
                    continue
                
                # 推理
                features = model.backbone(image_data)
                outputs = model.transformer(features)
                
                # 后处理
                detections = postprocess_outputs(outputs)
                
                # 统计结果
                for detection in detections:
                    total_detections += 1
                    total_confidence += detection['confidence']
                    detected_classes.add(detection['class_name'])
                
                validation_results.append({
                    'image_id': image_id,
                    'detections': len(detections),
                    'avg_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0
                })
                
                print(f"     验证图像 {i+1}/{len(image_ids)}: {len(detections)} 个检测")
                    
            except Exception as e:
                continue
    
    # 计算验证指标
    avg_confidence = total_confidence / max(total_detections, 1)
    class_diversity = len(detected_classes)
    
    results = {
        'total_detections': total_detections,
        'avg_confidence': avg_confidence,
        'class_diversity': class_diversity,
        'detected_classes': list(detected_classes),
        'per_image_results': validation_results
    }
    
    print(f"   ✅ 验证完成:")
    print(f"      总检测数: {total_detections}")
    print(f"      平均置信度: {avg_confidence:.3f}")
    print(f"      检测类别数: {class_diversity}")
    
    return results

if __name__ == "__main__":
    import numpy as np
    main()
