#!/usr/bin/env python3
"""
测试完整训练脚本的基本功能
在实际运行50轮训练前，先验证脚本的各个组件是否正常工作
"""

import os
import sys
import json

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt

# 设置Jittor
jt.flags.use_cuda = 1

def test_data_loading():
    """测试数据加载功能"""
    print("🔄 测试数据加载...")
    
    try:
        # 导入数据加载函数
        from full_scale_training import load_coco_dataset
        
        data_dir = '/home/kyc/project/RT-DETR/data/coco2017_50'
        
        # 测试训练集加载
        train_images, train_annotations, train_dir = load_coco_dataset(data_dir, 'train')
        print(f"   ✅ 训练集: {len(train_images)}张图像")
        
        # 测试验证集加载
        val_images, val_annotations, val_dir = load_coco_dataset(data_dir, 'val')
        print(f"   ✅ 验证集: {len(val_images)}张图像")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 数据加载失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("🔄 测试模型创建...")
    
    try:
        from full_scale_training import create_model
        
        model = create_model()
        if model is not None:
            print("   ✅ 模型创建成功")
            return True, model
        else:
            print("   ❌ 模型创建失败")
            return False, None
            
    except Exception as e:
        print(f"   ❌ 模型创建异常: {e}")
        return False, None

def test_single_forward_pass(model):
    """测试单次前向传播"""
    print("🔄 测试前向传播...")
    
    try:
        # 创建虚拟输入
        dummy_input = jt.randn(1, 3, 640, 640)
        
        # 前向传播
        with jt.no_grad():
            outputs = model(dummy_input)
        
        print(f"   ✅ 前向传播成功")
        print(f"      输出键: {list(outputs.keys())}")
        print(f"      pred_logits形状: {outputs['pred_logits'].shape}")
        print(f"      pred_boxes形状: {outputs['pred_boxes'].shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_computation(model):
    """测试损失计算"""
    print("🔄 测试损失计算...")
    
    try:
        from jittor_rt_detr.src.nn.criterion.rtdetr_criterion import build_criterion

        # 创建损失函数
        criterion = build_criterion(num_classes=80)
        
        # 创建虚拟输入和目标
        dummy_input = jt.randn(1, 3, 640, 640)
        dummy_targets = [{
            'boxes': jt.array([[0.1, 0.1, 0.5, 0.5], [0.3, 0.3, 0.7, 0.7]], dtype=jt.float32),
            'labels': jt.array([0, 2], dtype=jt.int64)
        }]
        
        # 前向传播
        outputs = model(dummy_input)
        
        # 计算损失
        loss_dict = criterion(outputs, dummy_targets)
        total_loss = sum(loss_dict.values())
        
        print(f"   ✅ 损失计算成功")
        print(f"      损失项: {list(loss_dict.keys())}")
        print(f"      总损失: {total_loss.numpy().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 损失计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_postprocessing():
    """测试后处理功能"""
    print("🔄 测试后处理...")
    
    try:
        from full_scale_training import postprocess_outputs, create_model
        
        # 创建模型并生成输出
        model = create_model()
        dummy_input = jt.randn(1, 3, 640, 640)
        
        with jt.no_grad():
            outputs = model(dummy_input)
        
        # 后处理
        detections = postprocess_outputs(outputs)
        
        print(f"   ✅ 后处理成功")
        print(f"      检测数量: {len(detections)}")
        
        if detections:
            print(f"      示例检测: {detections[0]}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 后处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_save_load():
    """测试模型保存和加载"""
    print("🔄 测试模型保存/加载...")
    
    try:
        from full_scale_training import create_model, save_model, load_model
        
        # 创建模型
        model = create_model()
        
        # 保存模型
        test_save_path = '/tmp/test_model.pkl'
        save_model(model, test_save_path, epoch=0, loss=1.0, 
                  additional_info={'test': True})
        
        # 加载模型
        new_model = create_model()
        checkpoint = load_model(new_model, test_save_path)
        
        if checkpoint is not None:
            print(f"   ✅ 模型保存/加载成功")
            
            # 清理测试文件
            if os.path.exists(test_save_path):
                os.remove(test_save_path)
            
            return True
        else:
            return False
            
    except Exception as e:
        print(f"   ❌ 模型保存/加载失败: {e}")
        return False

def test_logger():
    """测试日志记录器"""
    print("🔄 测试日志记录...")
    
    try:
        from full_scale_training import TrainingLogger
        
        # 创建日志记录器
        logger = TrainingLogger('/tmp/test_logs')
        
        # 记录一些虚拟数据
        logger.log_training(0, 2.5)
        logger.log_training(1, 2.0)
        logger.log_training(2, 1.5)
        
        logger.log_validation(2, {
            'total_detections': 100,
            'avg_confidence': 0.75,
            'class_diversity': 10
        })
        
        # 保存日志
        logger.save_logs()
        logger.plot_training_curve()
        
        print(f"   ✅ 日志记录成功")
        
        # 清理测试文件
        import shutil
        if os.path.exists('/tmp/test_logs'):
            shutil.rmtree('/tmp/test_logs')
        
        return True
        
    except Exception as e:
        print(f"   ❌ 日志记录失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 RT-DETR完整训练脚本功能测试")
    print("=" * 60)
    
    tests = [
        ("数据加载", test_data_loading),
        ("模型创建", test_model_creation),
        ("后处理", test_postprocessing),
        ("模型保存/加载", test_model_save_load),
        ("日志记录", test_logger),
    ]
    
    results = {}
    model = None
    
    for test_name, test_func in tests:
        if test_name == "模型创建":
            success, model = test_func()
            results[test_name] = success
        elif test_name in ["前向传播", "损失计算"] and model is not None:
            results[test_name] = test_func(model)
        else:
            results[test_name] = test_func()
    
    # 如果模型创建成功，测试前向传播和损失计算
    if model is not None:
        print("🔄 测试前向传播...")
        results["前向传播"] = test_single_forward_pass(model)
        
        print("🔄 测试损失计算...")
        results["损失计算"] = test_loss_computation(model)
    
    # 输出测试结果
    print("\n📊 测试结果汇总:")
    print("=" * 60)
    
    all_passed = True
    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name:15} : {status}")
        if not success:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 所有测试通过！可以开始完整训练。")
        print("\n💡 运行完整训练:")
        print("   python experiments/full_scale_training.py")
    else:
        print("⚠️  部分测试失败，请检查问题后再进行完整训练。")

if __name__ == "__main__":
    main()
