#!/usr/bin/env python3
"""
RT-DETR 完整全面测试脚本
从基础功能到完整训练的全面验证
"""

import os
import sys
import json
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'jittor_rt_detr'))

import jittor as jt
from src.nn.model import RTDETR

# 设置Jittor
jt.flags.use_cuda = 1

class ComprehensiveTester:
    """RT-DETR 综合测试器"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        
    def log_test(self, test_name, success, details=""):
        """记录测试结果"""
        self.test_results[test_name] = {
            'success': success,
            'details': details,
            'timestamp': time.time() - self.start_time
        }
        
        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {details}")
    
    def test_1_environment(self):
        """测试1: 环境检查"""
        print("\n" + "="*60)
        print("🧪 测试1: 环境检查")
        print("="*60)
        
        try:
            # Jittor版本
            print(f"Jittor版本: {jt.__version__}")
            
            # CUDA可用性
            cuda_available = jt.flags.use_cuda
            print(f"CUDA可用: {cuda_available}")
            
            # 内存测试
            test_tensor = jt.randn(100, 100)
            memory_ok = test_tensor.shape == (100, 100)
            
            self.log_test("环境检查", True, f"Jittor {jt.__version__}, CUDA: {cuda_available}")
            return True
            
        except Exception as e:
            self.log_test("环境检查", False, f"错误: {e}")
            return False
    
    def test_2_model_creation(self):
        """测试2: 模型创建"""
        print("\n" + "="*60)
        print("🧪 测试2: 模型创建")
        print("="*60)
        
        try:
            # 创建不同类别数的模型
            for num_classes in [2, 10, 80]:
                model = RTDETR(num_classes=num_classes)
                model = model.float32()
                
                # 检查模型参数
                total_params = sum(p.numel() for p in model.parameters())
                print(f"  {num_classes}类模型: {total_params:,} 参数")
                
                # 检查关键层
                has_encoder = hasattr(model, 'backbone')
                has_decoder = hasattr(model, 'decoder')
                
                if not (has_encoder and has_decoder):
                    raise Exception(f"{num_classes}类模型结构不完整")
            
            self.log_test("模型创建", True, "2/10/80类模型创建成功")
            return True
            
        except Exception as e:
            self.log_test("模型创建", False, f"错误: {e}")
            return False
    
    def test_3_forward_pass(self):
        """测试3: 前向传播"""
        print("\n" + "="*60)
        print("🧪 测试3: 前向传播")
        print("="*60)
        
        try:
            model = RTDETR(num_classes=10)
            model = model.float32()
            model.eval()
            
            # 测试不同批次大小
            for batch_size in [1, 2, 4]:
                dummy_input = jt.randn(batch_size, 3, 640, 640)
                
                with jt.no_grad():
                    outputs = model(dummy_input)
                
                logits, boxes, enc_logits, enc_boxes = outputs
                
                # 检查输出形状
                expected_queries = 300
                expected_classes = 10
                
                if logits.shape != (6, batch_size, expected_queries, expected_classes):
                    raise Exception(f"Logits形状错误: {logits.shape}")
                
                if boxes.shape != (6, batch_size, expected_queries, 4):
                    raise Exception(f"Boxes形状错误: {boxes.shape}")
                
                print(f"  批次大小 {batch_size}: 输出形状正确")
            
            self.log_test("前向传播", True, "所有批次大小测试通过")
            return True
            
        except Exception as e:
            self.log_test("前向传播", False, f"错误: {e}")
            return False
    
    def test_4_data_loading(self):
        """测试4: 数据加载"""
        print("\n" + "="*60)
        print("🧪 测试4: 数据加载")
        print("="*60)
        
        try:
            # 检查数据文件
            data_dir = "data/coco2017_50/train2017"
            ann_file = "data/coco2017_50/annotations/instances_train2017.json"
            
            if not os.path.exists(data_dir):
                raise Exception(f"数据目录不存在: {data_dir}")
            
            if not os.path.exists(ann_file):
                raise Exception(f"标注文件不存在: {ann_file}")
            
            # 加载标注文件
            with open(ann_file, 'r') as f:
                coco_data = json.load(f)
            
            num_images = len(coco_data['images'])
            num_annotations = len(coco_data['annotations'])
            num_categories = len(coco_data['categories'])
            
            print(f"  图片数量: {num_images}")
            print(f"  标注数量: {num_annotations}")
            print(f"  类别数量: {num_categories}")
            
            # 检查图片文件
            image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
            print(f"  实际图片文件: {len(image_files)}")
            
            if len(image_files) == 0:
                raise Exception("没有找到图片文件")
            
            # 测试加载一张图片
            test_img_path = os.path.join(data_dir, image_files[0])
            test_img = Image.open(test_img_path)
            print(f"  测试图片尺寸: {test_img.size}")
            
            self.log_test("数据加载", True, f"{num_images}张图片, {num_categories}个类别")
            return True
            
        except Exception as e:
            self.log_test("数据加载", False, f"错误: {e}")
            return False
    
    def test_5_simple_training(self):
        """测试5: 简单训练"""
        print("\n" + "="*60)
        print("🧪 测试5: 简单训练 (过拟合测试)")
        print("="*60)
        
        try:
            # 创建简单模型
            model = RTDETR(num_classes=10)
            model = model.float32()
            model.train()
            
            # 创建简单优化器
            optimizer = jt.optim.Adam(model.parameters(), lr=1e-4)
            
            # 创建固定的虚拟数据
            dummy_image = jt.randn(1, 3, 640, 640)
            dummy_targets = [{
                'labels': jt.array([0, 1]),  # 两个目标
                'boxes': jt.array([[0.5, 0.5, 0.2, 0.2], [0.3, 0.7, 0.1, 0.1]])
            }]
            
            print("  开始过拟合测试...")
            losses = []
            
            for epoch in range(10):
                # 前向传播
                outputs = model(dummy_image)
                logits, boxes, _, _ = outputs
                
                # 简单损失计算
                pred_logits = logits[-1][0][:2]  # 前2个查询
                pred_boxes = boxes[-1][0][:2]
                
                # 分类损失
                cls_loss = jt.nn.cross_entropy_loss(pred_logits, dummy_targets[0]['labels'])
                
                # 回归损失
                bbox_loss = jt.nn.l1_loss(pred_boxes, dummy_targets[0]['boxes'])
                
                total_loss = cls_loss + bbox_loss
                
                # 反向传播
                optimizer.zero_grad()
                optimizer.backward(total_loss)
                optimizer.step()
                
                loss_value = float(total_loss.data)
                losses.append(loss_value)
                
                if epoch % 2 == 0:
                    print(f"    Epoch {epoch}: Loss = {loss_value:.4f}")
                
                # 检查NaN
                if np.isnan(loss_value):
                    raise Exception(f"损失变为NaN (Epoch {epoch})")
            
            # 检查损失是否下降
            initial_loss = losses[0]
            final_loss = losses[-1]
            loss_reduction = (initial_loss - final_loss) / initial_loss
            
            print(f"  初始损失: {initial_loss:.4f}")
            print(f"  最终损失: {final_loss:.4f}")
            print(f"  损失下降: {loss_reduction:.2%}")
            
            if loss_reduction > 0.1:  # 损失下降超过10%
                self.log_test("简单训练", True, f"损失下降 {loss_reduction:.2%}")
                return True
            else:
                self.log_test("简单训练", False, f"损失下降不足: {loss_reduction:.2%}")
                return False
            
        except Exception as e:
            self.log_test("简单训练", False, f"错误: {e}")
            return False
    
    def test_6_inference_pipeline(self):
        """测试6: 推理流程"""
        print("\n" + "="*60)
        print("🧪 测试6: 推理流程")
        print("="*60)
        
        try:
            # 检查是否有训练好的模型
            model_paths = [
                "checkpoints/balanced_rt_detr_best_model.pkl",
                "checkpoints/balanced_rt_detr_final_model.pkl"
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                self.log_test("推理流程", False, "没有找到训练好的模型")
                return False
            
            # 加载模型
            model = RTDETR(num_classes=80)
            model = model.float32()
            state_dict = jt.load(model_path)
            model.load_state_dict(state_dict)
            model.eval()
            
            print(f"  使用模型: {model_path}")
            
            # 加载测试图片
            data_dir = "data/coco2017_50/train2017"
            image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
            test_img_path = os.path.join(data_dir, image_files[0])
            
            # 预处理
            image = Image.open(test_img_path).convert('RGB')
            original_size = image.size
            
            resized_image = image.resize((640, 640), Image.LANCZOS)
            img_array = np.array(resized_image, dtype=np.float32) / 255.0
            
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_array = (img_array - mean) / std
            img_tensor = jt.array(img_array.transpose(2, 0, 1), dtype='float32').unsqueeze(0)
            
            # 推理
            with jt.no_grad():
                outputs = model(img_tensor)
            
            logits, boxes, _, _ = outputs
            
            # 后处理
            pred_logits = logits[-1][0]
            pred_boxes = boxes[-1][0]
            
            logits_np = pred_logits.stop_grad().numpy()
            boxes_np = pred_boxes.stop_grad().numpy()
            
            # 计算置信度
            exp_logits = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
            scores = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            max_scores = np.max(scores, axis=1)
            
            print(f"  推理完成")
            print(f"  最高置信度: {np.max(max_scores):.3f}")
            print(f"  平均置信度: {np.mean(max_scores):.3f}")
            print(f"  >0.1置信度的数量: {np.sum(max_scores > 0.1)}")
            
            self.log_test("推理流程", True, f"推理成功, 最高置信度: {np.max(max_scores):.3f}")
            return True
            
        except Exception as e:
            self.log_test("推理流程", False, f"错误: {e}")
            return False
    
    def generate_test_report(self):
        """生成测试报告"""
        print("\n" + "="*60)
        print("📊 测试报告")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['success'])
        
        print(f"总测试数: {total_tests}")
        print(f"通过测试: {passed_tests}")
        print(f"失败测试: {total_tests - passed_tests}")
        print(f"通过率: {passed_tests/total_tests:.1%}")
        print(f"总耗时: {time.time() - self.start_time:.1f}秒")
        
        print(f"\n详细结果:")
        for test_name, result in self.test_results.items():
            status = "✅" if result['success'] else "❌"
            print(f"  {status} {test_name}: {result['details']}")
        
        # 保存报告
        report_path = "test_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\n📄 详细报告已保存: {report_path}")
        
        return passed_tests == total_tests
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始RT-DETR完整全面测试")
        print("="*60)
        
        # 按顺序运行所有测试
        tests = [
            self.test_1_environment,
            self.test_2_model_creation,
            self.test_3_forward_pass,
            self.test_4_data_loading,
            self.test_5_simple_training,
            self.test_6_inference_pipeline
        ]
        
        for test_func in tests:
            try:
                success = test_func()
                if not success:
                    print(f"\n⚠️ 测试失败，但继续执行后续测试...")
            except Exception as e:
                print(f"\n❌ 测试异常: {e}")
        
        # 生成最终报告
        all_passed = self.generate_test_report()
        
        if all_passed:
            print("\n🎉 所有测试通过！RT-DETR系统完全正常！")
        else:
            print("\n⚠️ 部分测试失败，请查看详细报告")
        
        return all_passed

def main():
    tester = ComprehensiveTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎯 建议下一步:")
        print("  1. 运行小规模训练验证学习能力")
        print("  2. 使用对比可视化分析推理效果")
        print("  3. 尝试不同的训练策略和超参数")
    else:
        print("\n🔧 建议修复:")
        print("  1. 检查失败的测试项")
        print("  2. 确保环境配置正确")
        print("  3. 验证数据文件完整性")

if __name__ == "__main__":
    main()
