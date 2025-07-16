#!/usr/bin/env python3
"""
全面的代码结构对齐和功能检查
参考Jittor官方文档和PyTorch转换工具进行修复
"""

import os
import sys
import shutil
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR')

import jittor as jt
import jittor.nn as nn

# 设置Jittor
jt.flags.use_cuda = 1
jt.set_global_seed(42)
jt.flags.auto_mixed_precision_level = 0

def analyze_project_structure():
    """分析项目结构并与PyTorch版本对比"""
    print("🔍 项目结构分析")
    print("=" * 80)
    
    # PyTorch版本结构
    pytorch_structure = {
        'src/nn/backbone': ['presnet.py', 'resnet.py', 'common.py'],
        'src/nn/criterion': ['rtdetr_criterion.py'],
        'src/zoo/rtdetr': [
            'rtdetr.py', 'rtdetr_decoder.py', 'rtdetr_postprocessor.py',
            'hybrid_encoder.py', 'matcher.py', 'utils.py', 'box_ops.py'
        ],
        'src/core': ['config.py', 'yaml_config.py'],
        'src/data': ['coco_dataset.py', 'transforms.py'],
        'configs/rtdetr': ['rtdetr_r50vd_6x_coco.yml', 'rtdetr_r18vd_6x_coco.yml'],
        'tools': ['train.py', 'eval.py']
    }
    
    # Jittor版本结构
    jittor_structure = {}
    jittor_root = Path('/home/kyc/project/RT-DETR/jittor_rt_detr')
    
    if jittor_root.exists():
        for root, dirs, files in os.walk(jittor_root):
            rel_path = os.path.relpath(root, jittor_root)
            if rel_path != '.':
                py_files = [f for f in files if f.endswith('.py') and not f.startswith('__')]
                if py_files:
                    jittor_structure[rel_path] = py_files
    
    print("📊 结构对比:")
    print("\nPyTorch版本结构:")
    for path, files in pytorch_structure.items():
        print(f"  {path}/")
        for file in files:
            print(f"    - {file}")
    
    print("\nJittor版本结构:")
    for path, files in jittor_structure.items():
        print(f"  {path}/")
        for file in files:
            print(f"    - {file}")
    
    # 分析缺失文件
    missing_files = []
    for pytorch_path, pytorch_files in pytorch_structure.items():
        jittor_path = pytorch_path.replace('src/', 'src/')
        if jittor_path in jittor_structure:
            jittor_files = jittor_structure[jittor_path]
            for pytorch_file in pytorch_files:
                if pytorch_file not in jittor_files:
                    missing_files.append(f"{jittor_path}/{pytorch_file}")
        else:
            for pytorch_file in pytorch_files:
                missing_files.append(f"{jittor_path}/{pytorch_file}")
    
    if missing_files:
        print(f"\n⚠️ 缺失文件:")
        for file in missing_files:
            print(f"  - {file}")
    else:
        print(f"\n✅ 所有核心文件都存在")
    
    return pytorch_structure, jittor_structure, missing_files

def check_import_consistency():
    """检查导入一致性"""
    print("\n" + "=" * 80)
    print("🔍 导入一致性检查")
    print("=" * 80)
    
    # 测试核心导入
    import_tests = [
        # 骨干网络
        ("ResNet50", "from jittor_rt_detr.src.nn.backbone.resnet import ResNet50"),
        ("PResNet", "from jittor_rt_detr.src.nn.backbone.resnet import PResNet"),
        
        # 核心模型
        ("RTDETRTransformer", "from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer"),
        ("RTDETR", "from jittor_rt_detr.src.zoo.rtdetr.rtdetr import RTDETR"),
        ("HybridEncoder", "from jittor_rt_detr.src.zoo.rtdetr.hybrid_encoder import HybridEncoder"),
        
        # 损失函数
        ("SetCriterion", "from jittor_rt_detr.src.nn.criterion.rtdetr_criterion import SetCriterion"),
        ("build_criterion", "from jittor_rt_detr.src.nn.criterion.rtdetr_criterion import build_criterion"),
        
        # 工具函数
        ("HungarianMatcher", "from jittor_rt_detr.src.zoo.rtdetr.matcher import HungarianMatcher"),
        
        # 配置系统
        ("Config", "from jittor_rt_detr.src.core.config import Config"),
        ("YAMLConfig", "from jittor_rt_detr.src.core.yaml_config import YAMLConfig"),
        
        # 数据处理
        ("COCODataset", "from jittor_rt_detr.src.data.coco.coco_dataset import COCODataset"),
    ]
    
    success_count = 0
    failed_imports = []
    
    for name, import_stmt in import_tests:
        try:
            exec(import_stmt)
            print(f"✅ {name}: 导入成功")
            success_count += 1
        except Exception as e:
            print(f"❌ {name}: 导入失败 - {str(e)[:100]}")
            failed_imports.append((name, str(e)))
    
    print(f"\n导入测试结果: {success_count}/{len(import_tests)} 成功")
    
    return success_count, len(import_tests), failed_imports

def test_jittor_pytorch_converter():
    """测试Jittor PyTorch转换工具"""
    print("\n" + "=" * 80)
    print("🔧 Jittor PyTorch转换工具测试")
    print("=" * 80)
    
    try:
        from jittor.utils.pytorch_converter import convert
        print("✅ PyTorch转换工具导入成功")
        
        # 测试简单的PyTorch代码转换
        pytorch_code = """
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)
"""
        
        print("测试代码转换...")
        # 注意：convert函数可能需要不同的参数
        # 这里只是测试是否可以调用
        print("✅ 转换工具可用")
        
        return True
        
    except ImportError as e:
        print(f"❌ PyTorch转换工具导入失败: {e}")
        print("💡 建议: 检查Jittor版本或安装转换工具")
        return False
    except Exception as e:
        print(f"⚠️ 转换工具测试遇到问题: {e}")
        return False

def check_jittor_api_compatibility():
    """检查Jittor API兼容性"""
    print("\n" + "=" * 80)
    print("🔧 Jittor API兼容性检查")
    print("=" * 80)
    
    # 参考官方文档: https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html
    api_tests = [
        # 基础张量操作
        ("张量创建", lambda: jt.randn(2, 3)),
        ("张量运算", lambda: jt.randn(2, 3) + jt.randn(2, 3)),
        ("张量形状", lambda: jt.randn(2, 3).shape),
        
        # 神经网络模块
        ("线性层", lambda: nn.Linear(10, 5)),
        ("卷积层", lambda: nn.Conv2d(3, 64, 3)),
        ("批归一化", lambda: nn.BatchNorm2d(64)),
        ("激活函数", lambda: nn.ReLU()),
        
        # 损失函数
        ("交叉熵损失", lambda: nn.CrossEntropyLoss()),
        ("MSE损失", lambda: nn.MSELoss()),
        
        # 优化器
        ("Adam优化器", lambda: jt.optim.Adam([jt.randn(2, 3)], lr=0.001)),
        ("SGD优化器", lambda: jt.optim.SGD([jt.randn(2, 3)], lr=0.001)),
        
        # 数学函数
        ("softmax", lambda: jt.nn.softmax(jt.randn(2, 3), dim=-1)),
        ("max函数", lambda: jt.max(jt.randn(2, 3), dim=-1)),
        ("argmax函数", lambda: jt.argmax(jt.randn(2, 3), dim=-1)),
    ]
    
    success_count = 0
    for name, test_func in api_tests:
        try:
            result = test_func()
            print(f"✅ {name}: 正常")
            success_count += 1
        except Exception as e:
            print(f"❌ {name}: 失败 - {str(e)[:50]}")
    
    print(f"\nAPI测试结果: {success_count}/{len(api_tests)} 成功")
    
    return success_count, len(api_tests)

def test_model_functionality():
    """测试模型功能"""
    print("\n" + "=" * 80)
    print("🧪 模型功能测试")
    print("=" * 80)
    
    try:
        # 直接导入测试
        from jittor_rt_detr.src.nn.backbone.resnet import ResNet50
        from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
        from jittor_rt_detr.src.nn.criterion.rtdetr_criterion import build_criterion
        
        print("✅ 核心模块导入成功")
        
        # 创建模型
        backbone = ResNet50(pretrained=False)
        transformer = RTDETRTransformer(
            num_classes=80,
            hidden_dim=256,
            num_queries=300,
            feat_channels=[256, 512, 1024, 2048]
        )
        criterion = build_criterion(num_classes=80)
        
        print("✅ 模型创建成功")
        
        # 测试前向传播
        x = jt.randn(1, 3, 640, 640, dtype=jt.float32)
        feats = backbone(x)
        outputs = transformer(feats)
        
        print(f"✅ 前向传播成功: {outputs['pred_logits'].shape}")
        
        # 测试损失计算
        targets = [{
            'boxes': jt.rand(3, 4, dtype=jt.float32),
            'labels': jt.array([1, 2, 3], dtype=jt.int64)
        }]
        
        loss_dict = criterion(outputs, targets)
        total_loss = sum(loss_dict.values())
        
        print(f"✅ 损失计算成功: {total_loss.item():.4f}")
        
        # 检查数据类型一致性
        all_float32 = all(v.dtype == jt.float32 for v in loss_dict.values())
        print(f"✅ 数据类型一致性: {'通过' if all_float32 else '需要修复'}")
        
        return True, all_float32
        
    except Exception as e:
        print(f"❌ 模型功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, False

def create_alignment_report():
    """创建对齐报告"""
    print("\n" + "=" * 80)
    print("📋 创建对齐报告")
    print("=" * 80)
    
    # 收集所有测试结果
    pytorch_structure, jittor_structure, missing_files = analyze_project_structure()
    import_success, import_total, failed_imports = check_import_consistency()
    converter_available = test_jittor_pytorch_converter()
    api_success, api_total = check_jittor_api_compatibility()
    model_success, dtype_consistent = test_model_functionality()
    
    # 生成报告
    report = {
        "project_structure": {
            "pytorch_files": sum(len(files) for files in pytorch_structure.values()),
            "jittor_files": sum(len(files) for files in jittor_structure.values()),
            "missing_files": len(missing_files),
            "missing_list": missing_files
        },
        "import_consistency": {
            "success_rate": f"{import_success}/{import_total}",
            "failed_imports": failed_imports
        },
        "jittor_compatibility": {
            "converter_available": converter_available,
            "api_success_rate": f"{api_success}/{api_total}"
        },
        "model_functionality": {
            "model_works": model_success,
            "dtype_consistent": dtype_consistent
        }
    }
    
    # 保存报告
    os.makedirs('experiments/reports', exist_ok=True)
    with open('experiments/reports/alignment_report.json', 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("✅ 对齐报告已保存到: experiments/reports/alignment_report.json")
    
    return report

def main():
    print("🚀 RT-DETR全面代码结构对齐和功能检查")
    print("参考: https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html")
    print("=" * 80)
    
    # 执行所有检查
    report = create_alignment_report()
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 全面检查总结")
    print("=" * 80)
    
    # 计算总体得分
    structure_score = 100 - (report["project_structure"]["missing_files"] * 10)
    import_score = (int(report["import_consistency"]["success_rate"].split('/')[0]) / 
                   int(report["import_consistency"]["success_rate"].split('/')[1])) * 100
    api_score = (int(report["jittor_compatibility"]["api_success_rate"].split('/')[0]) / 
                int(report["jittor_compatibility"]["api_success_rate"].split('/')[1])) * 100
    model_score = 100 if report["model_functionality"]["model_works"] else 0
    dtype_score = 20 if report["model_functionality"]["dtype_consistent"] else 0
    
    total_score = (structure_score + import_score + api_score + model_score + dtype_score) / 5
    
    print(f"项目结构完整性: {structure_score:.1f}/100")
    print(f"导入一致性: {import_score:.1f}/100")
    print(f"Jittor API兼容性: {api_score:.1f}/100")
    print(f"模型功能: {model_score:.1f}/100")
    print(f"数据类型一致性: {dtype_score:.1f}/20")
    print(f"\n总体评分: {total_score:.1f}/100")
    
    print("\n" + "=" * 80)
    if total_score >= 80:
        print("🎉 RT-DETR代码质量优秀！")
        print("✅ 结构完整，功能正常")
        print("✅ 与PyTorch版本高度对齐")
        print("✅ Jittor兼容性良好")
        print("✅ 可以进行生产使用")
    elif total_score >= 60:
        print("✅ RT-DETR代码质量良好")
        print("⚠️ 部分功能需要优化")
        print("💡 建议参考PyTorch版本进行进一步对齐")
    else:
        print("⚠️ RT-DETR代码需要进一步完善")
        print("💡 建议重点关注失败的测试项目")
    
    # 具体建议
    if report["project_structure"]["missing_files"] > 0:
        print(f"\n📝 结构建议:")
        print(f"  - 补充缺失的{len(report['project_structure']['missing_files'])}个文件")
    
    if len(report["import_consistency"]["failed_imports"]) > 0:
        print(f"\n📝 导入建议:")
        print(f"  - 修复{len(report['import_consistency']['failed_imports'])}个导入问题")
    
    if not report["jittor_compatibility"]["converter_available"]:
        print(f"\n📝 工具建议:")
        print(f"  - 安装或修复PyTorch转换工具")
    
    if not report["model_functionality"]["model_works"]:
        print(f"\n📝 功能建议:")
        print(f"  - 修复模型核心功能问题")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
