#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RT-DETR Jittor 项目安装脚本
自动安装依赖、配置环境并验证安装
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path


class ProjectInstaller:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.requirements_file = self.project_root / "requirements.txt"
        self.python_version = sys.version_info

    def check_python_version(self):
        """检查Python版本"""
        print("检查Python版本...")

        if self.python_version < (3, 7):
            print(
                f"❌ Python版本过低: {self.python_version.major}.{self.python_version.minor}")
            print("需要Python 3.7或更高版本")
            return False
        elif self.python_version >= (3, 10):
            print(
                f"⚠️  Python版本: {self.python_version.major}.{self.python_version.minor}")
            print("建议使用Python 3.7-3.9以获得最佳兼容性")
        else:
            print(
                f"✅ Python版本: {self.python_version.major}.{self.python_version.minor}")

        return True

    def check_system_info(self):
        """检查系统信息"""
        print("\n系统信息:")
        print(f"操作系统: {platform.system()} {platform.release()}")
        print(f"架构: {platform.machine()}")
        print(f"Python路径: {sys.executable}")

    def install_jittor(self):
        """安装Jittor框架"""
        print("\n安装Jittor框架...")

        try:
            # 检查是否已安装Jittor
            import jittor
            print(f"✅ Jittor已安装，版本: {jittor.__version__}")
            return True
        except ImportError:
            print("Jittor未安装，开始安装...")

        try:
            # 安装Jittor (CPU版本)
            print("安装Jittor CPU版本...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "jittor"
            ])

            # 验证安装
            import jittor
            print(f"✅ Jittor安装成功，版本: {jittor.__version__}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Jittor安装失败: {e}")
            return False
        except ImportError as e:
            print(f"❌ Jittor导入失败: {e}")
            return False

    def install_requirements(self):
        """安装项目依赖"""
        print("\n安装项目依赖...")

        if not self.requirements_file.exists():
            print("❌ requirements.txt文件不存在")
            return False

        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(
                    self.requirements_file)
            ])
            print("✅ 项目依赖安装成功")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ 依赖安装失败: {e}")
            return False

    def verify_installation(self):
        """验证安装"""
        print("\n验证安装...")

        # 检查关键模块
        modules_to_check = [
            'jittor',
            'numpy',
            'PIL',
            'tqdm',
            'matplotlib',
            'cv2',
            'scipy'
        ]

        all_success = True
        for module in modules_to_check:
            try:
                if module == 'PIL':
                    import PIL
                    print(f"✅ {module}: {PIL.__version__}")
                elif module == 'cv2':
                    import cv2
                    print(f"✅ {module}: {cv2.__version__}")
                else:
                    imported_module = __import__(module)
                    version = getattr(
                        imported_module, '__version__', 'unknown')
                    print(f"✅ {module}: {version}")
            except ImportError as e:
                print(f"❌ {module}: 导入失败 - {e}")
                all_success = False

        return all_success

    def test_jittor_functionality(self):
        """测试Jittor功能"""
        print("\n测试Jittor功能...")

        try:
            import jittor as jt

            # 测试基本操作
            x = jt.array([1, 2, 3, 4])
            y = x * 2
            print(f"✅ 基本运算测试通过: {y}")

            # 测试GPU支持
            if jt.has_cuda:
                print("✅ CUDA支持可用")
                jt.flags.use_cuda = 1
                x_gpu = jt.array([1, 2, 3, 4])
                print(f"✅ GPU运算测试通过: {x_gpu}")
            else:
                print("⚠️  CUDA支持不可用，将使用CPU")

            return True

        except Exception as e:
            print(f"❌ Jittor功能测试失败: {e}")
            return False

    def create_directories(self):
        """创建必要的目录"""
        print("\n创建项目目录...")

        directories = [
            "data",
            "checkpoints",
            "experiments",
            "benchmark_results"
        ]

        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            print(f"✅ 创建目录: {directory}")

    def download_sample_data(self):
        """下载示例数据"""
        print("\n准备示例数据...")

        try:
            # 运行数据准备脚本
            subprocess.check_call([
                sys.executable, "prepare_data.py", "--subset_only", "--subset_size", "10"
            ])
            print("✅ 示例数据准备完成")
            return True

        except subprocess.CalledProcessError as e:
            print(f"⚠️  示例数据准备失败: {e}")
            print("您可以稍后手动运行: python prepare_data.py")
            return False

    def run_quick_test(self):
        """运行快速测试"""
        print("\n运行快速测试...")

        try:
            # 测试模型导入
            from model import RTDETR
            print("✅ 模型导入成功")

            # 测试数据集导入
            from dataset import COCODataset
            print("✅ 数据集导入成功")

            # 测试损失函数导入
            from loss import DETRLoss
            print("✅ 损失函数导入成功")

            print("✅ 所有模块导入成功")
            return True

        except Exception as e:
            print(f"❌ 快速测试失败: {e}")
            return False

    def generate_config_file(self):
        """生成配置文件"""
        print("\n生成配置文件...")

        config_content = """# RT-DETR Jittor 配置文件

# 数据配置
DATA_DIR = "data"
COCO_DIR = "data/coco"
TRAIN_IMG_DIR = "data/coco/train2017"
VAL_IMG_DIR = "data/coco/val2017"
TRAIN_ANN_FILE = "data/coco/annotations/instances_train2017.json"
VAL_ANN_FILE = "data/coco/annotations/instances_val2017.json"

# 模型配置
NUM_CLASSES = 80
NUM_QUERIES = 300
EMBED_DIM = 256
NUM_HEADS = 8
DEC_DEPTH = 6

# 训练配置
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 20
LR_DROP_EPOCH = 5
CLIP_MAX_NORM = 0.1

# 推理配置
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5

# 路径配置
CHECKPOINT_DIR = "checkpoints"
EXPERIMENT_DIR = "experiments"
BENCHMARK_DIR = "benchmark_results"
"""

        config_file = self.project_root / "config.py"
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)

        print("✅ 配置文件已生成: config.py")

    def print_usage_instructions(self):
        """打印使用说明"""
        print("\n" + "="*60)
        print("🎉 安装完成！")
        print("="*60)

        print("\n📋 使用说明:")
        print("1. 准备数据:")
        print("   python prepare_data.py --download")

        print("\n2. 开始训练:")
        print("   python train.py --subset_size 100 --epochs 20")

        print("\n3. 测试模型:")
        print(
            "   python test.py --weights checkpoints/model_epoch_20.pkl --img_path test.png")

        print("\n4. 可视化结果:")
        print(
            "   python vis.py --weights checkpoints/model_epoch_20.pkl --img_path test.png")

        print("\n5. 性能对比:")
        print("   python benchmark.py")

        print("\n📚 更多信息请查看 README.md")
        print("="*60)

    def install(self, skip_tests=False):
        """执行完整安装流程"""
        print("🚀 开始安装 RT-DETR Jittor 项目...")

        # 检查Python版本
        if not self.check_python_version():
            return False

        # 显示系统信息
        self.check_system_info()

        # 安装Jittor
        if not self.install_jittor():
            return False

        # 安装项目依赖
        if not self.install_requirements():
            return False

        # 验证安装
        if not self.verify_installation():
            return False

        # 测试Jittor功能
        if not self.test_jittor_functionality():
            return False

        # 创建目录
        self.create_directories()

        # 生成配置文件
        self.generate_config_file()

        # 下载示例数据
        self.download_sample_data()

        # 运行快速测试
        if not skip_tests:
            if not self.run_quick_test():
                print("⚠️  快速测试失败，但安装可能仍然可用")

        # 打印使用说明
        self.print_usage_instructions()

        return True


def main():
    parser = argparse.ArgumentParser(description="RT-DETR Jittor 项目安装工具")
    parser.add_argument('--skip-tests', action='store_true',
                        help='跳过快速测试')
    parser.add_argument('--gpu', action='store_true',
                        help='安装GPU版本的Jittor')

    args = parser.parse_args()

    installer = ProjectInstaller()

    if installer.install(skip_tests=args.skip_tests):
        print("\n✅ 安装成功完成！")
        sys.exit(0)
    else:
        print("\n❌ 安装失败！")
        print("请检查错误信息并重试")
        sys.exit(1)


if __name__ == '__main__':
    main()
