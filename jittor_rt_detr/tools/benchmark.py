#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能对比脚本
用于比较Jittor和PyTorch版本的性能
"""

import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class BenchmarkResult:
    def __init__(self, framework, model_name):
        self.framework = framework
        self.model_name = model_name
        self.results = {
            'training_time': [],
            'memory_usage': [],
            'inference_speed': [],
            'loss_curve': [],
            'accuracy': []
        }

    def add_training_result(self, epoch, loss, time_taken, memory_used):
        self.results['loss_curve'].append({'epoch': epoch, 'loss': loss})
        self.results['training_time'].append(time_taken)
        self.results['memory_usage'].append(memory_used)

    def add_inference_result(self, speed, accuracy):
        self.results['inference_speed'].append(speed)
        self.results['accuracy'].append(accuracy)

    def get_average_metrics(self):
        return {
            'avg_training_time': np.mean(self.results['training_time']),
            'avg_memory_usage': np.mean(self.results['memory_usage']),
            'avg_inference_speed': np.mean(self.results['inference_speed']),
            'final_loss': self.results['loss_curve'][-1]['loss'] if self.results['loss_curve'] else None,
            'final_accuracy': self.results['accuracy'][-1] if self.results['accuracy'] else None
        }


class BenchmarkRunner:
    def __init__(self, output_dir="benchmark_results"):
        self.output_dir = output_dir
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_framework_result(self, framework, result):
        self.results[framework] = result

    def generate_comparison_report(self):
        """生成对比报告"""
        report_file = f"{self.output_dir}/benchmark_report_{self.timestamp}.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# 性能对比报告 - {self.timestamp}\n\n")

            # 框架信息
            f.write("## 测试框架\n\n")
            for framework, result in self.results.items():
                f.write(f"- **{framework}**: {result.model_name}\n")

            # 平均指标对比
            f.write("\n## 平均性能指标\n\n")
            f.write("| 指标 | " + " | ".join(self.results.keys()) + " |\n")
            f.write("|------|" + "|".join(["-----"]
                    * len(self.results)) + "|\n")

            metrics = ['avg_training_time', 'avg_memory_usage',
                       'avg_inference_speed', 'final_loss']
            metric_names = ['训练时间(分钟)', '内存使用(GB)', '推理速度(ms)', '最终损失']

            for metric, name in zip(metrics, metric_names):
                values = []
                for framework, result in self.results.items():
                    avg_metrics = result.get_average_metrics()
                    value = avg_metrics.get(metric, 'N/A')
                    if isinstance(value, float):
                        if 'time' in metric:
                            values.append(f"{value:.1f}")
                        elif 'memory' in metric:
                            values.append(f"{value:.1f}")
                        elif 'speed' in metric:
                            values.append(f"{value:.1f}")
                        else:
                            values.append(f"{value:.4f}")
                    else:
                        values.append(str(value))

                f.write(f"| {name} | " + " | ".join(values) + " |\n")

            # 性能提升计算
            if len(self.results) >= 2:
                f.write("\n## 性能提升分析\n\n")
                frameworks = list(self.results.keys())
                baseline = frameworks[0]
                comparison = frameworks[1]

                baseline_metrics = self.results[baseline].get_average_metrics()
                comparison_metrics = self.results[comparison].get_average_metrics(
                )

                f.write(f"以 {baseline} 为基准，{comparison} 的性能表现：\n\n")

                for metric, name in zip(metrics, metric_names):
                    baseline_val = baseline_metrics.get(metric)
                    comparison_val = comparison_metrics.get(metric)

                    if baseline_val and comparison_val and isinstance(baseline_val, (int, float)) and isinstance(comparison_val, (int, float)):
                        if baseline_val != 0:
                            improvement = (
                                (comparison_val - baseline_val) / baseline_val) * 100
                            direction = "提升" if improvement < 0 else "下降"
                            f.write(
                                f"- **{name}**: {improvement:.1f}% {direction}\n")

        print(f"对比报告已生成: {report_file}")
        return report_file

    def plot_comparison_charts(self):
        """绘制对比图表"""
        if len(self.results) < 2:
            print("需要至少两个框架的结果才能进行对比")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Framework Performance Comparison', fontsize=16)

        # 损失曲线对比
        ax1 = axes[0, 0]
        for framework, result in self.results.items():
            epochs = [r['epoch'] for r in result.results['loss_curve']]
            losses = [r['loss'] for r in result.results['loss_curve']]
            ax1.plot(epochs, losses, label=framework, linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 训练时间对比
        ax2 = axes[0, 1]
        frameworks = list(self.results.keys())
        training_times = [np.mean(result.results['training_time'])
                          for result in self.results.values()]
        ax2.bar(frameworks, training_times, color=['blue', 'orange'])
        ax2.set_ylabel('Training Time (minutes)')
        ax2.set_title('Average Training Time')

        # 内存使用对比
        ax3 = axes[1, 0]
        memory_usage = [np.mean(result.results['memory_usage'])
                        for result in self.results.values()]
        ax3.bar(frameworks, memory_usage, color=['green', 'red'])
        ax3.set_ylabel('Memory Usage (GB)')
        ax3.set_title('Average Memory Usage')

        # 推理速度对比
        ax4 = axes[1, 1]
        inference_speeds = [np.mean(result.results['inference_speed'])
                            for result in self.results.values()]
        ax4.bar(frameworks, inference_speeds, color=['purple', 'brown'])
        ax4.set_ylabel('Inference Speed (ms)')
        ax4.set_title('Average Inference Speed')

        plt.tight_layout()

        # 保存图表
        chart_file = f"{self.output_dir}/benchmark_charts_{self.timestamp}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"对比图表已保存: {chart_file}")

    def save_results(self):
        """保存结果到JSON文件"""
        results_data = {}
        for framework, result in self.results.items():
            results_data[framework] = {
                'model_name': result.model_name,
                'results': result.results,
                'average_metrics': result.get_average_metrics()
            }

        json_file = f"{self.output_dir}/benchmark_results_{self.timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        print(f"结果数据已保存: {json_file}")


def simulate_jittor_results():
    """模拟Jittor框架的结果"""
    result = BenchmarkResult("Jittor", "RT-DETR")

    # 模拟训练过程
    for epoch in range(1, 21):
        loss = 8.0 * np.exp(-epoch * 0.1) + np.random.normal(0, 0.1)
        training_time = 0.9 + np.random.normal(0, 0.05)  # 分钟
        memory_usage = 3.5 + np.random.normal(0, 0.2)    # GB

        result.add_training_result(epoch, loss, training_time, memory_usage)

    # 模拟推理结果
    for _ in range(10):
        inference_speed = 28 + np.random.normal(0, 2)  # ms
        accuracy = 0.75 + np.random.normal(0, 0.02)
        result.add_inference_result(inference_speed, accuracy)

    return result


def simulate_pytorch_results():
    """模拟PyTorch框架的结果"""
    result = BenchmarkResult("PyTorch", "RT-DETR")

    # 模拟训练过程
    for epoch in range(1, 21):
        loss = 8.0 * np.exp(-epoch * 0.1) + np.random.normal(0, 0.1)
        training_time = 0.75 + np.random.normal(0, 0.05)  # 分钟
        memory_usage = 4.0 + np.random.normal(0, 0.2)     # GB

        result.add_training_result(epoch, loss, training_time, memory_usage)

    # 模拟推理结果
    for _ in range(10):
        inference_speed = 25 + np.random.normal(0, 2)  # ms
        accuracy = 0.75 + np.random.normal(0, 0.02)
        result.add_inference_result(inference_speed, accuracy)

    return result


def main():
    parser = argparse.ArgumentParser(description="性能对比工具")
    parser.add_argument('--output_dir', type=str, default='benchmark_results',
                        help='输出目录')
    args = parser.parse_args()

    # 创建输出目录
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # 创建对比运行器
    runner = BenchmarkRunner(args.output_dir)

    print("开始性能对比测试...")

    # 模拟Jittor结果
    print("模拟Jittor框架结果...")
    jittor_result = simulate_jittor_results()
    runner.add_framework_result("Jittor", jittor_result)

    # 模拟PyTorch结果
    print("模拟PyTorch框架结果...")
    pytorch_result = simulate_pytorch_results()
    runner.add_framework_result("PyTorch", pytorch_result)

    # 生成报告和图表
    print("生成对比报告...")
    runner.generate_comparison_report()
    runner.plot_comparison_charts()
    runner.save_results()

    print("性能对比测试完成！")


if __name__ == '__main__':
    main()
