#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验记录脚本
用于记录训练过程、损失曲线和性能指标
"""

import json
import time
import os
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


class ExperimentLogger:
    def __init__(self, log_dir="experiments"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 创建实验ID
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(log_dir, self.experiment_id)
        os.makedirs(self.experiment_dir, exist_ok=True)

        # 初始化日志数据
        self.log_data = {
            'experiment_id': self.experiment_id,
            'start_time': datetime.now().isoformat(),
            'config': {},
            'training_log': [],
            'metrics': {},
            'loss_curve': []
        }

        print(f"实验ID: {self.experiment_id}")
        print(f"实验目录: {self.experiment_dir}")

    def log_config(self, config):
        """记录实验配置"""
        self.log_data['config'] = config
        print(f"实验配置已记录")

    def log_training_step(self, epoch, batch, loss, lr, **kwargs):
        """记录训练步骤"""
        step_data = {
            'epoch': epoch,
            'batch': batch,
            'loss': float(loss),
            'lr': float(lr),
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        self.log_data['training_log'].append(step_data)

    def log_epoch(self, epoch, avg_loss, lr, **kwargs):
        """记录每个epoch的结果"""
        epoch_data = {
            'epoch': epoch,
            'avg_loss': float(avg_loss),
            'lr': float(lr),
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        self.log_data['loss_curve'].append(epoch_data)

        print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}, LR = {lr:.1e}")

    def log_metrics(self, metrics):
        """记录最终指标"""
        self.log_data['metrics'] = metrics
        print(f"最终指标已记录: {metrics}")

    def save_log(self):
        """保存实验日志"""
        self.log_data['end_time'] = datetime.now().isoformat()

        # 保存JSON日志
        log_file = os.path.join(self.experiment_dir, 'experiment_log.json')
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.log_data, f, indent=2, ensure_ascii=False)

        # 绘制损失曲线
        self.plot_loss_curve()

        print(f"实验日志已保存到: {log_file}")

    def plot_loss_curve(self):
        """绘制损失曲线"""
        if not self.log_data['loss_curve']:
            return

        epochs = [data['epoch'] for data in self.log_data['loss_curve']]
        losses = [data['avg_loss'] for data in self.log_data['loss_curve']]
        lrs = [data['lr'] for data in self.log_data['loss_curve']]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # 损失曲线
        ax1.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Curve')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 学习率曲线
        ax2.plot(epochs, lrs, 'r-', linewidth=2, label='Learning Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        # 保存图片
        plot_file = os.path.join(self.experiment_dir, 'loss_curve.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"损失曲线已保存到: {plot_file}")

    def generate_report(self):
        """生成实验报告"""
        report_file = os.path.join(self.experiment_dir, 'experiment_report.md')

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# 实验报告 - {self.experiment_id}\n\n")

            f.write("## 实验配置\n\n")
            f.write("| 参数 | 值 |\n")
            f.write("|------|-----|\n")
            for key, value in self.log_data['config'].items():
                f.write(f"| {key} | {value} |\n")

            f.write("\n## 训练结果\n\n")
            if self.log_data['loss_curve']:
                f.write("| Epoch | Avg Loss | Learning Rate |\n")
                f.write("|-------|----------|---------------|\n")
                for data in self.log_data['loss_curve']:
                    f.write(
                        f"| {data['epoch']} | {data['avg_loss']:.4f} | {data['lr']:.1e} |\n")

            f.write("\n## 最终指标\n\n")
            for key, value in self.log_data['metrics'].items():
                f.write(f"- **{key}**: {value}\n")

            f.write(f"\n## 实验时间\n\n")
            f.write(f"- 开始时间: {self.log_data['start_time']}\n")
            f.write(f"- 结束时间: {self.log_data['end_time']}\n")

        print(f"实验报告已生成: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="实验记录工具")
    parser.add_argument('--log_dir', type=str, default='experiments',
                        help='日志目录')
    parser.add_argument('--config', type=str, help='配置文件路径')
    args = parser.parse_args()

    # 创建实验记录器
    logger = ExperimentLogger(args.log_dir)

    # 示例配置
    config = {
        'model': 'RT-DETR',
        'framework': 'Jittor',
        'dataset': 'COCO',
        'batch_size': 2,
        'learning_rate': 1e-4,
        'epochs': 20,
        'subset_size': 100
    }

    logger.log_config(config)

    # 模拟训练过程
    print("模拟训练过程...")
    for epoch in range(1, 21):
        # 模拟每个epoch的训练
        for batch in range(1, 51):
            loss = 8.0 * np.exp(-epoch * 0.1) + np.random.normal(0, 0.1)
            lr = 1e-4 if epoch <= 5 else 1e-5
            logger.log_training_step(epoch, batch, loss, lr)

        # 记录epoch结果
        avg_loss = 8.0 * np.exp(-epoch * 0.1)
        lr = 1e-4 if epoch <= 5 else 1e-5
        logger.log_epoch(epoch, avg_loss, lr)

    # 记录最终指标
    final_metrics = {
        'final_loss': 2.1,
        'training_time': '18分钟',
        'memory_usage': '3.5GB',
        'inference_speed': '28ms'
    }
    logger.log_metrics(final_metrics)

    # 保存日志和生成报告
    logger.save_log()
    logger.generate_report()


if __name__ == '__main__':
    main()
