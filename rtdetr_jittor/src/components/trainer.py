#!/usr/bin/env python3
"""
训练组件
提供RT-DETR训练功能，参考ultimate_sanity_check.py的验证实现

支持EMA (Exponential Moving Average) 训练
"""

import os
import sys
import time
import pickle
import numpy as np

import jittor as jt

# 导入EMA模块
try:
    from src.optim.ema import ModelEMA, create_ema
    EMA_AVAILABLE = True
except ImportError:
    EMA_AVAILABLE = False
    ModelEMA = None
    create_ema = None


class RTDETRTrainer:
    """
    RT-DETR训练器
    参考ultimate_sanity_check.py的验证实现
    """
    def __init__(self, model, criterion, optimizer, save_dir="./results",
                 use_ema=False, ema_decay=0.9999, ema_warmups=2000):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.save_dir = save_dir

        # EMA支持
        self.ema = None
        if use_ema:
            if EMA_AVAILABLE:
                self.ema = ModelEMA(model, decay=ema_decay, warmups=ema_warmups)
                print(f"EMA enabled: decay={ema_decay}, warmups={ema_warmups}")
            else:
                print("Warning: EMA requested but module not available")

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 训练历史
        self.train_losses = []
        self.epoch_times = []

        print(f"Training results will be saved to: {save_dir}")
    
    def train_epoch(self, dataset, epoch, total_epochs):
        """
        训练一个epoch
        参考ultimate_sanity_check.py的实现
        """
        self.model.train()
        epoch_losses = []
        epoch_start_time = time.time()
        
        # 随机打乱数据
        indices = np.random.permutation(len(dataset))
        
        for i, idx in enumerate(indices):
            # 加载数据
            images, targets = dataset[idx]
            
            # 添加batch维度
            images = images.unsqueeze(0)
            targets = [targets]
            
            # 前向传播
            outputs = self.model(images, targets)
            
            # 损失计算
            loss_dict = self.criterion(outputs, targets)
            total_loss = sum(loss_dict.values())
            
            # 反向传播 - 使用验证过的方法
            self.optimizer.step(total_loss)

            # 关键: 强制执行计算图，防止Jittor惰性求值导致的内存泄漏
            jt.sync_all()

            # EMA更新 - 在优化器步骤后更新
            if self.ema is not None:
                self.ema.update(self.model)

            epoch_losses.append(float(total_loss.item()))
            
            # 打印进度（每10%或前几个batch）
            if i < 5 or (i + 1) % max(1, len(indices) // 10) == 0:
                print(f"     Batch {i+1}/{len(indices)}: 损失 = {total_loss.item():.4f}")
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = np.mean(epoch_losses)
        
        # 记录历史
        self.train_losses.append(avg_loss)
        self.epoch_times.append(epoch_time)
        
        print(f"   Epoch {epoch+1:3d}/{total_epochs}: 平均损失 = {avg_loss:.4f}, 用时 = {epoch_time:.1f}s")
        
        return avg_loss, epoch_time
    
    def train(self, dataset, num_epochs, lr_decay_epochs=None, lr_decay_factor=0.5):
        """
        完整训练流程
        
        Args:
            dataset: 训练数据集
            num_epochs: 训练轮数
            lr_decay_epochs: 学习率衰减的epoch列表
            lr_decay_factor: 学习率衰减因子
        """
        print(f"🚀 开始训练 {len(dataset)} 张图像，{num_epochs} 轮")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # 训练一个epoch
            avg_loss, epoch_time = self.train_epoch(dataset, epoch, num_epochs)
            
            # 学习率衰减
            if lr_decay_epochs and (epoch + 1) in lr_decay_epochs:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= lr_decay_factor
                print(f"   学习率衰减到: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        total_time = time.time() - start_time
        
        print(f"\n✅ 训练完成!")
        print(f"   初始损失: {self.train_losses[0]:.4f}")
        print(f"   最终损失: {self.train_losses[-1]:.4f}")
        print(f"   损失下降: {self.train_losses[0] - self.train_losses[-1]:.4f}")
        print(f"   总训练时间: {total_time:.1f}秒")
        print(f"   平均每轮时间: {total_time/num_epochs:.1f}秒")
        
        return {
            'train_losses': self.train_losses,
            'epoch_times': self.epoch_times,
            'total_time': total_time,
            'num_epochs': num_epochs,
            'final_loss': self.train_losses[-1],
            'loss_reduction': self.train_losses[0] - self.train_losses[-1]
        }
    
    def save_model(self, filename="rtdetr_model.pkl"):
        """保存模型"""
        model_path = os.path.join(self.save_dir, filename)
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")

        # 如果有EMA，也保存EMA模型
        if self.ema is not None:
            ema_filename = filename.replace('.pkl', '_ema.pkl')
            ema_path = os.path.join(self.save_dir, ema_filename)
            self.ema.module.save(ema_path)
            print(f"EMA model saved to: {ema_path}")

        return model_path

    def save_checkpoint(self, filename="checkpoint.pkl"):
        """保存完整的检查点（包括模型、优化器、EMA状态）"""
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'epoch_times': self.epoch_times,
        }
        if self.ema is not None:
            checkpoint['ema'] = self.ema.state_dict()

        checkpoint_path = os.path.join(self.save_dir, filename)
        jt.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = jt.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'epoch_times' in checkpoint:
            self.epoch_times = checkpoint['epoch_times']

        if self.ema is not None and 'ema' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema'])
            print('Loading ema.state_dict')

        print(f"Checkpoint loaded from: {checkpoint_path}")

    def get_eval_model(self):
        """
        获取用于评估的模型

        如果启用了EMA，返回EMA模型；否则返回原始模型
        """
        if self.ema is not None:
            return self.ema.module
        return self.model
    
    def save_training_results(self, results, filename="training_results.pkl"):
        """保存训练结果"""
        results_path = os.path.join(self.save_dir, filename)
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"📊 训练结果保存到: {results_path}")
        return results_path
    
    def get_training_stats(self):
        """获取训练统计信息"""
        if not self.train_losses:
            return None
        
        return {
            'total_epochs': len(self.train_losses),
            'initial_loss': self.train_losses[0],
            'final_loss': self.train_losses[-1],
            'best_loss': min(self.train_losses),
            'loss_reduction': self.train_losses[0] - self.train_losses[-1],
            'loss_reduction_percent': (self.train_losses[0] - self.train_losses[-1]) / self.train_losses[0] * 100,
            'total_time': sum(self.epoch_times),
            'avg_epoch_time': np.mean(self.epoch_times),
            'convergence_rate': (self.train_losses[0] - self.train_losses[-1]) / sum(self.epoch_times)
        }
    
    def print_training_summary(self):
        """打印训练总结"""
        stats = self.get_training_stats()
        if stats is None:
            print("⚠️ 没有训练历史数据")
            return
        
        print(f"\n📊 训练总结:")
        print(f"   训练轮数: {stats['total_epochs']}")
        print(f"   初始损失: {stats['initial_loss']:.4f}")
        print(f"   最终损失: {stats['final_loss']:.4f}")
        print(f"   最佳损失: {stats['best_loss']:.4f}")
        print(f"   损失下降: {stats['loss_reduction']:.4f} ({stats['loss_reduction_percent']:.1f}%)")
        print(f"   总训练时间: {stats['total_time']:.1f}秒")
        print(f"   平均每轮时间: {stats['avg_epoch_time']:.1f}秒")
        print(f"   收敛速度: {stats['convergence_rate']:.6f} 损失/秒")

def create_trainer(model, criterion, optimizer, save_dir="./results",
                   use_ema=False, ema_decay=0.9999, ema_warmups=2000):
    """
    创建训练器的工厂函数

    Args:
        model: RT-DETR模型
        criterion: 损失函数
        optimizer: 优化器
        save_dir: 保存目录
        use_ema: 是否使用EMA
        ema_decay: EMA衰减率
        ema_warmups: EMA预热步数

    Returns:
        trainer: RT-DETR训练器
    """
    trainer = RTDETRTrainer(
        model, criterion, optimizer, save_dir,
        use_ema=use_ema, ema_decay=ema_decay, ema_warmups=ema_warmups
    )
    return trainer


def quick_train(model, criterion, optimizer, dataset, num_epochs=50, save_dir="./results",
                use_ema=True, ema_decay=0.9999, ema_warmups=2000):
    """
    快速训练函数

    Args:
        model: RT-DETR模型
        criterion: 损失函数
        optimizer: 优化器
        dataset: 训练数据集
        num_epochs: 训练轮数
        save_dir: 保存目录
        use_ema: 是否使用EMA (默认True)
        ema_decay: EMA衰减率
        ema_warmups: EMA预热步数

    Returns:
        trainer: 训练器
        results: 训练结果
    """
    # 创建训练器 (默认启用EMA)
    trainer = create_trainer(
        model, criterion, optimizer, save_dir,
        use_ema=use_ema, ema_decay=ema_decay, ema_warmups=ema_warmups
    )

    # 开始训练
    results = trainer.train(dataset, num_epochs, lr_decay_epochs=[30, 40], lr_decay_factor=0.5)

    # 保存模型和结果
    trainer.save_model("rtdetr_trained.pkl")
    trainer.save_training_results(results)

    # 打印总结
    trainer.print_training_summary()

    return trainer, results

if __name__ == "__main__":
    # 测试训练组件
    print("🧪 测试RT-DETR训练组件")
    print("=" * 50)
    
    try:
        print("⚠️ 训练组件需要配合模型和数据集组件使用")
        print("   请参考train_script.py中的完整使用示例")
        
        print(f"\n🎉 RT-DETR训练组件验证完成!")
        
    except Exception as e:
        print(f"❌ 训练组件测试失败: {e}")
        import traceback
        traceback.print_exc()
