"""
Jittor RT-DETR 优化器模块

包含:
- ModelEMA: 模型指数移动平均
"""

from .ema import ModelEMA, create_ema

__all__ = ['ModelEMA', 'create_ema']
