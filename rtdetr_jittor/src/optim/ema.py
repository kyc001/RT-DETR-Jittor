"""
Model Exponential Moving Average (EMA) for Jittor
参考: rtdetr_pytorch/src/optim/ema.py

重要注意事项 (针对Jittor的惰性求值特性):
- Jittor使用惰性求值机制，所有计算只创建计算图，实际执行只在需要结果时
- EMA的target模型参数不会被频繁使用，可能导致计算图无限增长和内存泄漏
- 解决方案：在EMA更新后调用 jt.sync_all() 来强制执行计算并清理图
"""

import math
import jittor as jt
import jittor.nn as nn
from copy import deepcopy


__all__ = ['ModelEMA']


class ModelEMA:
    """
    Model Exponential Moving Average - Jittor版本

    保持模型state_dict中所有参数和缓冲区的移动平均。
    平滑版本的权重对于某些训练方案的良好表现是必要的。

    参考: https://github.com/rwightman/pytorch-image-models

    使用方法:
        ema = ModelEMA(model, decay=0.9999, warmups=2000)
        for batch in dataloader:
            loss = criterion(model(batch), targets)
            optimizer.step(loss)
            ema.update(model)  # 在优化器更新后调用

        # 评估时使用EMA模型
        results = evaluate(ema.module)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, warmups: int = 2000):
        """
        初始化EMA

        Args:
            model: 原始模型
            decay: 基础衰减率 (默认0.9999)
            warmups: 预热步数，用于指数增长衰减率 (默认2000)
        """
        super().__init__()

        # 创建EMA模型副本 (FP32)
        # 深拷贝模型并设置为评估模式
        self.module = deepcopy(model)
        self.module.eval()

        # 禁用EMA模块的梯度计算
        for p in self.module.parameters():
            p.stop_grad()

        self.decay = decay
        self.warmups = warmups
        self.updates = 0  # EMA更新次数

        # 衰减函数 - 指数增长到目标衰减率 (帮助早期训练)
        # 在warmup期间，衰减率从0逐渐增长到decay
        self.decay_fn = lambda x: decay * (1 - math.exp(-x / warmups))

    def update(self, model: nn.Module):
        """
        更新EMA参数

        重要: 在每个batch的优化器更新后调用此方法

        Args:
            model: 当前训练模型
        """
        # 更新计数
        self.updates += 1
        d = self.decay_fn(self.updates)

        # 获取当前模型的state_dict
        model_state = model.state_dict()

        # 更新EMA模型的参数
        for name, ema_param in self.module.state_dict().items():
            if name in model_state:
                model_param = model_state[name]

                # 只更新浮点数类型的参数
                if ema_param.dtype in [jt.float32, jt.float16, jt.float64]:
                    # EMA更新: ema_param = d * ema_param + (1 - d) * model_param
                    # 使用stop_grad()确保不会计算梯度
                    new_val = d * ema_param + (1 - d) * model_param.stop_grad()
                    ema_param.update(new_val)

        # 关键: 调用sync_all强制执行计算，防止Jittor惰性求值导致的内存泄漏
        # 这会立即执行所有pending的计算并释放中间结果
        jt.sync_all()

    def state_dict(self):
        """
        获取EMA状态字典

        Returns:
            dict: 包含module state_dict, updates, warmups
        """
        return {
            'module': self.module.state_dict(),
            'updates': self.updates,
            'warmups': self.warmups
        }

    def load_state_dict(self, state):
        """
        加载EMA状态

        Args:
            state: 之前保存的状态字典
        """
        self.module.load_state_dict(state['module'])
        if 'updates' in state:
            self.updates = state['updates']
        if 'warmups' in state:
            self.warmups = state['warmups']

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        """
        从原始模型更新EMA模型的属性

        Args:
            model: 原始模型
            include: 要包含的属性列表
            exclude: 要排除的属性列表
        """
        self._copy_attr(self.module, model, include, exclude)

    @staticmethod
    def _copy_attr(a, b, include=(), exclude=()):
        """
        从b复制属性到a

        Args:
            a: 目标对象
            b: 源对象
            include: 要包含的属性
            exclude: 要排除的属性
        """
        for k, v in b.__dict__.items():
            if (len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(a, k, v)

    def __repr__(self):
        return f'ModelEMA(decay={self.decay}, warmups={self.warmups}, updates={self.updates})'

    def extra_repr(self) -> str:
        return f'decay={self.decay}, warmups={self.warmups}'


def create_ema(model, decay=0.9999, warmups=2000):
    """
    创建EMA的工厂函数

    Args:
        model: 原始模型
        decay: 衰减率
        warmups: 预热步数

    Returns:
        ModelEMA实例
    """
    return ModelEMA(model, decay=decay, warmups=warmups)
