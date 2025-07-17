"""
RT-DETR组件化模块
提供可复用的RT-DETR组件，参考ultimate_sanity_check.py的验证实现

组件包括:
- model: RT-DETR模型组件
- dataset: COCO数据集加载组件  
- trainer: 训练组件
- visualizer: 可视化组件

使用示例:
    from jittor_rt_detr.src.components.model import create_rtdetr_model, create_optimizer
    from jittor_rt_detr.src.components.dataset import create_coco_dataset
    from jittor_rt_detr.src.components.trainer import quick_train
    from jittor_rt_detr.src.components.visualizer import create_visualizer
"""

# 导入主要的工厂函数
from .model import create_rtdetr_model, create_optimizer, RTDETRModel
from .dataset import create_coco_dataset, COCODataset
from .trainer import create_trainer, quick_train, RTDETRTrainer
from .visualizer import create_visualizer, RTDETRVisualizer

__version__ = "1.0.0"
__author__ = "RT-DETR Jittor Implementation"

__all__ = [
    # 模型组件
    'create_rtdetr_model',
    'create_optimizer', 
    'RTDETRModel',
    
    # 数据集组件
    'create_coco_dataset',
    'COCODataset',
    
    # 训练组件
    'create_trainer',
    'quick_train',
    'RTDETRTrainer',
    
    # 可视化组件
    'create_visualizer',
    'RTDETRVisualizer'
]
