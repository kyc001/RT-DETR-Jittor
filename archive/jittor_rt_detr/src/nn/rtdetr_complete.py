"""
完整的RT-DETR模型 - 严格按照PyTorch版本实现
参考: rtdetr_pytorch/src/zoo/rtdetr/rtdetr.py
"""

import jittor as jt
import jittor.nn as nn
import numpy as np
from .model import ResNet50
from .rtdetr_pytorch_aligned import RTDETRTransformer


class RTDETR(nn.Module):
    """
    RT-DETR模型 - 严格按照PyTorch版本的组合架构
    """
    
    def __init__(self, num_classes=80, backbone=None, encoder=None, decoder=None, multi_scale=None):
        super().__init__()
        
        # 如果没有提供组件，使用默认配置
        if backbone is None:
            self.backbone = ResNet50()
        else:
            self.backbone = backbone
            
        if decoder is None:
            # 使用默认的RT-DETR Transformer配置
            self.decoder = RTDETRTransformer(
                num_classes=num_classes,
                hidden_dim=256,
                num_queries=300,
                feat_channels=[512, 1024, 2048],  # ResNet50的输出通道
                feat_strides=[8, 16, 32],
                num_levels=3,
                num_decoder_points=4,
                nhead=8,
                num_decoder_layers=6,
                dim_feedforward=1024,
                dropout=0.0,
                activation="relu",
                num_denoising=100,
                label_noise_ratio=0.5,
                box_noise_scale=1.0,
                learnt_init_query=False
            )
        else:
            self.decoder = decoder
            
        # 编码器在RT-DETR中是集成在decoder中的
        self.encoder = encoder
        self.multi_scale = multi_scale
        
    def execute(self, x, targets=None):
        """前向传播 - 严格按照PyTorch版本"""
        # 多尺度训练（如果启用）
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = jt.nn.interpolate(x, size=[sz, sz])
            
        # 骨干网络特征提取
        x = self.backbone(x)
        
        # 编码器处理（如果有独立的编码器）
        if self.encoder is not None:
            x = self.encoder(x)
            
        # 解码器处理
        x = self.decoder(x, targets)

        return x
    
    def deploy(self):
        """部署模式"""
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self


def build_rtdetr(num_classes=80, **kwargs):
    """
    构建RT-DETR模型的工厂函数
    严格按照PyTorch版本的配置
    """
    # 默认配置
    default_config = {
        'hidden_dim': 256,
        'num_queries': 300,
        'feat_channels': [512, 1024, 2048],
        'feat_strides': [8, 16, 32],
        'num_levels': 3,
        'num_decoder_points': 4,
        'nhead': 8,
        'num_decoder_layers': 6,
        'dim_feedforward': 1024,
        'dropout': 0.0,
        'activation': "relu",
        'num_denoising': 100,
        'label_noise_ratio': 0.5,
        'box_noise_scale': 1.0,
        'learnt_init_query': False
    }
    
    # 更新配置
    default_config.update(kwargs)
    default_config['num_classes'] = num_classes
    
    # 创建组件
    backbone = ResNet50()
    decoder = RTDETRTransformer(**default_config)
    
    # 创建模型
    model = RTDETR(
        num_classes=num_classes,
        backbone=backbone,
        encoder=None,  # RT-DETR没有独立的编码器
        decoder=decoder,
        multi_scale=None
    )
    
    return model


class RTDETRPostProcessor(nn.Module):
    """
    RT-DETR后处理器 - 严格按照PyTorch版本实现
    参考: rtdetr_pytorch/src/zoo/rtdetr/rtdetr_postprocessor.py
    """
    
    def __init__(self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False):
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.remap_mscoco_category = remap_mscoco_category 
        self.deploy_mode = False 

    def execute(self, outputs, orig_target_sizes):
        """后处理"""
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']

        # 转换边界框格式
        bbox_pred = self.box_cxcywh_to_xyxy(boxes)
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = jt.sigmoid(logits)
            scores, index = jt.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            
            # 收集对应的边界框
            batch_size = bbox_pred.shape[0]
            boxes = []
            for i in range(batch_size):
                boxes.append(bbox_pred[i][index[i]])
            boxes = jt.stack(boxes)
            
        else:
            scores = jt.nn.softmax(logits, dim=-1)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            boxes = bbox_pred

        # 过滤低置信度检测
        valid_mask = scores > 0.3
        
        results = []
        for i in range(len(scores)):
            valid_indices = jt.where(valid_mask[i])[0]
            if len(valid_indices) > 0:
                result = {
                    'scores': scores[i][valid_indices],
                    'labels': labels[i][valid_indices],
                    'boxes': boxes[i][valid_indices]
                }
            else:
                result = {
                    'scores': jt.array([]),
                    'labels': jt.array([]),
                    'boxes': jt.array([]).reshape(0, 4)
                }
            results.append(result)

        return results

    @staticmethod
    def box_cxcywh_to_xyxy(x):
        """将中心点格式转换为左上右下格式"""
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return jt.stack(b, dim=-1)


# 为了兼容性，保持原有的接口
class RTDETRModel(RTDETR):
    """兼容性包装器"""
    
    def __init__(self, num_classes=80, **kwargs):
        # 创建默认的RT-DETR配置
        backbone = ResNet50()
        decoder = RTDETRTransformer(
            num_classes=num_classes,
            hidden_dim=256,
            num_queries=300,
            feat_channels=[512, 1024, 2048],
            feat_strides=[8, 16, 32],
            num_levels=3,
            num_decoder_points=4,
            nhead=8,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.0,
            activation="relu",
            num_denoising=100,
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False
        )
        
        super().__init__(
            num_classes=num_classes,
            backbone=backbone,
            encoder=None,
            decoder=decoder,
            multi_scale=None
        )


def create_rtdetr_model(num_classes=80, **kwargs):
    """创建RT-DETR模型的便捷函数"""
    return build_rtdetr(num_classes=num_classes, **kwargs)


# 导出的类和函数
__all__ = [
    'RTDETR', 
    'RTDETRModel', 
    'RTDETRPostProcessor',
    'build_rtdetr', 
    'create_rtdetr_model'
]
