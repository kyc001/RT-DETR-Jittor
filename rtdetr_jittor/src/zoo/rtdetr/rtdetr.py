"""RT-DETR main model implementation for Jittor
Aligned with PyTorch version structure
"""

import jittor as jt
import jittor.nn as nn
import numpy as np

__all__ = ['RTDETR']


class RTDETR(nn.Module):
    """RT-DETR main model class - 与PyTorch版本对齐"""

    def __init__(self, backbone, encoder=None, decoder=None, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.multi_scale = multi_scale

    def execute(self, x, targets=None):
        """Forward pass - aligned with PyTorch version"""
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = nn.interpolate(x, size=[sz, sz])

        # Backbone前向传播
        feats = self.backbone(x)
        
        # 如果有编码器，使用编码器
        if self.encoder is not None:
            feats = self.encoder(feats)
        
        # 解码器前向传播
        if self.decoder is not None:
            outputs = self.decoder(feats, targets)
        else:
            outputs = feats

        return outputs

    def deploy(self):
        """Deploy mode"""
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self


def build_rtdetr(backbone_name='resnet50', num_classes=80, **kwargs):
    """构建RT-DETR模型 - 工厂函数"""
    from ...nn.backbone.resnet import ResNet50, PResNet
    from .rtdetr_decoder import RTDETRTransformer
    from .hybrid_encoder import HybridEncoder
    
    # 创建骨干网络
    if backbone_name == 'resnet50':
        backbone = ResNet50(pretrained=kwargs.get('pretrained', False))
        feat_channels = [256, 512, 1024, 2048]
    elif backbone_name == 'presnet50':
        backbone = PResNet(depth=50, pretrained=kwargs.get('pretrained', False))
        feat_channels = [256, 512, 1024, 2048]
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    # 创建编码器
    encoder = HybridEncoder(
        in_channels=feat_channels[-3:],  # 使用后3个特征图
        hidden_dim=kwargs.get('hidden_dim', 256),
        num_heads=kwargs.get('num_heads', 8),
        num_layers=kwargs.get('num_encoder_layers', 1)
    )
    
    # 创建解码器
    decoder = RTDETRTransformer(
        num_classes=num_classes,
        hidden_dim=kwargs.get('hidden_dim', 256),
        num_queries=kwargs.get('num_queries', 300),
        feat_channels=feat_channels
    )
    
    # 创建完整模型
    model = RTDETR(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        multi_scale=kwargs.get('multi_scale', None)
    )
    
    return model
