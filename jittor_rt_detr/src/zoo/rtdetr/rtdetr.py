"""RT-DETR main model implementation for Jittor
Aligned with PyTorch version structure
"""

import jittor as jt
import jittor.nn as nn
import numpy as np

__all__ = ['RTDETR']


class RTDETR(nn.Module):
    """RT-DETR main model class"""

    def __init__(self, backbone, encoder, decoder, multi_scale=None):
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

        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x

    def deploy(self):
        """Deploy mode"""
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
