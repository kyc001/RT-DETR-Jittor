"""Build functions for RT-DETR models"""

import jittor as jt
from .rtdetr import RTDETR
from .rtdetr_decoder import build_rtdetr_complete
from ...nn.backbone import ResNet50
from ...nn.criterion import build_criterion

def build_rtdetr_model(num_classes=80, hidden_dim=256, num_queries=300):
    """Build complete RT-DETR model with standard components"""
    
    # Build backbone
    backbone = ResNet50()
    
    # Build complete model (includes encoder and decoder)
    model = build_rtdetr_complete(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_queries=num_queries
    )
    
    return model

def build_rtdetr_criterion(num_classes=80):
    """Build RT-DETR criterion (loss function)"""
    return build_criterion(num_classes)

__all__ = ['build_rtdetr_model', 'build_rtdetr_criterion']
