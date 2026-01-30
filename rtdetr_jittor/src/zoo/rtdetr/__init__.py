"""RT-DETR model implementations
Fully aligned with PyTorch version structure
"""

# 安全导入，避免循环导入问题
try:
    from .rtdetr import RTDETR
except ImportError:
    RTDETR = None

try:
    from .rtdetr_decoder import RTDETRTransformer, MSDeformableAttention
except ImportError:
    RTDETRTransformer = None
    MSDeformableAttention = None

try:
    from .matcher import HungarianMatcher, build_matcher
except ImportError:
    HungarianMatcher = None
    build_matcher = None

try:
    from .hybrid_encoder import HybridEncoder
except ImportError:
    HybridEncoder = None

# 简化导入，避免复杂依赖
def build_criterion(num_classes=80):
    """构建损失函数"""
    try:
        from ..rtdetr.rtdetr_criterion import build_criterion as _build_criterion
        return _build_criterion(num_classes)
    except ImportError:
        return None

__all__ = [
    'RTDETR',
    'RTDETRTransformer',
    'MSDeformableAttention',
    'HungarianMatcher',
    'HybridEncoder',
    'build_criterion',
    'build_matcher'
]
