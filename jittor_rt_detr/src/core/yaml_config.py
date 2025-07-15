"""YAML configuration system - simplified version"""

from .config import Config

class YAMLConfig(Config):
    """YAML-based configuration - simplified version without yaml dependency"""
    
    def __init__(self, config_path: str = None, **kwargs):
        # 默认配置
        config_dict = {
            'num_classes': 80,
            'hidden_dim': 256,
            'num_queries': 300,
            'lr': 1e-4,
            'epochs': 50,
            'batch_size': 2,
            'weight_decay': 1e-4
        }
        
        # 更新配置
        for key, value in kwargs.items():
            if value is not None:
                config_dict[key] = value
        
        super().__init__(config_dict)
        self.yaml_cfg = config_dict
