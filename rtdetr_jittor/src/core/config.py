"""Configuration system for Jittor RT-DETR"""

import os
from typing import Dict, Any

class Config:
    """Configuration class"""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        self.config_dict = config_dict or {}
    
    def __getitem__(self, key):
        return self.config_dict[key]
    
    def __setitem__(self, key, value):
        self.config_dict[key] = value
    
    def get(self, key, default=None):
        return self.config_dict.get(key, default)

def load_config(config_path: str = None) -> Config:
    """Load configuration - simplified version without YAML"""
    if config_path and os.path.exists(config_path):
        # 简化版本，返回默认配置
        default_config = {
            'num_classes': 80,
            'hidden_dim': 256,
            'num_queries': 300,
            'lr': 1e-4,
            'epochs': 50
        }
        return Config(default_config)
    else:
        return Config()
