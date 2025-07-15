"""
RT-DETR Jittor版本训练配置文件
"""

# 数据配置
DATA_CONFIG = {
    'data_dir': "data/coco2017_50/train2017",
    'ann_file': "data/coco2017_50/annotations/instances_train2017.json",
    'target_size': 640,
    'max_images': 50,  # 设置为None使用全部图片
    'batch_size': 2,
}

# 模型配置
MODEL_CONFIG = {
    'hidden_dim': 256,
    'num_queries': 300,
}

# 训练配置
TRAINING_CONFIG = {
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'print_freq': 10,  # 每10个batch打印一次
}

# 损失权重配置
LOSS_WEIGHTS = {
    'loss_focal': 2,
    'loss_bbox': 5, 
    'loss_giou': 2,
    # 辅助损失权重会自动生成
}

# Focal Loss配置
FOCAL_LOSS_CONFIG = {
    'alpha': 0.25,
    'gamma': 2.0,
}

# 保存配置
SAVE_CONFIG = {
    'model_save_path': "checkpoints/rtdetr_jittor.pkl",
    'history_save_path': "results/training_history.png",
    'log_save_path': "results/training.log",
}

# Jittor配置
JITTOR_CONFIG = {
    'use_cuda': True,
    'seed': 42,
    'auto_mixed_precision_level': 0,  # 强制float32
}
