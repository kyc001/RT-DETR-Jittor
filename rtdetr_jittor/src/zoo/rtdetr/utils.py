
"""
RT-DETR工具函数 - 严格按照PyTorch版本实现
参考: rtdetr_pytorch/src/zoo/rtdetr/utils.py
"""

import math
import jittor as jt
import jittor.nn as nn


def tile(x, repeats):
    """
    实现PyTorch风格的tile函数
    repeats: list of int，每个维度的重复次数
    """
    if isinstance(repeats, int):
        repeats = [repeats]

    # 确保repeats的长度与x的维度相同
    while len(repeats) < len(x.shape):
        repeats = [1] + list(repeats)

    result = x
    for i, rep in enumerate(repeats):
        if rep > 1:
            result = jt.concat([result] * rep, dim=i)
    return result


def inverse_sigmoid(x, eps=1e-5):
    """逆sigmoid函数 - 严格按照PyTorch版本"""
    x = jt.clamp(x, min_v=0., max_v=1.)
    x_clipped = jt.clamp(x, min_v=eps)
    one_minus_x_clipped = jt.clamp(1 - x, min_v=eps)
    return jt.log(x_clipped / one_minus_x_clipped)


def deformable_attention_core_func(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights):
    """
    可变形注意力核心函数 - 严格按照PyTorch版本实现

    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    # 分割value到不同层级
    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = jt.split(value, split_shape, dim=1)

    # 转换采样位置到网格坐标 [-1, 1]
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []

    for level, (h, w) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).view(bs * n_head, c, h, w)

        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).view(bs * n_head, Len_q, n_points, 2)

        sampling_value_l_ = nn.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)

    # 重新组织注意力权重
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).view(
        bs * n_head, 1, Len_q, n_levels * n_points)

    # 计算加权输出
    output = (jt.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(bs, n_head * c, Len_q)

    return output.transpose(1, 2)


def bias_init_with_prob(prior_prob=0.01):
    """根据先验概率初始化偏置值"""
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init


def get_activation(act, inplace=True):
    """获取激活函数"""
    if act is None:
        return nn.Identity()

    act = act.lower()

    if act == 'silu':
        m = nn.SiLU()
    elif act == 'relu':
        m = nn.ReLU()
    elif act == 'leaky_relu':
        m = nn.LeakyReLU()
    elif act == 'gelu':
        m = nn.GELU()
    elif isinstance(act, nn.Module):
        m = act
    else:
        raise RuntimeError(f'Unsupported activation: {act}')

    # Jittor的激活函数可能没有inplace参数
    if hasattr(m, 'inplace'):
        m.inplace = inplace

    return m


class MLP(nn.Module):
    """多层感知机 - 严格按照PyTorch版本实现"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)

        layers = []
        for n, k in zip([input_dim] + h, h + [output_dim]):
            layers.append(nn.Linear(n, k))
        self.layers = nn.ModuleList(layers)

        self.act = get_activation(act)

    def execute(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    """创建模块的N个副本"""
    return nn.ModuleList([module for _ in range(N)])


def _get_activation_fn(activation):
    """获取激活函数"""
    return get_activation(activation)


# 简化的grid_sample实现（如果Jittor版本不完整）
def simple_grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    """
    简化的grid_sample实现
    这是一个后备方案，如果Jittor的grid_sample不可用
    """
    try:
        return jt.nn.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    except:
        # 如果grid_sample不可用，使用简化的最近邻插值
        N, C, H, W = input.shape
        N_grid, H_grid, W_grid, _ = grid.shape

        # 将grid坐标从[-1,1]转换到[0,H-1]和[0,W-1]
        grid_x = (grid[:, :, :, 0] + 1) * (W - 1) / 2
        grid_y = (grid[:, :, :, 1] + 1) * (H - 1) / 2

        # 使用最近邻插值
        grid_x = jt.clamp(jt.round(grid_x).int64(), 0, W-1)
        grid_y = jt.clamp(jt.round(grid_y).int64(), 0, H-1)

        # 创建输出张量
        output = jt.zeros((N, C, H_grid, W_grid))

        for n in range(N):
            for h in range(H_grid):
                for w in range(W_grid):
                    x = grid_x[n, h, w]
                    y = grid_y[n, h, w]
                    output[n, :, h, w] = input[n, :, y, x]

        return output


# 导出的函数
__all__ = [
    'tile',
    'inverse_sigmoid',
    'deformable_attention_core_func',
    'bias_init_with_prob',
    'get_activation',
    'MLP',
    '_get_clones',
    '_get_activation_fn',
    'simple_grid_sample'
]
