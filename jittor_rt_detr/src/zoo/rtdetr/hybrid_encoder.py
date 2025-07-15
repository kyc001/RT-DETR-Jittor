
"""
多尺度可变形注意力模块 - 严格按照PyTorch版本实现
参考: rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py
"""

import math
import jittor as jt
import jittor.nn as nn
from .utils import deformable_attention_core_func


class MSDeformableAttention(nn.Module):
    """
    多尺度可变形注意力模块 - 严格按照PyTorch版本实现
    """

    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4):
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = deformable_attention_core_func

        self._reset_parameters()

    def _reset_parameters(self):
        """参数初始化 - 严格按照PyTorch版本"""
        # sampling_offsets初始化
        jt.init.constant_(self.sampling_offsets.weight, 0)

        # 创建初始网格
        thetas = jt.arange(self.num_heads, dtype=jt.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = jt.stack([jt.cos(thetas), jt.sin(thetas)], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdims=True)
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)

        # 缩放因子
        scaling = jt.arange(1, self.num_points + 1, dtype=jt.float32).view(1, 1, -1, 1)
        grid_init *= scaling

        # 设置偏置
        with jt.no_grad():
            self.sampling_offsets.bias.assign(grid_init.flatten())

        # attention_weights初始化
        jt.init.constant_(self.attention_weights.weight, 0)
        jt.init.constant_(self.attention_weights.bias, 0)

        # 投影层初始化
        jt.init.xavier_uniform_(self.value_proj.weight)
        jt.init.constant_(self.value_proj.bias, 0)
        jt.init.xavier_uniform_(self.output_proj.weight)
        jt.init.constant_(self.output_proj.bias, 0)

    def execute(self, query, reference_points, value, value_spatial_shapes, value_level_start_index, value_mask=None):
        """
        前向传播 - 严格按照PyTorch版本

        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1]
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ...]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length]
        """
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1]).sum() == num_value

        # 值投影
        value = self.value_proj(value)
        if value_mask is not None:
            value = value * (1 - value_mask.unsqueeze(-1).float())
        value = value.view(bs, num_value, self.num_heads, self.head_dim)

        # 计算采样偏移和注意力权重
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = jt.nn.softmax(attention_weights, -1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points)

        # 计算采样位置
        if reference_points.shape[-1] == 2:
            offset_normalizer = jt.stack([value_spatial_shapes[..., 1], value_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + \
                sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + \
                sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead.")

        # 执行多尺度可变形注意力
        output = self.ms_deformable_attn_core(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)

        # 输出投影
        output = self.output_proj(output)

        return output
