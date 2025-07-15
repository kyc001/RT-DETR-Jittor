
"""
多尺度可变形注意力模块 - 严格按照PyTorch版本实现
参考: rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py
"""

import math
import jittor as jt
import jittor.nn as nn


def deformable_attention_core_func(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights):
    """
    多尺度可变形注意力的核心函数
    这是一个简化的Jittor实现，替代PyTorch版本的CUDA核心函数
    """
    N, S, M, D = value.shape
    _, Lq, M, L, P, _ = sampling_locations.shape

    # 初始化输出
    output = jt.zeros((N, Lq, M * D), dtype=value.dtype)

    for level in range(L):
        H, W = value_spatial_shapes[level]
        start_idx = value_level_start_index[level]
        end_idx = start_idx + H * W

        level_value = value[:, start_idx:end_idx].view(N, H, W, M, D)
        level_sampling_loc = sampling_locations[:, :, :, level, :, :]  # N, Lq, M, P, 2
        level_attention_weight = attention_weights[:, :, :, level, :]  # N, Lq, M, P

        # 对每个采样点进行处理
        for p in range(P):
            loc = level_sampling_loc[:, :, :, p, :]  # N, Lq, M, 2
            weight = level_attention_weight[:, :, :, p]  # N, Lq, M

            # 将归一化坐标转换为像素坐标
            x = loc[:, :, :, 0] * W
            y = loc[:, :, :, 1] * H

            # 使用Jittor的clamp函数
            x = jt.clamp(jt.round(x).int64(), 0, W-1)
            y = jt.clamp(jt.round(y).int64(), 0, H-1)

            # 采样值 - 使用简单的最近邻插值
            # 创建索引
            batch_idx = jt.arange(N).unsqueeze(1).unsqueeze(1).expand(N, Lq, M)

            # 采样
            sampled = level_value[batch_idx, y, x]  # N, Lq, M, D

            # 加权累加
            weighted_sampled = sampled * weight.unsqueeze(-1)  # N, Lq, M, D
            output += weighted_sampled.view(N, Lq, M * D)

    return output.view(N, Lq, M * D)


class MSDeformableAttention(nn.Module):
    """
    多尺度可变形注意力模块
    严格按照PyTorch版本实现
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
        jt.init.constant_(self.sampling_offsets.weight, 0.)

        # 创建初始网格
        thetas = jt.arange(self.num_heads, dtype=jt.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = jt.stack([jt.cos(thetas), jt.sin(thetas)], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdims=True)[0]).view(
            self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        # 设置偏置
        self.sampling_offsets.bias = jt.array(grid_init.view(-1))

        # attention_weights初始化
        jt.init.constant_(self.attention_weights.weight, 0.)
        jt.init.constant_(self.attention_weights.bias, 0.)

        # 投影层初始化
        jt.init.xavier_uniform_(self.value_proj.weight)
        jt.init.constant_(self.value_proj.bias, 0.)
        jt.init.xavier_uniform_(self.output_proj.weight)
        jt.init.constant_(self.output_proj.bias, 0.)

    def execute(self, query, reference_points, value, value_spatial_shapes, value_level_start_index, value_mask=None):
        """
        前向传播

        Args:
            query: [N, Len_q, C] 查询特征
            reference_points: [N, Len_q, num_levels, 2] 参考点坐标
            value: [N, Len_v, C] 值特征
            value_spatial_shapes: [num_levels, 2] 每层的空间形状
            value_level_start_index: [num_levels] 每层的起始索引
            value_mask: [N, Len_v] 值的掩码
        """
        N, Len_q, _ = query.shape
        N, Len_v, _ = value.shape
        assert (value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1]).sum() == Len_v

        # 值投影
        value = self.value_proj(value)
        if value_mask is not None:
            value = value * (1 - value_mask.unsqueeze(-1).float())
        value = value.view(N, Len_v, self.num_heads, self.head_dim)

        # 计算采样偏移和注意力权重
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.num_heads, self.num_levels * self.num_points)
        attention_weights = jt.nn.softmax(attention_weights, -1).view(
            N, Len_q, self.num_heads, self.num_levels, self.num_points)

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


class MSDeformableAttnTransformerEncoderLayer(nn.Module):
    """
    多尺度可变形注意力Transformer编码器层
    """

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu", n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # 自注意力
        self.self_attn = MSDeformableAttention(d_model, n_heads, n_levels, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def execute(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # 自注意力
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 前馈网络
        src = self.forward_ffn(src)

        return src


class MSDeformableAttnTransformerEncoder(nn.Module):
    """
    多尺度可变形注意力Transformer编码器
    """

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = jt.meshgrid(jt.linspace(0.5, H_ - 0.5, H_, dtype=jt.float32),
                                       jt.linspace(0.5, W_ - 0.5, W_, dtype=jt.float32))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = jt.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = jt.concat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def execute(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output
