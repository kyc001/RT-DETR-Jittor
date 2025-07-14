from jittor.utils.pytorch_converter import convert

# 从PyTorch版本提取完整的MSDeformableAttention
pytorch_code = """
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class MSDeformableAttention(nn.Module):
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

        self._reset_parameters()

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def forward(self, query, reference_points, value, value_spatial_shapes, value_level_start_index, value_mask=None):
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, num_value, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).view(bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, -1).view(bs, num_query, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([value_spatial_shapes[..., 1], value_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead.")

        # 使用简化的可变形注意力核心函数
        output = self.deformable_attention_core_func(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output

    def deformable_attention_core_func(self, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights):
        bs, _, num_heads, head_dim = value.shape
        _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
        
        value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        
        for lid, (H_, W_) in enumerate(value_spatial_shapes):
            # bs, H_*W_, num_heads, head_dim -> bs, H_*W_, num_heads*head_dim -> bs, num_heads*head_dim, H_*W_ -> bs*num_heads, head_dim, H_, W_
            value_l_ = value_list[lid].flatten(2).transpose(1, 2).reshape(bs*num_heads, head_dim, H_, W_)
            # bs, num_queries, num_heads, num_points, 2 -> bs, num_heads, num_queries, num_points, 2 -> bs*num_heads, num_queries, num_points, 2
            sampling_grid_l_ = sampling_grids[:, :, :, lid].transpose(1, 2).flatten(0, 1)
            # bs*num_heads, head_dim, num_queries, num_points
            sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False)
            sampling_value_list.append(sampling_value_l_)
        
        # (bs, num_queries, num_heads, num_levels, num_points) -> (bs, num_heads, num_queries, num_levels, num_points) -> (bs*num_heads, 1, num_queries, num_levels*num_points)
        attention_weights = attention_weights.transpose(1, 2).reshape(bs*num_heads, 1, num_queries, num_levels*num_points)
        output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(bs, num_heads*head_dim, num_queries)
        return output.transpose(1, 2).contiguous()
"""

# 转换为Jittor代码
jittor_code = convert(pytorch_code)
print("=== 转换后的MSDeformableAttention Jittor代码 ===")
print(jittor_code)

# 保存转换后的代码
with open("jittor_rt_detr/src/nn/msdeformable_attention_pytorch_aligned.py", "w") as f:
    f.write(jittor_code)
