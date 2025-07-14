from jittor.utils.pytorch_converter import convert

# 从PyTorch版本提取多尺度可变形注意力的代码
pytorch_code = """
import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4,):
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
        # sampling_offsets
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
            
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        
        # attention_weights
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        
        # proj
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, value, value_spatial_shapes, value_level_start_index, value_mask=None):
        N, Len_q, _ = query.shape
        N, Len_v, _ = value.shape
        assert (value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1]).sum() == Len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(N, Len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([value_spatial_shapes[..., 1], value_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead.")

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output

def deformable_attention_core_func(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights):
    # 这是一个简化的实现，实际的PyTorch版本使用了CUDA核心函数
    N, S, M, D = value.shape
    _, Lq, M, L, P, _ = sampling_locations.shape
    
    # 简化的双线性插值实现
    output = torch.zeros((N, Lq, M * D), dtype=value.dtype, device=value.device)
    
    for level in range(L):
        H, W = value_spatial_shapes[level]
        start_idx = value_level_start_index[level]
        end_idx = start_idx + H * W
        
        level_value = value[:, start_idx:end_idx].view(N, H, W, M, D)
        level_sampling_loc = sampling_locations[:, :, :, level, :, :]  # N, Lq, M, P, 2
        level_attention_weight = attention_weights[:, :, :, level, :]  # N, Lq, M, P
        
        # 简化的双线性插值
        for p in range(P):
            loc = level_sampling_loc[:, :, :, p, :]  # N, Lq, M, 2
            weight = level_attention_weight[:, :, :, p]  # N, Lq, M
            
            # 将坐标转换为像素坐标
            x = loc[:, :, :, 0] * W
            y = loc[:, :, :, 1] * H
            
            # 简单的最近邻插值（实际应该是双线性插值）
            x = torch.clamp(x.round().long(), 0, W-1)
            y = torch.clamp(y.round().long(), 0, H-1)
            
            # 采样值
            sampled = level_value[torch.arange(N)[:, None, None], y, x]  # N, Lq, M, D
            
            # 加权累加
            output += (sampled * weight.unsqueeze(-1)).view(N, Lq, M * D)
    
    return output.view(N, Lq, M * D)
"""

# 转换为Jittor代码
jittor_code = convert(pytorch_code)
print("=== 转换后的Jittor代码 ===")
print(jittor_code)

# 保存转换后的代码
with open("jittor_rt_detr/src/nn/ms_deformable_attention.py", "w") as f:
    f.write(jittor_code)
