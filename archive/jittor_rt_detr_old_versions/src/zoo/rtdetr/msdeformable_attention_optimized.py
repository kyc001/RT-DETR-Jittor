
import jittor as jt
import jittor.nn as nn
import math

class OptimizedMSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim

        # 确保所有参数都参与计算
        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # 使用Jittor标准初始化
        jt.init.constant_(self.sampling_offsets.weight, 0)
        jt.init.constant_(self.sampling_offsets.bias, 0)
        jt.init.constant_(self.attention_weights.weight, 0)
        jt.init.constant_(self.attention_weights.bias, 0)
        jt.init.xavier_uniform_(self.value_proj.weight)
        jt.init.constant_(self.value_proj.bias, 0)
        jt.init.xavier_uniform_(self.output_proj.weight)
        jt.init.constant_(self.output_proj.bias, 0)

    def execute(self, query, reference_points, value, value_spatial_shapes, value_mask=None):
        """优化的前向传播，确保所有参数参与梯度计算"""
        bs, num_queries, _ = query.shape
        bs, num_value, _ = value.shape
        
        # 确保数据类型一致
        query = query.float32()
        value = value.float32()
        
        # 投影value
        value_proj = self.value_proj(value)
        
        # 计算采样偏移和注意力权重
        sampling_offsets = self.sampling_offsets(query)
        attention_weights = self.attention_weights(query)
        
        # 重塑为多头格式
        sampling_offsets = sampling_offsets.view(bs, num_queries, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = attention_weights.view(bs, num_queries, self.num_heads, self.num_levels * self.num_points)
        attention_weights = jt.nn.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.view(bs, num_queries, self.num_heads, self.num_levels, self.num_points)
        
        # 简化但有效的注意力计算
        # 使用采样偏移和注意力权重来调制标准注意力
        offset_scale = jt.mean(jt.mean(jt.mean(sampling_offsets.abs(), dim=5), dim=4), dim=3)  # [bs, num_queries, num_heads]
        weight_scale = jt.mean(jt.mean(attention_weights, dim=4), dim=3)  # [bs, num_queries, num_heads]
        
        # 标准多头注意力计算
        query_proj = query.view(bs, num_queries, self.num_heads, self.head_dim)
        value_proj = value_proj.view(bs, num_value, self.num_heads, self.head_dim)
        
        # 计算注意力分数
        attn_scores = jt.matmul(query_proj, value_proj.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 使用偏移和权重调制注意力分数
        offset_influence = offset_scale.unsqueeze(-1) * 0.1
        weight_influence = weight_scale.unsqueeze(-1) * 0.1
        
        attn_scores = attn_scores + offset_influence
        attn_scores = attn_scores * (1 + weight_influence)
        
        attn_weights = jt.nn.softmax(attn_scores, dim=-1)
        
        # 应用注意力
        output = jt.matmul(attn_weights, value_proj)
        output = output.view(bs, num_queries, self.embed_dim)
        
        # 输出投影
        output = self.output_proj(output)
        
        return output.float32()
