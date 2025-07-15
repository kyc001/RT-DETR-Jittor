
import jittor as jt
from jittor import init
from jittor import nn
import math

class MSDeformableAttention(nn.Module):

    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = ((num_heads * num_levels) * num_points)
        self.head_dim = (embed_dim // num_heads)
        assert ((self.head_dim * num_heads) == self.embed_dim)
        self.sampling_offsets = nn.Linear(embed_dim, (self.total_points * 2))
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        init.constant_(self.sampling_offsets.weight, value=0)
        thetas = (torch.arange(self.num_heads, dtype=torch.float32) * ((2.0 * math.pi) / self.num_heads))
        grid_init = torch.stack([thetas.cos(), thetas.sin()], (- 1))
        grid_init = (grid_init / grid_init.abs().max((- 1), keepdim=True)[0])
        grid_init = grid_init.view((self.num_heads, 1, 1, 2)).repeat(1, self.num_levels, self.num_points, 1)
        scaling = torch.arange(1, (self.num_points + 1), dtype=torch.float32).view((1, 1, (- 1), 1))
        grid_init *= scaling
        self.sampling_offsets.bias.data = grid_init.view((- 1))
        init.constant_(self.attention_weights.weight, value=0)
        init.constant_(self.attention_weights.bias, value=0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, value=0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, value=0)

    def execute(self, query, reference_points, value, value_spatial_shapes, value_mask=None):
        '\n        Args:\n            query (Tensor): [bs, query_length, C]\n            reference_points (Tensor): [bs, query_length, n_levels, 2]\n            value (Tensor): [bs, value_length, C]\n            value_spatial_shapes (List): [n_levels, 2]\n        '
        (bs, num_queries, _) = query.shape
        (bs, num_value, _) = value.shape
        value = self.value_proj(value)
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view((bs, num_queries, self.num_heads, self.num_levels, self.num_points, 2))
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view((bs, num_queries, self.num_heads, (self.num_levels * self.num_points)))
        attention_weights = F.softmax(attention_weights, (- 1))
        attention_weights = attention_weights.view((bs, num_queries, self.num_heads, self.num_levels, self.num_points))
        offset_normalizer = torch.stack([torch.tensor(s) for s in value_spatial_shapes], 0)
        offset_normalizer = offset_normalizer.to(sampling_offsets.device).float()
        sampling_offsets = (sampling_offsets / offset_normalizer[None, None, None, :, None, :])
        value = value.view((bs, num_value, self.num_heads, self.head_dim))
        query_reshaped = query.view((bs, num_queries, self.num_heads, self.head_dim))
        attn_scores = (torch.matmul(query_reshaped, value.transpose((- 2), (- 1))) / math.sqrt(self.head_dim))
        offset_influence = sampling_offsets.mean(dim=(3, 4, 5))
        attn_scores = (attn_scores + (offset_influence.unsqueeze((- 1)) * 0.1))
        weight_influence = attention_weights.mean(dim=(3, 4))
        attn_scores = (attn_scores * (1 + (weight_influence.unsqueeze((- 1)) * 0.1)))
        attn_weights = F.softmax(attn_scores, dim=(- 1))
        output = torch.matmul(attn_weights, value)
        output = output.view((bs, num_queries, self.embed_dim))
        output = self.output_proj(output)
        return output
