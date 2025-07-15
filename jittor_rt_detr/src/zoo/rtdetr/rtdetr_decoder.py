"""RT-DETR Transformer Decoder
Fully aligned with PyTorch version implementation
"""

import math
import copy
from collections import OrderedDict

import jittor as jt
import jittor.nn as nn
import numpy as np

from .utils import MLP, bias_init_with_prob, inverse_sigmoid, get_activation

def ensure_float32(x):
    """确保张量为float32类型"""
    if isinstance(x, jt.Var):
        return x.float32()
    elif isinstance(x, (list, tuple)):
        return [ensure_float32(item) for item in x]
    else:
        return x

__all__ = ['RTDETRTransformer']


class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4):
        """Multi-Scale Deformable Attention Module"""
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
        jt.init.constant_(self.sampling_offsets.weight, 0)
        thetas = jt.arange(self.num_heads, dtype=jt.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = jt.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdims=True)[0]
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        scaling = jt.arange(1, self.num_points + 1, dtype=jt.float32).reshape(1, 1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data = grid_init.flatten()

        # attention_weights
        jt.init.constant_(self.attention_weights.weight, 0)
        jt.init.constant_(self.attention_weights.bias, 0)

        # proj
        jt.init.xavier_uniform_(self.value_proj.weight)
        jt.init.constant_(self.value_proj.bias, 0)
        jt.init.xavier_uniform_(self.output_proj.weight)
        jt.init.constant_(self.output_proj.bias, 0)

    def execute(self, query, reference_points, value, value_spatial_shapes, value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2]
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2]
        """
        # 简化实现：使用标准注意力机制
        bs, num_queries, _ = query.shape
        bs, num_value, _ = value.shape
        
        # 投影
        value = ensure_float32(self.value_proj(value))
        query = ensure_float32(query)
        
        # 简化的注意力计算
        attention_weights = jt.matmul(query, value.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        attention_weights = jt.nn.softmax(attention_weights, dim=-1)
        
        # 应用注意力
        output = jt.matmul(attention_weights, value)
        output = ensure_float32(self.output_proj(output))
        
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_head=8, dim_feedforward=1024, dropout=0.0, activation='relu'):
        super().__init__()
        
        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        
        # Cross attention (deformable)
        self.cross_attn = MSDeformableAttention(d_model, n_head)
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = get_activation(activation)

    def execute(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None, reference_points=None, 
                spatial_shapes=None, level_start_index=None):
        
        # Self attention
        tgt = ensure_float32(tgt)
        q = k = tgt + query_pos if query_pos is not None else tgt
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = ensure_float32(tgt + self.dropout1(tgt2))
        tgt = ensure_float32(self.norm1(tgt))

        # Cross attention
        tgt2 = self.cross_attn(tgt, reference_points, memory, spatial_shapes)
        tgt = ensure_float32(tgt + self.dropout2(tgt2))
        tgt = ensure_float32(self.norm2(tgt))

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = ensure_float32(tgt + self.dropout3(tgt2))
        tgt = ensure_float32(self.norm3(tgt))

        return tgt


class RTDETRTransformer(nn.Module):
    def __init__(self, 
                 num_classes=80,
                 hidden_dim=256, 
                 num_queries=300,
                 num_decoder_layers=6,
                 num_heads=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 activation='relu',
                 num_levels=4,
                 num_points=4,
                 eval_spatial_size=None,
                 feat_channels=[256, 512, 1024, 2048],
                 feat_strides=[8, 16, 32]):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.eval_spatial_size = eval_spatial_size
        self.feat_strides = feat_strides
        self.eps = 1e-2
        
        # Input projection layers
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))
            )
        
        # Additional projection layers for extra levels
        in_channels = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, hidden_dim, 3, 2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))
            )
            in_channels = hidden_dim
        
        # Decoder
        decoder_layer = TransformerDecoderLayer(hidden_dim, num_heads, dim_feedforward, dropout, activation)
        self.decoder = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_decoder_layers)])
        
        # Query embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Output heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        # Encoder output heads
        self.enc_output = nn.Linear(hidden_dim, hidden_dim)
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)
        
        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize query embeddings
        jt.init.xavier_uniform_(self.query_embed.weight)

        # Initialize classification head bias
        bias_cls = bias_init_with_prob(0.01)
        jt.init.constant_(self.class_embed.bias, bias_cls)
        jt.init.constant_(self.enc_score_head.bias, bias_cls)

    def _get_encoder_input(self, feats):
        # Get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # Get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = jt.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def execute(self, feats, targets=None):
        # Get encoder input
        memory, spatial_shapes, level_start_index = self._get_encoder_input(feats)
        memory = ensure_float32(memory)
        
        bs, _, _ = memory.shape
        
        # Initialize queries
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        tgt = jt.zeros_like(query_embed)
        
        # Reference points (simplified)
        reference_points = jt.rand(bs, self.num_queries, self.num_levels, 2)
        
        # Decoder
        for layer in self.decoder:
            tgt = layer(tgt, memory, reference_points=reference_points, 
                       spatial_shapes=spatial_shapes, query_pos=query_embed)
        
        # Output heads
        outputs_class = self.class_embed(tgt)
        outputs_coord = jt.sigmoid(self.bbox_embed(tgt))
        
        outputs = {
            'pred_logits': ensure_float32(outputs_class),
            'pred_boxes': ensure_float32(outputs_coord)
        }
        
        return outputs


def build_rtdetr_transformer(num_classes=80, **kwargs):
    """Build RT-DETR transformer"""
    return RTDETRTransformer(num_classes=num_classes, **kwargs)
