"""Hybrid Encoder for RT-DETR"""

import jittor as jt
import jittor.nn as nn
import math

def ensure_float32(x):
    """确保张量为float32类型"""
    if isinstance(x, jt.Var):
        return x.float32()
    else:
        return jt.array(x, dtype=jt.float32)

class MultiHeadAttention(nn.Module):
    """多头注意力机制 - Jittor兼容版本"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def execute(self, query, key=None, value=None, attn_mask=None):
        if key is None:
            key = query
        if value is None:
            value = query
            
        batch_size, seq_len, embed_dim = query.shape
        
        # 投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 重塑为多头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力
        scores = jt.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attn_mask is not None:
            scores = scores + attn_mask
        
        attn_weights = jt.nn.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        out = jt.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        return self.out_proj(out)

class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层 - Jittor兼容版本"""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def execute(self, src, src_mask=None):
        # 自注意力
        src2 = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 前馈网络
        src2 = self.linear2(self.dropout(jt.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

class HybridEncoder(nn.Module):
    """混合编码器 - Jittor兼容版本"""
    
    def __init__(self, embed_dim=256, num_heads=8, num_layers=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 编码器层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1
            ) for _ in range(num_layers)
        ])
        
        # 位置编码
        self.pos_embed = nn.Parameter(jt.randn(1000, embed_dim) * 0.02)
        
    def execute(self, src, pos_embed=None):
        """前向传播"""
        batch_size, seq_len, embed_dim = src.shape
        
        # 添加位置编码
        if pos_embed is None and seq_len <= 1000:
            pos_embed = self.pos_embed[:seq_len].unsqueeze(0)
            src = src + pos_embed
        
        # 通过编码器层
        output = src
        for layer in self.layers:
            output = layer(output)
        
        return ensure_float32(output)

# 为了兼容性，添加一些常用的函数
def build_hybrid_encoder(embed_dim=256, num_heads=8, num_layers=6):
    """构建混合编码器"""
    return HybridEncoder(embed_dim, num_heads, num_layers)
