"""
数据类型安全的RT-DETR实现
完全独立实现，不修改原始源码，彻底解决float32/float64混合问题
"""

import jittor as jt
import jittor.nn as nn
import numpy as np
import math


def safe_float32(tensor):
    """安全地将任何tensor转换为float32"""
    if isinstance(tensor, jt.Var):
        return tensor.float32()
    elif isinstance(tensor, np.ndarray):
        return jt.array(tensor.astype(np.float32), dtype=jt.float32)
    else:
        return jt.array(tensor, dtype=jt.float32)


class DTypeSafeLinear(nn.Module):
    """数据类型安全的线性层"""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 强制权重为float32
        self.weight = jt.randn(out_features, in_features, dtype=jt.float32) * math.sqrt(2.0 / in_features)
        if bias:
            self.bias = jt.zeros(out_features, dtype=jt.float32)
        else:
            self.bias = None
    
    def execute(self, x):
        # 强制输入为float32
        x = safe_float32(x)
        
        # 矩阵乘法
        output = jt.matmul(x, self.weight.transpose(0, 1))
        
        if self.bias is not None:
            output = output + self.bias
        
        return safe_float32(output)


class DTypeSafeConv2d(nn.Module):
    """数据类型安全的卷积层"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 强制权重为float32
        fan_in = in_channels * kernel_size * kernel_size
        self.weight = jt.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=jt.float32) * math.sqrt(2.0 / fan_in)
        
        if bias:
            self.bias = jt.zeros(out_channels, dtype=jt.float32)
        else:
            self.bias = None
    
    def execute(self, x):
        # 强制输入为float32
        x = safe_float32(x)
        
        # 卷积操作
        output = jt.nn.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)
        
        return safe_float32(output)


class DTypeSafeMultiHeadAttention(nn.Module):
    """数据类型安全的多头注意力"""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = DTypeSafeLinear(d_model, d_model)
        self.k_proj = DTypeSafeLinear(d_model, d_model)
        self.v_proj = DTypeSafeLinear(d_model, d_model)
        self.out_proj = DTypeSafeLinear(d_model, d_model)
        
        self.scale = safe_float32(1.0 / math.sqrt(self.head_dim))
    
    def execute(self, query, key, value):
        batch_size, seq_len, _ = query.shape
        
        # 投影并重塑
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力计算
        scores = jt.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = jt.softmax(scores, dim=-1)
        attn_output = jt.matmul(attn_weights, v)
        
        # 重塑并输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)
        
        return safe_float32(output)


class DTypeSafeTransformerLayer(nn.Module):
    """数据类型安全的Transformer层"""
    
    def __init__(self, d_model, num_heads, dim_feedforward=2048):
        super().__init__()
        self.self_attn = DTypeSafeMultiHeadAttention(d_model, num_heads)
        self.linear1 = DTypeSafeLinear(d_model, dim_feedforward)
        self.linear2 = DTypeSafeLinear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def execute(self, x):
        # 自注意力
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.linear2(jt.relu(self.linear1(x)))
        x = self.norm2(x + self.dropout(ff_output))
        
        return safe_float32(x)


class DTypeSafeBackbone(nn.Module):
    """数据类型安全的骨干网络（简化版ResNet）"""
    
    def __init__(self):
        super().__init__()
        # 简化的特征提取网络
        self.conv1 = DTypeSafeConv2d(3, 64, 7, stride=2, padding=3)
        self.conv2 = DTypeSafeConv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = DTypeSafeConv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = DTypeSafeConv2d(256, 512, 3, stride=2, padding=1)
        
        self.pool = nn.AdaptiveAvgPool2d((20, 20))  # 固定输出尺寸
    
    def execute(self, x):
        # 强制输入为float32
        x = safe_float32(x)
        
        # 特征提取
        x = jt.relu(self.conv1(x))
        x = jt.relu(self.conv2(x))
        x = jt.relu(self.conv3(x))
        x = jt.relu(self.conv4(x))
        
        # 池化到固定尺寸
        x = self.pool(x)
        
        return safe_float32(x)


class DTypeSafeRTDETR(nn.Module):
    """数据类型安全的RT-DETR模型"""
    
    def __init__(self, num_classes=80, hidden_dim=256, num_queries=300, num_decoder_layers=6):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        
        # 骨干网络
        self.backbone = DTypeSafeBackbone()
        
        # 特征投影
        self.input_proj = DTypeSafeConv2d(512, hidden_dim, 1)
        
        # 查询嵌入
        self.query_embed = jt.randn(num_queries, hidden_dim, dtype=jt.float32) * 0.1
        
        # Transformer解码器
        self.decoder_layers = nn.ModuleList([
            DTypeSafeTransformerLayer(hidden_dim, 8, hidden_dim * 4)
            for _ in range(num_decoder_layers)
        ])
        
        # 预测头
        self.class_head = DTypeSafeLinear(hidden_dim, num_classes)
        self.bbox_head = DTypeSafeLinear(hidden_dim, 4)
        
        # 辅助预测头（用于深度监督）
        self.aux_class_heads = nn.ModuleList([
            DTypeSafeLinear(hidden_dim, num_classes) 
            for _ in range(num_decoder_layers - 1)
        ])
        self.aux_bbox_heads = nn.ModuleList([
            DTypeSafeLinear(hidden_dim, 4) 
            for _ in range(num_decoder_layers - 1)
        ])
    
    def execute(self, x, targets=None):
        # 强制输入为float32
        x = safe_float32(x)
        batch_size = x.shape[0]
        
        # 特征提取
        features = self.backbone(x)  # [bs, 512, 20, 20]
        
        # 特征投影
        proj_features = self.input_proj(features)  # [bs, hidden_dim, 20, 20]
        
        # 展平特征
        bs, c, h, w = proj_features.shape
        src = proj_features.flatten(2).transpose(1, 2)  # [bs, h*w, hidden_dim]
        src = safe_float32(src)
        
        # 查询嵌入
        query_embed = self.query_embed.unsqueeze(0).repeat(batch_size, 1, 1)  # [bs, num_queries, hidden_dim]
        query_embed = safe_float32(query_embed)
        
        # Transformer解码器
        decoder_output = query_embed
        aux_outputs = []
        
        for i, layer in enumerate(self.decoder_layers):
            decoder_output = layer(decoder_output)
            decoder_output = safe_float32(decoder_output)
            
            # 辅助输出
            if i < len(self.decoder_layers) - 1 and self.training:
                aux_logits = self.aux_class_heads[i](decoder_output)
                aux_boxes = jt.sigmoid(self.aux_bbox_heads[i](decoder_output))
                aux_outputs.append({
                    'pred_logits': safe_float32(aux_logits),
                    'pred_boxes': safe_float32(aux_boxes)
                })
        
        # 最终预测
        pred_logits = self.class_head(decoder_output)
        pred_boxes = jt.sigmoid(self.bbox_head(decoder_output))
        
        outputs = {
            'pred_logits': safe_float32(pred_logits),
            'pred_boxes': safe_float32(pred_boxes)
        }
        
        if self.training and aux_outputs:
            outputs['aux_outputs'] = aux_outputs
        
        return outputs


def build_dtype_safe_rtdetr(num_classes=80, hidden_dim=256, num_queries=300):
    """构建数据类型安全的RT-DETR模型"""
    model = DTypeSafeRTDETR(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        num_decoder_layers=6
    )
    
    # 强制所有参数为float32
    def init_weights(m):
        if hasattr(m, 'weight') and m.weight is not None:
            if m.weight.dtype != jt.float32:
                m.weight.data = m.weight.data.float32()
        if hasattr(m, 'bias') and m.bias is not None:
            if m.bias.dtype != jt.float32:
                m.bias.data = m.bias.data.float32()
    
    model.apply(init_weights)
    
    return model
