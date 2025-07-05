# model.py

import jittor as jt
import jittor.nn as nn

# ======================================================================
# 1. ResNet Backbone Implementation (与原版一致，无需修改)
# ======================================================================


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv(planes, planes, kernel_size=3,
                             stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(planes)
        self.conv3 = nn.Conv(planes, planes * self.expansion,
                             kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def execute(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv(3, 64, kernel_size=7,
                             stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.Pool(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def execute(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # 返回三个不同尺度的特征图，用于后续的混合编码器
        return [c3, c4, c5]


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

# ======================================================================
# 2. Transformer Components (与原版一致，无需修改)
# ======================================================================


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def execute(self, query, key=None, value=None):
        if key is None:
            key = value = query
        B, N, C = query.shape
        Bk, Nk, Ck = key.shape
        q = self.q_proj(query).reshape(B, N, self.num_heads,
                                       self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(key).reshape(Bk, Nk, self.num_heads,
                                     self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(value if value is not None else key).reshape(
            Bk, Nk, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = nn.softmax(attn, dim=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)
        return out


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)), nn.ReLU(),
            nn.Linear(int(embed_dim * mlp_ratio),
                      embed_dim), nn.Dropout(dropout)
        )

    def execute(self, tgt, memory):
        tgt = tgt + self.self_attn(self.norm1(tgt))
        tgt = tgt + self.cross_attn(self.norm2(tgt), memory, memory)
        tgt = tgt + self.mlp(self.norm3(tgt))
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, depth):
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(
            embed_dim, num_heads) for _ in range(depth)])

    def execute(self, tgt, memory):
        output = []
        for layer in self.layers:
            tgt = layer(tgt, memory)
            output.append(tgt)
        return jt.stack(output, dim=0)


class DetectionHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, 4), nn.Sigmoid()
        )

    def execute(self, x):
        logits = self.class_embed(x)
        boxes = self.bbox_embed(x)
        return logits, boxes


# ======================================================================
# 3. ## <<< 新增模块：实现与论文对齐的混合编码器 >>>
# ======================================================================

class AIFI(nn.Module):
    """
    Attention-based Intra-scale Feature Interaction (AIFI) module.
    论文中用于处理最高层（C5）特征，进行尺度内信息交互。
    本质上是一个标准的 Transformer Encoder Layer。
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)), nn.ReLU(),
            nn.Linear(int(embed_dim * mlp_ratio),
                      embed_dim), nn.Dropout(dropout)
        )

    def execute(self, x):
        """ x: (B, H*W, C) """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CCFM(nn.Module):
    """
    CNN-based Cross-scale Feature Fusion (CCFM) module.
    论文中用于融合不同尺度的特征图。
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm(out_channels),
            nn.ReLU()
        )

    def execute(self, x1, x2):
        """ 融合两个特征图 x1 (高层) 和 x2 (低层) """
        x1_upsampled = nn.interpolate(x1, size=x2.shape[2:], mode='bilinear')
        x = jt.concat([x1_upsampled, x2], dim=1)
        return self.conv(x)


class HybridEncoder(nn.Module):
    """
    高效混合编码器 (Efficient Hybrid Encoder)。
    集成了 AIFI 和 CCFM，用于处理多尺度特征。
    """

    def __init__(self, in_channels, embed_dim, num_heads):
        super().__init__()
        # AIFI 模块仅作用于最高层特征 (C5)
        self.aifi = AIFI(embed_dim, num_heads)

        # CCFM 模块用于跨尺度融合
        self.ccfm1 = CCFM(embed_dim * 2, embed_dim)  # Fusion of C5 and C4
        self.ccfm2 = CCFM(embed_dim * 2, embed_dim)  # Fusion of C4' and C3

    def execute(self, features):
        """ features: [c3, c4, c5], 经过 1x1 卷积投影后的特征列表 """
        c3, c4, c5 = features

        # 1. AIFI on C5 (尺度内交互)
        B, C, H5, W5 = c5.shape
        c5_flat = c5.reshape(B, C, -1).transpose(0, 2, 1)  # (B, H5*W5, C)
        c5_aifi = self.aifi(c5_flat).transpose(0, 2, 1).reshape(B, C, H5, W5)

        # 2. CCFM: C5 -> C4 (跨尺度融合 1)
        fused_c4 = self.ccfm1(c5_aifi, c4)

        # 3. CCFM: C4' -> C3 (跨尺度融合 2)
        fused_c3 = self.ccfm2(fused_c4, c3)

        return [fused_c3, fused_c4, c5_aifi]


# ======================================================================
# 4. ## <<< 修改主模型：集成混合编码器和查询选择机制 >>>
# ======================================================================

class RTDETR(nn.Module):
    def __init__(self, num_classes=81, num_queries=300, embed_dim=256, num_heads=8, dec_depth=6):
        super().__init__()
        self.num_queries = num_queries

        # 1. 骨干网络 (Backbone)
        self.backbone = ResNet50()

        # 2. 为多尺度特征创建输入投射层 (Input Projection)
        self.input_proj = nn.ModuleList([
            nn.Conv(512, embed_dim, 1),   # from C3 (ResNet layer2 output)
            nn.Conv(1024, embed_dim, 1),  # from C4 (ResNet layer3 output)
            nn.Conv(2048, embed_dim, 1),  # from C5 (ResNet layer4 output)
        ])

        # 3. 高效混合编码器 (Hybrid Encoder)
        self.encoder = HybridEncoder(
            in_channels=[embed_dim]*3, embed_dim=embed_dim, num_heads=num_heads)

        # 4. Transformer 解码器 (Decoder)
        self.decoder = TransformerDecoder(embed_dim, num_heads, dec_depth)

        # 5. ## <<< 关键修改：IoU 感知查询选择 >>>
        # 移除固定的 self.query_embed
        # 增加一个用于编码器输出的预测头，以选择 Top-K 查询
        self.enc_output_head = DetectionHead(embed_dim, num_classes)

        # 6. 解码器各层共享的最终预测头
        self.dec_output_head = DetectionHead(embed_dim, num_classes)

    def execute(self, x):
        # ------------------ 编码器部分 ------------------
        # 1. 骨干网络提取多尺度特征
        features = self.backbone(x)  # [c3, c4, c5]

        # 2. 对特征进行 1x1 卷积投影
        projections = [proj(feat)
                       for proj, feat in zip(self.input_proj, features)]

        # 3. 特征通过混合编码器进行融合
        # enc_outputs: [[B,C,H3,W3], [B,C,H4,W4], [B,C,H5,W5]]
        enc_outputs = self.encoder(projections)

        # 4. 准备送入解码器的数据
        # 将编码器输出的多尺度特征展平并拼接，作为解码器的 memory
        enc_feats_flat = []
        for feat in enc_outputs:
            B, C, H, W = feat.shape
            enc_feats_flat.append(feat.reshape(B, C, -1))

        # memory: (B, N_all, C)
        memory = jt.concat(enc_feats_flat, dim=2).transpose(0, 2, 1)

        # 5. ## <<< 关键修改：执行 IoU 感知查询选择 >>>
        # (B, N_all, C) -> (B, N_all, num_classes)
        enc_logits, _ = self.enc_output_head(memory)

        # 使用最大分类得分作为选择标准
        # scores: (B, N_all)
        scores = enc_logits.max(dim=-1)

        # 在所有 batch 中，每个 batch 独立选取 top-k
        # topk_indices: (B, num_queries)
        _, topk_indices = jt.topk(scores, self.num_queries, dim=1)

        # 从 memory 中根据索引选出 top-k 特征作为查询
        # query: (B, num_queries, C)
        batch_indices = jt.arange(x.shape[0]).unsqueeze(1)
        query = memory[batch_indices, topk_indices]

        # ------------------ 解码器部分 ------------------
        # 6. 将选出的查询和 memory 送入解码器
        # hs: (depth, B, num_queries, embed_dim)
        hs = self.decoder(query, memory)

        # 7. 对解码器每一层的输出都应用预测头
        logits, boxes = self.dec_output_head(hs)

        return logits, boxes
