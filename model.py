# model.py

import jittor as jt
import jittor.nn as nn

# ======================================================================
# 1. ResNet Backbone Implementation
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

        # 返回三个不同尺度的特征图
        return [c3, c4, c5]


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

# ======================================================================
# 2. Transformer Components (with Auxiliary Loss support)
# ======================================================================

# MultiHeadSelfAttention (no changes needed)


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

# TransformerEncoderLayer and TransformerEncoder (no changes needed)


class TransformerEncoderLayer(nn.Module):
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
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, depth):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(
            embed_dim, num_heads) for _ in range(depth)])

    def execute(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# TransformerDecoderLayer (no changes needed)


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

# ## <<< 关键修改：Decoder 返回所有中间层结果 >>>


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
        # (depth, batch, num_queries, embed_dim)
        return jt.stack(output, dim=0)

# DetectionHead (no changes needed, but will be applied multiple times)


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
# 3. Main RT-DETR Model (Upgraded)
# ======================================================================


class RTDETR(nn.Module):
    def __init__(self, num_classes=81, num_queries=300, embed_dim=256, num_heads=8, enc_depth=1, dec_depth=6):
        super().__init__()
        # ## <<< 关键修改：使用 ResNet50 作为骨干网络 >>>
        self.backbone = ResNet50()

        # ## <<< 关键修改：为多尺度特征创建输入投射层 >>>
        self.input_proj = nn.ModuleList([
            nn.Conv(512, embed_dim, 1),  # from C3
            nn.Conv(1024, embed_dim, 1),  # from C4
            nn.Conv(2048, embed_dim, 1),  # from C5
        ])

        self.encoder = TransformerEncoder(embed_dim, num_heads, enc_depth)
        self.decoder = TransformerDecoder(embed_dim, num_heads, dec_depth)
        self.query_embed = nn.Parameter(jt.randn(num_queries, embed_dim))

        # ## <<< 关键修改：预测头不再包含 num_queries >>>
        self.head = DetectionHead(embed_dim, num_classes)

    def execute(self, x):
        # x: (batch, 3, H, W)

        # ## <<< 关键修改：处理多尺度特征 >>>
        features = self.backbone(x)
        projections = [proj(feat)
                       for proj, feat in zip(self.input_proj, features)]

        # 此处简化，只使用最后一层特征 (C5) 送入 Encoder
        # 完整的 RT-DETR 会有更复杂的特征融合模块
        feat = projections[-1]

        B, C, H, W = feat.shape
        feat = feat.reshape(B, C, -1).transpose(0, 2,
                                                1)  # (batch, HW, embed_dim)

        memory = self.encoder(feat)

        query_embed = jt.unsqueeze(self.query_embed, 0).expand(B, -1, -1)

        # ## <<< 关键修改：hs 现在是 (depth, batch, num_queries, embed_dim) >>>
        hs = self.decoder(query_embed, memory)

        # ## <<< 关键修改：对 Decoder 的每一层输出都应用预测头 >>>
        logits, boxes = self.head(hs)

        return logits, boxes
