import jittor as jt
import jittor.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm(out_channels)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv(in_channels, out_channels, 1, stride)
        else:
            self.downsample = None

    def execute(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.act(out)
        return out


class CSPDarkNetTiny(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv(in_channels, 32, 3, 1, 1),
            nn.BatchNorm(32),
            nn.ReLU()
        )
        self.layer1 = BasicBlock(32, 64, stride=2)
        self.layer2 = BasicBlock(64, 128, stride=2)
        self.layer3 = BasicBlock(128, 256, stride=2)
        self.layer4 = BasicBlock(256, 512, stride=2)

    def execute(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


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
        # 支持自注意力和交叉注意力
        if key is None:
            key = value = query
        B, N, C = query.shape
        Bk, Nk, Ck = key.shape
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3)  # (B, num_heads, N, head_dim)
        k = self.k_proj(key).reshape(Bk, Nk, self.num_heads,
                                     self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(value if value is not None else key).reshape(
            Bk, Nk, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = nn.softmax(attn, dim=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)
        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def execute(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, depth):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads) for _ in range(depth)
        ])

    def execute(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def execute(self, tgt, memory):
        tgt = tgt + self.self_attn(self.norm1(tgt))
        tgt = tgt + self.cross_attn(self.norm2(tgt), memory, memory)
        tgt = tgt + self.mlp(self.norm3(tgt))
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, depth):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads) for _ in range(depth)
        ])

    def execute(self, tgt, memory):
        for layer in self.layers:
            tgt = layer(tgt, memory)
        return tgt


class DetectionHead(nn.Module):
    def __init__(self, embed_dim, num_classes, num_queries):
        super().__init__()
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 4),
            nn.Sigmoid()  # 归一化到0-1
        )
        self.num_queries = num_queries

    def execute(self, x):
        # x: (batch, num_queries, embed_dim)
        logits = self.class_embed(x)  # (batch, num_queries, num_classes)
        boxes = self.bbox_embed(x)    # (batch, num_queries, 4)
        return logits, boxes


class RTDETR(nn.Module):
    def __init__(self, num_classes=80, num_queries=100, embed_dim=256, num_heads=8, enc_depth=6, dec_depth=6):
        super().__init__()
        self.backbone = CSPDarkNetTiny()
        self.input_proj = nn.Conv(512, embed_dim, 1)
        self.encoder = TransformerEncoder(embed_dim, num_heads, enc_depth)
        self.decoder = TransformerDecoder(embed_dim, num_heads, dec_depth)
        self.query_embed = nn.Parameter(jt.randn(num_queries, embed_dim))
        self.head = DetectionHead(embed_dim, num_classes, num_queries)

    def execute(self, x):
        # x: (batch, 3, H, W)
        feat = self.backbone(x)  # (batch, 512, H/16, W/16)
        feat = self.input_proj(feat)  # (batch, embed_dim, H/16, W/16)
        B, C, H, W = feat.shape
        feat = feat.reshape(B, C, -1).transpose(0, 2,
                                                1)  # (batch, HW, embed_dim)
        memory = self.encoder(feat)   # (batch, HW, embed_dim)
        query_embed = jt.unsqueeze(self.query_embed, 0).expand(
            B, -1, -1)  # (batch, num_queries, embed_dim)
        # (batch, num_queries, embed_dim)
        hs = self.decoder(query_embed, memory)
        logits, boxes = self.head(hs)
        return logits, boxes

# 后续将集成 Detection Head
