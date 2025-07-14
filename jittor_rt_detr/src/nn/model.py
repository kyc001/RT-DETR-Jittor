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


class MLP(nn.Module):
    """多层感知机 (参考PyTorch版本)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        layers = []
        for n, k in zip([input_dim] + h, h + [output_dim]):
            layers.append(nn.Linear(n, k))
        self.layers = nn.ModuleList(layers)

    def execute(self, x):
        for i, layer in enumerate(self.layers):
            x = jt.nn.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


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
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.eps = 1e-2

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

        # 5. 编码器输出处理 (参考PyTorch版本)
        self.enc_output = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.enc_score_head = nn.Linear(embed_dim, num_classes)
        self.enc_bbox_head = MLP(embed_dim, embed_dim, 4, num_layers=3)

        # 6. 解码器各层的预测头 (参考PyTorch版本)
        self.dec_score_head = nn.ModuleList([
            nn.Linear(embed_dim, num_classes)
            for _ in range(dec_depth)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(embed_dim, embed_dim, 4, num_layers=3)
            for _ in range(dec_depth)
        ])

        # 7. 查询位置编码头
        self.query_pos_head = MLP(4, 2 * embed_dim, embed_dim, num_layers=2)

        self._reset_parameters()

    def _reset_parameters(self):
        """参数初始化 (参考PyTorch版本)"""
        # 初始化偏置
        bias_value = -2.19  # bias_init_with_prob(0.01) ≈ -2.19

        # 编码器头初始化
        jt.init.constant_(self.enc_score_head.bias, bias_value)
        jt.init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        jt.init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        # 解码器头初始化
        for cls_head, reg_head in zip(self.dec_score_head, self.dec_bbox_head):
            jt.init.constant_(cls_head.bias, bias_value)
            jt.init.constant_(reg_head.layers[-1].weight, 0)
            jt.init.constant_(reg_head.layers[-1].bias, 0)

    def _generate_anchors(self, spatial_shapes, grid_size=0.05):
        """生成锚点 (参考PyTorch版本)"""
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            # 创建网格
            grid_y = jt.arange(h, dtype=jt.float32)
            grid_x = jt.arange(w, dtype=jt.float32)
            grid_y, grid_x = jt.meshgrid(grid_y, grid_x)
            grid_xy = jt.stack([grid_x, grid_y], dim=-1)

            # 归一化坐标
            valid_WH = jt.array([w, h], dtype=jt.float32)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH

            # 设置宽高
            wh = jt.ones_like(grid_xy) * grid_size * (2.0 ** lvl)

            # 组合为锚点 [x, y, w, h]
            anchor = jt.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(anchor)

        anchors = jt.concat(anchors, dim=1)

        # 创建有效掩码
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(dim=-1).unsqueeze(-1)

        # 应用logit变换
        anchors = jt.log(anchors / (1 - anchors))
        anchors = jt.where(valid_mask, anchors, float('inf'))

        return anchors, valid_mask

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

        # 5. 生成锚点 (参考PyTorch版本)
        spatial_shapes = []
        for feat in enc_outputs:
            _, _, H, W = feat.shape
            spatial_shapes.append((H, W))

        anchors, valid_mask = self._generate_anchors(spatial_shapes)
        B = x.shape[0]
        anchors = anchors.expand(B, -1, -1)  # (B, N_all, 4)
        valid_mask = valid_mask.expand(B, -1, -1)  # (B, N_all, 1)

        # 6. 应用有效掩码到memory
        memory = valid_mask.to(memory.dtype) * memory

        # 7. 编码器输出处理 (参考PyTorch版本)
        output_memory = self.enc_output(memory)
        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        # 8. Top-K查询选择 (参考PyTorch版本)
        _, topk_ind = jt.topk(enc_outputs_class.max(dim=-1), self.num_queries, dim=1)

        # 收集top-k的特征和坐标
        batch_indices = jt.arange(B).unsqueeze(1).expand(-1, self.num_queries)
        topk_coords = enc_outputs_coord_unact[batch_indices, topk_ind]  # (B, num_queries, 4)
        topk_memory = output_memory[batch_indices, topk_ind]  # (B, num_queries, C)

        # 编码器的top-k预测
        enc_topk_logits = enc_outputs_class[batch_indices, topk_ind]  # (B, num_queries, num_classes)
        enc_topk_bboxes = jt.sigmoid(topk_coords)  # (B, num_queries, 4)

        # ------------------ 解码器部分 ------------------
        # 9. 准备解码器输入
        # 查询位置编码
        query_pos = self.query_pos_head(topk_coords)  # (B, num_queries, embed_dim)
        target = topk_memory + query_pos  # (B, num_queries, embed_dim)

        # 10. 解码器前向传播
        # hs: (depth, B, num_queries, embed_dim)
        hs = self.decoder(target, memory)

        # 11. 对解码器各层输出进行预测
        all_pred_logits = []
        all_pred_boxes = []

        for i, hidden_state in enumerate(hs):
            pred_logits = self.dec_score_head[i](hidden_state)
            pred_boxes_delta = self.dec_bbox_head[i](hidden_state)

            # 相对于参考点的增量预测
            pred_boxes = jt.sigmoid(pred_boxes_delta + topk_coords)

            all_pred_logits.append(pred_logits)
            all_pred_boxes.append(pred_boxes)

        # 12. 堆叠所有层的输出
        all_pred_logits = jt.stack(all_pred_logits, dim=0)  # (depth, B, num_queries, num_classes)
        all_pred_boxes = jt.stack(all_pred_boxes, dim=0)    # (depth, B, num_queries, 4)

        return all_pred_logits, all_pred_boxes, enc_topk_logits, enc_topk_bboxes
