"""
RT-DETR模型 - 严格按照PyTorch版本实现
参考: rtdetr_pytorch/src/zoo/rtdetr/
"""

import math
import jittor as jt
import jittor.nn as nn
from .ms_deformable_attention import MSDeformableAttention


class MultiheadAttention(nn.Module):
    """简单的多头注意力实现"""

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def execute(self, query, key, value, attn_mask=None):
        batch_size, seq_len, embed_dim = query.shape

        # 投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 重塑为多头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力
        scores = jt.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float('-inf'))

        attn_weights = jt.nn.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力
        attn_output = jt.matmul(attn_weights, v)

        # 重塑回原始形状
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # 输出投影
        output = self.out_proj(attn_output)

        return output, attn_weights


def bias_init_with_prob(prior_prob):
    """计算偏置初始化值"""
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init


def inverse_sigmoid(x, eps=1e-5):
    """逆sigmoid函数"""
    x = jt.clamp(x, eps, 1 - eps)
    x1 = x
    x2 = 1 - x
    return jt.log(x1 / x2)


class MLP(nn.Module):
    """多层感知机"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        layers = []
        for n, k in zip([input_dim] + h, h + [output_dim]):
            layers.append(nn.Linear(n, k))
        self.layers = nn.ModuleList(layers)
        self.act = nn.ReLU() if act == 'relu' else nn.GELU()

    def execute(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DeformableDetrTransformerDecoderLayer(nn.Module):
    """可变形DETR Transformer解码器层"""
    
    def __init__(self, d_model=256, n_head=8, dim_feedforward=1024, dropout=0.0, 
                 activation="relu", n_levels=4, n_points=4):
        super().__init__()

        # 交叉注意力
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 自注意力
        self.self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def execute(self, tgt, query_pos, reference_points, src, src_spatial_shapes, 
                level_start_index, src_padding_mask=None, attn_mask=None):
        # 自注意力
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2, _ = self.self_attn(q, k, tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 交叉注意力
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                              reference_points,
                              src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 前馈网络
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableDetrTransformerDecoder(nn.Module):
    """可变形DETR Transformer解码器"""
    
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx

    def execute(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index,
                bbox_head, score_head, query_pos_head, attn_mask=None):
        output = tgt
        intermediate = []
        intermediate_reference_points = []
        
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                       * jt.concat([src_spatial_shapes, src_spatial_shapes], -1)[None, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_spatial_shapes[None, None]
            
            query_pos = query_pos_head(reference_points_input.flatten(-2))

            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes,
                          src_level_start_index, attn_mask=attn_mask)

            # 更新参考点
            if bbox_head is not None:
                tmp = bbox_head[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = jt.sigmoid(new_reference_points)
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = jt.sigmoid(new_reference_points)
                reference_points = new_reference_points.detach()

            intermediate.append(output)
            intermediate_reference_points.append(reference_points)

        if bbox_head is not None:
            return jt.stack(intermediate), jt.stack(intermediate_reference_points)
        else:
            return jt.stack(intermediate), reference_points


class RTDETRTransformer(nn.Module):
    """
    RT-DETR Transformer - 严格按照PyTorch版本实现
    """
    
    def __init__(self, num_classes=80, hidden_dim=256, num_queries=300,
                 feat_channels=[512, 1024, 2048], feat_strides=[8, 16, 32], 
                 num_levels=3, num_decoder_points=4, nhead=8, num_decoder_layers=6, 
                 dim_feedforward=1024, dropout=0.0, activation="relu",
                 num_denoising=100, label_noise_ratio=0.5, box_noise_scale=1.0, 
                 learnt_init_query=False):
        super(RTDETRTransformer, self).__init__()
        
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) <= num_levels

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = 1e-2

        self.num_decoder_layers = num_decoder_layers
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # backbone特征投影
        self._build_input_proj_layer(feat_channels)

        # Transformer模块
        decoder_layer = DeformableDetrTransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points)
        self.decoder = DeformableDetrTransformerDecoder(hidden_dim, decoder_layer, num_decoder_layers, eval_idx=-1)

        # 去噪部分
        self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim)
        
        # 解码器嵌入
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # 编码器头
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim)
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # 解码器头
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(num_decoder_layers)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        """参数初始化"""
        # 类别和边界框头初始化
        bias_cls = bias_init_with_prob(0.01)
        
        # 编码器头
        jt.init.constant_(self.enc_score_head.bias, bias_cls)
        jt.init.constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        jt.init.constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        
        # 解码器头
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            jt.init.constant_(cls_.bias, bias_cls)
            jt.init.constant_(reg_.layers[-1].weight, 0.)
            jt.init.constant_(reg_.layers[-1].bias, 0.)

        # 其他层初始化
        jt.init.xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            jt.init.xavier_uniform_(self.tgt_embed.weight)
        jt.init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        jt.init.xavier_uniform_(self.query_pos_head.layers[1].weight)

    def _build_input_proj_layer(self, feat_channels):
        """构建输入投影层"""
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(32, self.hidden_dim)
                )
            )
        
        # 为额外层添加投影
        in_channels = feat_channels[-1]
        for i in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(32, self.hidden_dim)
                )
            )
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        """获取编码器输入"""
        # 获取投影特征
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # 获取编码器输入
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose(1, 2))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], 每层的起始索引
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = jt.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _get_decoder_input(self, memory, spatial_shapes, denoising_class=None, denoising_bbox_unact=None):
        """获取解码器输入"""
        bs, _, _ = memory.shape

        # 准备解码器输入
        anchors, valid_mask = self._generate_anchors(spatial_shapes)
        memory = valid_mask.to(memory.dtype) * memory

        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = jt.topk(enc_outputs_class.max(-1), self.num_queries, dim=1)

        # 收集top-k特征
        batch_indices = jt.arange(bs).unsqueeze(1).expand(-1, self.num_queries)
        reference_points_unact = enc_outputs_coord_unact[batch_indices, topk_ind]

        enc_topk_bboxes = jt.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = jt.concat([denoising_bbox_unact, reference_points_unact], 1)

        enc_topk_logits = enc_outputs_class[batch_indices, topk_ind]

        # 提取区域特征
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        else:
            target = output_memory[batch_indices, topk_ind]
            target = target.detach()

        if denoising_class is not None:
            target = jt.concat([denoising_class, target], 1)

        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits

    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05):
        """生成锚点"""
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = jt.meshgrid(jt.arange(h, dtype=jt.float32), jt.arange(w, dtype=jt.float32))
            grid_xy = jt.stack([grid_x, grid_y], -1)
            valid_WH = jt.array([w, h], dtype=jt.float32)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = jt.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchors.append(jt.concat([grid_xy, wh], -1).view(-1, h * w, 4))

        anchors = jt.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1).unsqueeze(-1)
        anchors = jt.log(anchors / (1 - anchors))
        anchors = jt.where(valid_mask, anchors, float('inf'))
        return anchors, valid_mask

    def execute(self, feats, targets=None):
        """前向传播"""
        # 输入投影和嵌入
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)

        # 准备去噪训练（简化版本，不包含完整的去噪逻辑）
        denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(memory, spatial_shapes, denoising_class, denoising_bbox_unact)

        # 解码器
        out_bboxes, out_logits = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            jt.array(spatial_shapes),
            jt.array(level_start_index),
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask)

        # 应用预测头
        outputs_class = []
        outputs_coord = []
        for lvl in range(out_logits.shape[0]):
            outputs_class.append(self.dec_score_head[lvl](out_logits[lvl]))
            outputs_coord.append(jt.sigmoid(self.dec_bbox_head[lvl](out_bboxes[lvl])))

        outputs_class = jt.stack(outputs_class)
        outputs_coord = jt.stack(outputs_coord)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        # 辅助输出
        if self.training:
            out['aux_outputs'] = self._set_aux_loss(outputs_class[:-1], outputs_coord[:-1])
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))

        return out

    def _set_aux_loss(self, outputs_class, outputs_coord):
        """设置辅助损失"""
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class, outputs_coord)]
