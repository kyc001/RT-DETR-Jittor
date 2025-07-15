
import jittor as jt
from jittor import init
from jittor import nn
import math

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = ([hidden_dim] * (num_layers - 1))
        self.layers = nn.ModuleList((nn.Linear(n, k) for (n, k) in zip(([input_dim] + h), (h + [output_dim]))))
        self.act = (nn.ReLU() if (act == 'relu') else nn.GELU())

    def execute(self, x):
        for (i, layer) in enumerate(self.layers):
            x = (self.act(layer(x)) if (i < (self.num_layers - 1)) else layer(x))
        return x

def bias_init_with_prob(prior_prob):
    bias_init = float((- math.log(((1 - prior_prob) / prior_prob))))
    return bias_init

def inverse_sigmoid(x, eps=1e-05):
    x = x.clamp(0, 1)
    x1 = 
    raise RuntimeError('origin source: <x.clamp(min=eps)>, There are needed 3 args in Pytorch clamp function, but you only provide 1')
    x2 = 
    raise RuntimeError('origin source: <(1 - x).clamp(min=eps)>, There are needed 3 args in Pytorch clamp function, but you only provide 1')
    return torch.log((x1 / x2))

class RTDETRTransformer(nn.Module):

    def __init__(self, num_classes=80, hidden_dim=256, num_queries=300, position_embed_type='sine', feat_channels=[512, 1024, 2048], feat_strides=[8, 16, 32], num_levels=3, num_decoder_points=4, nhead=8, num_decoder_layers=6, dim_feedforward=1024, dropout=0.0, activation='relu', num_denoising=100, label_noise_ratio=0.5, box_noise_scale=1.0, learnt_init_query=False):
        super(RTDETRTransformer, self).__init__()
        assert (position_embed_type in ['sine', 'learned']), f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert (len(feat_channels) <= num_levels)
        assert (len(feat_strides) <= num_levels)
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = 0.01
        self.num_decoder_layers = num_decoder_layers
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self._build_input_proj_layer(feat_channels)
        decoder_layer = DeformableDetrTransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points)
        self.decoder = DeformableDetrTransformerDecoder(hidden_dim, decoder_layer, num_decoder_layers, eval_idx=(- 1))
        self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim)
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, (2 * hidden_dim), hidden_dim, num_layers=2)
        self.enc_output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        self.dec_score_head = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_decoder_layers)])
        self.dec_bbox_head = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(num_decoder_layers)])
        self._reset_parameters()

    def _reset_parameters(self):
        bias_cls = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, value=bias_cls)
        init.constant_(self.enc_bbox_head.layers[(- 1)].weight, value=0.0)
        init.constant_(self.enc_bbox_head.layers[(- 1)].bias, value=0.0)
        for (cls_, reg_) in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, value=bias_cls)
            init.constant_(reg_.layers[(- 1)].weight, value=0.0)
            init.constant_(reg_.layers[(- 1)].bias, value=0.0)
        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(nn.Sequential(nn.Conv(in_channels, self.hidden_dim, 1, bias=False), nn.GroupNorm(32, self.hidden_dim, affine=None)))
        in_channels = feat_channels[(- 1)]
        for i in range((self.num_levels - len(feat_channels))):
            self.input_proj.append(nn.Sequential(nn.Conv(in_channels, self.hidden_dim, 3, stride=2, padding=1, bias=False), nn.GroupNorm(32, self.hidden_dim, affine=None)))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        proj_feats = [self.input_proj[i](feat) for (i, feat) in enumerate(feats)]
        if (self.num_levels > len(proj_feats)):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if (i == len_srcs):
                    proj_feats.append(self.input_proj[i](feats[(- 1)]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[(- 1)]))
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0]
        for (i, feat) in enumerate(proj_feats):
            (_, _, h, w) = feat.shape
            feat_flatten.append(feat.flatten(start_dim=2).transpose(1, 2))
            spatial_shapes.append([h, w])
            level_start_index.append(((h * w) + level_start_index[(- 1)]))
        feat_flatten = jt.contrib.concat(feat_flatten, dim=1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _get_decoder_input(self, memory, spatial_shapes, denoising_class=None, denoising_bbox_unact=None):
        (bs, _, _) = memory.shape
        (anchors, valid_mask) = self._generate_anchors(spatial_shapes)
        memory = (valid_mask.to(memory.dtype) * memory)
        output_memory = self.enc_output(memory)
        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = (self.enc_bbox_head(output_memory) + anchors)
        (_, topk_ind) = torch.topk(enc_outputs_class.max((- 1)).values, self.num_queries, dim=1)
        reference_points_unact = enc_outputs_coord_unact.gather(dim=1, index=topk_ind.unsqueeze((- 1)).repeat(1, 1, enc_outputs_coord_unact.shape[(- 1)]))
        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if (denoising_bbox_unact is not None):
            reference_points_unact = jt.contrib.concat([denoising_bbox_unact, reference_points_unact], dim=1)
        enc_topk_logits = enc_outputs_class.gather(dim=1, index=topk_ind.unsqueeze((- 1)).repeat(1, 1, enc_outputs_class.shape[(- 1)]))
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        else:
            target = output_memory.gather(dim=1, index=topk_ind.unsqueeze((- 1)).repeat(1, 1, output_memory.shape[(- 1)]))
            target = target.detach()
        if (denoising_class is not None):
            target = jt.contrib.concat([denoising_class, target], dim=1)
        return (target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits)

    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device='cpu'):
        if (spatial_shapes is None):
            spatial_shapes = [[int((self.eval_spatial_sizes[i][0] / s)), int((self.eval_spatial_sizes[i][1] / s))] for (i, s) in enumerate(self.feat_strides)]
        anchors = []
        for (lvl, (h, w)) in enumerate(spatial_shapes):
            (grid_y, grid_x) = torch.meshgrid(torch.arange(h, dtype=dtype), torch.arange(w, dtype=dtype))
            grid_xy = torch.stack([grid_x, grid_y], (- 1))
            valid_WH = torch.tensor([w, h]).to(dtype)
            grid_xy = ((grid_xy.unsqueeze(0) + 0.5) / valid_WH)
            wh = ((torch.ones_like(grid_xy) * grid_size) * (2.0 ** lvl))
            anchors.append(torch.cat([grid_xy, wh], (- 1)).view(((- 1), (h * w), 4)))
        anchors = jt.contrib.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < (1 - self.eps))).all((- 1), keepdim=True)
        anchors = torch.log((anchors / (1 - anchors)))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return (anchors, valid_mask)

    def execute(self, feats, targets=None):
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)
        if (self.training and (self.num_denoising > 0)):
            (denoising_class, denoising_bbox_unact, attn_mask, dn_meta) = get_contrastive_denoising_training_group(targets, self.num_classes, self.num_queries, self.denoising_class_embed, num_denoising=self.num_denoising, label_noise_ratio=self.label_noise_ratio, box_noise_scale=self.box_noise_scale)
        else:
            (denoising_class, denoising_bbox_unact, attn_mask, dn_meta) = (None, None, None, None)
        (target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits) = self._get_decoder_input(memory, spatial_shapes, denoising_class, denoising_bbox_unact)
        (out_bboxes, out_logits) = self.decoder(target, init_ref_points_unact, memory, spatial_shapes, level_start_index, self.dec_bbox_head, self.dec_score_head, self.query_pos_head, attn_mask=attn_mask)
        if (self.training and (dn_meta is not None)):
            (dn_out_bboxes, out_bboxes) = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            (dn_out_logits, out_logits) = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
        out = {'pred_logits': out_logits[(- 1)], 'pred_boxes': out_bboxes[(- 1)]}
        if (self.training and self.aux_loss):
            out['aux_outputs'] = self._set_aux_loss(out_logits[:(- 1)], out_bboxes[:(- 1)])
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))
            if (self.training and (dn_meta is not None)):
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta
        return out

    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for (a, b) in zip(outputs_class, outputs_coord)]
