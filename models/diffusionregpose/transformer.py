import math, random
import copy
from typing import Optional, List, Union
from util.misc import inverse_sigmoid
import torch
from torch import nn, Tensor
from .transformer_deformable import DeformableTransformerEncoderLayer, DeformableTransformerDecoderLayer
from .utils import gen_encoder_output_proposals, MLP, gen_sineembed_for_position
from .ops.modules.ms_deform_attn import MSDeformAttn
from detectron2.modeling.poolers import ROIPooler



def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Transformer(nn.Module):

    def __init__(self, d_model=256, nhead=8,
                 num_queries=300,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=4,
                 num_patterns=0,
                 modulate_hw_attn=False,
                 # for deformable encoder
                 deformable_encoder=False,
                 deformable_decoder=False,
                 num_feature_levels=1,
                 enc_n_points=4,
                 dec_n_points=4,
                 # init query
                 learnable_tgt_init=False,
                 random_refpoints_xy=False,
                 # two stage
                 two_stage_type='no',
                 two_stage_learn_wh=False,
                 two_stage_keep_all_tokens=False,
                 # evo of #anchors
                 dec_layer_number=None,
                 rm_self_attn_layers=None,
                 # for detach
                 rm_detach=None,
                 decoder_sa_type='sa',
                 module_seq=['sa', 'ca', 'ffn'],
                 # for pose
                 embed_init_tgt=False,
                 num_body_points=17,
                 num_box_decoder_layers=2,
                 num_group=100
                 ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.deformable_encoder = deformable_encoder
        self.deformable_decoder = deformable_decoder
        self.two_stage_keep_all_tokens = two_stage_keep_all_tokens
        self.num_queries = num_queries
        self.random_refpoints_xy = random_refpoints_xy
        # Build RoI.
        box_pooler = self._init_box_pooler()
        self.box_pooler = box_pooler
        assert query_dim == 4

        if num_feature_levels > 1:
            assert deformable_encoder, "only support deformable_encoder for num_feature_levels > 1"

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']

        # choose encoder layer type
        if deformable_encoder:
            encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                              dropout, activation,
                                                              num_feature_levels, nhead, enc_n_points)
        else:
            raise NotImplementedError
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers,
            encoder_norm, d_model=d_model,
            num_queries=num_queries,
            deformable_encoder=deformable_encoder,
            two_stage_type=two_stage_type
        )

        # choose decoder layer type
        if deformable_decoder:
            decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                              dropout, activation,
                                                              num_feature_levels, nhead, dec_n_points,
                                                              decoder_sa_type=decoder_sa_type,
                                                              module_seq=module_seq)

        else:
            raise NotImplementedError
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before,
                                                    num_feature_levels=num_feature_levels)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model, query_dim=query_dim,
                                          modulate_hw_attn=modulate_hw_attn,
                                          num_feature_levels=num_feature_levels,
                                          deformable_decoder=deformable_decoder,
                                          dec_layer_number=dec_layer_number,
                                          num_body_points=num_body_points,
                                          num_box_decoder_layers=num_box_decoder_layers,
                                          num_group=num_group
                                          )

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries  # useful for single stage model only
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0
        if self.num_patterns > 0:
            assert two_stage_type == 'no'
            self.patterns = nn.Embedding(self.num_patterns, d_model)
        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
            else:
                self.level_embed = None

        self.learnable_tgt_init = learnable_tgt_init
        assert learnable_tgt_init, "why not learnable_tgt_init"
        self.embed_init_tgt = embed_init_tgt
        if (two_stage_type != 'no' and embed_init_tgt) or (two_stage_type == 'no'):
            self.tgt_embed = nn.Embedding(self.num_queries, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.tgt_embed = None

        # for two stage
        self.two_stage_type = two_stage_type
        self.two_stage_learn_wh = two_stage_learn_wh
        assert two_stage_type in ['no', 'standard', 'early', 'combine', 'enceachlayer',
                                  'enclayer1'], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type in ['standard', 'combine', 'enceachlayer', 'enclayer1']:
            # anchor selection at the output of encoder
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)

            if two_stage_learn_wh:
                # import ipdb; ipdb.set_trace()
                self.two_stage_wh_embedding = nn.Embedding(1, 2)
            else:
                self.two_stage_wh_embedding = None

        if two_stage_type in ['early', 'combine']:
            # anchor selection at the output of backbone
            self.enc_output_backbone = nn.Linear(d_model, d_model)
            self.enc_output_norm_backbone = nn.LayerNorm(d_model)

        if two_stage_type == 'no':
            self.init_ref_points(num_queries)  # init self.refpoint_embed

        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None
        self.enc_out_pose_embed = None

        # evolution of anchors
        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            if self.two_stage_type != 'no' or num_patterns == 0:
                assert dec_layer_number[
                           0] == num_queries, f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries})"
            else:
                assert dec_layer_number[
                           0] == num_queries * num_patterns, f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries}) * num_patterns({num_patterns})"

        self._reset_parameters()

        self.d_model = d_model
        time_dim = d_model * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # block time mlp
        self.block_time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_model * 4, d_model * 1))

        self.rm_self_attn_layers = rm_self_attn_layers
        if rm_self_attn_layers is not None:
            # assert len(rm_self_attn_layers) == num_decoder_layers
            print("Removing the self-attn in {} decoder layers".format(rm_self_attn_layers))
            for lid, dec_layer in enumerate(self.decoder.layers):
                if lid in rm_self_attn_layers:
                    dec_layer.rm_self_attn_modules()

        self.rm_detach = rm_detach
        if self.rm_detach:
            assert isinstance(rm_detach, list)
            assert any([i in ['enc_ref', 'enc_tgt', 'dec'] for i in rm_detach])
        self.decoder.rm_detach = rm_detach

    @staticmethod
    def _init_box_pooler():

        # in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        in_features = ['p2', 'p3', 'p4', 'p5']
        # pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_resolution = 7
        # pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        pooler_scales = (0.125, 0.0625, 0.03125, 0.015625)
        # sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        sampling_ratio = 2
        # pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_type = 'ROIAlignV2'

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        # in_channels = [input_shape[f].channels for f in in_features]
        in_channels = [256, 256, 256, 256]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

        if self.two_stage_learn_wh:
            nn.init.constant_(self.two_stage_wh_embedding.weight, math.log(0.05 / (1 - 0.05)))

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)

        if self.random_refpoints_xy:
            # import ipdb; ipdb.set_trace()
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

    def forward(self, images_whwh, srcs, masks, x_boxes, refpoint_embed, t, pos_embeds, tgt, attn_mask=None,
                attn_mask2=None):
        # prepare input for encoder
        time_emb = self.time_mlp(t)  # tdy

        # fc_feature = fc_feature * (scale + 1) + shift

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)  # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c

            # # #tdy
            scale_shift = self.block_time_mlp(time_emb)
            scale_shift = torch.repeat_interleave(scale_shift.unsqueeze(1), h * w * 2, dim=1)
            scale, shift = scale_shift.chunk(2, dim=1)
            # src = src * (scale + 1) + shift
            pos_embed = pos_embed * (scale + 1) + shift

            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # two stage
        if self.two_stage_type in ['early', 'combine']:
            output_memory, output_proposals = gen_encoder_output_proposals(src_flatten, mask_flatten, spatial_shapes)
            output_memory = self.enc_output_norm_backbone(self.enc_output_backbone(output_memory))

            # gather boxes
            topk = self.num_queries
            enc_outputs_class = self.encoder.class_embed[0](output_memory)
            enc_topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]  # bs, nq
            enc_refpoint_embed = torch.gather(output_proposals, 1, enc_topk_proposals.unsqueeze(-1).repeat(1, 1, 4))

            src_flatten = output_memory
        else:
            enc_topk_proposals = enc_refpoint_embed = None

        #########################################################
        # Begin Encoder
        #########################################################
        # bs, nq = x_boxes.shape[:2]
        # ref_token_index = torch.randint(bs, nq)
        memory, enc_intermediate_output, enc_intermediate_refpoints = self.encoder(
            src_flatten,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            ref_token_index=enc_topk_proposals,  # bs, nq
            ref_token_coord=x_boxes,  # bs, nq, 4
            # ref_token_coord=enc_refpoint_embed, # bs, nq, 4
        )
        #########################################################
        # End Encoder
        # - memory: bs, \sum{hw}, c
        # - mask_flatten: bs, \sum{hw}
        # - lvl_pos_embed_flatten: bs, \sum{hw}, c
        # - enc_intermediate_output: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        # - enc_intermediate_refpoints: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        #########################################################

        if self.two_stage_type in ['standard', 'combine', 'enceachlayer', 'enclayer1']:
            if self.two_stage_learn_wh:
                # import ipdb; ipdb.set_trace()
                input_hw = self.two_stage_wh_embedding.weight[0]
            else:
                input_hw = None
            output_memory, output_proposals = gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes,
                                                                           input_hw)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))

            enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
            enc_outputs_coord_unselected = self.enc_out_bbox_embed(
                output_memory) + output_proposals  # (bs, \sum{hw}, 4) unsigmoid
            topk = self.num_queries
            topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]  # bs, nq

            # gather boxes
            refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1,
                                                   topk_proposals.unsqueeze(-1).repeat(1, 1, 4))  # unsigmoid
            refpoint_embed_ = refpoint_embed_undetach.detach()
            init_box_proposal = torch.gather(output_proposals, 1,
                                             topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).sigmoid()  # sigmoid

            # gather tgt
            tgt_undetach = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))
            if self.embed_init_tgt:
                tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)  # nq, bs, d_model
            else:
                tgt_ = tgt_undetach.detach()

            if self.training:  # tdy
                # import ipdb; ipdb.set_trace()
                # refpoint_embed_ [:, :100] = x_boxes
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

        elif self.two_stage_type == 'early':
            refpoint_embed_undetach = self.enc_out_bbox_embed(
                enc_intermediate_output[-1]) + enc_refpoint_embed  # unsigmoid, (bs, nq, 4)
            refpoint_embed = refpoint_embed_undetach.detach()  #

            tgt_undetach = enc_intermediate_output[-1]  # bs, nq, d_model
            tgt = tgt_undetach.detach()
        elif self.two_stage_type == 'no':
            tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)  # nq, bs, d_model
            refpoint_embed_ = self.refpoint_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)  # nq, bs, 4

            if refpoint_embed is not None:
                # import ipdb; ipdb.set_trace()
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

            # pat embed
            if self.num_patterns > 0:
                tgt_embed = tgt.repeat(1, self.num_patterns, 1)
                refpoint_embed = refpoint_embed.repeat(1, self.num_patterns, 1)
                tgt_pat = self.patterns.weight[None, :, :].repeat_interleave(self.num_queries,
                                                                             1)  # 1, n_q*n_pat, d_model
                tgt = tgt_embed + tgt_pat

            init_box_proposal = refpoint_embed_.sigmoid()

        else:
            raise NotImplementedError("unknown two_stage_type {}".format(self.two_stage_type))

        #########################################################
        # Begin Decoder
        #########################################################
        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=memory.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten.transpose(0, 1),
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios, tgt_mask=attn_mask, tgt_mask2=attn_mask2,
            x_boxes=x_boxes,
        )
        #########################################################
        # End Decoder
        # hs: n_dec, bs, nq, d_model
        # references: n_dec+1, bs, nq, query_dim
        #########################################################

        #########################################################
        # Begin postprocess
        #########################################################
        if self.two_stage_type == 'standard':
            if self.two_stage_keep_all_tokens:
                hs_enc = output_memory.unsqueeze(0)
                ref_enc = enc_outputs_coord_unselected.unsqueeze(0)
                init_box_proposal = output_proposals
                # import ipdb; ipdb.set_trace()
            else:
                hs_enc = tgt_undetach.unsqueeze(0)
                ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
        elif self.two_stage_type in ['combine', 'early']:
            hs_enc = enc_intermediate_output
            hs_enc = torch.cat((hs_enc, tgt_undetach.unsqueeze(0)), dim=0)  # nenc+1, bs, nq, c
            n_layer_hs_enc = hs_enc.shape[0]
            assert n_layer_hs_enc == self.num_encoder_layers + 1

            ref_enc = enc_intermediate_refpoints
            ref_enc = torch.cat((ref_enc, refpoint_embed_undetach.sigmoid().unsqueeze(0)), dim=0)  # nenc+1, bs, nq, 4
        elif self.two_stage_type in ['enceachlayer', 'enclayer1']:
            hs_enc = enc_intermediate_output
            hs_enc = torch.cat((hs_enc, tgt_undetach.unsqueeze(0)), dim=0)  # nenc, bs, nq, c
            n_layer_hs_enc = hs_enc.shape[0]
            assert n_layer_hs_enc == self.num_encoder_layers

            ref_enc = enc_intermediate_refpoints
            ref_enc = torch.cat((ref_enc, refpoint_embed_undetach.sigmoid().unsqueeze(0)),
                                dim=0)  # nenc, bs, nq, 4
        else:
            hs_enc = ref_enc = None

        return hs, references, hs_enc, ref_enc, init_box_proposal


class TransformerEncoder(nn.Module):

    def __init__(self,
                 encoder_layer, num_layers, norm=None, d_model=256,
                 num_queries=300,
                 deformable_encoder=False,
                 enc_layer_share=False, enc_layer_dropout_prob=None,
                 two_stage_type='no',
                 ):
        super().__init__()
        # prepare layers
        if num_layers > 0:
            self.layers = _get_clones(encoder_layer, num_layers, layer_share=enc_layer_share)
        else:
            self.layers = []
            del encoder_layer

        self.query_scale = None
        self.num_queries = num_queries
        self.deformable_encoder = deformable_encoder
        self.num_layers = num_layers
        self.norm = norm
        self.d_model = d_model

        self.enc_layer_dropout_prob = enc_layer_dropout_prob
        if enc_layer_dropout_prob is not None:
            assert isinstance(enc_layer_dropout_prob, list)
            assert len(enc_layer_dropout_prob) == num_layers
            for i in enc_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self.two_stage_type = two_stage_type
        if two_stage_type in ['enceachlayer', 'enclayer1']:
            _proj_layer = nn.Linear(d_model, d_model)
            _norm_layer = nn.LayerNorm(d_model)
            if two_stage_type == 'enclayer1':
                self.enc_norm = nn.ModuleList([_norm_layer])
                self.enc_proj = nn.ModuleList([_proj_layer])
            else:
                self.enc_norm = nn.ModuleList([copy.deepcopy(_norm_layer) for i in range(num_layers - 1)])
                self.enc_proj = nn.ModuleList([copy.deepcopy(_proj_layer) for i in range(num_layers - 1)])

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self,
                src: Tensor,
                pos: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                key_padding_mask: Tensor,
                ref_token_index: Optional[Tensor] = None,
                ref_token_coord: Optional[Tensor] = None,

                ):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]

            - ref_token_index: bs, nq
            - ref_token_coord: bs, nq, 4
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus:
            - output: [bs, sum(hi*wi), 256]
        """
        if self.two_stage_type in ['no', 'standard', 'enceachlayer', 'enclayer1']:
            assert ref_token_index is None

        output = src

        # preparation and reshape
        if self.num_layers > 0:
            if self.deformable_encoder:
                reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
                # import ipdb; ipdb.set_trace()

        intermediate_output = []
        intermediate_ref = []
        if ref_token_index is not None:
            out_i = torch.gather(output, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, self.d_model))
            intermediate_output.append(out_i)
            intermediate_ref.append(ref_token_coord)

        # intermediate_coord = []
        # main process
        for layer_id, layer in enumerate(self.layers):
            mask_ratio = 0.1 * (6 - layer_id)
            # main process
            dropflag = False
            if self.enc_layer_dropout_prob is not None:
                prob = random.random()
                if prob < self.enc_layer_dropout_prob[layer_id]:
                    dropflag = True

            if not dropflag:
                if self.deformable_encoder:

                    output = layer(src=output, pos=pos, reference_points=reference_points,
                                   spatial_shapes=spatial_shapes, level_start_index=level_start_index,
                                   key_padding_mask=key_padding_mask)
                else:
                    output = layer(src=output.transpose(0, 1), pos=pos.transpose(0, 1),
                                   key_padding_mask=key_padding_mask).transpose(0, 1)

            if ((layer_id == 0 and self.two_stage_type in ['enceachlayer', 'enclayer1']) \
                or (self.two_stage_type == 'enceachlayer')) \
                    and (layer_id != self.num_layers - 1):
                output_memory, output_proposals = gen_encoder_output_proposals(output, key_padding_mask, spatial_shapes)
                output_memory = self.enc_norm[layer_id](self.enc_proj[layer_id](output_memory))

                # gather boxes
                topk = self.num_queries
                enc_outputs_class = self.class_embed[layer_id](output_memory)
                ref_token_index = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]  # bs, nq
                ref_token_coord = torch.gather(output_proposals, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, 4))

                output = output_memory

            # aux loss
            if (layer_id != self.num_layers - 1) and ref_token_index is not None:
                out_i = torch.gather(output, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, self.d_model))
                intermediate_output.append(out_i)
                intermediate_ref.append(ref_token_coord)

        if self.norm is not None:
            output = self.norm(output)

        if ref_token_index is not None:
            intermediate_output = torch.stack(intermediate_output)  # n_enc/n_enc-1, bs, \sum{hw}, d_model
            intermediate_ref = torch.stack(intermediate_ref)
        else:
            intermediate_output = intermediate_ref = None

        return output, intermediate_output, intermediate_ref


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None,
                 return_intermediate=False,
                 d_model=256, query_dim=4,
                 modulate_hw_attn=False,
                 num_feature_levels=1,
                 deformable_decoder=False,
                 dec_layer_number=None,  # number of queries each layer in decoder
                 dec_layer_share=False,
                 dec_layer_dropout_prob=None,
                 num_box_decoder_layers=2,
                 num_body_points=17,
                 num_dn=100,
                 num_group=100
                 ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers, layer_share=dec_layer_share)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None

        self.num_body_points = num_body_points
        self.query_scale = None
        self.bbox_embed = None
        self.class_embed = None
        self.pose_embed = None
        self.pose_hw_embed = None
        self.num_box_decoder_layers = num_box_decoder_layers
        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder

        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:
            self.ref_anchor_head = None

        self.box_pred_damping = None

        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_layers
            # assert dec_layer_number[0] ==

        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            raise NotImplementedError
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0
        self.num_group = num_group
        self.emb_Q = nn.Linear(2, d_model)
        self.emb_K = nn.Linear(2, d_model)
        self.emb_V = nn.Linear(2, 2)

        self.emb_SQ = nn.Linear(2, d_model)
        self.emb_SK = nn.Linear(2, d_model)
        self.emb_SV = nn.Linear(2, 2)

        self.rm_detach = None
        self.num_dn = num_dn
        self.hw = nn.Embedding(self.num_body_points, 2)
        self.keypoint_embed = nn.Embedding(self.num_body_points, d_model)
        self.kpt_index = [x for x in range(self.num_group * (self.num_body_points + 1)) if
                          x % (self.num_body_points + 1) != 0]

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                tgt_mask2: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
                # for memory
                level_start_index: Optional[Tensor] = None,  # num_levels
                spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
                valid_ratios: Optional[Tensor] = None,
                x_boxes: Optional[Tensor] = None,

                ):
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        effect_num_dn = self.num_dn if self.training else 0
        inter_select_number = self.num_group
        for layer_id, layer in enumerate(self.layers):
            if self.deformable_decoder:
                if reference_points.shape[-1] == 4:
                    reference_points_input = reference_points[:, :, None] \
                                             * torch.cat([valid_ratios, valid_ratios], -1)[None, :]  # nq, bs, nlevel, 4
                else:
                    assert reference_points.shape[-1] == 2
                    reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]
                query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])  # nq, bs, 256*2
            else:
                query_sine_embed = gen_sineembed_for_position(reference_points)  # nq, bs, 256*2
                reference_points_input = None

            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos
            if not self.deformable_decoder:
                query_sine_embed = query_sine_embed[..., :self.d_model] * self.query_pos_sine_scale(output)

            # modulated HW attentions
            if not self.deformable_decoder and self.modulate_hw_attn:
                refHW_cond = self.ref_anchor_head(output).sigmoid()  # nq, bs, 2
                query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / reference_points[..., 2]).unsqueeze(
                    -1)
                query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / reference_points[..., 3]).unsqueeze(
                    -1)

            dropflag = False
            if self.dec_layer_dropout_prob is not None:
                prob = random.random()
                if prob < self.dec_layer_dropout_prob[layer_id]:
                    dropflag = True
            if not dropflag:
                output = layer(
                    tgt=output,
                    tgt_query_pos=query_pos,
                    tgt_query_sine_embed=query_sine_embed,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    tgt_reference_points=reference_points_input,

                    memory=memory,
                    memory_key_padding_mask=memory_key_padding_mask,
                    memory_level_start_index=level_start_index,
                    memory_spatial_shapes=spatial_shapes,
                    memory_pos=pos,
                    self_attn_mask=tgt_mask,
                    cross_attn_mask=memory_mask
                )

            intermediate.append(self.norm(output))



            # human update
            if layer_id < self.num_box_decoder_layers:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

            # query expansion
            if layer_id == self.num_box_decoder_layers - 1:
                dn_output = output[:effect_num_dn]
                dn_new_reference_points = new_reference_points[:effect_num_dn]
                class_unselected = self.class_embed[layer_id](output)[effect_num_dn:]
                topk_proposals = torch.topk(class_unselected.max(-1)[0], inter_select_number, dim=0)[1]
                new_reference_points_for_box = torch.gather(new_reference_points[effect_num_dn:], 0,
                                                            topk_proposals.unsqueeze(-1).repeat(1, 1, 4))

                new_output_for_box = torch.gather(output[effect_num_dn:], 0,
                                                  topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))

                new_output_for_keypoint = new_output_for_box[:, None, :, :] + self.keypoint_embed.weight[None, :, None,
                                                                              :]
                if self.num_body_points == 17:
                    delta_xy = self.pose_embed[-1](new_output_for_keypoint)[..., :2]
                else:
                    delta_xy = self.pose_embed[0](new_output_for_keypoint)[..., :2]
                keypoint_xy = (inverse_sigmoid(new_reference_points_for_box[..., :2][:, None]) + delta_xy).sigmoid()
                num_queries, _, bs, _ = keypoint_xy.shape
                keypoint_wh_weight = self.hw.weight.unsqueeze(0).unsqueeze(-2).repeat(num_queries, 1, bs, 1).sigmoid()
                keypoint_wh = keypoint_wh_weight * new_reference_points_for_box[..., 2:][:, None]
                new_reference_points_for_keypoint = torch.cat((keypoint_xy, keypoint_wh), dim=-1) # 100, 14, 1, 14

                #-------------------------- only kpts_box --------------------------#
                x_boxes_k = new_reference_points_for_keypoint.permute(2, 0, 1, 3).view(bs, -1, 4) # bs 1400 4
                x_boxes_x = x_boxes[:, :, :self.num_body_points*2][:, :, 0::2].unsqueeze(-1)
                x_boxes_y = x_boxes[:, :, :self.num_body_points * 2][:, :, 1::2].unsqueeze(-1) # bs 100 14 1
                x_boxes_xy = torch.cat((x_boxes_x, x_boxes_y),dim=-1).view(bs, -1, 2) # bs 1400 2
                self_att_q = self.emb_SQ(x_boxes_xy.to(torch.float32))
                self_att_k = self.emb_SK(x_boxes_xy.to(torch.float32))
                self_att_v = self.emb_SV(x_boxes_xy.to(torch.float32))
                self_att = (self_att_q * (self_att_q.shape[1] ** 0.5)) @ self_att_k.transpose(-2, -1)
                self_att = self_att.softmax(dim=-1)
                x_boxes_xy = (self_att) @ self_att_v

                kv_xy = x_boxes_k[:, :, :2]
                Q = self.emb_Q(x_boxes_xy.to(torch.float32) + kv_xy)
                K = self.emb_K(kv_xy)
                V = self.emb_V(kv_xy)
                atten = (Q * (Q.shape[1] ** 0.5)) @ K.transpose(-2, -1)
                # m_r = torch.ones_like(atten) * mask_ratio
                # atten = atten + torch.bernoulli(m_r) * -1e12
                atten = atten.softmax(dim=-1)
                new_reference_points_for_keypoint_xy = (atten) @ V
                new_reference_points_for_keypoint_xy = new_reference_points_for_keypoint_xy.view(bs, -1, self.num_body_points, 2).permute(1, 2, 0, 3)
                new_reference_points_for_keypoint[:, :, :, :2] = new_reference_points_for_keypoint_xy
                #-------------------------- only kpts_box --------------------------#

                new_reference_points = torch.cat(
                    (new_reference_points_for_box.unsqueeze(1), new_reference_points_for_keypoint), dim=1).flatten(0, 1)

                output = torch.cat((new_output_for_box.unsqueeze(1), new_output_for_keypoint), dim=1).flatten(0, 1)
                new_reference_points = torch.cat((dn_new_reference_points, new_reference_points), dim=0)
                output = torch.cat((dn_output, output), dim=0)
                tgt_mask = tgt_mask2

            # human-to-keypoints update
            if layer_id >= self.num_box_decoder_layers:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                output_bbox_dn = output[:effect_num_dn]
                output_bbox_norm = output[effect_num_dn:][0::(self.num_body_points + 1)]
                reference_before_sigmoid_bbox_dn = reference_before_sigmoid[:effect_num_dn]
                reference_before_sigmoid_bbox_norm = reference_before_sigmoid[effect_num_dn:][
                                                     0::(self.num_body_points + 1)]
                delta_unsig_dn = self.bbox_embed[layer_id](output_bbox_dn)
                delta_unsig_norm = self.bbox_embed[layer_id](output_bbox_norm)
                outputs_unsig_dn = delta_unsig_dn + reference_before_sigmoid_bbox_dn
                outputs_unsig_norm = delta_unsig_norm + reference_before_sigmoid_bbox_norm
                new_reference_points_for_box_dn = outputs_unsig_dn.sigmoid()
                new_reference_points_for_box_norm = outputs_unsig_norm.sigmoid()
                output_kpt = output[effect_num_dn:].index_select(0, torch.tensor(self.kpt_index, device=output.device))
                delta_xy_unsig = self.pose_embed[layer_id - self.num_box_decoder_layers](output_kpt)
                outputs_unsig = reference_before_sigmoid[effect_num_dn:].index_select(0, torch.tensor(self.kpt_index,
                                                                                                      device=output.device)).clone()  ##
                delta_hw_unsig = self.pose_hw_embed[layer_id - self.num_box_decoder_layers](output_kpt)
                outputs_unsig[..., :2] += delta_xy_unsig[..., :2]
                outputs_unsig[..., 2:] += delta_hw_unsig
                new_reference_points_for_keypoint = outputs_unsig.sigmoid()

                bs = new_reference_points_for_box_norm.shape[1]
                new_reference_points_norm = torch.cat((new_reference_points_for_box_norm.unsqueeze(1),
                                                       new_reference_points_for_keypoint.view(-1, self.num_body_points,
                                                                                              bs, 4)), dim=1).flatten(0,
                                                                                                                      1)
                new_reference_points = torch.cat((new_reference_points_for_box_dn, new_reference_points_norm), dim=0)

            if self.rm_detach and 'dec' in self.rm_detach:
                reference_points = new_reference_points
            else:
                reference_points = new_reference_points.detach()
            ref_points.append(new_reference_points)
        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points]
        ]


def _get_clones(module, N, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=args.query_dim,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
        modulate_hw_attn=True,
        deformable_encoder=True,
        deformable_decoder=True,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        learnable_tgt_init=True,
        random_refpoints_xy=args.random_refpoints_xy,
        two_stage_type=args.two_stage_type,
        two_stage_learn_wh=args.two_stage_learn_wh,
        two_stage_keep_all_tokens=args.two_stage_keep_all_tokens,
        dec_layer_number=args.dec_layer_number,
        rm_self_attn_layers=args.rm_self_attn_layers,
        rm_detach=args.rm_detach,
        decoder_sa_type=args.decoder_sa_type,
        module_seq=args.decoder_module_seq,
        embed_init_tgt=args.embed_init_tgt,
        num_body_points=args.num_body_points,
        num_box_decoder_layers=args.num_box_decoder_layers,
        num_group=args.num_group
    )


