import copy
import math
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from util import box_ops
from util.keypoint_ops import keypoint_xyzxyz_to_xyxyzz
from util.misc import (NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid)
from .backbones import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
from .utils import MLP
from .postprocesses import PostProcess
from .criterion import SetCriterion
from ..registry import MODULE_BUILD_FUNCS
from util.box_ops import box_cxcywh_to_xyxy
from collections import namedtuple
import random

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

from torchvision.ops import boxes as box_ops
from torchvision.ops import nms  # noqa . for compatibility
from detectron2.layers import batched_nms

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def batched_nms(
        boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float
):
    """
    Same as torchvision.ops.boxes.batched_nms, but with float().
    """
    assert boxes.shape[-1] == 4
    # Note: Torchvision already has a strategy (https://github.com/pytorch/vision/issues/1311)
    # to decide whether to use coordinate trick or for loop to implement batched_nms. So we
    # just call it directly.
    # Fp16 does not have enough range for batched NMS, so adding float().
    return box_ops.batched_nms(boxes.float(), scores, idxs, iou_threshold)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


# repeat them
def get_indices_for_repeat(now_num, target_num, device):
    """
    Input:
        - now_num: int
        - target_num: int
    Output:
        - indices: tensor[target_num]
    """
    out_indice = []
    base_indice = torch.arange(now_num).to(device)
    multiplier = target_num // now_num
    out_indice.append(base_indice.repeat(multiplier))
    residue = target_num % now_num
    out_indice.append(base_indice[torch.randint(0, now_num, (residue,), device=device)])
    return torch.cat(out_indice)


def cal_area_2_torch(v):
    w = torch.max(v[:, :, 0], -1)[0] - torch.min(v[:, :, 0], -1)[0]
    h = torch.max(v[:, :, 1], -1)[0] - torch.min(v[:, :, 1], -1)[0]
    return w * w + h * h


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)  # betas 的元素变为0-0.999之间


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.to(t.device).gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class DiffusionRegPose(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries,
                 aux_loss=False, iter_update=True,
                 query_dim=4,
                 random_refpoints_xy=False,
                 fix_refpoints_hw=-1,
                 num_feature_levels=1,
                 nheads=8,
                 two_stage_type='no',
                 dec_pred_class_embed_share=False,
                 dec_pred_bbox_embed_share=False,
                 dec_pred_pose_embed_share=False,
                 two_stage_class_embed_share=True,
                 two_stage_bbox_embed_share=True,
                 dn_number=100,
                 dn_box_noise_scale=0.4,
                 dn_label_noise_ratio=0.5,
                 dn_batch_gt_fuse=False,
                 dn_labelbook_size=100,
                 dn_attn_mask_type_list=['group2group'],
                 cls_no_bias=False,
                 num_group=100,
                 num_body_points=17,
                 num_box_decoder_layers=2,
                 ):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)
        self.num_body_points = num_body_points
        self.num_box_decoder_layers = num_box_decoder_layers

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4
        self.random_refpoints_xy = random_refpoints_xy
        self.fix_refpoints_hw = fix_refpoints_hw

        # for dn training
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_batch_gt_fuse = dn_batch_gt_fuse
        self.dn_labelbook_size = dn_labelbook_size
        self.dn_attn_mask_type_list = dn_attn_mask_type_list
        assert all([i in ['match2dn', 'dn2dn', 'group2group'] for i in dn_attn_mask_type_list])
        assert not dn_batch_gt_fuse

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == 'no', "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = nn.Linear(hidden_dim, num_classes, bias=(not cls_no_bias))
        if not cls_no_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            _class_embed.bias.data = torch.ones(self.num_classes) * bias_value

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        _pose_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        _pose_hw_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        nn.init.constant_(_pose_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_pose_embed.layers[-1].bias.data, 0)

        self.num_group = num_group
        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)]
        if dec_pred_class_embed_share:
            class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        else:
            class_embed_layerlist = [copy.deepcopy(_class_embed) for i in range(transformer.num_decoder_layers)]

        if num_body_points == 17:
            if dec_pred_pose_embed_share:
                pose_embed_layerlist = [_pose_embed for i in
                                        range(transformer.num_decoder_layers - num_box_decoder_layers + 1)]
            else:
                pose_embed_layerlist = [copy.deepcopy(_pose_embed) for i in
                                        range(transformer.num_decoder_layers - num_box_decoder_layers + 1)]
        else:
            if dec_pred_pose_embed_share:
                pose_embed_layerlist = [_pose_embed for i in
                                        range(transformer.num_decoder_layers - num_box_decoder_layers)]
            else:
                pose_embed_layerlist = [copy.deepcopy(_pose_embed) for i in
                                        range(transformer.num_decoder_layers - num_box_decoder_layers)]

        pose_hw_embed_layerlist = [_pose_hw_embed for i in
                                   range(transformer.num_decoder_layers - num_box_decoder_layers)]

        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.pose_embed = nn.ModuleList(pose_embed_layerlist)
        self.pose_hw_embed = nn.ModuleList(pose_hw_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.pose_embed = self.pose_embed
        self.transformer.decoder.pose_hw_embed = self.pose_hw_embed
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.num_box_decoder_layers = num_box_decoder_layers
        self.transformer.decoder.num_body_points = num_body_points
        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ['no', 'standard'], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type != 'no':
            if two_stage_bbox_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed

            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)
            self.refpoint_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def prepare_for_dn2(self, targets):
        if not self.training:

            device = targets[0]['boxes'].device
            bs = len(targets)
            attn_mask_infere = torch.zeros(bs, self.nheads, self.num_group * (self.num_body_points + 1),
                                           self.num_group * (self.num_body_points + 1),
                                           device=device, dtype=torch.bool)
            group_bbox_kpt = (self.num_body_points + 1)
            group_nobbox_kpt = self.num_body_points
            kpt_index = [x for x in range(self.num_group * (self.num_body_points + 1)) if
                         x % (self.num_body_points + 1) == 0]
            for matchj in range(self.num_group * (self.num_body_points + 1)):
                sj = (matchj // group_bbox_kpt) * group_bbox_kpt
                ej = (matchj // group_bbox_kpt + 1) * group_bbox_kpt
                if sj > 0:
                    attn_mask_infere[:, :, matchj, :sj] = True
                if ej < self.num_group * (self.num_body_points + 1):
                    attn_mask_infere[:, :, matchj, ej:] = True
            for match_x in range(self.num_group * (self.num_body_points + 1)):
                if match_x % group_bbox_kpt == 0:
                    attn_mask_infere[:, :, match_x, kpt_index] = False

            attn_mask_infere = attn_mask_infere.flatten(0, 1)
            return None, None, None, attn_mask_infere, None

        # targets, dn_scalar, noise_scale = dn_args
        device = targets[0]['boxes'].device
        bs = len(targets)
        dn_number = self.dn_number
        dn_box_noise_scale = self.dn_box_noise_scale
        dn_label_noise_ratio = self.dn_label_noise_ratio

        # gather gt boxes and labels
        gt_boxes = [t['boxes'] for t in targets]
        gt_labels = [t['labels'] for t in targets]
        gt_keypoints = [t['keypoints'] for t in targets]


        if self.dn_batch_gt_fuse:
            raise NotImplementedError
            gt_boxes_bsall = torch.cat(gt_boxes)  # num_boxes, 4
            gt_labels_bsall = torch.cat(gt_labels)
            num_gt_bsall = gt_boxes_bsall.shape[0]
            if num_gt_bsall > 0:
                indices = get_indices_for_repeat(num_gt_bsall, dn_number, device)
                gt_boxes_expand = gt_boxes_bsall[indices][None].repeat(bs, 1, 1)  # bs, num_dn, 4
                gt_labels_expand = gt_labels_bsall[indices][None].repeat(bs, 1)  # bs, num_dn
            else:
                # all negative samples when no gt boxes
                gt_boxes_expand = torch.rand(bs, dn_number, 4, device=device)
                gt_labels_expand = torch.ones(bs, dn_number, dtype=torch.int64, device=device) * int(self.num_classes)
        else:
            gt_boxes_expand = []
            gt_labels_expand = []
            gt_keypoints_expand = []
            for idx, (gt_boxes_i, gt_labels_i, gt_keypoint_i) in enumerate(zip(gt_boxes, gt_labels, gt_keypoints)):
                num_gt_i = gt_boxes_i.shape[0]
                if num_gt_i > 0:
                    indices = get_indices_for_repeat(num_gt_i, dn_number, device)
                    gt_boxes_expand_i = gt_boxes_i[indices]  # num_dn, 4
                    gt_labels_expand_i = gt_labels_i[indices]
                    gt_keypoints_expand_i = gt_keypoint_i[indices]
                else:
                    # all negative samples when no gt boxes
                    gt_boxes_expand_i = torch.rand(dn_number, 4, device=device)
                    gt_labels_expand_i = torch.ones(dn_number, dtype=torch.int64, device=device) * int(self.num_classes)
                    gt_keypoints_expand_i = torch.rand(dn_number, self.num_body_points * 3, device=device)
                gt_boxes_expand.append(gt_boxes_expand_i)
                gt_labels_expand.append(gt_labels_expand_i)
                gt_keypoints_expand.append(gt_keypoints_expand_i)
            gt_boxes_expand = torch.stack(gt_boxes_expand)
            gt_labels_expand = torch.stack(gt_labels_expand)
            gt_keypoints_expand = torch.stack(gt_keypoints_expand)
        knwon_boxes_expand = gt_boxes_expand.clone()
        knwon_labels_expand = gt_labels_expand.clone()

        # add noise
        if dn_label_noise_ratio > 0:
            prob = torch.rand_like(knwon_labels_expand.float())
            chosen_indice = prob < dn_label_noise_ratio
            new_label = torch.randint_like(knwon_labels_expand[chosen_indice], 0,
                                           self.dn_labelbook_size)  # randomly put a new one here
            knwon_labels_expand[chosen_indice] = new_label

        if dn_box_noise_scale > 0:
            diff = torch.zeros_like(knwon_boxes_expand)
            diff[..., :2] = knwon_boxes_expand[..., 2:] / 2
            diff[..., 2:] = knwon_boxes_expand[..., 2:]
            knwon_boxes_expand += torch.mul((torch.rand_like(knwon_boxes_expand) * 2 - 1.0), diff) * dn_box_noise_scale
            knwon_boxes_expand = knwon_boxes_expand.clamp(min=0.0, max=1.0)

        input_query_label = self.label_enc(knwon_labels_expand)
        input_query_bbox = inverse_sigmoid(knwon_boxes_expand)

        # prepare mask

        if 'group2group' in self.dn_attn_mask_type_list:
            attn_mask = torch.zeros(bs, self.nheads, dn_number + self.num_queries, dn_number + self.num_queries,
                                    device=device, dtype=torch.bool)
            attn_mask[:, :, dn_number:, :dn_number] = True
            for idx, (gt_boxes_i, gt_labels_i) in enumerate(zip(gt_boxes, gt_labels)):
                num_gt_i = gt_boxes_i.shape[0]
                if num_gt_i == 0:
                    continue
                for matchi in range(dn_number):
                    si = (matchi // num_gt_i) * num_gt_i
                    ei = (matchi // num_gt_i + 1) * num_gt_i
                    if si > 0:
                        attn_mask[idx, :, matchi, :si] = True
                    if ei < dn_number:
                        attn_mask[idx, :, matchi, ei:dn_number] = True
            attn_mask = attn_mask.flatten(0, 1)

        if 'group2group' in self.dn_attn_mask_type_list:
            attn_mask2 = torch.zeros(bs, self.nheads, dn_number + self.num_group * (self.num_body_points + 1),
                                     dn_number + self.num_group * (self.num_body_points + 1),
                                     device=device, dtype=torch.bool)
            attn_mask2[:, :, dn_number:, :dn_number] = True
            group_bbox_kpt = (self.num_body_points + 1)
            group_nobbox_kpt = self.num_body_points
            kpt_index = [x for x in range(self.num_group * (self.num_body_points + 1)) if
                         x % (self.num_body_points + 1) == 0]
            for matchj in range(self.num_group * (self.num_body_points + 1)):
                sj = (matchj // group_bbox_kpt) * group_bbox_kpt
                ej = (matchj // group_bbox_kpt + 1) * group_bbox_kpt
                if sj > 0:
                    attn_mask2[:, :, dn_number:, dn_number:][:, :, matchj, :sj] = True
                if ej < self.num_group * (self.num_body_points + 1):
                    attn_mask2[:, :, dn_number:, dn_number:][:, :, matchj, ej:] = True

            for match_x in range(self.num_group * (self.num_body_points + 1)):
                if match_x % group_bbox_kpt == 0:
                    attn_mask2[:, :, dn_number:, dn_number:][:, :, match_x, kpt_index] = False

            for idx, (gt_boxes_i, gt_labels_i) in enumerate(zip(gt_boxes, gt_labels)):
                num_gt_i = gt_boxes_i.shape[0]
                if num_gt_i == 0:
                    continue
                for matchi in range(dn_number):
                    si = (matchi // num_gt_i) * num_gt_i
                    ei = (matchi // num_gt_i + 1) * num_gt_i
                    if si > 0:
                        attn_mask2[idx, :, matchi, :si] = True
                    if ei < dn_number:
                        attn_mask2[idx, :, matchi, ei:dn_number] = True
            attn_mask2 = attn_mask2.flatten(0, 1)

        mask_dict = {
            'pad_size': dn_number,
            'known_bboxs': gt_boxes_expand,
            'known_labels': gt_labels_expand,
            'known_keypoints': gt_keypoints_expand
        }

        return input_query_label, input_query_bbox, attn_mask, attn_mask2, mask_dict

    def dn_post_process2(self, outputs_class, outputs_coord, outputs_keypoints_list, mask_dict):
        if mask_dict and mask_dict['pad_size'] > 0:
            output_known_class = [outputs_class_i[:, :mask_dict['pad_size'], :] for outputs_class_i in outputs_class]
            output_known_coord = [outputs_coord_i[:, :mask_dict['pad_size'], :] for outputs_coord_i in outputs_coord]

            outputs_class = [outputs_class_i[:, mask_dict['pad_size']:, :] for outputs_class_i in outputs_class]
            outputs_coord = [outputs_coord_i[:, mask_dict['pad_size']:, :] for outputs_coord_i in outputs_coord]
            outputs_keypoint = outputs_keypoints_list

            mask_dict.update({
                'output_known_coord': output_known_coord,
                'output_known_class': output_known_class
            })
        return outputs_class, outputs_coord, outputs_keypoint

    def nms_core(self, pose_coord, heat_score):
        num_people, num_joints = pose_coord.shape
        pose_coord0 = torch.zeros_like(pose_coord.reshape(-1, self.num_body_points, 2))
        pose_coord0[:, :, 0] = pose_coord[:, :self.num_body_points]
        pose_coord0[:, :, 1] = pose_coord[:, self.num_body_points:]
        pose_area = cal_area_2_torch(pose_coord0)
        pose_area = cal_area_2_torch(pose_coord0)[:, None].repeat(1, num_people * self.num_body_points)
        pose_area = pose_area.reshape(num_people, num_people, self.num_body_points)

        pose_diff = pose_coord0[:, None, :, :] - pose_coord0
        pose_diff.pow_(2)
        pose_dist = pose_diff.sum(3)
        pose_dist.sqrt_()
        pose_thre = 0.04 * torch.sqrt(pose_area)  # 0.15 0.04-crowdpose
        pose_dist = (pose_dist < pose_thre).sum(2)

        nms_pose = pose_dist > 10  # 10-crowdpose # More than 10 key points show significant differences compared to other poses.
        nms_score = heat_score > 0.05
        nms_pose *= nms_score

        ignored_pose_inds = []
        keep_pose_inds = []
        for i in range(nms_pose.shape[0]):
            if i in ignored_pose_inds:
                continue
            keep_inds = nms_pose[i].nonzero().cpu().numpy()
            keep_inds = [list(kind)[0] for kind in keep_inds]
            keep_scores = heat_score[keep_inds]

            if len(keep_scores) != 0:
                ind = torch.argmax(keep_scores)
                keep_ind = keep_inds[ind]
            else:
                keep_ind = 0

            if keep_ind in ignored_pose_inds:
                continue
            keep_pose_inds += [keep_ind]
            ignored_pose_inds += list(set(keep_inds) - set(ignored_pose_inds))

        return keep_pose_inds


    def q_sample(self,  x_start, t, noise=None):

        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod.to(t.device), t,
                                        x_start.shape)  # extract the appropriate  t  index for a batch of indices
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod.to(t.device), t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise  # x_t

    def prepare_diffusion_concat(self, args, gt_kpts, w, h):  # gt_kpts x1y1x2y2...x14y14,z1z2...z14
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        args.num_timesteps = args.timesteps
        t = torch.randint(0, args.num_timesteps, (1,), device=args.device).long()
        noise = torch.randn(args.num_proposals, args.num_body_points * 2, device=args.device)

        num_gt = gt_kpts.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            x_start = torch.ones(self.num_proposals, self.num_body_points * 2).to(self.device) * 0.5

        if num_gt < args.num_proposals:
            gt_kpts = gt_kpts[:, :self.num_body_points * 2]  # xyxy...xy

            box_placeholder = torch.randn(args.num_proposals - num_gt, self.num_body_points * 2,
                                          device=args.device)
            x_start = torch.cat((gt_kpts, box_placeholder), dim=0)

        elif num_gt > args.num_proposals:
            gt_kpts = gt_kpts[:, :self.num_body_points * 2]
            select_mask = [True] * args.num_proposals + [False] * (num_gt - args.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_kpts[select_mask]

        else:
            x_start = gt_kpts

        x_start = (x_start * 2. - 1.) * args.scale  # Signal scaling

        # noise sample
        t = torch.tensor([0]).to(t.device)
        # print(t)
        x = self.q_sample(x_start=x_start, t=t, noise=noise)  # x_t or pb_crpt

        x = torch.clamp(x, min=-1 * args.scale, max=args.scale)
        x = ((x / args.scale) + 1) / 2.  # x_start = (x_start * 2. - 1.) * self.scale
        diff_kpts = x

        return diff_kpts, noise, t

    def prepare_targets(self, args, targets):
        new_targets = []
        diffused_kpts = []
        noises = []
        ts = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image['size']
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=args.device)
            gt_classes = targets_per_image['labels']  # 0 for background

            gt_boxes = targets_per_image['boxes']  # cxcywh
            gt_keypoints = targets_per_image['keypoints']  # XYXY... VVV
            gt_keypoints_buquan = targets_per_image['keypoints_buquan']  # XYXY... VVV 231015

            d_kpts, d_noise, d_t = self.prepare_diffusion_concat(args, gt_keypoints_buquan, w, h)

            diffused_kpts.append(d_kpts)
            noises.append(d_noise)
            ts.append(d_t)
            target["labels"] = gt_classes.to(args.device)
            target["boxes"] = gt_boxes.to(args.device)
            target["boxes_xyxy"] = targets_per_image['boxes'].to(args.device)
            target["image_size_xyxy"] = image_size_xyxy.to(args.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(args.device)
            target["area"] = targets_per_image['area'].to(args.device)
            target["keypoints"] = gt_keypoints
            target["orig_size"] = targets_per_image['orig_size']
            target["iscrowd"] = targets_per_image['iscrowd']
            target["image_id"] = targets_per_image['image_id']
            new_targets.append(target)

        return new_targets, torch.stack(diffused_kpts), torch.stack(noises), torch.stack(ts)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def oks(self, pred, gt):

        offset = (pred[:self.num_body_points] - gt[:self.num_body_points]) ** 2 + \
                 (pred[self.num_body_points:] - gt[self.num_body_points:]) ** 2
        return (offset ** 0.5).sum()

    def model_predictions(self, wh, mask_dict, srcs, masks, x, t, poss, input_query_label, attn_mask, attn_mask2,
                          time_cond, x_self_cond=None, clip_x_start=False):
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2

        attn_mask = None

        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(wh, srcs, masks, x_boxes, x_boxes, t, poss,
                                                                             input_query_label, attn_mask,
                                                                             attn_mask2)  # tdy
        # update human boxes
        effective_dn_number = self.dn_number if self.training else 0
        outputs_coord_list = []
        outputs_class = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_cls_embed, layer_hs) in enumerate(
                zip(reference[:-1], self.bbox_embed, self.class_embed, hs)):
            if dec_lid < self.num_box_decoder_layers:
                layer_delta_unsig = layer_bbox_embed(layer_hs)
                layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
                layer_outputs_unsig = layer_outputs_unsig.sigmoid()
                layer_cls = layer_cls_embed(layer_hs)
                outputs_coord_list.append(layer_outputs_unsig)
                outputs_class.append(layer_cls)
            else:
                layer_hs_bbox_dn = layer_hs[:, :effective_dn_number, :]
                layer_hs_bbox_norm = layer_hs[:, effective_dn_number:, :][:, 0::(self.num_body_points + 1), :]
                bs = layer_ref_sig.shape[0]
                reference_before_sigmoid_bbox_dn = layer_ref_sig[:, :effective_dn_number, :]
                reference_before_sigmoid_bbox_norm = layer_ref_sig[:, effective_dn_number:, :][:,
                                                     0::(self.num_body_points + 1), :]
                layer_delta_unsig_dn = layer_bbox_embed(layer_hs_bbox_dn)
                layer_delta_unsig_norm = layer_bbox_embed(layer_hs_bbox_norm)
                layer_outputs_unsig_dn = layer_delta_unsig_dn + inverse_sigmoid(reference_before_sigmoid_bbox_dn)
                layer_outputs_unsig_dn = layer_outputs_unsig_dn.sigmoid()
                layer_outputs_unsig_norm = layer_delta_unsig_norm + inverse_sigmoid(
                    reference_before_sigmoid_bbox_norm)
                layer_outputs_unsig_norm = layer_outputs_unsig_norm.sigmoid()
                layer_outputs_unsig = torch.cat((layer_outputs_unsig_dn, layer_outputs_unsig_norm), dim=1)
                layer_cls_dn = layer_cls_embed(layer_hs_bbox_dn)
                layer_cls_norm = layer_cls_embed(layer_hs_bbox_norm)
                layer_cls = torch.cat((layer_cls_dn, layer_cls_norm), dim=1)
                outputs_class.append(layer_cls)
                outputs_coord_list.append(layer_outputs_unsig)

        # update keypoints boxes
        outputs_keypoints_list = []
        outputs_keypoints_hw = []
        kpt_index = [x for x in range(self.num_group * (self.num_body_points + 1)) if
                     x % (self.num_body_points + 1) != 0]
        for dec_lid, (layer_ref_sig, layer_hs) in enumerate(zip(reference[:-1], hs)):
            if dec_lid < self.num_box_decoder_layers:
                assert isinstance(layer_hs, torch.Tensor)
                bs = layer_hs.shape[0]
                layer_res = layer_hs.new_zeros((bs, self.num_queries, self.num_body_points * 3))
                outputs_keypoints_list.append(layer_res)
            else:
                bs = layer_ref_sig.shape[0]
                layer_hs_kpt = layer_hs[:, effective_dn_number:, :].index_select(1, torch.tensor(kpt_index,
                                                                                                 device=layer_hs.device))
                delta_xy_unsig = self.pose_embed[dec_lid - self.num_box_decoder_layers](layer_hs_kpt)
                layer_ref_sig_kpt = layer_ref_sig[:, effective_dn_number:, :].index_select(1,
                                                                                           torch.tensor(kpt_index,
                                                                                                        device=layer_hs.device))
                layer_outputs_unsig_keypoints = delta_xy_unsig + inverse_sigmoid(layer_ref_sig_kpt[..., :2])
                vis_xy_unsig = torch.ones_like(layer_outputs_unsig_keypoints,
                                               device=layer_outputs_unsig_keypoints.device)
                xyv = torch.cat((layer_outputs_unsig_keypoints, vis_xy_unsig[:, :, 0].unsqueeze(-1)), dim=-1)
                xyv = xyv.sigmoid()
                layer_res = xyv.reshape((bs, self.num_group, self.num_body_points, 3)).flatten(2, 3)
                layer_hw = layer_ref_sig_kpt[..., 2:].reshape(bs, self.num_group, self.num_body_points, 2).flatten(
                    2, 3)
                layer_res = keypoint_xyzxyz_to_xyxyzz(layer_res)
                outputs_keypoints_list.append(layer_res)
                outputs_keypoints_hw.append(layer_hw)

        dn_mask_dict = mask_dict
        if self.dn_number > 0 and dn_mask_dict is not None:
            outputs_class, outputs_coord_list, outputs_keypoints_list = self.dn_post_process2(outputs_class,
                                                                                              outputs_coord_list,
                                                                                              outputs_keypoints_list,
                                                                                              dn_mask_dict)
            dn_class_input = dn_mask_dict['known_labels']
            dn_bbox_input = dn_mask_dict['known_bboxs']
            dn_class_pred = dn_mask_dict['output_known_class']
            dn_bbox_pred = dn_mask_dict['output_known_coord']

        for idx, (_out_class, _out_bbox, _out_keypoint) in enumerate(
                zip(outputs_class, outputs_coord_list, outputs_keypoints_list)):
            assert _out_class.shape[1] == _out_bbox.shape[1] == _out_keypoint.shape[1]

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord_list[-1],
               'pred_keypoints': outputs_keypoints_list[-1]}
        if self.dn_number > 0 and dn_mask_dict is not None:
            out.update(
                {
                    'dn_class_input': dn_class_input,
                    'dn_bbox_input': dn_bbox_input,
                    'dn_class_pred': dn_class_pred[-1],
                    'dn_bbox_pred': dn_bbox_pred[-1],
                    'num_tgt': dn_mask_dict['pad_size']
                }
            )

        kpts = out['pred_keypoints']  # xxx...yyy...zzz [bs 100 42] --->bs 1400 4
        x_start = kpts[:, :, :self.num_body_points * 2]
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), out

    def inference(self, box_cls, box_pred, kpt_pred):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """

        scores = torch.sigmoid(box_cls)
        labels = torch.arange(self.num_classes, device=self.device). \
            unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

        box_pred_per_imager, scores_per_imager, scores_per_image2r, labels_per_imager, kpt_pred_per_image_onesr = [], [], [], [], []

        for i, (scores_per_image22, scores_per_image, box_pred_per_image, kpt_pred_per_image) in enumerate(zip(
                box_cls, scores, box_pred, kpt_pred
        )):
            scores_per_image0, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
            scores_per_image2 = scores_per_image22.clone()
            scores_per_image = scores_per_image0
            labels_per_image = labels[topk_indices]
            box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1,
                                                                                                       4)  # 500*80, 4
            box_pred_per_image = box_pred_per_image[topk_indices]

            kpt_pred_per_image = kpt_pred_per_image.view(-1, 1, 3 * self.num_body_points).repeat(1,
                                                                                                 self.num_classes,
                                                                                                 1).view(-1,
                                                                                                         3 * self.num_body_points)  # 500*80, 4
            kpt_pred_per_image = kpt_pred_per_image[topk_indices].reshape(-1, self.num_body_points, 3)
            kpt_pred_per_image_ones = kpt_pred_per_image

            box_pred_per_imager.append(box_pred_per_image)
            scores_per_imager.append(scores_per_image)
            scores_per_image2r.append(scores_per_image2)
            labels_per_imager.append(labels_per_image)
            kpt_pred_per_image_onesr.append(kpt_pred_per_image_ones)

        if self.use_ensemble and self.sampling_timesteps > 1:
            return torch.cat(box_pred_per_imager, dim=0), torch.cat(scores_per_imager, dim=0), torch.cat(
                scores_per_image2r, dim=0), torch.cat(labels_per_imager, dim=0), torch.cat(kpt_pred_per_image_onesr,
                                                                                           dim=0)

        return torch.cat(box_pred_per_imager, dim=0), torch.cat(scores_per_imager, dim=0), torch.cat(scores_per_image2r,
                                                                                                     dim=0), torch.cat(
            labels_per_imager, dim=0), torch.cat(kpt_pred_per_image_onesr, dim=0)

    @torch.no_grad()
    def ddim_sample(self, wh, mask_dict, samples, srcs, masks, poss, input_query_label, attn_mask, attn_mask2,
                    clip_denoised=True, do_postprocess=True):

        batch = len(samples.mask)
        shape = (batch, self.num_proposals, self.num_body_points * 2)
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=self.device) * 1

        ensemble_score, ensemble_label, ensemble_coord, ensemble_kpts, ensemble_score2 = [], [], [], [], []
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            preds, out = self.model_predictions(wh, mask_dict, srcs, masks, img, time_cond, poss, input_query_label,
                                                attn_mask, attn_mask2,
                                                time_cond, self_cond, clip_x_start=clip_denoised)

            pred_noise, x_start = preds.pred_noise, preds.pred_x_start
            outputs_class = out['pred_logits']
            outputs_coord = out['pred_boxes']
            outputs_kpts = out['pred_keypoints']

            if not self.box_renewal:  # filter
                score_per_image, box_per_image = outputs_class[0], outputs_coord[0]
                threshold = 0.05  # 0.5-->3(num_remain)  0.07-->22(num_remain) 0.15-->11(num_remain) 0.10-->14(num_remain)  0.05-->45(num_remain)
                score_per_image = torch.sigmoid(score_per_image)
                value, _ = torch.max(score_per_image, -1, keepdim=False)
                keep_idx = value > threshold
                num_remain = torch.sum(keep_idx)
                print(num_remain)

                pred_noise = pred_noise[:, keep_idx, :]
                x_start = x_start[:, keep_idx, :]
                img = img[:, keep_idx, :]
            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = 0 * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            if not self.box_renewal:  # filter
                # replenish with randn boxes
                img = torch.cat(
                    (img, torch.randn(len(img), self.num_proposals - num_remain, self.num_body_points * 2,
                                      device=img.device)),
                    dim=1)

            if self.use_ensemble and self.sampling_timesteps > 1:
                box_pred_per_image, scores_per_image, scores_per_image2, labels_per_image, kpt_pred_per_image_ones = self.inference(
                    outputs_class,
                    outputs_coord,
                    outputs_kpts)

                labels_per_image = labels_per_image
                scores_per_image = scores_per_image
                scores_per_image2 = scores_per_image2
                box_pred_per_image = box_pred_per_image
                kpt_pred_per_image_ones = kpt_pred_per_image_ones.reshape(-1, 3 * self.num_body_points)

                ensemble_label.append(labels_per_image)
                ensemble_score.append(scores_per_image)
                ensemble_score2.append(scores_per_image2)
                ensemble_coord.append(box_pred_per_image)
                ensemble_kpts.append(kpt_pred_per_image_ones)

        if self.use_ensemble and self.sampling_timesteps > 1:

            labels_per_image = torch.cat(ensemble_label, dim=0)
            scores_per_image = torch.cat(ensemble_score, dim=0)
            scores_per_image2 = torch.cat(ensemble_score2, dim=0)
            box_pred_per_image = torch.cat(ensemble_coord, dim=0)
            kpts_per_image = torch.cat(ensemble_kpts, dim=0)

            if self.use_nms:  # STEP >=2
                box_pred_per_image_nms = box_pred_per_image * (wh.repeat(len(box_pred_per_image), 1))
                box_pred_per_image_nms = box_cxcywh_to_xyxy(box_pred_per_image_nms)
                # keep = batched_nms(box_pred_per_image_nms, scores_per_image, labels_per_image, 1) # box_nms
                # scores_per_image = scores_per_image[keep]
                # scores_per_image2 = scores_per_image2[keep]
                # box_pred_per_image = box_pred_per_image[keep]
                # kpts_per_image = kpts_per_image[keep]

                w_kpts = wh[:, 0].repeat(self.num_body_points, 1)
                h_kpts = wh[:, 1].repeat(self.num_body_points, 1)
                wh_kpts = torch.cat((w_kpts.squeeze(1), h_kpts.squeeze(1)), dim=0)
                Z_pred = kpts_per_image[:, 0:(self.num_body_points * 2)] * wh_kpts[None, :]
                keep_index = self.nms_core(Z_pred, scores_per_image)  # pose_nms

                # keep_index = list(set(keep_index + keep.cpu().numpy().tolist())) #Union
                # keep_index = list(set(keep_index) & set(keep))#Intersection

                scores_per_image = scores_per_image[keep_index]
                scores_per_image2 = scores_per_image2[keep_index]
                box_pred_per_image = box_pred_per_image[keep_index]
                kpts_per_image = kpts_per_image[keep_index]

            output = {'pred_logits': scores_per_image2.unsqueeze(0), 'pred_boxes': box_pred_per_image.unsqueeze(0),
                      'pred_keypoints': kpts_per_image.unsqueeze(0)
                      }
            results = output  # step > 2
        else:
            box_pred_per_image, scores_per_image, scores_per_image2, labels_per_image, \
            kpt_pred_per_image_ones = self.inference(outputs_class, outputs_coord, outputs_kpts)
            kpt_pred_per_image_ones = kpt_pred_per_image_ones.reshape(-1, 3 * self.num_body_points)
            if self.use_nms:
                box_pred_per_image_nms = box_pred_per_image * (wh.repeat(len(box_pred_per_image), 1))
                box_pred_per_image_nms = box_cxcywh_to_xyxy(box_pred_per_image_nms)
                keep = batched_nms(box_pred_per_image_nms, scores_per_image, labels_per_image, 0.85)  # box_nms
                # scores_per_image = scores_per_image[keep]
                # scores_per_image2 = scores_per_image2[keep]
                # box_pred_per_image = box_pred_per_image[keep]
                # kpts_per_image = kpt_pred_per_image_ones[keep]

                w_kpts = wh[:, 0].repeat(self.num_body_points, 1)
                h_kpts = wh[:, 1].repeat(self.num_body_points, 1)
                wh_kpts = torch.cat((w_kpts.squeeze(1), h_kpts.squeeze(1)), dim=0)
                Z_pred = kpt_pred_per_image_ones[:, 0:(self.num_body_points * 2)] * wh_kpts[None, :]
                keep_index = self.nms_core(Z_pred, scores_per_image)  # pose_nms

                # keep_index = list(set(keep_index + keep.cpu().numpy().tolist())) # pose_nms + box_nms 726
                # keep_index = list(set(keep_index + keep.cpu().numpy().tolist())) #Union

                # keep_index = [j for j in keep_index if j in keep]#jiaoji

                scores_per_image = scores_per_image[keep_index]
                scores_per_image2 = scores_per_image2[keep_index]
                box_pred_per_image = box_pred_per_image[keep_index]
                kpts_per_image = kpt_pred_per_image_ones[keep_index]

                output = {'pred_logits': scores_per_image2.unsqueeze(0), 'pred_boxes': box_pred_per_image.unsqueeze(0),
                          'pred_keypoints': kpts_per_image.unsqueeze(0)
                          }  # step = 1, 2
            else:
                output = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord, 'pred_keypoints': outputs_kpts
                          }
            results = output

        if not do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results

    def forward(self, args, samples: NestedTensor, targets: List = None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        if self.dn_number > 0 or targets is not None:
            input_query_label, input_query_bbox, attn_mask, attn_mask2, mask_dict = \
                self.prepare_for_dn2(targets)

        else:
            assert targets is None
            input_query_bbox = input_query_label = attn_mask = attn_mask1 = attn_mask2 = mask_dict = None
        self.training = not args.eval
        self.num_proposals = args.num_proposals
        self.device = args.device
        # build diffusion
        timesteps = args.timesteps
        sampling_timesteps = args.sampling_timesteps
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = args.scale
        self.box_renewal = True
        self.use_ensemble = True
        self.use_nms = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        images_whwh = list()
        for bi in targets:
            h, w = bi["size"][:2]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        # Prepare Proposals.
        if not self.training:
            results = self.ddim_sample(images_whwh, mask_dict, samples, srcs, masks, poss, input_query_label, attn_mask,
                                       attn_mask2)
            return results
        targets, x_boxes, noises, t = self.prepare_targets(args, targets)  # xyxy...xy
        t = t.squeeze(-1)

        x_boxes = x_boxes.clone().detach().requires_grad_(True)

        if self.training:
            hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(images_whwh, srcs, masks, x_boxes,
                                                                                 input_query_bbox, t, poss,
                                                                                 input_query_label, attn_mask,
                                                                                 attn_mask2)  # tdy

            # update human boxes
            effective_dn_number = self.dn_number if self.training else 0
            outputs_coord_list = []
            outputs_class = []
            for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_cls_embed, layer_hs) in enumerate(
                    zip(reference[:-1], self.bbox_embed, self.class_embed, hs)):
                if dec_lid < self.num_box_decoder_layers:
                    layer_delta_unsig = layer_bbox_embed(layer_hs)
                    layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
                    layer_outputs_unsig = layer_outputs_unsig.sigmoid()
                    layer_cls = layer_cls_embed(layer_hs)
                    # layer_outputs_unsig = torch.clamp(layer_outputs_unsig, min=1e-6, max=1)
                    outputs_coord_list.append(layer_outputs_unsig)
                    outputs_class.append(layer_cls)
                else:
                    layer_hs_bbox_dn = layer_hs[:, :effective_dn_number, :]
                    layer_hs_bbox_norm = layer_hs[:, effective_dn_number:, :][:, 0::(self.num_body_points + 1), :]
                    bs = layer_ref_sig.shape[0]
                    reference_before_sigmoid_bbox_dn = layer_ref_sig[:, :effective_dn_number, :]
                    reference_before_sigmoid_bbox_norm = layer_ref_sig[:, effective_dn_number:, :][:,
                                                         0::(self.num_body_points + 1), :]
                    layer_delta_unsig_dn = layer_bbox_embed(layer_hs_bbox_dn)
                    layer_delta_unsig_norm = layer_bbox_embed(layer_hs_bbox_norm)
                    layer_outputs_unsig_dn = layer_delta_unsig_dn + inverse_sigmoid(reference_before_sigmoid_bbox_dn)
                    layer_outputs_unsig_dn = layer_outputs_unsig_dn.sigmoid()
                    layer_outputs_unsig_norm = layer_delta_unsig_norm + inverse_sigmoid(
                        reference_before_sigmoid_bbox_norm)
                    layer_outputs_unsig_norm = layer_outputs_unsig_norm.sigmoid()
                    layer_outputs_unsig = torch.cat((layer_outputs_unsig_dn, layer_outputs_unsig_norm), dim=1)
                    layer_cls_dn = layer_cls_embed(layer_hs_bbox_dn)
                    layer_cls_norm = layer_cls_embed(layer_hs_bbox_norm)
                    layer_cls = torch.cat((layer_cls_dn, layer_cls_norm), dim=1)
                    outputs_class.append(layer_cls)
                    # layer_outputs_unsig = torch.clamp(layer_outputs_unsig, min=1e-6, max=1)
                    outputs_coord_list.append(layer_outputs_unsig)

            # update keypoints boxes
            outputs_keypoints_list = []
            outputs_keypoints_hw = []
            kpt_index = [x for x in range(self.num_group * (self.num_body_points + 1)) if
                         x % (self.num_body_points + 1) != 0]
            for dec_lid, (layer_ref_sig, layer_hs) in enumerate(zip(reference[:-1], hs)):
                if dec_lid < self.num_box_decoder_layers:
                    assert isinstance(layer_hs, torch.Tensor)
                    bs = layer_hs.shape[0]
                    layer_res = layer_hs.new_zeros((bs, self.num_queries, self.num_body_points * 3))
                    outputs_keypoints_list.append(layer_res)
                else:
                    bs = layer_ref_sig.shape[0]
                    layer_hs_kpt = layer_hs[:, effective_dn_number:, :].index_select(1, torch.tensor(kpt_index,
                                                                                                     device=layer_hs.device))
                    delta_xy_unsig = self.pose_embed[dec_lid - self.num_box_decoder_layers](layer_hs_kpt)
                    layer_ref_sig_kpt = layer_ref_sig[:, effective_dn_number:, :].index_select(1,
                                                                                               torch.tensor(kpt_index,
                                                                                                            device=layer_hs.device))
                    layer_outputs_unsig_keypoints = delta_xy_unsig + inverse_sigmoid(layer_ref_sig_kpt[..., :2])
                    vis_xy_unsig = torch.ones_like(layer_outputs_unsig_keypoints,
                                                   device=layer_outputs_unsig_keypoints.device)
                    xyv = torch.cat((layer_outputs_unsig_keypoints, vis_xy_unsig[:, :, 0].unsqueeze(-1)), dim=-1)
                    xyv = xyv.sigmoid()
                    layer_res = xyv.reshape((bs, self.num_group, self.num_body_points, 3)).flatten(2, 3)
                    layer_hw = layer_ref_sig_kpt[..., 2:].reshape(bs, self.num_group, self.num_body_points, 2).flatten(
                        2, 3)
                    layer_res = keypoint_xyzxyz_to_xyxyzz(layer_res)
                    outputs_keypoints_list.append(layer_res)
                    outputs_keypoints_hw.append(layer_hw)

            dn_mask_dict = mask_dict
            if self.dn_number > 0 and dn_mask_dict is not None:
                outputs_class, outputs_coord_list, outputs_keypoints_list = self.dn_post_process2(outputs_class,
                                                                                                  outputs_coord_list,
                                                                                                  outputs_keypoints_list,
                                                                                                  dn_mask_dict)
                dn_class_input = dn_mask_dict['known_labels']
                dn_bbox_input = dn_mask_dict['known_bboxs']
                dn_class_pred = dn_mask_dict['output_known_class']
                dn_bbox_pred = dn_mask_dict['output_known_coord']

            for idx, (_out_class, _out_bbox, _out_keypoint) in enumerate(
                    zip(outputs_class, outputs_coord_list, outputs_keypoints_list)):
                assert _out_class.shape[1] == _out_bbox.shape[1] == _out_keypoint.shape[1]

            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord_list[-1],
                   'pred_keypoints': outputs_keypoints_list[-1],
                   'pred_init_boxes': init_box_proposal}
            if self.dn_number > 0 and dn_mask_dict is not None:
                out.update(
                    {
                        'dn_class_input': dn_class_input,
                        'dn_bbox_input': dn_bbox_input,
                        'dn_class_pred': dn_class_pred[-1],
                        'dn_bbox_pred': dn_bbox_pred[-1],
                        'num_tgt': dn_mask_dict['pad_size']
                    }
                )

            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list, outputs_keypoints_list)
                if self.dn_number > 0 and dn_mask_dict is not None:
                    assert len(dn_class_pred[:-1]) == len(dn_bbox_pred[:-1]) == len(out["aux_outputs"])
                    for aux_out, dn_class_pred_i, dn_bbox_pred_i in zip(out["aux_outputs"], dn_class_pred,
                                                                        dn_bbox_pred):
                        aux_out.update({
                            'dn_class_input': dn_class_input,
                            'dn_bbox_input': dn_bbox_input,
                            'dn_class_pred': dn_class_pred_i,
                            'dn_bbox_pred': dn_bbox_pred_i,
                            'num_tgt': dn_mask_dict['pad_size']
                        })
            # for encoder output
            if hs_enc is not None:
                interm_coord = ref_enc[-1]
                interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])
                interm_pose = torch.zeros_like(outputs_keypoints_list[0])
                out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord,
                                         'pred_keypoints': interm_pose}

            return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_keypoints):
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_keypoints': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_keypoints[:-1])]


@MODULE_BUILD_FUNCS.registe_with_name(module_name='diffusionregpose')
def build_diffusionregpose(args):
    num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_class_embed_share = args.dec_pred_class_embed_share
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share

    model = DiffusionRegPose(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=args.random_refpoints_xy,
        fix_refpoints_hw=args.fix_refpoints_hw,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_class_embed_share=dec_pred_class_embed_share,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        # two stage
        two_stage_type=args.two_stage_type,

        # box_share
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,

        dn_number=args.dn_number if args.use_dn else 0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_batch_gt_fuse=args.dn_batch_gt_fuse,
        dn_attn_mask_type_list=args.dn_attn_mask_type_list,
        dn_labelbook_size=dn_labelbook_size,

        cls_no_bias=args.cls_no_bias,
        num_group=args.num_group,
        num_body_points=args.num_body_points,
        num_box_decoder_layers=args.num_box_decoder_layers
    )
    matcher = build_matcher(args)

    # prepare weight dict
    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'loss_bbox': args.bbox_loss_coef,
        "loss_keypoints": args.keypoints_loss_coef,
        # "loss_graph": args.graph_loss_coef,
        "loss_oks": args.oks_loss_coef
    }
    weight_dict['loss_giou'] = args.giou_loss_coef

    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    if args.use_dn:
        weight_dict.update({
            'dn_loss_ce': args.dn_label_coef,
            'dn_loss_bbox': args.bbox_loss_coef * args.dn_bbox_coef,
            'dn_loss_giou': args.giou_loss_coef * args.dn_bbox_coef,
        })

    clean_weight_dict = copy.deepcopy(weight_dict)

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            for k, v in clean_weight_dict.items():
                if i < args.num_box_decoder_layers and 'keypoints' in k:
                    continue
                aux_weight_dict.update({k + f'_{i}': v})
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != 'no':
        interm_weight_dict = {}
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        _coeff_weight_dict = {
            'loss_ce': 1.0,
            'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
            'loss_keypoints': 1.0 if not no_interm_box_loss else 0.0,
            # 'loss_graph': 1.0 if not no_interm_box_loss else 0.0,
            'loss_oks': 1.0 if not no_interm_box_loss else 0.0,
        }
        try:
            interm_loss_coef = args.interm_loss_coef
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update(
            {k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items() if
             'keypoints' not in k})
        weight_dict.update(interm_weight_dict)

        interm_weight_dict.update({k + f'_query_expand': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in
                                   clean_weight_dict_wo_dn.items()})
        weight_dict.update(interm_weight_dict)

    losses = ['labels', 'boxes', "keypoints"]
    if args.dn_number > 0:
        losses += ["dn_label", "dn_bbox"]
    losses += ["matching"]

    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses,
                             num_box_decoder_layers=args.num_box_decoder_layers, num_body_points=args.num_body_points)
    criterion.to(device)
    postprocessors = {
        'bbox': PostProcess(num_select=args.num_select, nms_iou_threshold=args.nms_iou_threshold,
                            num_body_points=args.num_body_points),
    }

    return model, criterion, postprocessors