import copy
import os
import math
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
import random
from torch import Tensor
from util import box_ops
from util.keypoint_ops import keypoint_xyzxyz_to_xyxyzz
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .backbones import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
from .utils import PoseProjector, sigmoid_focal_loss, MLP
from .postprocesses import PostProcess
from .criterion import SetCriterion
from ..registry import MODULE_BUILD_FUNCS
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from collections import namedtuple
import random
from random import randint
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

import cv2
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms  # noqa . for compatibility
import numpy as np
import scipy.io as sio
from detectron2.layers import batched_nms


avg = np.array([[2.32475093, 1.57779447], [1.46855015, 1.58440983], [2.53009383, 2.43933295], [1.26832261, 2.42280815], [2.47537063, 2.79984638], [1.32206972, 2.73854381], [2.15573752, 3.62561172], [1.64350402, 3.6283541 ], [2.21628696, 4.86076251], [1.6072907,  4.85418122], [2.18580336, 6.18096722], [1.62752404, 6.17869719],[1.85912873, 0.36010667],[1.86695228, 1.29348862]])



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


class EDPose(nn.Module):
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

        # # repeat them
        # def get_indices_for_repeat(now_num, target_num, device='cuda'):
        #     """
        #     Input:
        #         - now_num: int
        #         - target_num: int
        #     Output:
        #         - indices: tensor[target_num]
        #     """
        #     out_indice = []
        #     base_indice = torch.arange(now_num).to(device)
        #     multiplier = target_num // now_num
        #     out_indice.append(base_indice.repeat(multiplier))
        #     residue = target_num % now_num
        #     out_indice.append(base_indice[torch.randint(0, now_num, (residue,), device=device)])
        #     return torch.cat(out_indice)

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
        pose_thre = 0.04 * torch.sqrt(pose_area) # 0.15 0.04-crowdpose
        pose_dist = (pose_dist < pose_thre).sum(2)

        nms_pose = pose_dist > 10 # 10-crowdpose

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

    def vis_noise_pose(self, x_start, w, h, str_img):
        color_bone = [
            [0.10, 0.96, 0.16],
            [0.90, 0.00, 1.00],
            [0.19, 0.88, 0.90],
            [0.90, 1.00, 0.00],
            [0.63, 0.93, 0.94],
            [0.82, 0.11, 0.55],
            [1.00, 0.38, 0.50],
            [0.93, 0.15, 0.34],
            [0.10, 0.29, 0.98],
            [1.00, 0.50, 1.00],
            [0.04, 0.59, 0.27],
            [0.90, 0.63, 0.79],
            [0.94, 0.90, 0.55]]
        sks = [[12, 13], [1, 13], [0, 13], [0, 2], [2, 4], [1, 3], [3, 5], [13, 7], [13, 6], [7, 9], [9, 11],
               [6, 8], [8, 10]]
        # for si, sk in enumerate(sks):
        #     color_line = [color_bone[si][0] * 255, color_bone[si][1] * 255, color_bone[si][2] * 255]
        #     cv2.line(img, (int(x[sk[0]]), int(y[sk[0]])), (int(x[sk[1]]), int(y[sk[1]])), color=color_line,
        #              thickness=10)


        img = cv2.imread('/mnt/sde/tdy/crowdpose/images/101659.jpg') # 102265
        x_boxes_np = x_start.cpu().numpy()
        kpts_x = x_boxes_np[:, 0::2]
        kpts_y = x_boxes_np[:, 1::2]
        # x, y = kpts_x[0], kpts_y[0]

        # kpts_z = x_boxes_np[:, 1::2]
        w_np = w.cpu().numpy()
        h_np = h.cpu().numpy()
        img_ori = cv2.resize(img, (800, 1200))


        img = np.ones([3000, 3000, 3]) * 255
        det = 1000
        img[det:1200+det, det:800+det,:] = img_ori
        flag = -1
        for x, y, in zip(kpts_x, kpts_y):
            for si, sk in enumerate(sks):
                color_line = [color_bone[si][0] * 255, color_bone[si][1] * 255, color_bone[si][2] * 255]
                cv2.line(img, (int(x[sk[0]] * w_np + det), int(y[sk[0]] * h_np + det)),
                         (int(x[sk[1]] * w_np + det), int(y[sk[1]] * h_np + det)), color=color_line,
                         thickness=10)

        for kpt_x, kpt_y in zip(kpts_x, kpts_y):
            flag += 1
            if flag<2:
                pass
                for kpt_x0, kpt_y0 in zip(kpt_x, kpt_y):
                    cv2.circle(img, (int(kpt_x0 * w_np+det), int(kpt_y0 * h_np+det)), radius=8, color=(255, 103, 37),
                               thickness=-1)
            else:
                for kpt_x0, kpt_y0 in zip(kpt_x, kpt_y):
                    cv2.circle(img, (int(kpt_x0 * w_np+det), int(kpt_y0 * h_np+det)), radius=8, color=(255,0,255), #(255,0,255)pinhong
                               thickness=-1)

        # cv2.rectangle(img, (int(det), int(det)), (int(w_np+det), int(h_np+det)), (225, 89, 110), thickness=3)

        cv2.imwrite('/mnt/sdc/tandayi/vis/' + str_img + '.jpg', img) # enlarge

    # def vis_noise_pose(self, x_start, w, h, str_img):
    #     img = cv2.imread('/mnt/sde/tdy/crowdpose/images/118844.jpg') # 102265
    #     x_boxes_np = x_start.cpu().numpy()
    #     kpts_x = x_boxes_np[:, 0::2]
    #     kpts_y = x_boxes_np[:, 1::2]
    #     # kpts_z = x_boxes_np[:, 1::2]
    #     w_np = w.cpu().numpy()
    #     h_np = h.cpu().numpy()
    #     img_ori = cv2.resize(img, (608, 912))
    #
    #     img = np.ones([1500, 1500, 3]) * 255
    #     img[0:912, 0:608,:] = img_ori
    #     flag = -1
    #     for kpt_x, kpt_y in zip(kpts_x, kpts_y):
    #         flag += 1
    #         if flag==198 or flag==199:
    #             for kpt_x0, kpt_y0 in zip(kpt_x, kpt_y):
    #                 cv2.circle(img, (int(kpt_x0 * w_np), int(kpt_y0 * h_np)), radius=8, color=(0, 255, 0),
    #                            thickness=-1)
    #         else:
    #             for kpt_x0, kpt_y0 in zip(kpt_x, kpt_y):
    #                 cv2.circle(img, (int(kpt_x0 * w_np), int(kpt_y0 * h_np)), radius=8, color=(100, 30, 210),
    #                            thickness=-1)
    #
    #     cv2.rectangle(img, (int(0), int(0)), (int(w_np), int(h_np)), (225, 89, 110), thickness=3)
    #
    #     cv2.imwrite('/mnt/sdc/tandayi/vis/' + str_img + '.jpg', img)

    def q_sample(self, args, opt, x_start, t, noise=None):

        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(opt.sqrt_alphas_cumprod.to(t.device), t,
                                        x_start.shape)  # extract the appropriate  t  index for a batch of indices
        sqrt_one_minus_alphas_cumprod_t = extract(opt.sqrt_one_minus_alphas_cumprod.to(t.device), t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise  # x_t

    def prepare_diffusion_concat(self, args, opt, gt_kpts, w, h):#gt_kpts x1y1x2y2...x14y14,z1z2...z14
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        args.num_timesteps = args.timesteps
        t = torch.randint(0, args.num_timesteps, (1,), device=args.device).long()
        # t = torch.ones_like(t) * 999
        noise = torch.randn(args.num_proposals, args.num_body_points * 2, device=args.device)

        num_gt = gt_kpts.shape[0]

        # img = np.zeros((1500, 1500, 3))
        # kx = gt_kpts[1, 0:28][0::2]
        # ky = gt_kpts[1, 0:28][1::2]
        # for x, y in zip(kx, ky):
        #     cv2.circle(img, (int(x * 608), int(y * 913)), radius=10, color=(0, 0, 255), thickness=10)
        # cv2.imwrite('/mnt/hdd/home/tandayi/code/2D HPE/ED-Pose/vis6/img.jpg', img)

        # num_gt = 0
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            # gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=args.device)
            #
            x_start = torch.ones(self.num_proposals, self.num_body_points * 2).to(self.device) * 0.5
            # num_gt = 1


        if num_gt < args.num_proposals:
            gt_kpts = gt_kpts[:, :self.num_body_points * 2] # xyxy...xy
            # num_proposals = args.num_proposals
            # device = args.device
            # indices = get_indices_for_repeat(num_gt, num_proposals, device)
            # gt_boxes_expand = gt_kpts[indices]  # num_dn, 4

            # box_placeholder = torch.randn(args.num_proposals - num_gt, self.num_body_points*2,
            #                               device=args.device)#
            # box_placeholder = torch.randn(args.num_proposals - num_gt, self.num_body_points * 2,
            #                               device=args.device) + 0.5  # 721
            #
            # box_placeholder = torch.randn(args.num_proposals-num_gt, self.num_body_points * 2,
            #                               device=args.device)*0.5 + 0.5  # 721
            # box_placeholder = torch.randn(args.num_proposals-num_gt, self.num_body_points * 2,
            #                               device=args.device)
            box_placeholder = torch.zeros(args.num_proposals - num_gt, self.num_body_points * 2,
                                          device=args.device)  # 723

            # img_1k = np.zeros((1500, 1500, 3))
            # for tg in box_placeholder_half:
            #     # img_1k = np.zeros((1500, 1500, 3))
            #     # img_1k[:h, :w] = img_r
            #     color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            #     kpts = tg
            #     kpts_x = kpts[:28][0::2]
            #     kpts_y = kpts[:28][1::2]
            #     kpt_id = 0
            #     for (x, y) in zip(kpts_x, kpts_y):
            #         kpt_id += 1
            #         if 1:
            #             cv2.circle(img_1k, (int(x * w), int(y * h)), radius=10, color=color, thickness=-1)
            #             cv2.putText(img_1k, str(kpt_id), (int(x * w) + 10, int(y * h) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            #                         color, 4)
            #
            # cv2.imwrite('/mnt/hdd/home/tandayi/code/2D HPE/ED-Pose/covar0/' + str(2) + '.jpg', img_1k)
            # c = 0

            # box_placeholder = torch.clip(box_placeholder, min=1e-4)
#             # #########template---------------------------
#             shape = (1, 100, 28)
#             img = torch.randn(shape[0], shape[1] * 2, device=self.device) * 0.5 + 0.5
#             size = torch.abs(torch.randn(100, device=self.device))*5
            
#             yFile = '/root/code/data.mat'
#             datay = sio.loadmat(yFile)
#             avg = datay['avg_left']
#             avgxy = avg.reshape(-1)  # xyxy...xyxy
#             imgx = img[:, 0::2].unsqueeze(-1).repeat(1, 1, 14)
#             imgy = img[:, 1::2].unsqueeze(-1).repeat(1, 1, 14)
            
#             avg_pose_x = torch.from_numpy(avgxy[0::2]).unsqueeze(0).unsqueeze(0).repeat(shape[0], shape[1], 1)
#             avg_pose_y = torch.from_numpy(avgxy[1::2]).unsqueeze(0).unsqueeze(0).repeat(shape[0], shape[1], 1)
#             img0 = avg_pose_x.to(imgx.device) / (30) + imgx  # head_h = 30
#             img1 = avg_pose_y.to(imgx.device) / (30) + imgy
            
#             img = torch.zeros((1, self.num_proposals, self.num_body_points * 2)).to(imgx.device)
#             # img[:, :, 0::2] = img0 * size.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 14)
#             # img[:, :, 1::2] = img1 * size.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 14)
#             img[:, :, 0::2] = img0
#             img[:, :, 1::2] = img1
#             #
#             # # img = torch.randn(shape, device=self.device) * 0.5+0.5
#             # # img = torch.zeros(shape, device=self.device)
#             #
#             # img1500 = np.zeros((2000, 2000))
#             # cv2.rectangle(img1500, ((int(0)), int(0)), ((int(w)), int(h)), (0, 255, 0), 2)
#             #
#             # for imx, imy in zip(img[:, :, 0::2], img[:, :, 1::2]):
#             #     for ix, iy in zip(imx, imy):
#             #         for x, y in zip(ix, iy):
#             #             cv2.circle(img1500, (int((x) * w), int((y) * h)),
#             #                        radius=5, color=(251, 0, 125), thickness=-1)
#             #
#             # cv2.imwrite(
#             #     '/mnt/sdc/tandayi/vis/' + '2.jpg',
#             #     img1500)
            
#             box_placeholder = img.squeeze(0)[:self.num_group - num_gt]

#             #########template-------------
            indices = get_indices_for_repeat(num_gt, args.num_proposals - num_gt, args.device)
            gt_boxes_expand = gt_kpts[indices]  # num_dn, 4
            box_placeholder = gt_boxes_expand

            # yFile = '/root/code/ED-Pose10_coco_724/datasets/data.mat'
            # datay = sio.loadmat(yFile)
            # avg = torch.from_numpy(datay['avg_left_img'])
            # avg1 = avg.reshape(-1).to(gt_kpts.device).to(torch.float32)  # xyxyxy...xy
            # sigma = torch.from_numpy(datay['covariance_left_img']).to(gt_kpts.device).to(torch.float32)
            # box_placeholder = torch.randn(args.num_proposals - num_gt, self.num_body_points * 2,
            #                               device=args.device) @ sigma + avg1[None, :]
            # a = 0
    

            x_start = torch.cat((gt_kpts, box_placeholder), dim=0)
            # x_start = torch.cat((box_placeholder, gt_kpts), dim=0)
            # x_start = box_placeholder
            # x_start += noise
            c = 0


        elif num_gt > args.num_proposals:
            gt_kpts = gt_kpts[:, :self.num_body_points*2]
            select_mask = [True] * args.num_proposals + [False] * (num_gt - args.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_kpts[select_mask]

        else:
            x_start = gt_kpts

        # x_start = torch.randn(args.num_proposals, self.num_body_points * 2,
        #                               device=args.device) / 2. + 0.5  #

        # self.vis_noise_pose(x_start, w, h, str_img='x_start')

        x_start = (x_start * 2. - 1.) * args.scale  # Signal scaling
        # x_start = x_start * args.scale  # Signal scaling

        # self.vis_noise_pose(x_start, w, h, str_img='x_start_scale')

        # noise sample
        # t = torch.tensor([1981]).to(t.device)
        # print(t)
        x = self.q_sample(args, opt, x_start=x_start, t=t, noise=noise)  # x_t or pb_crpt

        x = torch.clamp(x, min=-1 * args.scale, max=args.scale)
        x = ((x / args.scale) + 1) / 2.  # x_start = (x_start * 2. - 1.) * self.scale 相反
        # x = (x / args.scale) # x_start = (x_start * 2. - 1.) * self.scale 相反

        # self.vis_noise_pose(x, w, h, str_img='x_q_sample')


        # diff_boxes = box_cxcywh_to_xyxy(x)
        diff_kpts = x  # XYXYXY
        # diff_boxes = inverse_sigmoid(x)

        return diff_kpts, noise, t

    def prepare_targets(self, args, opt, targets):
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
            gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes)  # xyxy
            gt_keypoints = targets_per_image['keypoints'] # XYXY... VVV
            gt_keypoints_buquan = targets_per_image['keypoints_buquan']  # XYXY... VVV 231015

            # #normalized by human boxes-------------------------------
            # gt_keypoints_norm = gt_keypoints.clone()
            # gt_keypoints_norm[:, :self.num_body_points*2][:, 0::2] = \
            #     (gt_keypoints[:, :self.num_body_points*2][:, 0::2] -
            #      gt_boxes_xyxy[:, 0].unsqueeze(1).repeat(1, self.num_body_points)) \
            #     / gt_boxes[:, 2].unsqueeze(1).repeat(1, self.num_body_points)
            # gt_keypoints_norm[:, :self.num_body_points * 2][:, 1::2] = \
            #     (gt_keypoints[:, :self.num_body_points * 2][:, 1::2] -
            #      gt_boxes_xyxy[:, 1].unsqueeze(1).repeat(1, self.num_body_points)) / \
            #     gt_boxes[:, 3].unsqueeze(1).repeat(1, self.num_body_points)
            #
            # gt_keypoints = gt_keypoints_norm.clone()
            # # normalized by human boxes-------------------------------

            a = 0



            # img_r = cv2.imread('/mnt/hdd/home/tandayi/data/crowdpose/images/' + str(100051) + '.jpg')
            # img_1k = np.zeros((1500, 1500, 3))
            # img_1k[:335, :500] = img_r
            # for tg in gt_keypoints:
            #     # img_1k = np.zeros((1500, 1500, 3))
            #     # img_1k[:h, :w] = img_r
            #     color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            #     kpts = tg
            #     kpts_x = kpts[:28][0::2]
            #     kpts_y = kpts[:28][1::2]
            #     kpts_v = kpts[28:]
            #     kpt_id = 0
            #     for (x, y, v) in zip(kpts_x, kpts_y, kpts_v):
            #         kpt_id += 1
            #         if v < 1:
            #             cv2.circle(img_1k, (int(x * w), int(y * h)), radius=10, color=color, thickness=-1)
            #             cv2.putText(img_1k, str(kpt_id), (int(x * w) + 10, int(y * h) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            #                         color, 4)
            #
            #         else:
            #             cv2.circle(img_1k, (int(x * w), int(y * h)), radius=5, color=color, thickness=-1)
            #             # cv2.putText(img_1k, str(kpt_id), (int(x) + 10, int(y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color,
            #             #             4)
            #
            # cv2.imwrite('/mnt/hdd/home/tandayi/code/2D HPE/ED-Pose/covar0/' + str(1) + '.jpg', img_1k)
            # c = 0
            d_kpts, d_noise, d_t = self.prepare_diffusion_concat(args, opt, gt_keypoints_buquan, w, h)

            # d_boxes, d_noise, d_t = self.prepare_diffusion_concat(args, opt, gt_boxes)  # gt_boxes cxcywh --->d_boxes: cxcywh

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

        offset = (pred[:self.num_body_points] - gt[:self.num_body_points])**2 + \
                 (pred[self.num_body_points:] - gt[self.num_body_points:])**2
        head_len = (pred[12] - pred[13]) ** 2 + (pred[26] - pred[27]) ** 2

        # return (offset ** 0.5 / (head_len ** 0.5 + 1e-5)).sum()
        return (offset ** 0.5).sum()

    def model_predictions(self, wh, mask_dict, srcs, masks, x, t, poss, input_query_label, attn_mask, attn_mask2,
                          time_cond, x_self_cond=None, clip_x_start=False):
        # self.scale = 1.5
        # yFile = '/root/code/ED-Pose10_coco_724/datasets/data.mat'
        # datay = sio.loadmat(yFile)
        # avg = torch.from_numpy(datay['avg_left_img'])
        # avg1 = avg.reshape(-1).to(self.device).to(torch.float32)  # xyxyxy...xy
        # sigma = torch.from_numpy(datay['covariance_left_img']).to(self.device).to(torch.float32)
        # x_boxes = torch.randn(1, 100, self.num_body_points * 2,
        #                                   device=self.device) @ sigma + avg1[None, :]
        # a = 0
        # x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_boxes = x
        x_boxes = ((x_boxes / self.scale) + 1) / 2

        # x_boxes_np = x_boxes.cpu().numpy()
        # kpts_x = x_boxes_np[:, :, 0::2]
        # kpts_y = x_boxes_np[:, :, 1::2]
        # wh_np = wh.cpu().numpy()
        #
        # img = np.zeros([1500, 1500, 3])
        # for kpt_x, kpt_y in zip(kpts_x[0], kpts_y[0]):
        #     for kpt_x0, kpt_y0 in zip(kpt_x, kpt_y):
        #         a  = int(kpt_x0 * wh_np[0][0])
        #         cv2.circle(img, (int(kpt_x0 * wh_np[0][0]), int(kpt_y0 * wh_np[0][1])), radius=3, color=(125, 89, 0), thickness=-1)
        #
        # cv2.imwrite('/mnt/hdd/home/tandayi/code/2D HPE/ED-Pose/vis_random/1/noise.jpg', img)



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
        # outputs_class, outputs_coord, outputs_kpt, outputs_kpt_bbox = self.head(backbone_feats, backbone_feats, x_boxes, t, None)
        outputs_coord = out['pred_boxes']
        kpts = out['pred_keypoints']  # xxx...yyy...zzz [bs 100 42] --->bs 1400 4
        # kpts = kpts.view(bs, -1, 3, self.num_body_points)  # bs 100 3 14
        # kpts_x, kpts_y = kpts[:, :, 0, :], kpts[:, :, 1, :]
        # keypoints_hw = outputs_keypoints_hw[-1].view(bs, -1, self.num_body_points, 2)  # bs 100 14 2
        # kpts_w, kpts_h = keypoints_hw[:, :, :, 0], keypoints_hw[:, :, :, 1]
        # kpts_box = torch.cat((kpts_x.unsqueeze(-1), kpts_y.unsqueeze(-1), kpts_w.unsqueeze(-1), kpts_h.unsqueeze(-1)),
        #                      dim=-1)
        # kpts_box = kpts_box.view(bs, -1, 4)
        x_start = kpts[:, :, :self.num_body_points*2]

        # 111111111111111111111111111111111111111111111111111111111

        # x_start = outputs_coord # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        # x_start = outputs_coord[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        # x_start = x_start / images_whwh[:, None, :]
        # x_start = box_xyxy_to_cxcywh(x_start)
        x_start = (x_start * 2 - 1.) * self.scale
        # x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
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

        results = []

        # if self.use_focal or self.use_fed_loss:
        if 1:
            scores = torch.sigmoid(box_cls)
            # scores = box_cls
            scores_np = scores.cpu().numpy()
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
                return torch.cat(box_pred_per_imager, dim=0), torch.cat(scores_per_imager, dim=0), torch.cat(scores_per_image2r, dim=0), torch.cat(labels_per_imager, dim=0), torch.cat(kpt_pred_per_image_onesr, dim=0)

                # if self.use_ensemble and self.sampling_timesteps > 1:
                #     return box_pred_per_image, scores_per_image, scores_per_image2, labels_per_image, kpt_pred_per_image_ones

        return torch.cat(box_pred_per_imager, dim=0), torch.cat(scores_per_imager, dim=0), torch.cat(scores_per_image2r, dim=0), torch.cat(labels_per_imager, dim=0), torch.cat(kpt_pred_per_image_onesr, dim=0)

    @torch.no_grad()
    def ddim_sample(self, wh, mask_dict, samples, srcs, masks, poss, input_query_label, attn_mask, attn_mask2,
                    clip_denoised=True, do_postprocess=True):

        batch = len(samples.mask)
        shape = (batch, self.num_proposals, self.num_body_points*2)
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times1 = torch.linspace(3, 10, 5)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # img = torch.randn(shape[0], shape[1] * 2, device=self.device)
        # yFile = '/mnt/hdd/home/tandayi/code/2D HPE/ED-Pose/covar/data.mat'
        # datay = sio.loadmat(yFile)
        # avg = datay['avg_left']
        # avgxy = avg.reshape(-1)  # xyxy...xyxy
        # imgx = img[:, 0::2].unsqueeze(-1).repeat(1, 1, 14)
        # imgy = img[:, 1::2].unsqueeze(-1).repeat(1, 1, 14)
        #
        # avg_pose_x = torch.from_numpy(avgxy[0::2]).unsqueeze(0).unsqueeze(0).repeat(shape[0], shape[1], 1)
        # avg_pose_y = torch.from_numpy(avgxy[1::2]).unsqueeze(0).unsqueeze(0).repeat(shape[0], shape[1], 1)
        # img0 = avg_pose_x.to(imgx.device) / (30)   + imgx # head_h = 30
        # img1 = avg_pose_y.to(imgx.device) / (30)   + imgy
        #
        # img = torch.zeros((batch, self.num_proposals, self.num_body_points * 2)).to(imgx.device)
        # img[:, :, 0::2] = img0
        # img[:, :, 1::2] = img1

        img = torch.randn(shape, device=self.device) * 1
        # img = torch.zeros(shape, device=self.device)

        # img1500 = np.zeros((samples.mask.size(1), samples.mask.size(2)))

        # for imx, imy in zip(img[:, :, 0::2], img[:, :, 1::2]):
        #     for ix, iy in zip(imx, imy):
        #         for x, y in zip(ix, iy):
        #             cv2.circle(img1500, (int((x) * samples.mask.size(1)), int((y)* samples.mask.size(2))), radius=5, color=(251,0, 125), thickness=-1)
        #
        # cv2.imwrite(
        #     '/mnt/hdd/home/tandayi/code/2D HPE/ED-Pose/covar/' + '2.jpg',
        #     img1500)

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
            keep_idx = None

            if self.box_renewal:  # filter
                score_per_image, box_per_image = outputs_class[0], outputs_coord[0]
                threshold = 0.1
                score_per_image = torch.sigmoid(score_per_image)
                value, _ = torch.max(score_per_image, -1, keepdim=False)
                keep_idx = value > threshold
                num_remain = torch.sum(keep_idx)

                pred_noise = pred_noise[:, keep_idx, :]
                x_start = x_start[:, keep_idx, :]
                img = img[:, keep_idx, :]
            if time_next < 0:
                img = x_start
                # ################Li-Qingyun
                # box_pred_per_image, scores_per_image, labels_per_image = self.inference(outputs_class[-1], outputs_coord[-1],images.image_sizes)
                # ensemble_score.append(scores_per_image)
                # ensemble_label.append(labels_per_image)
                # ensemble_coord.append(box_pred_per_image)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            # img = x_start # 723

            if self.box_renewal:  # filter
                # replenish with randn boxes
                box_0 = outputs_coord
                img = torch.cat(
                    (img, torch.randn(len(img), self.num_proposals - num_remain, self.num_body_points*2, device=img.device)),
                    dim=1)

            if self.use_ensemble and self.sampling_timesteps > 1:
                box_pred_per_image, scores_per_image, scores_per_image2, labels_per_image, kpt_pred_per_image_ones = self.inference(outputs_class,
                                                                                        outputs_coord,
                                                                                        outputs_kpts)
                # ensemble_score.append(outputs_class)
                # ensemble_coord.append(outputs_coord)
                # ensemble_kpts.append(outputs_kpts)
                # kpt_pred_per_image_ones = kpt_pred_per_image_ones.reshape(100, -1)
                # scores, topk_indices = scores_per_image.topk(self.num_proposals*2, sorted=True)
                bs = int(len(scores_per_image)/self.num_proposals)
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

            if not self.use_nms:
                box_pred_per_image_nms = box_pred_per_image * (wh.repeat(len(box_pred_per_image), 1))
                box_pred_per_image_nms = box_cxcywh_to_xyxy(box_pred_per_image_nms)
                keep = batched_nms(box_pred_per_image_nms, scores_per_image, labels_per_image, 0.95) # box_nms
                # scores_per_image = scores_per_image[keep]
                # scores_per_image2 = scores_per_image2[keep]
                # box_pred_per_image = box_pred_per_image[keep]
                # kpts_per_image = kpts_per_image[keep]


                w_kpts = wh[:, 0].repeat(self.num_body_points, 1)
                h_kpts = wh[:, 1].repeat(self.num_body_points, 1)
                wh_kpts = torch.cat((w_kpts.squeeze(1), h_kpts.squeeze(1)), dim=0)
                Z_pred = kpts_per_image[:, 0:(self.num_body_points * 2)] * wh_kpts[None, :]
                keep_index = self.nms_core(Z_pred, scores_per_image) # pose_nms

                scores_per_image = scores_per_image[keep_index]
                scores_per_image2 = scores_per_image2[keep_index]
                box_pred_per_image = box_pred_per_image[keep_index]
                kpts_per_image = kpts_per_image[keep_index]


            output = {'pred_logits': scores_per_image2.unsqueeze(0), 'pred_boxes': box_pred_per_image.unsqueeze(0), 'pred_keypoints': kpts_per_image.unsqueeze(0)
                      }
            results = output # step > 2
        else:
            box_pred_per_image, scores_per_image, scores_per_image2, labels_per_image, \
            kpt_pred_per_image_ones = self.inference(outputs_class,outputs_coord,outputs_kpts)
            kpt_pred_per_image_ones = kpt_pred_per_image_ones.reshape(-1, 3 * self.num_body_points)
            if not self.use_nms:
                box_pred_per_image_nms = box_pred_per_image * (wh.repeat(len(box_pred_per_image), 1))
                box_pred_per_image_nms = box_cxcywh_to_xyxy(box_pred_per_image_nms)
                keep = batched_nms(box_pred_per_image_nms, scores_per_image, labels_per_image, 0.95) # box_nms
                # scores_per_image = scores_per_image[keep]
                # scores_per_image2 = scores_per_image2[keep]
                # box_pred_per_image = box_pred_per_image[keep]
                # kpts_per_image = kpts_per_image[keep]


                w_kpts = wh[:, 0].repeat(self.num_body_points, 1)
                h_kpts = wh[:, 1].repeat(self.num_body_points, 1)
                wh_kpts = torch.cat((w_kpts.squeeze(1), h_kpts.squeeze(1)), dim=0)
                Z_pred = kpt_pred_per_image_ones[:, 0:(self.num_body_points * 2)] * wh_kpts[None, :]
                keep_index = self.nms_core(Z_pred, scores_per_image) # pose_nms

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
                outputs_kpts = outputs_kpts[:, :, :28].squeeze(0)


                # self.vis_noise_pose(outputs_kpts, torch.tensor(608).to(outputs_kpts.device), torch.tensor(912).to(outputs_kpts.device), str_img='output_100_kpts')

                # def vis_noise_pose(self, x_start, w, h, str_img):
                #     img = cv2.imread('/mnt/sde/tdy/crowdpose/images/118844.jpg')  # 118844
                #     x_boxes_np = x_start.cpu().numpy()
                #     kpts_x = x_boxes_np[:, 0::2]
                #     kpts_y = x_boxes_np[:, 1::2]
                #     # kpts_z = x_boxes_np[:, 1::2]
                #     w_np = w.cpu().numpy()
                #     h_np = h.cpu().numpy()
                #     img_ori = cv2.resize(img, (608, 912))
                #
                #     img = np.ones([1500, 1500, 3]) * 255
                #     img[0:912, 0:608, :] = img_ori
                #     flag = -1
                #     for kpt_x, kpt_y in zip(kpts_x, kpts_y):
                #         flag += 1
                #         if flag == 198 or flag == 199:
                #             for kpt_x0, kpt_y0 in zip(kpt_x, kpt_y):
                #                 cv2.circle(img, (int(kpt_x0 * w_np), int(kpt_y0 * h_np)), radius=8, color=(0, 255, 0),
                #                            thickness=-1)
                #         else:
                #             for kpt_x0, kpt_y0 in zip(kpt_x, kpt_y):
                #                 cv2.circle(img, (int(kpt_x0 * w_np), int(kpt_y0 * h_np)), radius=8,
                #                            color=(100, 30, 210),
                #                            thickness=-1)
                #
                #     cv2.rectangle(img, (int(0), int(0)), (int(w_np), int(h_np)), (225, 89, 110), thickness=3)
                #
                #     cv2.imwrite('/mnt/sdc/tandayi/vis/' + str_img + '.jpg', img)
                #









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

    def forward(self, args, opt, samples: NestedTensor, targets: List = None):
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
        img = samples.tensors.permute(0, 2, 3, 1).cpu().numpy()[0]
        w, h = samples.tensors.shape[2:]
        # cv2.imwrite('/mnt/sdc/tandayi/vis/0.jpg', img)

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
        self.training = args.is_train
        self.num_proposals = args.num_proposals
        self.device = args.device
        self.num_timesteps = args.timesteps
        self.sampling_timesteps = args.sampling_timesteps
        self.ddim_sampling_eta = args.ddim_sampling_eta
        self.self_condition = args.self_condition
        self.scale = args.scale
        self.sqrt_recip_alphas_cumprod = opt.sqrt_recip_alphas_cumprod
        self.sqrt_recipm1_alphas_cumprod = opt.sqrt_recipm1_alphas_cumprod
        self.box_renewal = True
        self.use_ensemble = True
        self.use_nms = True
        self.alphas_cumprod = opt.alphas_cumprod


        targets, x_boxes, noises, t = self.prepare_targets(args, opt, targets)  # xyxy...xy
        images_whwh = list()
        for bi in targets:
            h, w = bi["image_size_xyxy"][:2]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        # Prepare Proposals.
        if not self.training:
            results = self.ddim_sample(images_whwh, mask_dict, samples, srcs, masks, poss, input_query_label, attn_mask,
                                       attn_mask2)
            return results
        t = t.squeeze(-1)
        # x_boxes = torch.tensor(x_boxes, dtype=torch.float32)
        x_boxes = x_boxes.clone().detach().requires_grad_(True)

        if self.training:
            # hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(srcs, masks, input_query_bbox, t, poss, input_query_label, attn_mask,attn_mask2)
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


@MODULE_BUILD_FUNCS.registe_with_name(module_name='edpose')
def build_edpose(args):
    num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_class_embed_share = args.dec_pred_class_embed_share
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share

    model = EDPose(
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