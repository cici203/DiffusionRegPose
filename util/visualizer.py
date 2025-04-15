import os, sys
from textwrap import wrap
import torch
import numpy as np
import cv2
import datetime

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from pycocotools import mask as maskUtils
from matplotlib import transforms

from util.utils import renorm
from models.diffusionregpose.utils import OKSLoss


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


class ColorMap():
    def __init__(self, basergb=[255, 255, 0]):
        self.basergb = np.array(basergb)

    def __call__(self, attnmap):
        assert attnmap.dtype == np.uint8
        h, w = attnmap.shape
        res = self.basergb.copy()
        res = res[None][None].repeat(h, 0).repeat(w, 1)  # h, w, 3
        attn1 = attnmap.copy()[..., None]  # h, w, 1
        res = np.concatenate((res, attn1), axis=-1).astype(np.uint8)
        return res


def rainbow_text(x, y, ls, lc, **kw):
    """
    Take a list of strings ``ls`` and colors ``lc`` and place them next to each
    other, with text ls[i] being shown in color lc[i].

    This example shows how to do both vertical and horizontal text, and will
    pass all keyword arguments to plt.text, so you can set the font size,
    family, etc.
    """
    t = plt.gca().transData
    fig = plt.gcf()
    plt.show()

    # horizontal version
    for s, c in zip(ls, lc):
        text = plt.text(x, y, " " + s + " ", color=c, transform=t, **kw)
        text.draw(fig.canvas.get_renderer())
        ex = text.get_window_extent()
        t = transforms.offset_copy(text._transform, x=ex.width, units='dots')


class COCOVisualizer():
    def __init__(self, coco=None, tokenlizer=None) -> None:
        from pycocotools.coco import COCO
        # json_file = r'/mnt/sde/tdy/crowdpose/json/crowdpose_test.json'
        json_file = r'/mnt/sde/tdy/crowdpose/json/crowdpose_trainval.json'
        coco = COCO(json_file)
        self.coco = coco
        self.num_kpts = 14
        self.thr = 0.1
        self.oks = OKSLoss(linear=True,
                           num_keypoints=self.num_kpts,
                           eps=1e-6,
                           reduction='mean',
                           loss_weight=1.0)

    def visualize(self, img, tgt, caption=None, dpi=180, savedir='vis'):
        """
        img: tensor(3, H, W)
        tgt: make sure they are all on cpu.
            must have items: 'image_id', 'boxes', 'size'
        """
        imgIds = self.coco.getImgIds(imgIds=[int(tgt['image_id'])])
        img_read = cv2.imread('/mnt/sde/tdy/crowdpose/images/' + str(imgIds[0]) + '.jpg')
        img_resize = img_read
        img = torch.from_numpy(img_resize).permute(2, 0, 1)
        plt.figure(dpi=dpi)
        plt.rcParams['font.size'] = '5'
        img = renorm(img).permute(1, 2, 0)
        self.addtgt(tgt, img.numpy())

    def addtgt(self, tgt, img):

        assert 'boxes' in tgt
        ax = plt.gca()
        H, W = tgt['size'].tolist()
        numbox = tgt['boxes'].shape[0]
        color_kpt = [[0.00, 0.00, 0.00],
                     [1.00, 1.00, 1.00],
                     [1.00, 0.00, 0.00],
                     [1.00, 1, 00., 0.00],
                     [0.50, 0.16, 0.16],
                     [0.00, 0.00, 1.00],
                     [0.69, 0.88, 0.90],
                     [0.00, 1.00, 0.00],
                     [0.63, 0.13, 0.94],
                     [0.82, 0.71, 0.55],
                     [1.00, 0.38, 0.00],
                     [0.53, 0.15, 0.34],
                     [1.00, 0.39, 0.28],
                     [1.00, 0.00, 1.00],
                     [0.04, 0.09, 0.27],
                     [0.20, 0.63, 0.79],
                     [0.94, 0.90, 0.55]]

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
        color = []
        polygons = []
        boxes = []
        len_valid = 0
        sks = [[12, 13], [1, 13], [0, 13], [0, 2], [2, 4], [1, 3], [3, 5], [13, 7], [13, 6], [7, 9], [9, 11], [6, 8],
               [8, 10]]
        h, w = int(tgt['gt_ori_size'][0].cpu()), int(tgt['gt_ori_size'][1].cpu())
        c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        for box_gt, gt_kpts in zip(tgt['gt_boxes'].cpu(), tgt['gt_kpts'].cpu()):
            gt_box = box_cxcywh_to_xyxy(box_gt)
            gt_box = gt_box * torch.Tensor([w, h, w, h])  # cxcxwh
            gt_kpts[: self.num_kpts * 2] = gt_kpts[: self.num_kpts * 2] * (
                torch.Tensor([w, h]).repeat(self.num_kpts))  # cxcxwh
            gt_kpts_x = gt_kpts[:self.num_kpts * 2][0: self.num_kpts * 2: 2].numpy()
            gt_kpts_y = gt_kpts[:self.num_kpts * 2][1: self.num_kpts * 2: 2].numpy()
            gt_kpts_v = gt_kpts[self.num_kpts * 2:].numpy()
            i = 0
            for x, y, v in zip(gt_kpts_x, gt_kpts_y, gt_kpts_v):
                if v == 0:
                    continue
                else:
                    i += 1
                    c_kpt = [color_kpt[i][0] * 255, color_kpt[i][1] * 255, color_kpt[i][2] * 255]
                    cv2.circle(img, (int(x), int(y)), radius=3, color=c_kpt, thickness=-1)
            for sk in sks:
                if np.all(gt_kpts_v[sk] > 0):
                    color_line = [0, 255, 0]
                    cv2.line(img, (int(gt_kpts_x[sk[0]]), int(gt_kpts_y[sk[0]])),
                             (int(gt_kpts_x[sk[1]]), int(gt_kpts_y[sk[1]])),
                             color=color_line, thickness=2)

            box_np = gt_box.numpy()
            cv2.rectangle(img, ((int(box_np[0])), int(box_np[1])), ((int(box_np[2])), int(box_np[3])), (0, 255, 0), 2)

        for box, score in zip(tgt['boxes'].cpu(), tgt['scores'].cpu()):
            if score < torch.tensor(self.thr):
                continue
            len_valid += 1
            box_np = box.numpy()
            unnormbbox = box * torch.Tensor([W, H, W, H])
            unnormbbox[:2] -= unnormbbox[2:] / 2
            [bbox_x, bbox_y, bbox_w, bbox_h] = unnormbbox.tolist()
            boxes.append([bbox_x, bbox_y, bbox_w, bbox_h])
            cv2.rectangle(img, (int(box_np[0]), int(box_np[1])), (int(box_np[2]), int(box_np[3])), (0, 0, 255), 2)
            poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h],
                    [bbox_x + bbox_w, bbox_y]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons.append(Polygon(np_poly))
            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            color.append(c)

        p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.1)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)

        if 'strings_positive' in tgt:
            assert len(tgt['strings_positive']) == numbox, f"{len(tgt['strings_positive'])} = {numbox}, "
            for idx, strlist in enumerate(tgt['strings_positive']):
                cate_id = int(tgt['labels'][idx])
                _string = str(cate_id) + ':' + ' '.join(strlist)
                bbox_x, bbox_y, bbox_w, bbox_h = boxes[idx]
                ax.text(bbox_x, bbox_y, _string, color='black', bbox={'facecolor': color[idx], 'alpha': 0.6, 'pad': 1})

        if 'box_label' in tgt:
            assert len(tgt['box_label']) == numbox, f"{len(tgt['box_label'])} = {numbox}, "
            for idx, bl in enumerate(tgt['box_label']):
                _string = str(bl)
                bbox_x, bbox_y, bbox_w, bbox_h = boxes[idx]
                ax.text(bbox_x, bbox_y, _string, color='black', bbox={'facecolor': color[idx], 'alpha': 0.6, 'pad': 1})

        if 'caption' in tgt:
            ax.set_title(tgt['caption'], wrap=True)

        if 'attn' in tgt:
            if isinstance(tgt['attn'], tuple):
                tgt['attn'] = [tgt['attn']]
            for item in tgt['attn']:
                attn_map, basergb = item
                attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-3)
                attn_map = (attn_map * 255).astype(np.uint8)
                cm = ColorMap(basergb)
                heatmap = cm(attn_map)
                ax.imshow(heatmap)

        # calculate the oks
        oks = np.zeros((len(tgt['keypoints']), len(tgt['gt_kpts'])))
        for i, Z_pred in enumerate(tgt['keypoints']):
            x_p = Z_pred[0: self.num_kpts * 3: 3].cpu() / (torch.tensor(w).repeat(self.num_kpts))
            y_p = Z_pred[1: self.num_kpts * 3: 3].cpu() / (torch.tensor(h).repeat(self.num_kpts))
            j = -1
            for Z_gt, targets_area in zip(tgt['gt_kpts'], tgt['gt_area']):
                j += 1
                z_gt = Z_gt[: self.num_kpts * 2].cpu()  # xyxy...xy
                v_gt = Z_gt[self.num_kpts * 2:].cpu()
                z_pred = torch.zeros_like(z_gt)
                z_pred[0: self.num_kpts * 2: 2] = x_p
                z_pred[1: self.num_kpts * 2: 2] = y_p
                oks[i, j] = 1 - self.oks(z_pred.unsqueeze(0), z_gt.unsqueeze(0), v_gt.unsqueeze(0),
                                         targets_area.unsqueeze(0).cpu(), weight=None, avg_factor=None,
                                         reduction_override=None)
        oks_max = np.max(oks, axis=1)
        oks_max_0 = np.max(oks, axis=0)
        oks_max_0 = round(np.mean(oks_max_0), 5)

        if 'keypoints' in tgt:
            # turn skeleton into zero-based index
            sks = [[12, 13], [1, 13], [0, 13], [0, 2], [2, 4], [1, 3], [3, 5], [13, 7], [13, 6], [7, 9], [9, 11],
                   [6, 8], [8, 10]]
            for box, ann, score, oks in zip(tgt['boxes'], tgt['keypoints'], tgt['scores'], oks_max):
                if score < torch.tensor(self.thr):
                    continue
                box_np = box.cpu().numpy()
                kp = np.array(ann.cpu())
                x = kp[0: 42: 3]
                y = kp[1: 42: 3]
                v = kp[2: 42: 3]
                for i in range(14):
                    c_kpt = [color_kpt[i][0] * 255, color_kpt[i][1] * 255, color_kpt[i][2] * 255]
                    cv2.circle(img, (int(x[i]), int(y[i])), radius=3, color=c_kpt, thickness=-1)
                for si, sk in enumerate(sks):
                    if np.all(v[sk] > 0):
                        color_line = [color_bone[si][0] * 255, color_bone[si][1] * 255, color_bone[si][2] * 255]
                        cv2.line(img, (int(x[sk[0]]), int(y[sk[0]])), (int(x[sk[1]]), int(y[sk[1]])), color=color_line,
                                 thickness=5)
                        plt.plot(x[sk],y[sk], linewidth=1, color=c)

                cv2.putText(img, 'o:' + str(round(oks, 2)), (int(box_np[0]), int(box_np[1]) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
            cv2.imwrite('/mnt/sdc/tandayi/vis/vis_gt_pred_train/' + str(
                int(tgt['image_id'])) + '-avg_gt_oks:' + str(round(oks_max_0, 4)) + '_022diff' + '.jpg', img)
        ax.set_axis_off()

    def showAnns(self, anns, draw_bbox=False):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
            datasetType = 'instances'
        elif 'caption' in anns[0]:
            datasetType = 'captions'
        else:
            raise Exception('datasetType not supported')
        if datasetType == 'instances':
            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            for ann in anns:
                c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
                if 'segmentation' in ann:
                    if type(ann['segmentation']) == list:
                        # polygon
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                            polygons.append(Polygon(poly))
                            color.append(c)
                    else:
                        # mask
                        t = self.imgs[ann['image_id']]
                        if type(ann['segmentation']['counts']) == list:
                            rle = maskUtils.frPyObjects([ann['segmentation']], t['height'], t['width'])
                        else:
                            rle = [ann['segmentation']]
                        m = maskUtils.decode(rle)
                        img = np.ones((m.shape[0], m.shape[1], 3))
                        if ann['iscrowd'] == 1:
                            color_mask = np.array([2.0, 166.0, 101.0]) / 255
                        if ann['iscrowd'] == 0:
                            color_mask = np.random.random((1, 3)).tolist()[0]
                        for i in range(3):
                            img[:, :, i] = color_mask[i]
                        ax.imshow(np.dstack((img, m * 0.5)))

                if 'keypoints' in ann and type(ann['keypoints']) == list:
                    # turn skeleton into zero-based index
                    sks = np.array(self.loadCats(ann['category_id'])[0]['skeleton']) - 1
                    kp = np.array(ann['keypoints'])
                    x = kp[0::3]
                    y = kp[1::3]
                    v = kp[2::3]
                    for sk in sks:
                        if np.all(v[sk] > 0):
                            plt.plot(x[sk], y[sk], linewidth=3, color=c)
                    plt.plot(x[v > 0], y[v > 0], 'o', markersize=8, markerfacecolor=c, markeredgecolor='k',
                             markeredgewidth=2)
                    plt.plot(x[v > 1], y[v > 1], 'o', markersize=8, markerfacecolor=c, markeredgecolor=c,
                             markeredgewidth=2)

                if draw_bbox:
                    [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                    poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h],
                            [bbox_x + bbox_w, bbox_y]]
                    np_poly = np.array(poly).reshape((4, 2))
                    polygons.append(Polygon(np_poly))
                    color.append(c)

            p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
            ax.add_collection(p)
        elif datasetType == 'captions':
            for ann in anns:
                print(ann['caption'])
