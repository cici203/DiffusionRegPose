"""
COCO dataset which returns image_id for evaluation.
"""
import json
import os
import random
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision
from numpy.core.defchararray import array
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from util.box_ops import box_cxcywh_to_xyxy, box_iou
import datasets.transforms_coco as T
from datasets.data_util import preparing_dataset
__all__ = ['build']
from scipy import linalg
from scipy.io import savemat
import scipy.io as sio
import matplotlib.pyplot as plt

class CocoDetection(torch.utils.data.Dataset):
    def __init__(self, root_path, image_set, transforms, return_masks):
        super(CocoDetection, self).__init__()
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        if image_set == "train":
            self.img_folder = root_path / "images"/ "train2017"
            self.coco = COCO(root_path / "annotations/person_keypoints_train2017.json")
            imgIds = sorted(self.coco.getImgIds())
            self.all_imgIds = []
            covariance = []
            # r = random.randint(0, 10)
            for image_id in imgIds:
                if self.coco.getAnnIds(imgIds=image_id) == []:
                    continue
                ann_ids = self.coco.getAnnIds(imgIds=image_id)
                target = self.coco.loadAnns(ann_ids)
                num_keypoints = [obj["num_keypoints"] for obj in target]
                if sum(num_keypoints) == 0:
                    continue

                # if (image_id + 0) % 6 == 0:
                #     self.all_imgIds.append(image_id)
                self.all_imgIds.append(image_id)

            #     ##################统计协方差和均值
            #     # img = cv2.imread('/mnt/hdd/home/tandayi/data/crowdpose/images/' + str(image_id) + '.jpg')
            #     upper_index = [0, 1, 2, 3, 4, 5, 12, 13]
            #     under_index = [6, 7, 8, 9, 10, 11]
            #     # upper_index = [12, 13, 0,2,4] #left
            #     upper_index = [12, 13, 1, 3, 5]  # right
            #     upper_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ,15, 16]  # all
            #     under_index = [6, 7, 8, 9, 10, 11]
            #     for tg in target:
            #         # tg['bbox]: left_top_x, left_top_y, w, h
            #         box_w, box_h, box_left_top = tg['bbox'][2], tg['bbox'][3], tg['bbox'][:2]
            #         bbox_lt_x, bbox_lt_y = box_left_top[0], box_left_top[1]
            #         bbox_bd_x, bbox_bd_y = box_left_top[0] + box_w, box_left_top[1] + box_h
            #         # cv2.rectangle(img, (int(bbox_lt_x), int(bbox_lt_y)), (int(bbox_bd_x), int(bbox_bd_y)), (100, 0, 245), thickness=3)
            #         # kpts_x = np.array(tg['keypoints'][0::3])
            #         # kpts_y = np.array(tg['keypoints'][1::3])
            #         # for x, y in zip(kpts_x, kpts_y):
            #         #     cv2.circle(img, (int(x), int(y)), radius=3, color=(155, 0, 255), thickness=-1)
            #         #
            #         # cv2.imwrite('/mnt/hdd/home/tandayi/code/2D HPE/ED-Pose/covar/bbox.jpg', img)
            #         # w = self.coco.imgs[tg['image_id']]['width']
            #         # h = self.coco.imgs[tg['image_id']]['height']
            #         # normalize (kpts_x - box_left_top_x) / box_x, (kpts_y - box_left_top_y) / box_y
            #         kpts_x, kpts_y, kpts_v = [], [], []
            #         for i in range(17):
            #             if i in upper_index:
            #                 kpts_x.append(tg['keypoints'][0::3][i])
            #                 kpts_y.append(tg['keypoints'][1::3][i])
            #                 kpts_v.append(tg['keypoints'][2::3][i])
            #
            #         # kpts_vis = tg['keypoints'][2::3]
            #         minv = np.min(np.array(kpts_v))
            #         if minv == 0:
            #             continue
            #         else:
            #             head_h = ((np.array(kpts_x)[6] - np.array(kpts_x)[12]) ** 2 \
            #                       + (np.array(kpts_y)[6] - np.array(kpts_y)[
            #                         12]) ** 2) ** 0.5 #肩胯距离
            #             kpts_x_norm_box = (np.array(kpts_x) - bbox_lt_x) / np.array(
            #                 ([head_h for _ in range(len(upper_index))]))
            #             kpts_y_norm_box = (np.array(kpts_y) - bbox_lt_y) / np.array(
            #                 [head_h for _ in range(len(upper_index))])
            #             kpts_x, kpts_y = kpts_x_norm_box, kpts_y_norm_box
            #             flag = -1
            #             co = np.zeros((len(upper_index), 2))
            #
            #             kpts_mx, kpts_my = kpts_x.mean(), kpts_y.mean()  # mean
            #             kpts_rx, kpts_ry = kpts_x - kpts_mx, kpts_y - kpts_my  # relative
            #             for (x, y, v) in zip(kpts_x, kpts_y, kpts_v):
            #                 flag += 1
            #                 co[flag,] = np.append(x, y)
            #         covariance.append(co)
            #
            # covariance0 = np.array(covariance)
            # # cov = covariance0 > 0
            # avg_17_xy = np.mean(np.array(covariance), axis=0)  # (14, 2)
            #
            # covar_xyxy = np.array(covariance).reshape(-1, len(upper_index) * 2)  # xy...xy
            # covar_x = np.array(covariance)[:, :, 0]
            # covar_y = np.array(covariance)[:, :, 1]
            # covar_xxyy = np.stack([covar_x, covar_y], axis=1).reshape(-1, len(upper_index) * 2)  # xxxx...yyyyy
            # c = np.cov(covar_xyxy.T)
            # file_name = '/home/tandayi/code/2D_HPE/ED-Pose/data_coco.mat'
            # savemat(file_name, {'covariance_all_left': covar_xyxy, 'covariance_left': c, 'avg_left': avg_17_xy})
            # plt.matshow(c)
            # plt.savefig('/home/tandayi/code/2D_HPE/ED-Pose/sigma_crowdpose.jpg')
            # c = 0
        else:
            self.img_folder = root_path/ 'images' / "val2017"
            # self.img_folder = '/mnt/hdd/home/tandayi/coco-annotator/datasets/3'
            self.coco = COCO(root_path / "annotations/person_keypoints_val2017.json")
            imgIds = sorted(self.coco.getImgIds())
            self.all_imgIds = []
            for image_id in imgIds:

                self.all_imgIds.append(image_id)

    def __len__(self):
        # return 1
        return len(self.all_imgIds)



    def __getitem__(self, idx):
        # idx = 4052
        image_idx = idx
        image_id = self.all_imgIds[idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        target = self.coco.loadAnns(ann_ids)

        target = {'image_id': image_id, 'annotations': target}
        img = Image.open(self.img_folder/ self.coco.loadImgs(image_id)[0]['file_name'])
        # img = Image.open(self.img_folder / '000000000139.jpg')

        # buquan

        # h, w = img.height, img.width
        tgs = target['annotations']
        # upper_index = [12, 13, 0, 2, 4]
        # upper_index = [12, 13, 1, 3, 5]
        upper_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # all
        # under_index = [6, 7, 8, 9, 10, 11]
        # img_r = cv2.imread('/mnt/hdd/home/tandayi/data/crowdpose/images/' + str(image_id) + '.jpg')
        # img_1k = np.zeros((1200, 1200, 3))
        # img_1k[:h, :w, :] = img_r

        for itg, tg in enumerate(tgs):
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

            kpts_x, kpts_y, kpts_v = [], [], []
            for i in range(17):
                if i in upper_index:
                    kpts_x.append(tg['keypoints'][0::3][i])
                    kpts_y.append(tg['keypoints'][1::3][i])
                    kpts_v.append(tg['keypoints'][2::3][i])
            if not (kpts_x[6] and kpts_x[12] and kpts_y[6] and kpts_y[12]):
                target['annotations'][itg]['mean_keypoints'] = target['annotations'][itg]['keypoints']
                target['annotations'][itg]['keypoints_buquan'] = target['annotations'][itg]['keypoints']
                continue

            head_h = ((np.array(kpts_x)[6] - np.array(kpts_x)[12]) ** 2 \
                      + (np.array(kpts_y)[6] - np.array(kpts_y)[
                        12]) ** 2) ** 0.5

            yFile = '/home/tandayi/code/2D_HPE/ED-Pose/data_coco.mat'
            datay = sio.loadmat(yFile)
            sigam = datay['covariance_left']
            avg = datay['avg_left']
            avgxy = avg.reshape(-1)  # xyxy...xyxy
            covar_all = datay['covariance_all_left']
            # plt.matshow(sigam)
            # plt.savefig('/mnt/hdd/home/tandayi/code/2D HPE/ED-Pose/covar/sigma.jpg')

            # tg['bbox]: left_top_x, left_top_y, w, h
            box_w, box_h, box_left_top = tg['bbox'][2], tg['bbox'][3], tg['bbox'][:2]
            bbox_lt_x, bbox_lt_y = box_left_top[0], box_left_top[1]
            bbox_bd_x, bbox_bd_y = box_left_top[0] + box_w, box_left_top[1] + box_h

            kpts_x_norm_box = (np.array(kpts_x) - bbox_lt_x) / np.array(([head_h for _ in range(len(upper_index))]))
            kpts_y_norm_box = (np.array(kpts_y) - bbox_lt_y) / np.array([head_h for _ in range(len(upper_index))])
            num_valid = (np.array(kpts_v) > 0).sum()
            # kpt_x = np.array(kpt['keypoints'][0::3]) / w
            # kpt_y = np.array(kpt['keypoints'][1::3]) / h

            # kpts_mx_norm_box = np.sum(kpts_x_norm_box * (np.array(kpts_v) > 0)/num_valid)
            # kpts_my_norm_box = np.sum(kpts_y_norm_box * (np.array(kpts_v) > 0) / num_valid)
            # kpt_x, kpt_y = kpts_x_norm_box - kpts_mx_norm_box, kpts_y_norm_box - kpts_my_norm_box
            kpt_x, kpt_y = kpts_x_norm_box, kpts_y_norm_box
            kpt_z = np.array(kpts_v)
            idx = np.argsort(kpt_z)
            sort_x = kpt_x[np.argsort(kpt_z)]
            sort_y = kpt_y[np.argsort(kpt_z)]

            # recovery_arr = np.zeros_like(sort_x)
            # for i, num in enumerate(sort_x):
            #     recovery_arr[idx[i]] = num

            N = np.sum(kpt_z == 0)

            if N == 0:
                target['annotations'][itg]['mean_keypoints'] = target['annotations'][itg]['keypoints']
                target['annotations'][itg]['keypoints_buquan'] = target['annotations'][itg]['keypoints']
                # for x, y in zip(sort_x, sort_y):
                #     cv2.circle(img_1k, (int(x * head_h + bbox_lt_x), int(y * head_h + bbox_lt_y)), radius=5, color=color, thickness=-1)
                continue
            else:
                index = np.zeros(len(upper_index) * 2).astype('int64')
                for i, ind in enumerate(idx):
                    index[i * 2] = ind * 2
                    index[i * 2 + 1] = ind * 2 + 1

                covar_all_new = covar_all[:, index]
                avgxy_new = avgxy[index]
                covar_all_newT = np.cov(covar_all_new[:].T)  # 28 28
                covar_all_newT = np.linalg.pinv(covar_all_newT)
                L = np.linalg.cholesky(covar_all_newT)
                test = L @ L.T

                xy_know = np.stack((sort_x, sort_y), axis=1).reshape(-1)

                B = L.T
                B0 = B[:, :N * 2]
                B1 = B[:, N * 2:]
                AA = B0
                b = B @ avgxy_new - B1 @ xy_know[N * 2:]
                x_unknow = np.linalg.inv(AA.T @ AA) @ AA.T @ b
                x_unknow = np.linalg.lstsq(AA, b, rcond=None)[0]
                X0_f = x_unknow

                x_recon, y_recon = X0_f[0::2], X0_f[1::2]

                # img_1k[:h, :w, :] = img_r
                # cv2.rectangle(img_1k, (int(bbox_lt_x), int(bbox_lt_y)), (int(bbox_bd_x), int(bbox_bd_y)), color,
                #               thickness=3)
                # for x, y in zip(sort_x, sort_y):
                #     cv2.circle(img_1k, (int(x * head_h + bbox_lt_x), int(y * head_h + bbox_lt_y)), radius=5,
                #                color=color, thickness=-1)
                #
                # for x, y in zip(x_recon, y_recon):
                #     cv2.circle(img_1k, (int(x * head_h + bbox_lt_x), int(y * head_h + bbox_lt_y)), radius=10,
                #                color=color, thickness=-1)

                c = 0
                re_sort_x = sort_x * head_h + bbox_lt_x
                re_sort_x[:len(x_recon)] = x_recon * head_h + bbox_lt_x
                re_sort_y = sort_y * head_h + bbox_lt_y
                re_sort_y[:len(y_recon)] = y_recon * head_h + bbox_lt_y

                # for x, y in zip(re_sort_x, re_sort_y):
                #     cv2.circle(img_1k, (int(x), int(y)), radius=5, color=color, thickness=-1)
                # cv2.imwrite(
                #     '/mnt/hdd/home/tandayi/code/2D HPE/ED-Pose/covar/' + str(image_idx) + '_+' + str(itg) + '.jpg',
                #     img_1k)

                avg_pose_x = avgxy_new[0::2] * head_h + bbox_lt_x
                avg_pose_y = avgxy_new[1::2] * head_h + bbox_lt_y

                # for x, y in zip(avg_pose_x, avg_pose_y):
                #     cv2.circle(img_1k, (int(x), int(y)), radius=5, color=color, thickness=-1)
                # cv2.imwrite('/mnt/hdd/home/tandayi/code/2D HPE/ED-Pose/covar/' + str(image_idx)  + '_' + str(itg) +  '.jpg', img_1k)

                recovery_arr_x = np.zeros_like(re_sort_x)
                for i, num in enumerate(re_sort_x):
                    recovery_arr_x[idx[i]] = num

                recovery_arr_y = np.zeros_like(re_sort_y)
                for i, num in enumerate(re_sort_y):
                    recovery_arr_y[idx[i]] = num

                recovery_mean_x = np.zeros_like(avg_pose_x)
                for i, num in enumerate(avg_pose_x):
                    recovery_mean_x[idx[i]] = num

                recovery_mean_y = np.zeros_like(avg_pose_y)
                for i, num in enumerate(avg_pose_y):
                    recovery_mean_y[idx[i]] = num

                re_kpts = np.zeros((17 * 3)).astype('int64')
                re_kpts[0::3] = recovery_arr_x.astype('int64')
                re_kpts[1::3] = recovery_arr_y.astype('int64')
                re_kpts[2::3] = kpt_z.astype('int64')

                re_kpts_mean = np.zeros((17 * 3)).astype('int64')
                re_kpts_mean[0::3] = recovery_mean_x.astype('int64')
                re_kpts_mean[1::3] = recovery_mean_y.astype('int64')
                re_kpts_mean[2::3] = kpt_z.astype('int64')

                # buquan
                # target['annotations'][itg]['keypoints_buquan'] = list(re_kpts)
                target['annotations'][itg]['keypoints_buquan'] = target['annotations'][itg]['keypoints']
                target['annotations'][itg]['mean_keypoints'] = list(re_kpts_mean)
                a = 0

        # tgs = target['annotations']
        # image_idx_0 = str(image_id).zfill(12)
        # img_r = cv2.imread('/mnt/sde/tdy/coco_img/images/train2017/' + image_idx_0 + '.jpg')
        # img_1k = np.zeros((1500, 1500, 3))
        # img_1k[:h, :w] = img_r
        # for tg in tgs:
        #     # img_1k = np.zeros((1500, 1500, 3))
        #     # img_1k[:h, :w] = img_r
        #     color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        #     kpts = tg['keypoints_buquan']
        #     # kpts = tg['mean_keypoints']
        #     kpts_x = kpts[0::3]
        #     kpts_y = kpts[1::3]
        #     kpts_v = kpts[2::3]
        #     kpt_id = 0
        #     for (x, y, v) in zip(kpts_x, kpts_y, kpts_v):
        #         kpt_id += 1
        #         if v < 1:
        #             cv2.circle(img_1k, (int(x), int(y)), radius=10, color=color, thickness=-1)
        #             cv2.putText(img_1k, str(kpt_id), (int(x) + 10, int(y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color,
        #                         4)
        #
        #         else:
        #             cv2.circle(img_1k, (int(x), int(y)), radius=5, color=color, thickness=-1)
        #             # cv2.putText(img_1k, str(kpt_id), (int(x) + 10, int(y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color,
        #             #             4)
        #
        # cv2.imwrite('/home/tandayi/code/2D_HPE/ED-Pose/' + str(image_idx) + '.jpg', img_1k)
        # c = 0


        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(img_array)
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        anno = [obj for obj in anno if obj['num_keypoints'] != 0]
        keypoints = [obj["keypoints"] for obj in anno]
        keypoints_buquan = [obj["keypoints_buquan"] for obj in anno]
        boxes = [obj["bbox"] for obj in anno]
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32).reshape(-1, 17, 3)
        keypoints_buquan = torch.as_tensor(keypoints_buquan, dtype=torch.float32).reshape(-1, 17, 3)
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        keypoints = keypoints[keep]
        keypoints_buquan = keypoints_buquan[keep]
        if self.return_masks:
            masks = masks[keep]
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints
            target["keypoints_buquan"] = keypoints_buquan
        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return image, target


def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]

    # update args from config files
    scales = getattr(args, 'data_aug_scales', scales)
    max_size = getattr(args, 'data_aug_max_size', max_size)
    scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
    scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

    # resize them
    data_aug_scale_overlap = getattr(args, 'data_aug_scale_overlap', None)
    if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
        data_aug_scale_overlap = float(data_aug_scale_overlap)
        scales = [int(i * data_aug_scale_overlap) for i in scales]
        max_size = int(max_size * data_aug_scale_overlap)
        scales2_resize = [int(i * data_aug_scale_overlap) for i in scales2_resize]
        scales2_crop = [int(i * data_aug_scale_overlap) for i in scales2_crop]
    datadict_for_print = {
        'scales': scales,
        'max_size': max_size,
        'scales2_resize': scales2_resize,
        'scales2_crop': scales2_crop
    }
    if image_set == 'train':
        if fix_size:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize([(max_size, max(scales))]),
                normalize,
            ])

        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(*scales2_crop),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])

    if image_set in ['val', 'test']:


        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    mode = 'person_keypoints'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "test": (root / "test2017", root / "annotations" / 'image_info_test-dev2017.json'),
    }
    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(root, image_set, transforms=make_coco_transforms(image_set, strong_aug=args.strong_aug),
                            return_masks=args.masks)

    return dataset

