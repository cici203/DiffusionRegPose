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
            self.img_folder = root_path / "images"/ "images"
            self.coco = COCO(root_path / "annotations/ochuman_coco_format_val_range_0.00_1.00.json")
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
            #
            #     for tg in target:
            #         w = self.coco.imgs[tg['image_id']]['width']
            #         h = self.coco.imgs[tg['image_id']]['height']
            #         kpts_x = np.array(tg['keypoints'][0::3]) / np.array(([w for _ in range(17)]))
            #         kpts_y = np.array(tg['keypoints'][1::3]) / np.array([h for _ in range(17)])
            #         kpts_vis = tg['keypoints'][2::3]
            #         minv = np.min(np.array(kpts_vis))
            #         if minv == 0:
            #             # continue
            #             flag = -1
            #             co = np.zeros((17, 2))
            #             for (x, y, v) in zip(kpts_x, kpts_y, kpts_vis):
            #                 flag += 1
            #                 co[flag,] = np.append(x, y)
            #         else:
            #             flag = -1
            #             co = np.zeros((17, 2))
            #             for (x, y, v) in zip(kpts_x, kpts_y, kpts_vis):
            #                 flag += 1
            #                 co[flag,] = np.append(x, y)
            #         covariance.append(co)
            #
            # covariance0 = np.array(covariance)
            # cov = covariance0 > 0
            # avg_17_xy = np.sum(covariance0, axis=0) / np.sum(cov, axis=0)  # (14, 2)
            #
            # covar0 = np.array(covariance).reshape(-1, 34)  # xy...xy
            # covar_x = np.array(covariance)[:, :, 0]
            # covar_y = np.array(covariance)[:, :, 1]
            # covar = np.stack([covar_x,covar_y],axis=1).reshape(-1, 34)
            # c = np.cov(covar.T)
            # file_name = '/mnt/hdd/home/tandayi/code/2D HPE/ED-Pose/covar/data.mat'
            # savemat(file_name, {'covariance': c, 'avg': avg_17_xy})
            # plt.matshow(c)
            # plt.savefig('/mnt/hdd/home/tandayi/code/2D HPE/ED-Pose/covar/sigma0.jpg')
        else:
            self.img_folder = root_path/ 'images' / "images"
            # self.img_folder = '/mnt/hdd/home/tandayi/coco-annotator/datasets/3'
            self.coco = COCO(root_path / "annotations/ochuman_coco_format_val_range_0.00_1.00.json")
            imgIds = sorted(self.coco.getImgIds())
            self.all_imgIds = []
            for image_id in imgIds:

                self.all_imgIds.append(image_id)

    def __len__(self):
        # return 1
        return len(self.all_imgIds)



    def __getitem__(self, idx):
        image_id = self.all_imgIds[idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        target = self.coco.loadAnns(ann_ids)

        target = {'image_id': image_id, 'annotations': target}
        img = Image.open(self.img_folder/ self.coco.loadImgs(image_id)[0]['file_name'])
        # img = Image.open(self.img_folder / '000000000139.jpg')
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
        boxes = [obj["bbox"] for obj in anno]
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32).reshape(-1, 17, 3)
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

