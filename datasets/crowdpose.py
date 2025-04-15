import json
import os
from pathlib import Path
import cv2
import numpy as np
import copy
import torch
import torch.utils.data
from PIL import Image
from crowdposetools.coco import COCO
import datasets.transforms_crowdpose as T
from scipy.io import savemat
import scipy.io as sio
__all__ = ['build']

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

class CocoDetection(torch.utils.data.Dataset):
    def __init__(self, root_path, image_set, transforms, return_masks, is_completion):
        super(CocoDetection, self).__init__()
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.is_completion = is_completion
        self.root_path = root_path
        if image_set == "train":
            self.img_folder = root_path / "images"
            self.coco = COCO(root_path / "json/crowdpose_trainval.json")
            imgIds = sorted(self.coco.getImgIds())
            self.all_imgIds = []
            covariance = []
            for image_id in imgIds:
                if self.coco.getAnnIds(imgIds=image_id) == []:
                    continue
                ann_ids = self.coco.getAnnIds(imgIds=image_id)
                target = self.coco.loadAnns(ann_ids)
                num_keypoints = [obj["num_keypoints"] for obj in target]
                if sum(num_keypoints) == 0:
                    continue
                self.all_imgIds.append(image_id)
                 #---------------------------Statistical covariance and mean---------------------------#
                kpts_index = list(range(14)) # all index
                for tg in target:
                    box_w, box_h, box_left_top = tg['bbox'][2], tg['bbox'][3], tg['bbox'][:2]
                    bbox_lt_x, bbox_lt_y = box_left_top[0], box_left_top[1]
                    kpts_x, kpts_y, kpts_v = [], [], []
                    for i in range(14):
                        if i in kpts_index:
                            kpts_x.append(tg['keypoints'][0::3][i])
                            kpts_y.append(tg['keypoints'][1::3][i])
                            kpts_v.append(tg['keypoints'][2::3][i])
                    minv = np.min(np.array(kpts_v))
                    if minv == 0:
                        continue
                    else:
                        head_h = ((np.array(kpts_x)[-1] - np.array(kpts_x)[-2]) ** 2 \
                                  + (np.array(kpts_y)[-1] - np.array(kpts_y)[
                                    -2]) ** 2) ** 0.5
                        kpts_x_norm_box = (np.array(kpts_x) - bbox_lt_x) / np.array(
                            ([head_h for _ in range(len(kpts_index))]))
                        kpts_y_norm_box = (np.array(kpts_y) - bbox_lt_y) / np.array(
                            [head_h for _ in range(len(kpts_index))])
                        kpts_x, kpts_y = kpts_x_norm_box, kpts_y_norm_box
                        flag = -1
                        co = np.zeros((len(kpts_index), 2))

                        for (x, y, v) in zip(kpts_x, kpts_y, kpts_v):
                            flag += 1
                            co[flag,] = np.append(x, y)
                    covariance.append(co)

            avg_14_xy = np.mean(np.array(covariance), axis=0) # (14, 2)

            covar_xyxy = np.array(covariance).reshape(-1, len(kpts_index)*2)  # xy...xy
            c = np.cov(covar_xyxy.T)
            file_name = os.path.join(root_path, "data.mat")
            savemat(file_name, {'covariance_all_left': covar_xyxy, 'covariance_left': c, 'avg_left': avg_14_xy})

        else:
            self.img_folder = root_path / "images"
            self.coco = COCO(root_path / "json/crowdpose_test.json")
            imgIds = sorted(self.coco.getImgIds())
            self.all_imgIds = []
            for image_id in imgIds:
                self.all_imgIds.append(image_id)

    def kpts_completion(self, target, img):
        tgs = target['annotations']
        kpts_index = list(range(14)) # all index
        for itg, tg in enumerate(tgs):

            kpts_x, kpts_y, kpts_v = [], [], []
            for i in range(14):
                if i in kpts_index:
                    kpts_x.append(tg['keypoints'][0::3][i])
                    kpts_y.append(tg['keypoints'][1::3][i])
                    kpts_v.append(tg['keypoints'][2::3][i])
            if not (kpts_x[-1] and kpts_x[-2] and kpts_y[-1] and kpts_y[-2]):
                target['annotations'][itg]['mean_keypoints'] = target['annotations'][itg]['keypoints']
                target['annotations'][itg]['keypoints_buquan'] = target['annotations'][itg]['keypoints']
                x = np.array(target['annotations'][itg]['keypoints'][0::3])
                y = np.array(target['annotations'][itg]['keypoints'][1::3])
                z = np.array(target['annotations'][itg]['keypoints'][2::3])

                re_kpts_ratio = np.zeros((14 * 3))
                re_kpts_ratio[0::3] = x / img.width
                re_kpts_ratio[1::3] = y / img.height
                re_kpts_ratio[2::3] = z.astype('int64')

                target['annotations'][itg]['keypoints_ratio'] = list(re_kpts_ratio)
                continue

            head_h = ((np.array(kpts_x)[-1] - np.array(kpts_x)[-2]) ** 2 \
                      + (np.array(kpts_y)[-1] - np.array(kpts_y)[
                        -2]) ** 2) ** 0.5

            yFile = os.path.join(self.root_path, "data.mat")
            datay = sio.loadmat(yFile)
            avg = datay['avg_left']
            avgxy = avg.reshape(-1)  # xyxy...xyxy
            covar_all = datay['covariance_all_left']

            box_w, box_h, box_left_top = tg['bbox'][2], tg['bbox'][3], tg['bbox'][:2]
            bbox_lt_x, bbox_lt_y = box_left_top[0], box_left_top[1]

            kpts_x_norm_box = (np.array(kpts_x) - bbox_lt_x) / np.array(([head_h for _ in range(len(kpts_index))]))
            kpts_y_norm_box = (np.array(kpts_y) - bbox_lt_y) / np.array([head_h for _ in range(len(kpts_index))])

            kpt_x, kpt_y = kpts_x_norm_box, kpts_y_norm_box
            kpt_z = np.array(kpts_v)
            idx = np.argsort(kpt_z)
            sort_x = kpt_x[np.argsort(kpt_z)]
            sort_y = kpt_y[np.argsort(kpt_z)]

            N = np.sum(kpt_z == 0)

            if N == 0:
                target['annotations'][itg]['mean_keypoints'] = target['annotations'][itg]['keypoints']
                target['annotations'][itg]['keypoints_buquan'] = target['annotations'][itg]['keypoints']
                x = np.array(target['annotations'][itg]['keypoints'][0::3])
                y = np.array(target['annotations'][itg]['keypoints'][1::3])
                z = np.array(target['annotations'][itg]['keypoints'][2::3])

                re_kpts_ratio = np.zeros((14 * 3))
                re_kpts_ratio[0::3] = x / img.width
                re_kpts_ratio[1::3] = y / img.height
                re_kpts_ratio[2::3] = z.astype('int64')

                target['annotations'][itg]['keypoints_ratio'] = list(re_kpts_ratio)
                continue
            else:
                index = np.zeros(len(kpts_index) * 2).astype('int64')
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
                x_unknow = np.linalg.lstsq(AA, b, rcond=None)[0]
                X0_f = x_unknow

                x_recon, y_recon = X0_f[0::2], X0_f[1::2]

                re_sort_x = sort_x * head_h + bbox_lt_x
                re_sort_x[:len(x_recon)] = x_recon * head_h + bbox_lt_x
                re_sort_y = sort_y * head_h + bbox_lt_y
                re_sort_y[:len(y_recon)] = y_recon * head_h + bbox_lt_y

                avg_pose_x = avgxy_new[0::2] * head_h + bbox_lt_x
                avg_pose_y = avgxy_new[1::2] * head_h + bbox_lt_y

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

                re_kpts = np.zeros((14 * 3)).astype('int64')
                re_kpts[0::3] = recovery_arr_x.astype('int64')
                re_kpts[1::3] = recovery_arr_y.astype('int64')
                re_kpts[2::3] = kpt_z.astype('int64')

                re_kpts_ratio = np.zeros((14 * 3))
                re_kpts_ratio[0::3] = recovery_arr_x / img.width
                re_kpts_ratio[1::3] = recovery_arr_y / img.height
                re_kpts_ratio[2::3] = kpt_z.astype('int64')

                re_kpts_mean = np.zeros((14 * 3)).astype('int64')
                re_kpts_mean[0::3] = recovery_mean_x.astype('int64')
                re_kpts_mean[1::3] = recovery_mean_y.astype('int64')
                re_kpts_mean[2::3] = kpt_z.astype('int64')

                target['annotations'][itg]['mean_keypoints'] = list(re_kpts_mean)
                target['annotations'][itg]['keypoints_ratio'] = list(re_kpts_ratio)

                if self.is_completion:
                    target['annotations'][itg]['keypoints_buquan'] = list(re_kpts)
                else:
                    target['annotations'][itg]['keypoints_buquan'] = target['annotations'][itg]['keypoints']

        return target

    def vis_completion(self, target, image_id, w, h):
        tgs = target['annotations']
        img_r = cv2.imread('/path/to/crowdpose/images/' + str(image_id) + '.jpg')
        img_1k = np.zeros((1500, 1500, 3))
        img_1k[:h, :w] = img_r
        for tg in tgs:
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            kpts = tg['keypoints_buquan']
            kpts_x = kpts[0::3]
            kpts_y = kpts[1::3]
            kpts_v = kpts[2::3]
            kpt_id = 0
            for (x, y, v) in zip(kpts_x, kpts_y, kpts_v):
                kpt_id += 1
                if v < 1:
                    cv2.circle(img_1k, (int(x), int(y)), radius=10, color=color, thickness=-1)
                    cv2.putText(img_1k, str(kpt_id), (int(x) + 10, int(y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color,
                                4)
                else:
                    cv2.circle(img_1k, (int(x), int(y)), radius=5, color=color, thickness=-1)
        cv2.imwrite('/path/to/image_save/' + str(image_id) + '.jpg', img_1k)


    def __len__(self):
        return len(self.all_imgIds)

    def __getitem__(self, idx):

        image_id = self.all_imgIds[idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        target = self.coco.loadAnns(ann_ids)

        target = {'image_id': image_id, 'annotations': target}
        img = Image.open(self.img_folder / self.coco.loadImgs(image_id)[0]['file_name'])

        # -------------------------KEYPOINTS COMPLETION# -------------------------
        target = self.kpts_completion(target, img)
        is_vis_completion = True
        if self.is_completion and is_vis_completion:
            h, w = img.height, img.width
            self.vis_completion(target, image_id, w, h)


        keypoint_ratio = copy.deepcopy(target)

        img, target = self.prepare(img, target)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, keypoint_ratio


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
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32).reshape(-1, 14, 3)
        keypoints_buquan = torch.as_tensor(keypoints_buquan, dtype=torch.float32).reshape(-1, 14, 3)
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
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["iscrowd"] = iscrowd[keep]
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return image, target


def make_coco_transforms(image_set, fix_size=False, args=None):
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
    print("data_aug_params:", json.dumps(datadict_for_print, indent=2))

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
    root = Path(args.crowdpose_path)
    dataset = CocoDetection(root, image_set, transforms=make_coco_transforms(image_set),
                            return_masks=args.masks, is_completion=args.is_completion)
    return dataset
