import math
import sys
import json
from typing import Iterable
from util.utils import to_device
import torch
import util.misc as utils
from util import box_ops
from util import visualizer
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms  # noqa . for compatibility
import numpy as np


import copy
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

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None, writer_dict=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0
    keypoints_ratio = []

    for samples, targets, keypoint_ratio in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        keypoints_ratio.append(keypoint_ratio)

        samples = samples.to(device)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(args, samples, targets)
            else:
                outputs = model(samples)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        weight_dict['loss_init_bbox'] = 2.0
        weight_dict['loss_init_giou'] = 1.0
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            accumulation_steps = 1
            losses = losses / accumulation_steps
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if (_cnt + 1) % accumulation_steps == 0:
                # optimizer the net
                optimizer.step()  # update parameters of net
                optimizer.zero_grad()  # reset gradient

        if args.onecyclelr:
            lr_scheduler.step()
        # if args.use_ema:
        #     if epoch >= args.ema_epoch:
        #         ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)

        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if _cnt % 1 == 0 and args.rank == 0:
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            # print(global_steps)
            a = np.array([metric_logger.meters['loss'].avg])
            writer.add_scalar('Loss_train_sum', np.array([metric_logger.meters['loss'].avg]), global_steps)
            writer.add_scalar('Loss_ce', metric_logger.meters['loss_ce'].avg, global_steps)
            writer.add_scalar('Loss_boxes', metric_logger.meters['loss_bbox'].avg, global_steps)
            writer.add_scalar('Loss_giou', metric_logger.meters['loss_giou'].avg, global_steps)
            writer.add_scalar('Loss_keypoints', metric_logger.meters['loss_keypoints'].avg, global_steps)
            # writer.add_scalar('Loss_keypoints', loss_dict['loss_keypoints'], global_steps)
            writer.add_scalar('Loss_oks', metric_logger.meters['loss_oks'].avg, global_steps)
            # writer.add_scalar('Loss_graph', metric_logger.meters['loss_graph'].avg, global_steps)
            writer.add_scalar('Loss_dn_ce', metric_logger.meters['dn_loss_ce'].avg, global_steps)
            writer.add_scalar('Loss_dn_bbox', metric_logger.meters['dn_loss_bbox'].avg, global_steps)
            writer.add_scalar('Loss_dn_giou', metric_logger.meters['dn_loss_giou'].avg, global_steps)
            writer.add_scalar('lr', metric_logger.meters['lr'].value, global_steps)

            writer_dict['train_global_steps'] = global_steps + 1

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break
    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    iou_types = tuple(k for k in ( 'bbox', 'keypoints'))
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    if args.dataset_file=="coco":
        from datasets.coco_eval import CocoEvaluator
        coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    elif args.dataset_file=="OCHuman":
        from datasets.coco_eval import CocoEvaluator
        coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    elif args.dataset_file=="crowdpose":
        from datasets.crowdpose_eval import CocoEvaluator
        coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    _cnt = 0

    pred = []
    keypoints_ratio = []
    for samples, targets, keypoint_ratio in metric_logger.log_every(data_loader, 10, header, logger=logger):
        keypoints_ratio.append(keypoint_ratio)
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(args, samples, targets)
            else:
                outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes) # 61

        score_results = copy.deepcopy(results)
        tgt = score_results[0]
        keep = batched_nms(tgt['boxes'], tgt['scores'], tgt['labels'], 1)
        num = torch.sum(score_results[0]['scores'] > 0)
        score_results[0]['labels'] = score_results[0]['labels'][keep]
        score_results[0]['keypoints'] = score_results[0]['keypoints'][keep]
        score_results[0]['scores'] = score_results[0]['scores'][keep]
        score_results[0]['boxes'] = score_results[0]['boxes'][keep]
        # results = score_results # 2024
        num1 = torch.sum(score_results[0]['scores'] >= 0)
        for i in range(int(num1)):
            result = {}
            result['image_id'] = targets[0]['image_id'].tolist()[0]
            result['category_id'] = score_results[0]['labels'][i].tolist()
            result['keypoints'] = score_results[0]['keypoints'][i].tolist()
            result['score'] = score_results[0]['scores'][i].tolist()
            # result['boxes'] = score_results[0]['boxes'][i].tolist()
            pred.append(result)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)
        _cnt += 1
        if args.vis_eval:
            vis = visualizer.COCOVisualizer().visualize
            tgt = results[0]
            tgt['gt_boxes'] = targets[0]['boxes']
            tgt['gt_kpts'] = targets[0]['keypoints']
            tgt['gt_area'] = targets[0]['area']
            tgt['gt_ori_size'] = targets[0]['orig_size']
            tgt['image_id'] = targets[0]['image_id']
            tgt['size'] = targets[0]['size']
            tgt['kpt_bbox'] = tgt['boxes']
            labels_per_image = torch.ones(100)
            keep = batched_nms(tgt['boxes'], tgt['scores'], tgt['labels'], 0.5)
            tgt['labels'] = tgt['labels'][keep]
            # tgt['size'] = tgt['size'][keep]
            tgt['kpt_bbox'] = tgt['kpt_bbox'][keep]
            tgt['scores'] = tgt['scores'][keep]
            tgt['boxes'] = tgt['boxes'][keep]
            tgt['keypoints'] = tgt['keypoints'][keep]
            img = samples.tensors[0].cpu()
            vis(img, tgt, caption=None, dpi=180,
                savedir=args.output_dir)  # tgt must have items: 'image_id', 'boxes', 'size'

        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break
    with open( args.output_dir + "result_crowdpose.json", 'w') as f:
         json.dump(pred, f)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
            stats['coco_eval_keypoints_detr'] = coco_evaluator.coco_eval['keypoints'].stats.tolist()
    visualizer.COCOVisualizer()
    return stats, coco_evaluator
