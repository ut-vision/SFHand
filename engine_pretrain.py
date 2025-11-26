# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import os
from typing import Iterable

import torch

import util.misc as misc
from util import box_ops
from einops import rearrange
import numpy as np
from util.meter import *
import torch
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm
try:
    from util.renderer import Renderer, render_one_hand
except:
    print("Renderer not found")

def build_transform(model_name, mode):

    mean, std = (0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258, 0.27577711)
    input_size = 336 if model_name.endswith("_336PX") else 224
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=3),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform
    
def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, scaler,
                    log_writer=None,
                    args=None,criterion=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        optimizer.zero_grad()
        vis_txt = [tensor.cuda(args.gpu, non_blocking=True) for tensor in batch[:3]]
        targets = {k: v.cuda(args.gpu, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch[3].items()}

        vis_txt[0] = vis_txt[0].permute(0, 2, 1, 3, 4) # [b t c h w -> b c t h w]
        vis_txt[1] = vis_txt[1].permute(0, 2, 1, 3, 4) # [b t c h w -> b c t h w]
        seq_len = vis_txt[0].shape[2]

        for stream_iter in range(seq_len - 1):
            with torch.cuda.amp.autocast():
                stream_inputs, stream_targets, \
                    hand_stat, hand_valid = arrange_stream_data(vis_txt, targets, seq_len, stream_iter, args.stream_len)

                outputs = model(*stream_inputs, hand_stat, hand_valid, roi_boxes=stream_targets['boxes'])
                loss_dict = criterion(outputs, stream_targets)
                weight_dict = criterion.weight_dict
                loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            scaler.scale(loss).backward()
            metric_logger.update(loss=loss.item())
            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            if log_writer is not None and (data_iter_step + 1) % 10 == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('train_loss', loss.item(), epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad(set_to_none=True)

        # clear kv cache after a streaming
        model.module.adapter.clear_cache()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, criterion, data_loader, args, device, vis=False, iou_thresholds=None):
    err_data = {}
    if iou_thresholds is None:
        iou_thresholds = [-1, 0, 0.25, 0.4, 0.5, 0.75]
    for thresh in iou_thresholds:
        err_data[thresh] = {'true_positives': 0, 'positives': 0, 'gt_len': 0, 'camt_error': [],
                            'mpjpe': [], 'pa-mpjpe': [], 'camt_error_last': []}

    model.eval()
    criterion.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    for batch in metric_logger.log_every(data_loader, 10, header):
        vis_txt = [tensor.to(device) for tensor in batch[:3]]
        targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch[3].items()}

        vis_txt[0] = vis_txt[0].permute(0, 2, 1, 3, 4)
        vis_txt[1] = vis_txt[1].permute(0, 2, 1, 3, 4)
        seq_len = vis_txt[0].shape[2]

        static_outputs, prev_hand_stat, prev_hand_valid, video_vis = None, None, None, None
        for stream_iter in range(seq_len - 1):
            with torch.cuda.amp.autocast():
                stream_inputs, stream_targets, \
                    hand_stat, hand_valid = arrange_stream_data(vis_txt, targets, seq_len, stream_iter, args.stream_len)
                if prev_hand_stat is not None:
                    hand_stat = prev_hand_stat
                    hand_valid = prev_hand_valid

                outputs = model(*stream_inputs, hand_stat, hand_valid, eval_mode=True)
                for k in outputs:
                    if 'pred' in k:
                        outputs[k] = outputs[k][:, model.module.mlp.seq_len // 2:, :, :]
                
                '''Uncomment the following code for auto-regressive prediction'''
                pred_hand_type = outputs['pred_logits'].argmax(-1)
                prev_hand_stat = torch.cat([outputs['pred_boxes'], outputs['pred_camera_t'], pred_hand_type.unsqueeze(-1),
                            outputs['pred_global_orient'], outputs['pred_hand_pose'], outputs['pred_betas']], dim=-1)
                prev_hand_valid = torch.ones_like(hand_valid) * prev_hand_stat.size(2)

                '''Uncomment the following code for sanity check (static prediction)'''
                # if static_outputs is None:
                #     hand_type_logits = F.one_hot(stream_in['hand_type'].long(), num_classes=3).float()
                #     static_outputs = {'pred_boxes': stream_in['boxes'],
                #                     'pred_logits': hand_type_logits.squeeze(-2),
                #                     'pred_camera_t': stream_in['camera_t'],
                #                     'pred_global_orient': stream_in['global_orient'],
                #                     'pred_hand_pose': stream_in['hand_pose'],
                #                     'pred_betas': stream_in['betas']}
                #     outputs = static_outputs
                # else:
                #     outputs = static_outputs
                loss_dict = criterion(outputs, stream_targets)
                weight_dict = criterion.weight_dict
                is_last = stream_iter == seq_len - 2
                matched_res = multi_eval(outputs, stream_targets, criterion, err_data, is_last_frame=is_last)

                if vis:
                    vis_frames = arrange_vis_frames(stream_inputs[1], matched_res, criterion, stream_iter, vis_gt=False, draw_box=True)
                    video_vis = vis_frames if video_vis is None else np.concatenate([video_vis, vis_frames], axis=1)
                    if stream_iter >= seq_len - 2:
                         visualize_video(video_vis, stream_targets['uid'], "render_results")
                         return
        model.module.adapter.clear_cache()

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = misc.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    for thresh in err_data.keys():
        tp, pos, gt_len = err_data[thresh]['true_positives'], err_data[thresh]['positives'], err_data[thresh]['gt_len']
        precision = tp / pos if pos > 0 else 0
        recall = pos / gt_len if gt_len > 0 else 0

        camt_error = torch.cat(err_data[thresh]['camt_error']).mean()
        camt_error_last = torch.cat(err_data[thresh]['camt_error_last']).mean()
        mpjpe_err = torch.cat(err_data[thresh]['mpjpe']).mean()
        pampjpe_err = torch.cat(err_data[thresh]['pa-mpjpe']).mean()
        # calculate and print
        # Create bordered single line format for easy comparison across thresholds
        content = f"Cls: {precision:.2f}, Recall: {recall:.2f}, ADE: {camt_error * 100:.2f}cm, " + \
            f"FDE: {camt_error_last * 100:.2f}cm, MPJPE: {mpjpe_err * 100:.2f}cm, PA-MPJPE: {pampjpe_err * 100:.2f}cm"
        print(f"┌─ IoU {thresh:.2f} ─{'─' * (len(content) - len(f'{thresh:.2f}') - 6)}┐")
        print(f"│ {content} │")
        print(f"└{'─' * (len(content) + 2)}┘")

    return err_data

def multi_eval(outputs, targets, criterion, err_data, is_last_frame=False):
    mano_layer = criterion.mano_layer
    indices = criterion.get_indices(outputs, targets)
    idx = criterion._get_src_permutation_idx(indices)
    if 'pred_logits' in outputs and 'pred_boxes' in outputs:
        # Prepare data for forecasting evaluation
        src_logits = rearrange(outputs['pred_logits'], "bs seq q cls -> (bs seq) q cls")
        src_logits = src_logits[idx]
        target_cls = rearrange(targets['hand_type'], "bs seq q -> (bs seq) q")
        target_classes = torch.cat([t[i] for t, (_, i) in zip(target_cls, indices)], dim=0)
        pred_classes = torch.argmax(src_logits[:, :2], dim=-1)
        correct_pred = (pred_classes == target_classes)

        src_boxes = rearrange(outputs['pred_boxes'], "bs seq q box -> (bs seq) q box")
        src_boxes = src_boxes[idx]
        target_boxes = rearrange(targets['boxes'], "bs seq q box -> (bs seq) q box")
        target_boxes = torch.cat([t[i] for t, (_, i) in zip(target_boxes, indices)], dim=0)

        # Prepare data for hand pose evaluation
        src_camt = rearrange(outputs['pred_camera_t'], "bs seq q camt -> (bs seq) q camt")
        src_camt = src_camt[idx]
        target_camt = rearrange(targets['camera_t'], "bs seq q camt -> (bs seq) q camt")
        target_camt = torch.cat([t[i] for t, (_, i) in zip(target_camt, indices)], dim=0)

        src_hp = rearrange(outputs['pred_hand_pose'], "bs seq q (p1 p2 p3) -> (bs seq) q p1 p2 p3",
                           p1=criterion.num_joints, p2=3, p3=3)
        src_hp = src_hp[idx]
        src_ort = rearrange(outputs['pred_global_orient'], "bs seq q (o1 o2 o3) -> (bs seq) q o1 o2 o3",
                           o1=1, o2=3, o3=3)
        src_ort = src_ort[idx]
        src_betas = rearrange(outputs['pred_betas'], "bs seq q betas -> (bs seq) q betas")
        src_betas = src_betas[idx]

        target_kp3d = rearrange(targets['keypoints_3d'], "bs seq q j1 j2 -> (bs seq) q j1 j2")
        target_kp3d = torch.cat([t[i] for t, (_, i) in zip(target_kp3d, indices)], dim=0)

        target_verts = rearrange(targets['vertices'], "bs seq q v1 v2 -> (bs seq) q v1 v2")
        target_verts = torch.cat([t[i] for t, (_, i) in zip(target_verts, indices)], dim=0)

        ious = torch.diag(box_ops.box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))[0])

        for thresh in err_data.keys():
            positives = (ious > thresh).sum().item()
            correct_predictions = (ious > thresh) & correct_pred
            tp = correct_predictions.sum().item()  # Calculate the number of correct predictions
            err_data[thresh]['true_positives'] += tp
            err_data[thresh]['positives'] += positives
            err_data[thresh]['gt_len'] += len(ious)

            # Compute MPJPE for hand pose
            camt_error = torch.norm(target_camt[correct_predictions] - src_camt[correct_predictions], dim=1)
            err_data[thresh]['camt_error'].append(camt_error)
            if is_last_frame:
                err_data[thresh]['camt_error_last'].append(camt_error)

            mano_output = mano_layer(global_orient=src_ort, hand_pose=src_hp, betas=src_betas, pose2rot=False)
            pred_joints = mano_output.joints
            pa_aligned_joints = misc.compute_similarity_transform(pred_joints, target_kp3d)
            wr_aligned_joints = pred_joints - pred_joints[:, :1, :] + target_kp3d[:, :1, :]
            mpjpe = torch.norm(wr_aligned_joints[correct_predictions] - target_kp3d[correct_predictions], dim=-1)
            pa_mpjpe = torch.norm(pa_aligned_joints[correct_predictions] - target_kp3d[correct_predictions], dim=-1)
            err_data[thresh]['mpjpe'].append(mpjpe.mean(-1))
            err_data[thresh]['pa-mpjpe'].append(pa_mpjpe.mean(-1))

        return {'idx': idx, 'focal': targets['focal'],
                'pred_boxes': src_boxes, 'target_boxes': target_boxes,
                'pred_camt': src_camt, 'target_camt': target_camt,
                'pred_ort': src_ort,
                'pred_hand_pose': src_hp,
                'pred_betas': src_betas,
                'target_kp3d': target_kp3d, 'target_verts': target_verts,
                'pred_hand_type': pred_classes, 'target_hand_type': target_classes}

def arrange_stream_data(vis_txt, targets, seq_len, stream_iter, stream_len=1):
    stream_target_keys = ['camera_t', 'hand_type', 'global_orient', 'hand_pose', 'valid', 'orig_size',
                                      'betas', 'keypoints_3d', 'vertices', 'boxes']
    stream_targets = targets.copy()
    stream_in = targets.copy()
    start_iter = max(0, stream_iter - stream_len + 1)
    end_iter = min(seq_len - 1, stream_iter + 1)

    for k in stream_target_keys:
        stream_in[k] = stream_targets[k][:, start_iter: end_iter]
        if k != 'valid':
            stream_in[k] = stream_in[k].view(*stream_in[k].shape[:3], -1)
        stream_targets[k] = stream_targets[k][:, stream_iter + 1: stream_iter + 2]
    hand_stat = torch.cat([stream_in['boxes'], stream_in['camera_t'], stream_in['hand_type'],
                            stream_in['global_orient'], stream_in['hand_pose'], stream_in['betas']], dim=-1)
    valid = stream_in['valid']

    stream_inputs = vis_txt.copy()
    stream_inputs[0] = stream_inputs[0][:, :, start_iter: end_iter]
    stream_inputs[1] = stream_inputs[1][:, :, end_iter: end_iter + 1]

    return stream_inputs, stream_targets, hand_stat, valid

def arrange_vis_frames(frames_rear, matched_res, criterion, stream_iter=None, vis_gt=False, draw_box=True):
    _, _, seq, h, w = frames_rear.shape
    batch_idx = matched_res['idx'][0] // seq
    seq_idx = matched_res['idx'][0] % seq
    mano_layer = criterion.mano_layer
    mano_output = mano_layer(global_orient=matched_res['pred_ort'],
                             hand_pose=matched_res['pred_hand_pose'],
                             betas=matched_res['pred_betas'])
    # pred_joints = mano_output.joints
    # pa_aligned_joints = misc.compute_similarity_transform(pred_joints, matched_res['target_kp3d'])
    pred_verts = mano_output.vertices
    pa_aligned_verts = misc.compute_similarity_transform(pred_verts, matched_res['target_verts'])

    frames_np = frames_rear.permute(0, 2, 3, 4, 1).cpu().numpy()
    frames_vis = frames_np[..., ::-1] # RGB to BGR
    renderer = Renderer(faces=mano_layer.faces)
    for i in tqdm(range(len(batch_idx)), desc=f"Rendering stream iter {stream_iter}"):
        b, s = int(batch_idx[i]), int(seq_idx[i])
        img = frames_vis[b, s].copy()

        focal = matched_res['focal'][b].cpu().numpy()
        box = matched_res['pred_boxes'][i]
        is_right = matched_res['pred_hand_type'][i].cpu().numpy()
        camt = matched_res['pred_camt'][i].cpu().numpy()
        verts = pa_aligned_verts[i].cpu().numpy()
        if vis_gt:
            box = matched_res['target_boxes'][i]
            is_right = matched_res['target_hand_type'][i].cpu().numpy()
            camt = matched_res['target_camt'][i].cpu().numpy()
            verts = matched_res['target_verts'][i].cpu().numpy()

        rendered_img = render_one_hand(
            renderer=renderer, 
            img=img, 
            img_hw=(h, w), 
            verts=verts, 
            camt=camt, 
            is_right=is_right, 
            focal=focal, 
            box=box, 
            draw_box=draw_box
        )
        frames_vis[b, s] = rendered_img
    return frames_vis

def visualize_video(video_vis, uid, save_dir='render_results'):
    num, seq_len, height, width, _ = video_vis.shape
    os.makedirs(save_dir, exist_ok=True)
    for b in tqdm(range(num), desc="Saving videos"):
        video = video_vis[b] # (seq, h, w, 3)
        path = os.path.join(save_dir, f"{uid[b]}.mp4")

        duration = 3.0
        fps = seq_len / duration
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, fps, (width, height))
        
        for frame_idx in range(seq_len):
            frame = video[frame_idx]  # (h, w, 3)
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            out.write(frame)
        out.release()
