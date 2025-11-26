import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from util import box_ops
from util.misc import accuracy, get_world_size, is_dist_avail_and_initialized
from model.mano_wrapper import MANO

def loss_boxes(outputs, targets, num_boxes):
    losses = F.l1_loss(outputs, targets, reduction='none').sum() / num_boxes
    return losses


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, mano_cfg):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.mano_layer = MANO(**{k.lower(): v for k, v in dict(mano_cfg).items()})
        self.num_joints = mano_cfg.NUM_HAND_JOINTS

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = rearrange(outputs['pred_logits'], "bs seq q cls -> (bs seq) q cls")
        target_cls = rearrange(targets['hand_type'], "bs seq q -> (bs seq) q")

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(target_cls, indices)]).to(torch.int64)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        pred_logits = rearrange(pred_logits, "bs seq q cls -> (bs seq) q cls")
        tgt_lengths = rearrange(targets['valid'], "bs seq c -> (bs seq c)")

        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = rearrange(outputs['pred_boxes'], "bs seq q box -> (bs seq) q box")
        src_boxes = src_boxes[idx]
        target_boxes = rearrange(targets['boxes'], "bs seq q box -> (bs seq) q box")

        target_boxes = torch.cat([t[i] for t, (_, i) in zip(target_boxes, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_pose(self, outputs, targets, indices, num_boxes):
        assert 'pred_camera_t' in outputs
        assert 'pred_global_orient' in outputs
        assert 'pred_hand_pose' in outputs
        assert 'pred_betas' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_camt = rearrange(outputs['pred_camera_t'], "bs seq q t -> (bs seq) q t")
        src_ort = rearrange(outputs['pred_global_orient'], "bs seq q (o1 o2 o3) -> (bs seq) q o1 o2 o3",
                            o1=1, o2=3, o3=3)
        src_pose = rearrange(outputs['pred_hand_pose'], "bs seq q (p1 p2 p3) -> (bs seq) q p1 p2 p3",
                             p1=self.num_joints, p2=3, p3=3)
        src_beta = rearrange(outputs['pred_betas'], "bs seq q beta -> (bs seq) q beta")
        src_camt = src_camt[idx]
        src_ort = src_ort[idx]
        src_pose = src_pose[idx]
        src_beta = src_beta[idx]

        target_camt = rearrange(targets['camera_t'], "bs seq q t -> (bs seq) q t")
        target_ort = rearrange(targets['global_orient'], "bs seq q o1 o2 o3 -> (bs seq) q o1 o2 o3")
        target_pose = rearrange(targets['hand_pose'], "bs seq q p1 p2 p3 -> (bs seq) q p1 p2 p3")
        target_beta = rearrange(targets['betas'], "bs seq q beta -> (bs seq) q beta")
        target_kp3d = rearrange(targets['keypoints_3d'], "bs seq q j1 j2 -> (bs seq) q j1 j2")

        target_camt = torch.cat([t[i] for t, (_, i) in zip(target_camt, indices)], dim=0)
        target_ort = torch.cat([t[i] for t, (_, i) in zip(target_ort, indices)], dim=0)
        target_pose = torch.cat([t[i] for t, (_, i) in zip(target_pose, indices)], dim=0)
        target_beta = torch.cat([t[i] for t, (_, i) in zip(target_beta, indices)], dim=0)
        target_kp3d = torch.cat([t[i] for t, (_, i) in zip(target_kp3d, indices)], dim=0)

        loss_camt = F.l1_loss(src_camt, target_camt, reduction='none')
        loss_ort = F.l1_loss(src_ort, target_ort, reduction='none')
        loss_pose = F.l1_loss(src_pose, target_pose, reduction='none')
        loss_beta = F.l1_loss(src_beta, target_beta, reduction='none')
        loss_mano = loss_pose.sum() + loss_beta.sum() + loss_camt.sum() + loss_ort.sum()

        mano_out = self.mano_layer(global_orient=src_ort, hand_pose=src_pose, betas=src_beta, pose2rot=False)
        loss_kp3d = F.l1_loss(mano_out.joints, target_kp3d, reduction='none')

        losses = {}
        losses['loss_pose'] = (loss_mano + loss_kp3d.sum()) / num_boxes
        return losses

    def loss_itc(self, outputs, targets, indices, num_boxes):
        """Compute the image-text contrastive loss
        video_embed: (bs, seq, d) - video features
        text_embed: (bs, d) - text features
        """
        assert 'video_embed' in outputs
        assert 'text_embed' in outputs
        
        video_embed = outputs['video_embed']  # (bs, seq, d)
        text_embed = outputs['text_embed']    # (bs, d)

        video_embed, text_embed = F.normalize(video_embed, dim=-1), F.normalize(text_embed, dim=-1)

        # Compute similarity matrix between video tokens and text
        # (bs, seq, d) @ (bs, d, 1) -> (bs, seq, 1) -> (bs, seq)
        sim_matrix = torch.bmm(video_embed, text_embed.unsqueeze(-1)).squeeze(-1)  # (bs, seq)
        
        # Find the token with maximum similarity for each batch
        max_sim_indices = torch.argmax(sim_matrix, dim=1)  # (bs,)
        
        # Gather the best video tokens for each batch
        best_video_tokens = torch.gather(video_embed, 1, 
                                       max_sim_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, video_embed.shape[-1]))  # (bs, 1, d)
        best_video_tokens = best_video_tokens.squeeze(1)  # (bs, d)
        
        # Get logit scale from outputs or use default
        logit_scale = outputs.get('logit_scale', 1.0)
        # Compute logits for contrastive learning (similar to CLIP)
        logits_per_video = logit_scale * best_video_tokens @ text_embed.T  # (bs, bs)
        logits_per_text = logit_scale * text_embed @ best_video_tokens.T   # (bs, bs)
        
        # Create labels for contrastive learning (diagonal should be positive pairs)
        batch_size = video_embed.shape[0]
        device = video_embed.device
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        
        # Compute VLP loss (similar to CLIP loss)
        vlp_loss = (
            F.cross_entropy(logits_per_video, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2
        # Compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_video, dim=-1)
            correct = pred.eq(labels).sum()
            acc = 100 * correct / logits_per_video.size(0)
        
        return {'loss_itc': vlp_loss, 'itc_acc': acc}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def get_indices(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        return indices

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'pose': self.loss_pose,
            'itc': self.loss_itc,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        # print(indices, len(indices))

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = torch.sum(targets["valid"])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        return losses

