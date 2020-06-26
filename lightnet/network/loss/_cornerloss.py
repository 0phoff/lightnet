#
#   Darknet RegionLoss
#   Copyright EAVISE
#

import math
import numpy as np
import torch
import torch.nn as nn

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

__all__ = ['CornerLoss']


class CornerLoss(nn.modules.loss._Loss):
    """ TODO
    """
    def __init__(self, stride=4, gaussian_iou=0.3, heatmap_scale=1, pull_scale=1, push_scale=1, offset_scale=1, inter_scale=1):
        super().__init__()
        self.stride = stride
        self.gaussian_iou = gaussian_iou
        self.heatmap_scale = heatmap_scale
        self.pull_scale = pull_scale
        self.push_scale = push_scale
        self.offset_scale = offset_scale
        self.inter_scale = inter_scale

        self.eps = 1e-4
        self.l1 = nn.SmoothL1Loss()

    def extra_repr(self):
        repr_str = f'stride={self.stride}, gaussian_iou={self.gaussian_iou}\n'
        repr_str += f'heatmap_scale={self.heatmap_scale}, pull_scale={self.pull_scale}, push_scale={self.push_scale}, offset_scale={self.offset_scale}, inter_scale={self.inter_scale}'
        return repr_str

    def forward(self, output, target):
        """ Compute Corner loss.

        Args:
            output (tuple of 2 Tensors): Output from the network :class:`~lightnet.models.Cornernet` (training mode)
            target (brambox annotation dataframe): Brambox annotations
        """
        if isinstance(output, (list, tuple)):
            output, intermediate = output
        else:
            intermediate = None

        # Parameters
        device = output.device
        nB, nC, nH, nW = output.shape
        nClasses = (nC // 2) - 3

        # Split output
        output = output.view(nB, 2, -1, nH, nW)                                             # BATCH, TLBR, NUM_CLASSES+3, H, W
        out_heatmaps = output[:, :, :-3].sigmoid().clamp(min=1e-4, max=1-1e-4)              # BATCH, TLBR, NUM_CLASSES,   H, W
        out_embeddings = output[:, :, -3].view(nB, 2, -1)                                   # BATCH, TLBR,                 HxW
        out_offsets = output[:, :, -2:]                                                     # BATCH, TLBR, XY,            H, W

        # Split intermediate
        if intermediate is not None:
            intermediate = intermediate.view(nB, 2, -1, nH, nW)                             # BATCH, TLBR, NUM_CLASSES+3, H, W
            inter_heatmaps = intermediate[:, :, :-3].sigmoid().clamp(min=1e-4, max=1-1e-4)  # BATCH, TLBR, NUM_CLASSES,   H, W
            inter_embeddings = intermediate[:, :, -3].view(nB, 2, -1)                       # BATCH, TLBR,                 HxW
            inter_offsets = intermediate[:, :, -2:]                                         # BATCH, TLBR, XY,            H, W

        # Get ground truth tensors
        gt_heatmaps, gt_embeddings, gt_offsets, gt_mask = self.build_targets(target, nB, nClasses, nH, nW)
        gt_mask = gt_mask[:, :, None, ...].expand_as(gt_offsets)
        nGT = gt_mask.sum().item()

        # Losses
        if intermediate is not None:
            self.loss_heatmap = self.heatmap_scale * (
                self.focal_loss(out_heatmaps, gt_heatmaps)
                + self.inter_scale * self.focal_loss(inter_heatmaps, gt_heatmaps)
            )

            self.loss_embedding = (
                self.pushpull_loss(out_embeddings, gt_embeddings)
                + self.inter_scale * self.pushpull_loss(inter_embeddings, gt_embeddings)
            )

            if nGT > 0:
                self.loss_offset = self.offset_scale * (
                    self.l1(out_offsets[gt_mask], gt_offsets[gt_mask])
                    + self.inter_scale * self.l1(inter_offsets[gt_mask], gt_offsets[gt_mask])
                )
            else:
                self.loss_offset = torch.tensor(0.0)

            self.loss = (self.loss_heatmap + self.loss_embedding + self.loss_offset) / (1 + self.inter_scale)
        else:
            self.loss_heatmap = self.heatmap_scale * self.focal_loss(out_heatmaps, gt_heatmaps)
            self.loss_embedding = self.pushpull_loss(out_embeddings, gt_embeddings)
            if nGT > 0:
                self.loss_offset = self.offset_scale * self.l1(out_offsets[gt_mask], gt_offsets[gt_mask])
            else:
                self.loss_offset = torch.tensor(0.0)

            self.loss = self.loss_heatmap + self.loss_embedding + self.loss_offset

        # Loss
        return self.loss

    def build_targets(self, ground_truth, nB, nC, nH, nW):
        """ Convert ground truths to network output tensors """
        heatmaps = torch.zeros(nB, 2, nC, nH, nW, requires_grad=False)
        offsets = torch.zeros(nB, 2, 2, nH, nW, requires_grad=False)
        mask = torch.zeros(nB, 2, nH, nW, dtype=torch.bool, requires_grad=False)
        embedding = list()

        for b, gt_batch in ground_truth.groupby('batch_number', sort=False):
            # GT tensors
            class_id = torch.from_numpy(gt_batch.class_id.values).long()
            size = torch.from_numpy(gt_batch[['width', 'height']].values).float() / self.stride
            coords = torch.empty((gt_batch.shape[0], 4), requires_grad=False)
            coords[:, 0:2] = torch.from_numpy(gt_batch[['x_top_left', 'y_top_left']].values).float() / self.stride
            coords[:, 2:4] = coords[:, 0:2] + size
            coords_idx = coords.long()

            # Heatmaps
            if self.gaussian_iou:
                radii = gaussian_radius(size[:, 0], size[:, 1], self.gaussian_iou)
                coords0 = (coords_idx - radii[:, None]).clamp(min=0)
                coords1 = (coords_idx + radii[:, None] + 1)
                coords1[:, 0:3:2].clamp_(max=nW)
                coords1[:, 1:4:2].clamp_(max=nH)
                for idx, r in enumerate(radii):
                    g = create_gaussian(r, r/3+0.5)

                    sxtl = slice(coords0[idx, 0], coords1[idx, 0])
                    gxtl = slice(0, sxtl.stop - sxtl.start)
                    sytl = slice(coords0[idx, 1], coords1[idx, 1])
                    gytl = slice(0, sytl.stop - sytl.start)

                    sxbr = slice(coords0[idx, 2], coords1[idx, 2])
                    gxbr = slice(0, sxbr.stop - sxbr.start)
                    sybr = slice(coords0[idx, 3], coords1[idx, 3])
                    gybr = slice(0, sybr.stop - sybr.start)

                    heatmaps[b, 0, class_id[idx], sytl, sxtl] = torch.max(
                        g[gytl, gxtl],
                        heatmaps[b, 0, class_id[idx], sytl, sxtl]
                    )

                    heatmaps[b, 1, class_id[idx], sybr, sxbr] = torch.max(
                        g[gybr, gxbr],
                        heatmaps[b, 1, class_id[idx], sybr, sxbr]
                    )
            else:
                heatmaps[b, 0, class_id, coords_idx[:, 1], coords_idx[:, 0]] = 1
                heatmaps[b, 1, class_id, coords_idx[:, 3], coords_idx[:, 2]] = 1

            # Mask
            mask[b, 0, coords_idx[:, 1], coords_idx[:, 0]] = True
            mask[b, 1, coords_idx[:, 3], coords_idx[:, 2]] = True

            # Embeddings
            embedding.append(coords_idx[:, 1:4:2] * nW + coords_idx[:, 0:3:2])

            # Offsets
            off = (coords - coords_idx).transpose(0, 1).contiguous()
            offsets[b, 0, :, coords_idx[:, 1], coords_idx[:, 0]] = off[0:2]
            offsets[b, 1, :, coords_idx[:, 3], coords_idx[:, 2]] = off[2:4]

        return (heatmaps, embedding, offsets, mask)

    def focal_loss(self, pred, gt):
        p_mask = gt.eq(1)
        n_mask = ~p_mask

        p_pred = pred[p_mask]
        n_pred = pred[n_mask]

        p_loss = (p_pred.log() * (1 - p_pred) ** 2).sum()
        n_loss = ((1 - n_pred).log() * (n_pred ** 2) * ((1 - gt[n_mask]) ** 4)).sum()

        num_pos = p_mask.sum().item()
        if num_pos == 0:
            return -1 * n_loss
        else:
            return -1 * (p_loss + n_loss) / num_pos

    def pushpull_loss(self, pred, gt):
        push = 0
        pull = 0

        for b, gt_batch in enumerate(gt):
            nGT = gt_batch.shape[0]
            if nGT <= 0:
                continue

            pred_batch = torch.gather(pred[b], 1, gt_batch.transpose(0, 1))     # TLBR, nGT

            # Pull
            pred_mean = (pred_batch[0] + pred_batch[1]) / 2
            pull += ((pred_batch[0] - pred_mean) ** 2 + (pred_batch[1] - pred_mean) ** 2).sum() / (nGT + self.eps)

            # Push
            dist = (1 - torch.abs(pred_mean[..., None] - pred_mean[None, ...])).clamp(min=0)
            push += dist.triu(1).sum() / ((nGT - 1) * nGT + self.eps)

        return self.push_scale * push + self.pull_scale * pull


def gaussian_radius(box_width, box_height, iou):
    box_width = box_width.ceil().int()
    box_height = box_height.ceil().int()

    # a1 = 1
    b1 = (box_width + box_height)
    c1 = box_width * box_height * (1 - iou) / (1 + iou)
    sq1 = (b1 ** 2 - 4 * c1).sqrt()
    r1 = (b1 - sq1) / 2

    # a2  = 4
    b2 = 2 * (box_width + box_height)
    c2 = (1 - iou) * box_width * box_height
    sq2 = (b2 ** 2 - 16 * c2).sqrt()
    r2 = (b2 - sq2) / 8

    a3 = 4 * iou
    b3 = -2 * iou * (box_height + box_width)
    c3 = (iou - 1) * box_width * box_height
    sq3 = (b3 ** 2 - 4 * a3 * c3).sqrt()
    r3 = (b3 + sq3) / (2 * a3)

    rmin, _ = torch.min(torch.stack([r1, r2, r3], dim=0), dim=0)
    return rmin.int().clamp(min=0)


def create_gaussian(radius, sigma=1):
    x = torch.arange(-radius, radius+1)[None, ...]
    y = x.clone().view(-1, 1)
    return (-1 * (x**2 + y**2) / (2 * sigma**2)).exp()
