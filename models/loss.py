import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor


def _dice_loss(pred, target, target_roi_weight, need_sigmoid=True):
    """
        https://arxiv.org/pdf/1812.02427.pdf -> batch dice coefficient
    :param input: (N, C, d1, d2, d3, ...) -> binary mask
    :param target: (N, C, d1, d2, d3, ...) -> binary mask
    :param target_roi_weight: (N, C), whether roi is included in target.
            If N slices are from single volume, then the weights are same for N slices.
            weight for slice are same for slices from the same volume.
    :return:
    """
    if need_sigmoid:
        pred = pred.sigmoid()
    # If there is no ground truth for rois in one volume, then we ignore the dice between pred and target
    numerator = torch.sum(pred * target, dim=[2, 3]) * target_roi_weight  # -> shape (N, C)
    denominator = torch.sum(pred + target, dim=[2, 3]) * target_roi_weight  # -> shape (N, C)
    batch_numerator = torch.sum(numerator, dim=0)
    batch_denominator = torch.sum(denominator, dim=0)
    batch_roi_weight = torch.sum(target_roi_weight, dim=0).gt(0)  # -> shape (C)
    DSC = (2 * batch_numerator + 1) / (batch_denominator + 1)
    DSC_loss = torch.sum((1 - DSC) * batch_roi_weight)
    return DSC_loss


def _sigmoid_focol_loss(pred, target, target_roi_weight, alpha: float = 0.25, gamma: float = 2, need_sigmoid=True):
    if need_sigmoid:
        prob = pred.sigmoid()
    else:
        prob = pred
    # print(need_sigmoid)
    # print(prob.max(), prob.min())
    # print(target.max(), target.min())
    ce_loss = F.binary_cross_entropy(prob, target, reduction="none")
    p_t = prob * target + (1 - prob) * (1 - target)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss

    # loss [N, C, H, W]
    loss = torch.sum(loss, dim=[2, 3]) * target_roi_weight
    return loss.sum()


class Criterion(nn.Module):
    def __init__(self, config):
        super(Criterion, self).__init__()
        self.focal_loss_weight = config['loss']['focal_loss_weight']
        self.dice_loss_weight = config['loss']['dice_loss_weight']
        self.deep_supervision = config['deep_supervision']

    def forward(self, pred, target, target_roi_weight, deep_supervision=False, need_sigmoid=True,
                layer_weight: Optional[Tensor] = None):
        focal_loss = 0
        dice_loss = 0
        loss = 0
        loss_dict = {}
        if deep_supervision:
            for i, pred_single in enumerate(pred):
                if layer_weight is not None:
                    weight = layer_weight[i]
                else:
                    weight = 1
                focal_loss_ = _sigmoid_focol_loss(pred_single, target, target_roi_weight, need_sigmoid=need_sigmoid)
                dice_loss_ = _dice_loss(pred_single, target, target_roi_weight, need_sigmoid=need_sigmoid)
                loss_ = (focal_loss_ * self.focal_loss_weight + dice_loss_ * self.dice_loss_weight) * weight
                # print(focal_loss_, dice_loss_, loss_)
                loss_dict[f"layer_{i}"] = {
                    "focal_loss": focal_loss_.detach().item(),
                    "dice_loss": dice_loss_.detach().item(),
                    "loss": loss_.detach().item()
                }
                focal_loss += focal_loss_
                dice_loss += dice_loss_
                loss += loss_
        else:
            focal_loss = _sigmoid_focol_loss(pred[-1], target, target_roi_weight, need_sigmoid=need_sigmoid)
            dice_loss = _dice_loss(pred[-1], target, target_roi_weight, need_sigmoid=need_sigmoid)
            loss = focal_loss * self.focal_loss_weight + dice_loss * self.dice_loss_weight

            loss_dict = {
                "focal_loss": focal_loss.detach().item(),
                "dice_loss": dice_loss.detach().item(),
                "loss": loss.detach().item()
            }

        return loss, loss_dict