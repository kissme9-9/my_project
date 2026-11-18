"""损失函数：Dice loss、组合损失、边界损失。"""
import torch
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def combined_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target) + dice_loss(pred, target)


def boundary_loss_from_dist(pred, dist_map):
    """
    边界损失：pred 是 logits，dist_map 是 GT 的距离图。
    """
    pred = torch.sigmoid(pred)
    return (pred * dist_map).mean()
