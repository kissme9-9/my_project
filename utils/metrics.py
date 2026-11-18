"""
医学分割指标计算：Dice、HD95、边界带 Dice
HD95 用 scipy 距离变换实现，不依赖 medpy/exacta，避免版本不一致。
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion


def dice_score_np(pred, target, smooth=1e-6):
    """二值预测与目标的 Dice。pred, target 都是 0/1 数组。"""
    pred = pred.astype(np.bool_)
    target = target.astype(np.bool_)
    intersection = np.logical_and(pred, target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def hausdorff_95(pred, target, spacing=(1.0, 1.0)):
    """
    95% 豪斯多夫距离（像素单位）。
    pred, target: 0/1 二值数组。
    只在**边界轮廓**之间计算双向最近邻距离，取 95% 分位数。
    返回 HD95；若某一方为空或边界为空，返回 inf。
    """
    pred = pred.astype(np.bool_)
    target = target.astype(np.bool_)
    if pred.sum() == 0 or target.sum() == 0:
        return float('inf')

    # 提取边界轮廓：原 mask 与腐蚀后 mask 的异或（即最外层一圈像素）
    pred_boundary = pred ^ binary_erosion(pred)
    target_boundary = target ^ binary_erosion(target)

    if pred_boundary.sum() == 0 or target_boundary.sum() == 0:
        return float('inf')

    # target 边界上每个点到最近 pred 边界像素的距离
    dt_pred_boundary = distance_transform_edt(~pred_boundary, sampling=spacing)
    dist_target_to_pred = dt_pred_boundary[target_boundary]

    # pred 边界上每个点到最近 target 边界像素的距离
    dt_target_boundary = distance_transform_edt(~target_boundary, sampling=spacing)
    dist_pred_to_target = dt_target_boundary[pred_boundary]

    all_dist = np.concatenate([dist_target_to_pred, dist_pred_to_target])
    return float(np.percentile(all_dist, 95))


def boundary_dice(pred, target, erode_pixels=3, smooth=1e-6):
    """
    边界带 Dice：对 GT 做 erode_pixels 像素腐蚀，边界带 = GT - 腐蚀GT，
    只在该区域内计算 Dice。
    """
    target = target.astype(np.bool_)
    pred = pred.astype(np.bool_)
    if target.sum() == 0:
        return float('nan')

    eroded = target.copy()
    for _ in range(erode_pixels):
        eroded = binary_erosion(eroded)
    boundary_band = np.logical_and(target, ~eroded)

    if boundary_band.sum() == 0:
        return float('nan')

    pred_in_band = np.logical_and(pred, boundary_band)
    target_in_band = boundary_band
    intersection = pred_in_band.sum()
    return (2.0 * intersection + smooth) / (pred_in_band.sum() + target_in_band.sum() + smooth)


def compute_all_metrics(pred_bin, target_bin, spacing=(1.0, 1.0), erode=3):
    """计算单样本的 Dice、HD95、边界Dice。"""
    d = dice_score_np(pred_bin, target_bin)
    h = hausdorff_95(pred_bin, target_bin, spacing)
    bd = boundary_dice(pred_bin, target_bin, erode_pixels=erode)
    return {"dice": d, "hd95": h, "boundary_dice": bd}
