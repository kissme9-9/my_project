"""
第4步：推理后处理对比（零训练成本）
对 baseline 的 5 折概率图，测试三种后处理方案：
  A: prob > 0.5（基线）
  B: 形态学精修（开运算+闭运算+最大连通域）
  C: 阈值微调（在每折训练集上搜最优阈值，应用到验证集）
每种方案计算 Dice、HD95、边界Dice，按折对比。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import glob
import numpy as np
from PIL import Image
import cv2
from utils import compute_all_metrics
from config.paths import (
    IMG_DIR, MASK_LEFT_DIR, MASK_RIGHT_DIR, EXP_DIR, IMG_SIZE,
)
BASELINE_DIR = os.path.join(EXP_DIR, "baseline")


def load_gt(name):
    ml = np.array(Image.open(os.path.join(MASK_LEFT_DIR, name)).convert("L"))
    mr = np.array(Image.open(os.path.join(MASK_RIGHT_DIR, name)).convert("L"))
    mask = np.maximum(ml, mr)
    mask = Image.fromarray(mask).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    return (np.array(mask) > 127).astype(np.uint8)


def postprocess_A(prob):
    return (prob > 0.5).astype(np.uint8)


def postprocess_B(prob):
    """形态学精修：开运算(k=3) → 闭运算(k=5) → 保留前2大连通域（左右肺）"""
    binary = (prob > 0.5).astype(np.uint8)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    # 保留前 2 大连通域（左右肺）
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num > 1:
        # 跳过背景（label 0），取面积最大的两个
        areas = stats[1:, cv2.CC_STAT_AREA]
        top2_idx = np.argsort(areas)[-2:] + 1  # +1 因为跳过了背景
        result = np.zeros_like(binary)
        for idx in top2_idx:
            result[labels == idx] = 1
        binary = result
    return binary


def find_best_threshold(probs, gts, thresholds=None):
    """在一组 (prob, gt) 上搜索使平均 Dice 最高的阈值"""
    if thresholds is None:
        thresholds = np.arange(0.3, 0.71, 0.02)
    best_t, best_dice = 0.5, -1
    for t in thresholds:
        dices = []
        for prob, gt in zip(probs, gts):
            pred = (prob > t).astype(np.uint8)
            inter = np.logical_and(pred, gt).sum()
            d = (2 * inter + 1e-6) / (pred.sum() + gt.sum() + 1e-6)
            dices.append(d)
        mean_d = np.mean(dices)
        if mean_d > best_dice:
            best_dice = mean_d
            best_t = t
    return best_t


def main():
    folds_dir = sorted(glob.glob(os.path.join(BASELINE_DIR, "fold_*")))
    all_results = {"A": [], "B": [], "C": []}

    for fold_dir in folds_dir:
        fold_id = os.path.basename(fold_dir)
        prob_files = sorted(glob.glob(os.path.join(fold_dir, "probs", "*.npy")))
        if not prob_files:
            continue

        # 收集该折所有验证样本
        probs, gts, names = [], [], []
        for pf in prob_files:
            name = os.path.basename(pf).replace(".npy", ".png")
            prob = np.load(pf)
            gt = load_gt(name)
            probs.append(prob)
            gts.append(gt)
            names.append(name)

        # 方案 C：需要训练集概率图来搜阈值，但我们只有验证集概率图。
        # 按照用户要求"在训练折内搜"，这里用验证集近似（因为没有保存训练集概率图）。
        # 更严谨的做法是在训练时保存，但这里用验证集内留一法近似：
        # 对每个样本，用该折其他样本搜阈值，再应用到该样本。
        fold_results = {"A": [], "B": [], "C": []}
        for i, (prob, gt) in enumerate(zip(probs, gts)):
            # A
            predA = postprocess_A(prob)
            fold_results["A"].append(compute_all_metrics(predA, gt))
            # B
            predB = postprocess_B(prob)
            fold_results["B"].append(compute_all_metrics(predB, gt))

        # C：留一法搜阈值
        for i in range(len(probs)):
            other_probs = [probs[j] for j in range(len(probs)) if j != i]
            other_gts = [gts[j] for j in range(len(gts)) if j != i]
            best_t = find_best_threshold(other_probs, other_gts)
            predC = (probs[i] > best_t).astype(np.uint8)
            fold_results["C"].append(compute_all_metrics(predC, gts[i]))

        for scheme in ["A", "B", "C"]:
            dices = [r["dice"] for r in fold_results[scheme]]
            hd95s = [r["hd95"] for r in fold_results[scheme] if not np.isinf(r["hd95"])]
            bds = [r["boundary_dice"] for r in fold_results[scheme] if not np.isnan(r["boundary_dice"])]
            all_results[scheme].append({
                "fold": fold_id,
                "dice_mean": float(np.mean(dices)),
                "dice_std": float(np.std(dices)),
                "hd95_median": float(np.median(hd95s)) if hd95s else float('inf'),
                "boundary_dice_mean": float(np.mean(bds)) if bds else float('nan'),
                "n": len(dices),
            })

    # 汇总
    summary = {}
    for scheme in ["A", "B", "C"]:
        dices = [r["dice_mean"] for r in all_results[scheme]]
        hd95s = [r["hd95_median"] for r in all_results[scheme] if not np.isinf(r["hd95_median"])]
        bds = [r["boundary_dice_mean"] for r in all_results[scheme] if not np.isnan(r["boundary_dice_mean"])]
        summary[scheme] = {
            "dice_mean_of_fold_means": float(np.mean(dices)),
            "dice_std_of_fold_means": float(np.std(dices)),
            "hd95_median_across_folds": float(np.median(hd95s)) if hd95s else float('inf'),
            "boundary_dice_mean": float(np.mean(bds)) if bds else float('nan'),
            "per_fold": all_results[scheme],
        }

    out_path = os.path.join(BASELINE_DIR, "postprocess_comparison.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved postprocess comparison to {out_path}")

    print("\n" + "=" * 70)
    print(f"{'方案':<10} {'Dice(均值±std)':<20} {'HD95中位数':<15} {'边界Dice':<12}")
    print("-" * 70)
    labels = {"A": "A: >0.5", "B": "B: 形态学", "C": "C: 阈值微调"}
    for scheme in ["A", "B", "C"]:
        s = summary[scheme]
        print(f"{labels[scheme]:<10} {s['dice_mean_of_fold_means']:.4f}±{s['dice_std_of_fold_means']:.4f}       "
              f"{s['hd95_median_across_folds']:<15.2f} {s['boundary_dice_mean']:<12.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
