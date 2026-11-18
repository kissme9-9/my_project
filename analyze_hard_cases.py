"""
第2步：困难样本可视化排查
输入：experiments/baseline/fold_*/per_sample_metrics.json + probs/*.npy
输出：
  - hard_cases_top8.json：Top8 困难样本及跨折重复性
  - hard_case_analysis.png：四联图（原图/GT/概率热力图/误差图）
  - conclusion.txt：一句话结论
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import glob
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.paths import (
    IMG_DIR, MASK_LEFT_DIR, MASK_RIGHT_DIR, EXP_DIR, IMG_SIZE,
)
BASELINE_DIR = os.path.join(EXP_DIR, "baseline")


def load_img_and_mask(name):
    img = Image.open(os.path.join(IMG_DIR, name)).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255.0
    ml = np.array(Image.open(os.path.join(MASK_LEFT_DIR, name)).convert("L"))
    mr = np.array(Image.open(os.path.join(MASK_RIGHT_DIR, name)).convert("L"))
    mask = np.maximum(ml, mr)
    mask = Image.fromarray(mask).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    mask = (np.array(mask) > 127).astype(np.uint8)
    return img, mask


def main():
    # 收集所有折的逐样本指标
    all_metrics = []
    for fold_id in range(5):
        path = os.path.join(BASELINE_DIR, f"fold_{fold_id}", "per_sample_metrics.json")
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping")
            continue
        with open(path) as f:
            samples = json.load(f)
        for s in samples:
            s["fold"] = fold_id
            all_metrics.append(s)

    print(f"Total samples across folds: {len(all_metrics)}")

    # 按 HD95 从大到小排序（HD95 越大越差），过滤 inf
    valid = [s for s in all_metrics if s["hd95"] != float('inf')]
    valid.sort(key=lambda x: x["hd95"], reverse=True)

    # Top 8 困难样本（按样本名去重，取每个样本最差的一次）
    name_to_worst = {}
    for s in valid:
        name = s["name"]
        if name not in name_to_worst or s["hd95"] > name_to_worst[name]["hd95"]:
            name_to_worst[name] = s
    top8 = sorted(name_to_worst.values(), key=lambda x: x["hd95"], reverse=True)[:8]

    # 统计每个样本在 5 折中进入 Top8 的次数（这里用 HD95 > 中位数阈值近似）
    # 更准确：统计每个样本在多少折里 HD95 排前 8
    fold_top8_names = []
    for fold_id in range(5):
        fold_samples = [s for s in all_metrics if s["fold"] == fold_id and s["hd95"] != float('inf')]
        fold_samples.sort(key=lambda x: x["hd95"], reverse=True)
        fold_top8_names.append(set(s["name"] for s in fold_samples[:8]))

    for s in top8:
        count = sum(1 for names in fold_top8_names if s["name"] in names)
        s["cross_fold_count"] = count
        s["is_structural"] = count >= 3

    # 保存 Top8
    out_path = os.path.join(BASELINE_DIR, "hard_cases_top8.json")
    with open(out_path, "w") as f:
        json.dump(top8, f, indent=2)
    print(f"Saved Top8 hard cases to {out_path}")

    # 画四联图：原图 / GT / 概率热力图 / 误差图
    structural = [s for s in top8 if s["is_structural"]]
    print(f"Structural hard cases (>=3 folds): {len(structural)}")
    n_show = min(len(structural), 6)
    if n_show == 0:
        n_show = min(len(top8), 6)
        show_list = top8[:n_show]
    else:
        show_list = structural[:n_show]

    fig, axes = plt.subplots(n_show, 4, figsize=(16, 4 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]
    cols = ["Original", "Ground Truth", "Prob Heatmap", "Error (red=FP, blue=FN)"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=11)

    for row, s in enumerate(show_list):
        name = s["name"]
        img, gt = load_img_and_mask(name)
        # 加载该样本在其最差折的概率图
        fold_id = s["fold"]
        prob_path = os.path.join(BASELINE_DIR, f"fold_{fold_id}", "probs", name.replace(".png", ".npy"))
        if os.path.exists(prob_path):
            prob = np.load(prob_path)
        else:
            prob = np.zeros_like(gt, dtype=np.float32)
        pred = (prob > 0.5).astype(np.uint8)

        # 误差图：FP 红，FN 蓝
        error = np.zeros((*gt.shape, 3), dtype=np.float32)
        fp = np.logical_and(pred == 1, gt == 0)
        fn = np.logical_and(pred == 0, gt == 1)
        error[fp] = [1, 0, 0]
        error[fn] = [0, 0, 1]
        # 叠加原图灰度
        gray = np.stack([img] * 3, axis=-1) * 0.5
        error_overlay = np.clip(gray + error * 0.7, 0, 1)

        axes[row, 0].imshow(img, cmap="gray")
        axes[row, 1].imshow(gt, cmap="gray")
        axes[row, 2].imshow(img, cmap="gray")
        axes[row, 2].imshow(prob, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[row, 3].imshow(error_overlay)
        for ax in axes[row]:
            ax.axis("off")
        axes[row, 0].set_ylabel(f"{name}\nHD95={s['hd95']:.1f} Dice={s['dice']:.3f}\nfolds={s['cross_fold_count']}", fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(BASELINE_DIR, "hard_case_analysis.png")
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(f"Saved hard case analysis to {fig_path}")

    # 输出一句话结论
    fp_counts = []
    fn_counts = []
    regions = []
    for s in show_list:
        name = s["name"]
        img, gt = load_img_and_mask(name)
        fold_id = s["fold"]
        prob_path = os.path.join(BASELINE_DIR, f"fold_{fold_id}", "probs", name.replace(".png", ".npy"))
        if os.path.exists(prob_path):
            prob = np.load(prob_path)
            pred = (prob > 0.5).astype(np.uint8)
            fp = np.logical_and(pred == 1, gt == 0).sum()
            fn = np.logical_and(pred == 0, gt == 1).sum()
            fp_counts.append(fp)
            fn_counts.append(fn)
            # 判断误差区域：肺底（下方）/肺尖（上方）/心影旁（中间偏下）
            fn_mask = np.logical_and(pred == 0, gt == 1)
            if fn_mask.sum() > 0:
                ys, _ = np.where(fn_mask)
                if np.mean(ys) > gt.shape[0] * 0.7:
                    regions.append("肺底/膈肌")
                elif np.mean(ys) < gt.shape[0] * 0.3:
                    regions.append("肺尖")
                else:
                    regions.append("心影旁/中部")
            else:
                regions.append("无明显FN")

    from collections import Counter
    region_counter = Counter(regions)
    dominant_region = region_counter.most_common(1)[0][0] if regions else "未知"
    fp_total = sum(fp_counts)
    fn_total = sum(fn_counts)
    error_type = "假阴性(FN)为主" if fn_total > fp_total else "假阳性(FP)为主"
    n_structural = len(structural)

    conclusion = (f"困难样本主要误差区域为「{dominant_region}」，"
                  f"误差类型以「{error_type}」为主，"
                  f"其中跨折稳定（≥3折）的结构性困难样本有 {n_structural} 个。")
    with open(os.path.join(BASELINE_DIR, "conclusion.txt"), "w") as f:
        f.write(conclusion + "\n")
    print(f"\n结论：{conclusion}")


if __name__ == "__main__":
    main()
