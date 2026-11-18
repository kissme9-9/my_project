"""
5 折交叉验证训练脚本
用法：
  python kfold_train.py --augment baseline    # 第1步：基线（仅水平翻转）
  python kfold_train.py --augment combo       # 第3步：组合增强
  python kfold_train.py --augment baseline --folds 0,1   # 只跑前2折
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"

import sys
import time
import json
import argparse
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from models import AttentionUNet
from utils import MontgomeryDataset, combined_loss, compute_all_metrics
from config.paths import (
    IMG_DIR, MASK_LEFT_DIR, MASK_RIGHT_DIR, EXP_DIR,
    IMG_SIZE, BATCH_SIZE, EPOCHS, LR,
)

# ==================== 配置 ====================
SEED = 42
N_SPLITS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_out_dir(augment):
    return os.path.join(EXP_DIR, augment)


def log(msg, out_dir):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(out_dir, "train_log.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_folds(img_paths):
    """固定 5 折划分，保存到 folds.json"""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = []
    for train_idx, val_idx in kf.split(img_paths):
        folds.append({
            "train": [int(i) for i in train_idx],
            "val": [int(i) for i in val_idx],
        })
    return folds


def train_one_fold(fold_id, train_paths, val_paths, augment, out_dir):
    """训练单折，保存模型和验证集概率图，返回逐样本指标"""
    log(f"=== Fold {fold_id} | augment={augment} ===", out_dir)
    log(f"  train={len(train_paths)}, val={len(val_paths)}", out_dir)

    train_ds = MontgomeryDataset(train_paths, MASK_LEFT_DIR, MASK_RIGHT_DIR,
                                 size=IMG_SIZE, augment=augment)
    val_ds = MontgomeryDataset(val_paths, MASK_LEFT_DIR, MASK_RIGHT_DIR,
                               size=IMG_SIZE, augment=None)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = AttentionUNet(in_ch=1, out_ch=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_dice = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = combined_loss(preds, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)

        # 验证
        model.eval()
        v_loss, v_dice = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs)
                v_loss += combined_loss(preds, masks).item()
                pred_bin = (torch.sigmoid(preds) > 0.5).float()
                inter = (pred_bin * masks).sum()
                v_dice += (2 * inter + 1e-6) / (pred_bin.sum() + masks.sum() + 1e-6)
        v_loss /= len(val_loader)
        v_dice /= len(val_loader)
        scheduler.step(v_loss)
        dt = time.time() - t0
        log(f"  Epoch {epoch:02d}/{EPOCHS} | loss={epoch_loss:.4f} | "
            f"vloss={v_loss:.4f} | vdice={v_dice:.4f} | {dt:.0f}s", out_dir)

        if v_dice > best_dice:
            best_dice = v_dice
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 保存最佳模型
    fold_dir = os.path.join(out_dir, f"fold_{fold_id}")
    os.makedirs(fold_dir, exist_ok=True)
    torch.save(best_state, os.path.join(fold_dir, "best_model.pth"))
    log(f"  Fold {fold_id} best dice={best_dice:.4f}, model saved", out_dir)

    # 用最佳模型跑验证集，保存概率图 + 逐样本指标
    model.load_state_dict(best_state)
    model.eval()
    per_sample = []
    prob_dir = os.path.join(fold_dir, "probs")
    os.makedirs(prob_dir, exist_ok=True)

    with torch.no_grad():
        for imgs, masks, names in val_loader:
            imgs = imgs.to(DEVICE)
            probs = torch.sigmoid(model(imgs)).cpu().numpy()
            for i in range(len(names)):
                name = names[i]
                prob = probs[i, 0]
                np.save(os.path.join(prob_dir, name.replace(".png", ".npy")), prob)
                pred_bin = (prob > 0.5).astype(np.uint8)
                target_bin = masks[i, 0].numpy().astype(np.uint8)
                m = compute_all_metrics(pred_bin, target_bin)
                per_sample.append({"name": name, **m})

    # 保存逐样本指标
    with open(os.path.join(fold_dir, "per_sample_metrics.json"), "w") as f:
        json.dump(per_sample, f, indent=2)

    # 本折统计
    dices = [s["dice"] for s in per_sample]
    hd95s = [s["hd95"] for s in per_sample if not np.isinf(s["hd95"])]
    bds = [s["boundary_dice"] for s in per_sample if not np.isnan(s["boundary_dice"])]
    log(f"  Fold {fold_id} stats: dice={np.mean(dices):.4f}±{np.std(dices):.4f} | "
        f"hd95_median={np.median(hd95s):.2f} | bdice={np.mean(bds):.4f}", out_dir)

    # 释放内存
    del model, optimizer, scheduler, best_state
    import gc
    gc.collect()
    return per_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", default="baseline", choices=["baseline", "combo"])
    parser.add_argument("--folds", default="0,1,2,3,4", help="逗号分隔的折号")
    args = parser.parse_args()

    augment = "flip" if args.augment == "baseline" else "combo"
    out_dir = get_out_dir(args.augment)
    os.makedirs(out_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
    log(f"Total images: {len(img_paths)}", out_dir)
    log(f"Augment: {args.augment} ({augment})", out_dir)
    log(f"Device: {DEVICE}", out_dir)

    # 固定折划分
    folds_path = os.path.join(EXP_DIR, "folds.json")
    if os.path.exists(folds_path):
        with open(folds_path) as f:
            folds = json.load(f)
        log(f"Loaded existing folds from {folds_path}", out_dir)
    else:
        folds = build_folds(img_paths)
        with open(folds_path, "w") as f:
            json.dump(folds, f, indent=2)
        log(f"Saved folds to {folds_path}", out_dir)

    target_folds = [int(x) for x in args.folds.split(",")]
    all_per_sample = []

    for fold_id in target_folds:
        fold = folds[fold_id]
        train_paths = [img_paths[i] for i in fold["train"]]
        val_paths = [img_paths[i] for i in fold["val"]]
        ps = train_one_fold(fold_id, train_paths, val_paths, augment, out_dir)
        all_per_sample.append(ps)

    # 汇总所有折
    all_dice = [s["dice"] for ps in all_per_sample for s in ps]
    all_hd95 = [s["hd95"] for ps in all_per_sample for s in ps if not np.isinf(s["hd95"])]
    all_bd = [s["boundary_dice"] for ps in all_per_sample for s in ps if not np.isnan(s["boundary_dice"])]

    summary = {
        "augment": args.augment,
        "folds_run": target_folds,
        "dice_mean": float(np.mean(all_dice)),
        "dice_std": float(np.std(all_dice)),
        "hd95_median": float(np.median(all_hd95)),
        "boundary_dice_mean": float(np.mean(all_bd)),
        "n_samples": len(all_dice),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log("=" * 60, out_dir)
    log(f"SUMMARY [{args.augment}]", out_dir)
    log(f"  Dice: {summary['dice_mean']:.4f} ± {summary['dice_std']:.4f}", out_dir)
    log(f"  HD95 median: {summary['hd95_median']:.2f} px", out_dir)
    log(f"  Boundary Dice: {summary['boundary_dice_mean']:.4f}", out_dir)
    log(f"  Samples: {summary['n_samples']}", out_dir)
    log("=" * 60, out_dir)


if __name__ == "__main__":
    main()
