"""
第5步：边界损失实验
在数据加载阶段预计算 GT 距离图，损失 = Dice_loss + alpha * Boundary_loss
用法：
  python boundary_loss_train.py --alpha 0.1 --folds 0,1   # 先跑1-2折快速筛选
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
from PIL import Image
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

from models import AttentionUNet
from utils import combined_loss, compute_all_metrics
from config.paths import (
    IMG_DIR, MASK_LEFT_DIR, MASK_RIGHT_DIR, EXP_DIR,
    IMG_SIZE, BATCH_SIZE, EPOCHS, LR,
)

SEED = 42
N_SPLITS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BoundaryDataset(Dataset):
    """返回 img, mask, dist_map"""
    def __init__(self, img_paths, mask_left_dir, mask_right_dir, size=256, augment='flip'):
        self.img_paths = img_paths
        self.mask_left_dir = mask_left_dir
        self.mask_right_dir = mask_right_dir
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.img_paths)

    def _load_mask(self, name):
        ml = np.array(Image.open(os.path.join(self.mask_left_dir, name)).convert("L"))
        mr = np.array(Image.open(os.path.join(self.mask_right_dir, name)).convert("L"))
        return np.maximum(ml, mr)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        name = os.path.basename(img_path)
        img = Image.open(img_path).convert("L")
        img = img.resize((self.size, self.size), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        mask = self._load_mask(name)
        mask = Image.fromarray(mask).resize((self.size, self.size), Image.NEAREST)
        mask = (np.array(mask) > 127).astype(np.float32)

        if self.augment == 'flip' and np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        # 预计算距离图：背景到前景边界的距离
        # 边界损失公式：L_boundary = (pred_prob * dist_map).mean()
        # dist_map: 前景内部为负距离（鼓励预测），外部为正距离（惩罚预测）
        # 标准实现：dist = distance_transform_edt(~mask) - distance_transform_edt(mask)
        fg = mask > 0.5
        dt_bg = distance_transform_edt(~fg)
        dt_fg = distance_transform_edt(fg)
        dist_map = dt_bg - dt_fg  # 前景内为负，背景为正
        dist_map = dist_map.astype(np.float32)

        img_t = torch.from_numpy(img).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        dist_t = torch.from_numpy(dist_map).unsqueeze(0)
        return img_t, mask_t, dist_t, name


def boundary_loss_fn(pred, dist_map):
    """pred: logits, dist_map: 距离图"""
    prob = torch.sigmoid(pred)
    return (prob * dist_map).mean()


def train_one_fold(fold_id, train_paths, val_paths, alpha, out_dir):
    fold_dir = os.path.join(out_dir, f"fold_{fold_id}")
    os.makedirs(fold_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train_log.txt")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"=== Fold {fold_id} | alpha={alpha} ===")
    train_ds = BoundaryDataset(train_paths, MASK_LEFT_DIR, MASK_RIGHT_DIR, size=IMG_SIZE, augment='flip')
    val_ds = BoundaryDataset(val_paths, MASK_LEFT_DIR, MASK_RIGHT_DIR, size=IMG_SIZE, augment=None)
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
        for imgs, masks, dists, _ in train_loader:
            imgs, masks, dists = imgs.to(DEVICE), masks.to(DEVICE), dists.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = combined_loss(preds, masks) + alpha * boundary_loss_fn(preds, dists)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)

        model.eval()
        v_loss, v_dice = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks, _, _ in val_loader:
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
        log(f"  Epoch {epoch:02d}/{EPOCHS} | loss={epoch_loss:.4f} | vdice={v_dice:.4f} | {dt:.0f}s")

        if v_dice > best_dice:
            best_dice = v_dice
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    torch.save(best_state, os.path.join(fold_dir, "best_model.pth"))
    log(f"  Fold {fold_id} best dice={best_dice:.4f}")

    # 验证集逐样本指标
    model.load_state_dict(best_state)
    model.eval()
    per_sample = []
    with torch.no_grad():
        for imgs, masks, _, names in val_loader:
            imgs = imgs.to(DEVICE)
            probs = torch.sigmoid(model(imgs)).cpu().numpy()
            for i in range(len(names)):
                pred_bin = (probs[i, 0] > 0.5).astype(np.uint8)
                target_bin = masks[i, 0].numpy().astype(np.uint8)
                m = compute_all_metrics(pred_bin, target_bin)
                per_sample.append({"name": names[i], **m})

    with open(os.path.join(fold_dir, "per_sample_metrics.json"), "w") as f:
        json.dump(per_sample, f, indent=2)

    dices = [s["dice"] for s in per_sample]
    hd95s = [s["hd95"] for s in per_sample if not np.isinf(s["hd95"])]
    bds = [s["boundary_dice"] for s in per_sample if not np.isnan(s["boundary_dice"])]
    log(f"  Fold {fold_id}: dice={np.mean(dices):.4f}±{np.std(dices):.4f} | "
        f"hd95_med={np.median(hd95s):.2f} | bdice={np.mean(bds):.4f}")

    # 释放内存
    del model, optimizer, scheduler, best_state
    import gc
    gc.collect()
    return per_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--folds", default="0,1")
    args = parser.parse_args()

    out_dir = os.path.join(EXP_DIR, f"boundary_alpha{args.alpha}")
    os.makedirs(out_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
    folds_path = os.path.join(EXP_DIR, "folds.json")
    with open(folds_path) as f:
        folds = json.load(f)

    target_folds = [int(x) for x in args.folds.split(",")]
    all_per_sample = []
    for fold_id in target_folds:
        fold = folds[fold_id]
        train_paths = [img_paths[i] for i in fold["train"]]
        val_paths = [img_paths[i] for i in fold["val"]]
        ps = train_one_fold(fold_id, train_paths, val_paths, args.alpha, out_dir)
        all_per_sample.append(ps)

    all_dice = [s["dice"] for ps in all_per_sample for s in ps]
    all_hd95 = [s["hd95"] for ps in all_per_sample for s in ps if not np.isinf(s["hd95"])]
    all_bd = [s["boundary_dice"] for ps in all_per_sample for s in ps if not np.isnan(s["boundary_dice"])]
    summary = {
        "alpha": args.alpha,
        "folds": target_folds,
        "dice_mean": float(np.mean(all_dice)),
        "dice_std": float(np.std(all_dice)),
        "hd95_median": float(np.median(all_hd95)),
        "boundary_dice_mean": float(np.mean(all_bd)),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSUMMARY [alpha={args.alpha}]:")
    print(f"  Dice: {summary['dice_mean']:.4f} ± {summary['dice_std']:.4f}")
    print(f"  HD95 median: {summary['hd95_median']:.2f}")
    print(f"  Boundary Dice: {summary['boundary_dice_mean']:.4f}")


if __name__ == "__main__":
    main()
