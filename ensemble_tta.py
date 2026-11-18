"""
TTA + 加权集成 + 形态学后处理 完整流程
配置对比：
  1. 基线单模型 + 形态学 (baseline best, 已有结果)
  2. 基线 5 折等权集成 + TTA + 形态学
  3. 基线 5 折加权集成 + TTA + 形态学
  4. 增强 5 折加权集成 + TTA + 形态学
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"

import json
import glob
import time
import csv
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from models import AttentionUNet
from utils import compute_all_metrics, morphology_postprocess
from config.paths import (
    IMG_DIR, MASK_LEFT_DIR, MASK_RIGHT_DIR, EXP_DIR, TTA_CACHE_DIR,
    IMG_SIZE, BATCH_SIZE,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXP_BASE = EXP_DIR
CACHE_DIR = TTA_CACHE_DIR


class ValDataset(Dataset):
    def __init__(self, img_paths, size=256):
        self.img_paths = img_paths
        self.size = size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        name = os.path.basename(img_path)
        img = Image.open(img_path).convert("L")
        img = img.resize((self.size, self.size), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0), name


def load_gt_mask(name):
    ml = np.array(Image.open(os.path.join(MASK_LEFT_DIR, name)).convert("L"))
    mr = np.array(Image.open(os.path.join(MASK_RIGHT_DIR, name)).convert("L"))
    mask = np.maximum(ml, mr)
    mask = Image.fromarray(mask).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    return (np.array(mask) > 127).astype(np.uint8)


def tta_inference(model, img_tensor):
    """
    P_TTA = 1/2 (sigmoid(f(x)) + flip(sigmoid(f(flip(x)))))
    img_tensor: (B, 1, H, W)
    """
    model.eval()
    with torch.no_grad():
        pred1 = torch.sigmoid(model(img_tensor))
        img_flip = torch.flip(img_tensor, dims=[3])  # 水平翻转
        pred2 = torch.sigmoid(model(img_flip))
        pred2 = torch.flip(pred2, dims=[3])  # 翻回来
    return (pred1 + pred2) / 2.0


def get_fold_weights(exp_name):
    """从逐折 Dice 计算权重 w_i = dice_i / sum(dice_j)"""
    with open(os.path.join(EXP_BASE, exp_name, "summary.json")) as f:
        s = json.load(f)
    dices = [pf["dice_mean"] for pf in s["per_fold"]]
    total = sum(dices)
    weights = [d / total for d in dices]
    return dices, weights


def run_all_tta(exp_name, img_paths, val_name_to_fold):
    """
    对 exp 的 5 个模型，在所有验证样本上跑 TTA。
    TTA 后概率图保存到 CACHE_DIR，文件名 {exp}_fold{fold}_{sample}.npy。
    返回 {name: {model_fold: prob}}。
    """
    exp_dir = os.path.join(EXP_BASE, exp_name)
    os.makedirs(CACHE_DIR, exist_ok=True)
    all_tta = {}  # name -> {model_fold: prob (256x256)}

    for model_fold in range(5):
        model_path = os.path.join(exp_dir, f"fold_{model_fold}", "best_model.pth")
        if not os.path.exists(model_path):
            print(f"  [WARN] {exp_name} fold_{model_fold} best_model.pth missing")
            continue
        model = AttentionUNet(in_ch=1, out_ch=1).to(DEVICE)
        state = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()

        ds = ValDataset(img_paths, size=IMG_SIZE)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        t0 = time.time()
        for imgs, names in loader:
            imgs = imgs.to(DEVICE)
            probs = tta_inference(model, imgs).cpu().numpy()
            for i, name in enumerate(names):
                if name not in all_tta:
                    all_tta[name] = {}
                prob = probs[i, 0]
                all_tta[name][model_fold] = prob
                # 保存 TTA 概率图缓存: {exp}_fold{fold}_{sample}.npy
                cache_name = f"{exp_name}_fold{model_fold}_{name.replace('.png', '.npy')}"
                np.save(os.path.join(CACHE_DIR, cache_name), prob)
        print(f"  {exp_name} model_fold={model_fold} TTA done: {time.time()-t0:.0f}s")

        del model
        import gc
        gc.collect()

    return all_tta


def load_tta_from_cache(exp_name, val_names, model_folds=range(5)):
    """
    从 CACHE_DIR 读取已保存的 TTA 概率图，返回 {name: {model_fold: prob}}。
    若某个 (name, model_fold) 缺失则跳过该模型对该样本的预测。
    """
    all_tta = {}
    missing = []
    for name in val_names:
        all_tta[name] = {}
        for mf in model_folds:
            cache_name = f"{exp_name}_fold{mf}_{name.replace('.png', '.npy')}"
            cache_path = os.path.join(CACHE_DIR, cache_name)
            if os.path.exists(cache_path):
                all_tta[name][mf] = np.load(cache_path)
            else:
                missing.append((name, mf))
    if missing:
        print(f"  [WARN] {exp_name}: {len(missing)} missing cache entries")
    return all_tta


def evaluate_ensemble(all_tta, val_names, weights_dict, use_morphology=True,
                      strict_oof=False, val_name_to_fold=None):
    """
    all_tta: {name: {model_fold: prob}}
    weights_dict: {model_fold: weight} or None for equal weight
    strict_oof: 若为 True，每个样本只用其所属验证折的模型（val_name_to_fold[name]）
                的 TTA 概率，不做跨折集成（严格 OOF，无泄漏）。
    """
    results = []
    for name in val_names:
        if name not in all_tta:
            continue
        model_probs = all_tta[name]

        if strict_oof:
            # 严格 OOF：只用该样本所属验证折的模型
            own_fold = val_name_to_fold[name]
            p_ens = model_probs[own_fold]
            fold_id = own_fold
        elif weights_dict is None:
            # 等权集成
            n = len(model_probs)
            p_ens = sum(model_probs.values()) / n
            fold_id = -1  # ensemble
        else:
            # 加权集成
            total_w = sum(weights_dict[mf] for mf in model_probs)
            p_ens = sum(weights_dict[mf] * model_probs[mf] for mf in model_probs) / total_w
            fold_id = -1  # ensemble

        pred_bin = (p_ens > 0.5).astype(np.uint8)
        if use_morphology:
            pred_bin = morphology_postprocess(pred_bin)
        gt = load_gt_mask(name)
        m = compute_all_metrics(pred_bin, gt)
        results.append({"name": name, "fold": fold_id, **m})
    return results


def summarize(results, label):
    dices = [r["dice"] for r in results]
    hd95s = [r["hd95"] for r in results if not np.isinf(r["hd95"])]
    bds = [r["boundary_dice"] for r in results if not np.isnan(r["boundary_dice"])]
    print(f"{label:<55s} dice={np.mean(dices):.4f}  hd95_med={np.median(hd95s):.2f}  bdice={np.mean(bds):.4f}")
    return {
        "label": label,
        "n": len(results),
        "dice_mean": float(np.mean(dices)),
        "dice_std": float(np.std(dices)),
        "hd95_median": float(np.median(hd95s)),
        "hd95_mean": float(np.mean(hd95s)),
        "boundary_dice_mean": float(np.mean(bds)),
    }


def main():
    # 加载折划分
    with open(os.path.join(EXP_BASE, "folds.json")) as f:
        folds = json.load(f)
    img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))

    # 建立 val_name -> 所属折 的映射
    val_name_to_fold = {}
    for fold_id, fold in enumerate(folds):
        for idx in fold["val"]:
            name = os.path.basename(img_paths[idx])
            val_name_to_fold[name] = fold_id
    val_names = sorted(val_name_to_fold.keys())
    print(f"Total validation samples: {len(val_names)}")

    # Step 1: 计算权重
    print("\n=== Step 1: 计算权重 ===")
    base_dices, base_weights = get_fold_weights("baseline")
    combo_dices, combo_weights = get_fold_weights("combo")
    print(f"Baseline fold dices: {[f'{d:.4f}' for d in base_dices]}")
    print(f"Baseline weights:    {[f'{w:.4f}' for w in base_weights]}")
    print(f"Combo fold dices:    {[f'{d:.4f}' for d in combo_dices]}")
    print(f"Combo weights:       {[f'{w:.4f}' for w in combo_weights]}")

    # Step 2: TTA 推理（优先从缓存加载，缺失才跑推理）
    print("\n=== Step 2: TTA 推理（优先读缓存） ===")
    base_tta = load_tta_from_cache("baseline", val_names)
    base_complete = all(len(base_tta.get(n, {})) == 5 for n in val_names)
    if base_complete:
        print(f"  baseline: cache hit ({len(base_tta)} samples, all 5 folds)")
    else:
        print("  baseline: cache incomplete, running inference...")
        base_tta = run_all_tta("baseline", img_paths, val_name_to_fold)

    combo_tta = load_tta_from_cache("combo", val_names)
    combo_complete = all(len(combo_tta.get(n, {})) == 5 for n in val_names)
    if combo_complete:
        print(f"  combo: cache hit ({len(combo_tta)} samples, all 5 folds)")
    else:
        print("  combo: cache incomplete, running inference...")
        combo_tta = run_all_tta("combo", img_paths, val_name_to_fold)
    print(f"Baseline TTA samples: {len(base_tta)}")
    print(f"Combo TTA samples: {len(combo_tta)}")

    # Step 3+4: 集成 + 形态学 + 评估
    print("\n=== Step 3-5: 集成 + 形态学 + 评估 ===")
    print(f"{'配置':<55s} {'Dice':<10s} {'HD95':<10s} {'bdice':<10s}")
    print("-" * 90)

    all_summaries = []
    all_per_sample = {}  # label -> per_sample list

    # 配置 1: 基线严格 OOF + TTA + 形态学（无跨折泄漏，真实泛化估计）
    r = evaluate_ensemble(base_tta, val_names, weights_dict=None, use_morphology=True,
                          strict_oof=True, val_name_to_fold=val_name_to_fold)
    label = "baseline strict-OOF + TTA + morph (no leakage)"
    all_summaries.append(summarize(r, label))
    all_per_sample[label] = r

    # 配置 2: 基线等权集成 + TTA + 形态学
    r = evaluate_ensemble(base_tta, val_names, weights_dict=None, use_morphology=True)
    label = "baseline equal-weight ensemble + TTA + morph"
    all_summaries.append(summarize(r, label))
    all_per_sample[label] = r

    # 配置 3: 基线加权集成 + TTA + 形态学
    w_dict = {i: base_weights[i] for i in range(5)}
    r = evaluate_ensemble(base_tta, val_names, weights_dict=w_dict, use_morphology=True)
    label = "baseline weighted ensemble + TTA + morph"
    all_summaries.append(summarize(r, label))
    all_per_sample[label] = r

    # 配置 4: 增强加权集成 + TTA + 形态学
    w_dict = {i: combo_weights[i] for i in range(5)}
    r = evaluate_ensemble(combo_tta, val_names, weights_dict=w_dict, use_morphology=True)
    label = "combo weighted ensemble + TTA + morph"
    all_summaries.append(summarize(r, label))
    all_per_sample[label] = r

    # 对照: 基线单模型 + 形态学 (从 postprocess_comparison.json 读取)
    with open(os.path.join(EXP_BASE, "baseline", "postprocess_comparison.json")) as f:
        pp = json.load(f)
    b_summary = {
        "label": "baseline single model + morph (reference)",
        "dice_mean": pp["B"]["dice_mean_of_fold_means"],
        "hd95_median": pp["B"]["hd95_median_across_folds"],
        "boundary_dice_mean": pp["B"]["boundary_dice_mean"],
    }
    all_summaries.insert(0, b_summary)
    print(f"{b_summary['label']:<55s} dice={b_summary['dice_mean']:.4f}  hd95_med={b_summary['hd95_median']:.2f}  bdice={b_summary['boundary_dice_mean']:.4f}")

    # 保存逐样本 CSV（每个配置一个文件，方便 pandas 分析）
    import csv
    csv_dir = os.path.join(EXP_BASE, "ensemble_csv")
    os.makedirs(csv_dir, exist_ok=True)
    for label, per_sample in all_per_sample.items():
        safe_name = label.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        csv_path = os.path.join(csv_dir, f"{safe_name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "fold", "dice", "hd95", "boundary_dice"])
            writer.writeheader()
            for row in per_sample:
                writer.writerow({k: row[k] for k in ["name", "fold", "dice", "hd95", "boundary_dice"]})
        print(f"  CSV saved: {csv_path}")

    # 保存 summary JSON
    out = {
        "weights": {
            "baseline_dices": base_dices,
            "baseline_weights": base_weights,
            "combo_dices": combo_dices,
            "combo_weights": combo_weights,
        },
        "configurations": all_summaries,
    }
    out_path = os.path.join(EXP_BASE, "ensemble_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
