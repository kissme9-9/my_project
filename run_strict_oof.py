"""
快速跑 strict_OOF：每个样本只用其所属验证折的模型做 TTA + 形态学。
只加载每个 fold 的模型，只推理该 fold 的验证集样本（~28张），无需全量推理。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"

import json
import glob
import time
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from models import AttentionUNet
from utils import compute_all_metrics, morphology_postprocess
from ensemble_tta import ValDataset, tta_inference, load_gt_mask, summarize
from config.paths import IMG_DIR, EXP_DIR, IMG_SIZE, BATCH_SIZE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXP_BASE = EXP_DIR


def run_strict_oof(exp_name):
    exp_dir = os.path.join(EXP_BASE, exp_name)
    with open(os.path.join(EXP_BASE, "folds.json")) as f:
        folds = json.load(f)
    img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))

    all_results = []
    for fold_id in range(5):
        model_path = os.path.join(exp_dir, f"fold_{fold_id}", "best_model.pth")
        if not os.path.exists(model_path):
            print(f"  [WARN] fold_{fold_id} best_model.pth missing")
            continue

        val_idx = folds[fold_id]["val"]
        val_paths = [img_paths[i] for i in val_idx]
        ds = ValDataset(val_paths, size=IMG_SIZE)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = AttentionUNet(in_ch=1, out_ch=1).to(DEVICE)
        state = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()

        t0 = time.time()
        fold_results = []
        with torch.no_grad():
            for imgs, names in loader:
                imgs = imgs.to(DEVICE)
                probs = tta_inference(model, imgs).cpu().numpy()
                for i, name in enumerate(names):
                    p = probs[i, 0]
                    pred_bin = (p > 0.5).astype(np.uint8)
                    pred_bin = morphology_postprocess(pred_bin)
                    gt = load_gt_mask(name)
                    m = compute_all_metrics(pred_bin, gt)
                    fold_results.append({"name": name, "fold": fold_id, **m})
        all_results.extend(fold_results)
        dices = [r["dice"] for r in fold_results]
        hd95s = [r["hd95"] for r in fold_results if not np.isinf(r["hd95"])]
        bds = [r["boundary_dice"] for r in fold_results if not np.isnan(r["boundary_dice"])]
        print(f"  {exp_name} fold_{fold_id}: n={len(fold_results)} "
              f"dice={np.mean(dices):.4f} hd95_med={np.median(hd95s):.2f} "
              f"bdice={np.mean(bds):.4f}  ({time.time()-t0:.0f}s)")

        del model
        import gc
        gc.collect()

    return all_results


def main():
    print("=== Strict OOF + TTA + Morphology ===")
    for exp in ["baseline", "combo"]:
        print(f"\n--- {exp} ---")
        results = run_strict_oof(exp)
        summary = summarize(results, f"{exp} strict-OOF + TTA + morph")
        # 保存逐样本结果
        out_path = os.path.join(EXP_BASE, f"{exp}_strict_oof_results.json")
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "per_sample": results}, f, indent=2, ensure_ascii=False)
        print(f"  -> saved to {out_path}")


if __name__ == "__main__":
    main()
