"""
单图推理脚本（部署配置）
输入：单张胸部 X 光图路径
输出：肺部分割掩膜(.png) + 概率图(.npy)
内部：加载 5 折模型 → TTA(水平翻转) → 等权平均 → 形态学后处理 → 保存

用法:
  python inference.py <img_path> [--exp baseline|combo] [--outdir <dir>]
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"

import argparse
import numpy as np
from PIL import Image
import torch

from models import AttentionUNet
from utils import morphology_postprocess
from config.paths import EXP_DIR, IMG_SIZE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXP_BASE = EXP_DIR


def load_image(img_path):
    """加载单张 X 光图，resize 到 256x256，归一化到 [0,1]，返回 (1,1,256,256) tensor 和原始图像。"""
    img = Image.open(img_path).convert("L")
    orig_size = img.size  # (W, H)
    img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0), orig_size, img_resized


def tta_single(model, img_tensor):
    """对单张图做 TTA：P = 1/2 (sigmoid(f(x)) + flip(sigmoid(f(flip(x)))))"""
    model.eval()
    with torch.no_grad():
        p1 = torch.sigmoid(model(img_tensor))
        img_flip = torch.flip(img_tensor, dims=[3])
        p2 = torch.sigmoid(model(img_flip))
        p2 = torch.flip(p2, dims=[3])
    return ((p1 + p2) / 2.0).cpu().numpy()[0, 0]


def segment_image(img_path, exp_name="baseline", output_dir=None):
    """
    对单张图做分割。
    返回: (mask_uint8, prob_float32)
    同时保存掩膜 PNG 和概率图 NPY 到 output_dir（若指定）。
    """
    img_tensor, orig_size, img_resized = load_image(img_path)
    img_tensor = img_tensor.to(DEVICE)

    exp_dir = os.path.join(EXP_BASE, exp_name)
    probs = []
    for fold_id in range(5):
        model_path = os.path.join(exp_dir, f"fold_{fold_id}", "best_model.pth")
        if not os.path.exists(model_path):
            continue
        model = AttentionUNet(in_ch=1, out_ch=1).to(DEVICE)
        state = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state)
        probs.append(tta_single(model, img_tensor))
        del model
        import gc
        gc.collect()

    if not probs:
        raise RuntimeError(f"No model found in {exp_dir}")

    # 等权平均
    p_ens = np.mean(probs, axis=0)

    # 二值化 + 形态学
    pred_bin = (p_ens > 0.5).astype(np.uint8)
    pred_bin = morphology_postprocess(pred_bin)

    # 恢复原始尺寸
    # 二值掩膜必须用最近邻，避免产生中间灰度值
    mask_pil = Image.fromarray(pred_bin * 255).resize(orig_size, Image.NEAREST)
    mask_np = np.array(mask_pil)
    # 概率图用双线性插值恢复原尺寸
    prob_pil = Image.fromarray((p_ens * 255).astype(np.uint8)).resize(orig_size, Image.BILINEAR)
    prob_np = np.array(prob_pil).astype(np.float32) / 255.0

    name = os.path.splitext(os.path.basename(img_path))[0]
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        mask_pil.save(os.path.join(output_dir, f"{name}_mask.png"))
        np.save(os.path.join(output_dir, f"{name}_prob.npy"), prob_np)
        # 保存叠加图（红色掩膜覆盖）
        img_orig = Image.open(img_path).convert("RGB")
        overlay = np.array(img_orig).copy()
        mask_full = mask_np > 127
        overlay[mask_full] = (overlay[mask_full] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
        Image.fromarray(overlay).save(os.path.join(output_dir, f"{name}_overlay.png"))

    return mask_np, prob_np


def main():
    parser = argparse.ArgumentParser(description="Lung segmentation inference")
    parser.add_argument("img_path", help="Path to chest X-ray image")
    parser.add_argument("--exp", default="baseline", choices=["baseline", "combo"],
                        help="Which experiment's 5-fold models to use (default: baseline)")
    parser.add_argument("--outdir", default=None, help="Output directory (default: same dir as image)")
    args = parser.parse_args()

    if not os.path.exists(args.img_path):
        raise FileNotFoundError(f"Image not found: {args.img_path}")

    output_dir = args.outdir or os.path.dirname(args.img_path)
    mask, prob = segment_image(args.img_path, exp_name=args.exp, output_dir=output_dir)

    print(f"Segmentation done.")
    print(f"  Mask: {mask.shape}, lung pixels: {mask.sum()}")
    print(f"  Prob range: [{prob.min():.4f}, {prob.max():.4f}]")
    if output_dir:
        name = os.path.splitext(os.path.basename(args.img_path))[0]
        print(f"  Saved to: {os.path.join(output_dir, name + '_mask.png')}")


if __name__ == "__main__":
    main()
