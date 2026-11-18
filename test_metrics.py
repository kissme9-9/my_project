"""
metrics.py 的单元测试，重点验证修复后的 hausdorff_95。
运行: python test_metrics.py
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.metrics import hausdorff_95, dice_score_np, boundary_dice, compute_all_metrics


def make_square(h, w, r, c, size):
    """构造一个 size x size 的掩膜，在 (r, c) 处有一个 h x w 的矩形。"""
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[r:r + h, c:c + w] = 1
    return mask


def test_identical_masks_hd95_zero():
    """两个完全相同的掩膜 → HD95 = 0"""
    a = make_square(20, 20, 10, 10, 50)
    hd = hausdorff_95(a, a)
    assert hd == 0.0, f"Expected 0.0, got {hd}"
    print(f"[PASS] identical masks: HD95 = {hd}")


def test_one_pixel_shift_hd95():
    """两个相差 1 像素的矩形 → HD95 ≈ 1.0"""
    a = make_square(20, 20, 10, 10, 50)
    b = make_square(20, 20, 11, 10, 50)  # 向下偏移 1 像素
    hd = hausdorff_95(a, b)
    # 边界大部分重合，偏移边的距离为 1
    assert 0.9 <= hd <= 1.1, f"Expected ~1.0, got {hd}"
    print(f"[PASS] 1px shift: HD95 = {hd:.4f}")


def test_nested_masks_hd95_nonzero():
    """一个掩膜完全在另一个内部 → HD95 应反映边界距离，不能是 0"""
    outer = make_square(30, 30, 10, 10, 60)
    inner = make_square(20, 20, 15, 15, 60)  # 完全在 outer 内部，四周各留 5px
    hd = hausdorff_95(inner, outer)
    # 边到边距离 5px，角点对角距离 sqrt(50)≈7.07，95%分位应在(5, 7.5)区间
    assert hd > 0, f"Expected > 0, got {hd}"
    assert 5.0 <= hd <= 7.5, f"Expected in (5, 7.5), got {hd}"
    print(f"[PASS] nested masks: HD95 = {hd:.4f}")


def test_empty_mask_hd95_inf():
    """空掩膜 → HD95 = inf"""
    a = make_square(20, 20, 10, 10, 50)
    empty = np.zeros((50, 50), dtype=np.uint8)
    hd = hausdorff_95(a, empty)
    assert np.isinf(hd), f"Expected inf, got {hd}"
    print(f"[PASS] empty mask: HD95 = {hd}")


def test_partial_overlap_hd95():
    """部分重叠的两个矩形 → HD95 应 > 0 且 < 最大偏移"""
    a = make_square(20, 20, 10, 10, 60)
    b = make_square(20, 20, 10, 20, 60)  # 向右偏移 10 像素，部分重叠
    hd = hausdorff_95(a, b)
    assert hd > 0, f"Expected > 0, got {hd}"
    assert hd <= 10.5, f"Expected <= 10.5, got {hd}"
    print(f"[PASS] partial overlap: HD95 = {hd:.4f}")


def test_dice_score():
    """Dice 基本验证：完全相同 = 1.0，无交集 = 0"""
    a = make_square(20, 20, 10, 10, 50)
    assert dice_score_np(a, a) == 1.0
    b = make_square(10, 10, 35, 35, 50)  # 不重叠
    assert dice_score_np(a, b) < 1e-3
    print(f"[PASS] dice_score: identical=1.0, disjoint={dice_score_np(a, b):.6f}")


def test_boundary_dice():
    """边界 Dice：完全相同 = 1.0（边界带内全匹配）"""
    a = make_square(20, 20, 10, 10, 50)
    bd = boundary_dice(a, a, erode_pixels=3)
    assert bd > 0.99, f"Expected ~1.0, got {bd}"
    print(f"[PASS] boundary_dice identical = {bd:.4f}")


def test_compute_all_metrics():
    """compute_all_metrics 返回完整字典"""
    a = make_square(20, 20, 10, 10, 50)
    b = make_square(20, 20, 11, 10, 50)
    m = compute_all_metrics(a, b)
    assert set(m.keys()) == {"dice", "hd95", "boundary_dice"}
    assert 0 < m["dice"] <= 1.0
    assert m["hd95"] > 0
    print(f"[PASS] compute_all_metrics keys OK; dice={m['dice']:.4f} hd95={m['hd95']:.4f}")


if __name__ == "__main__":
    test_identical_masks_hd95_zero()
    test_one_pixel_shift_hd95()
    test_nested_masks_hd95_nonzero()
    test_empty_mask_hd95_inf()
    test_partial_overlap_hd95()
    test_dice_score()
    test_boundary_dice()
    test_compute_all_metrics()
    print("\nAll tests passed!")
