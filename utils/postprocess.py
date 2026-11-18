"""形态学后处理：开运算→闭运算→自适应连通域保留。"""
import numpy as np
import cv2


def morphology_postprocess(binary, min_area_ratio=0.05, max_area_ratio=10.0):
    """
    开运算(k=3) → 闭运算(k=5) → 自适应连通域保留。
    连通域策略：
      - 若只有 1 个连通域：直接保留。
      - 若前 2 大连通域面积比 > max_area_ratio（默认 10:1）：说明次大的是噪点，
        退化为"保留最大连通域 + 面积超过最大值 min_area_ratio 的次大连通域"。
      - 否则：保留前 2 大连通域（左右肺）。
    """
    binary = binary.astype(np.uint8)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        order = np.argsort(areas)[::-1]
        largest_area = areas[order[0]]
        if num == 2:
            keep_idx = [order[0]]
        else:
            second_area = areas[order[1]]
            ratio = largest_area / second_area if second_area > 0 else float('inf')
            if ratio > max_area_ratio:
                keep_idx = [order[0]]
                for idx in order[1:]:
                    if areas[idx] >= largest_area * min_area_ratio:
                        keep_idx.append(idx)
                    else:
                        break
            else:
                keep_idx = order[:2].tolist()
        result = np.zeros_like(binary)
        for idx in keep_idx:
            result[labels == (idx + 1)] = 1
        binary = result
    return binary
