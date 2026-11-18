"""
路径配置。
优先从环境变量读取，未设置时使用相对于项目根目录的默认路径。
避免硬编码个人绝对路径。
"""
import os

# 项目根目录（本文件位于 config/ 下，根目录是上一级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录：可通过环境变量 LUNG_SEG_DATA_DIR 覆盖
DATA_DIR = os.environ.get(
    "LUNG_SEG_DATA_DIR",
    os.path.join(PROJECT_ROOT, "data", "MontgomerySet"),
)
IMG_DIR = os.path.join(DATA_DIR, "CXR_png")
MASK_LEFT_DIR = os.path.join(DATA_DIR, "ManualMask", "leftMask")
MASK_RIGHT_DIR = os.path.join(DATA_DIR, "ManualMask", "rightMask")

# 实验输出目录
EXP_DIR = os.environ.get(
    "LUNG_SEG_EXP_DIR",
    os.path.join(PROJECT_ROOT, "experiments"),
)
TTA_CACHE_DIR = os.path.join(EXP_DIR, "tta_cache")
ENSEMBLE_CSV_DIR = os.path.join(EXP_DIR, "ensemble_csv")

# 模型与训练超参
IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 15
LR = 1e-3
