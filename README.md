# Lung Segmentation — Attention U-Net with 5-Fold Ensemble + TTA

基于 **Attention U-Net** 的 Montgomery 肺部 X 光分割，CPU 环境下完成 5 折交叉验证、组合增强、后处理消融与 TTA 集成。

## 核心结果

| 配置 | Dice | HD95 (px) | 边界 Dice |
|---|---|---|---|
| 基线单模型 + 形态学 | 0.9625 | 4.24 | 0.8856 |
| 基线 strict-OOF + TTA + 形态学 | 0.9638 | 3.98 | 0.8873 |
| 基线 5 折集成 + TTA + 形态学 | **0.9684** | **3.00** | 0.8975 |
| 增强 5 折集成 + TTA + 形态学 | 0.9681 | **3.00** | **0.9033** |

> **边界质量优先**：增强 5 折集成 + TTA + 形态学（边界 Dice = 0.9033）。
> **综合最优**：基线 5 折集成 + TTA + 形态学（Dice = 0.9684，HD95 = 3.00 px）。

## 评估口径说明

- **集成口径**（Dice 0.9684）：对每个验证样本，用全部 5 折模型做 TTA 后等权/加权平均。这是部署时的实际配置，但评估时样本被除所属折外的其他模型见过，存在轻微泄漏，指标偏乐观。
- **strict-OOF 口径**（Dice 0.9638）：每个样本只用其所属验证折的模型（训练时未见过该样本），无泄漏，是更严谨的泛化估计。
- 两者差值（+0.0046）即为跨折集成带来的真实增益。

## 项目结构

```
lung_seg/
├── configs/               # 路径与超参配置（paths.py）
├── data/                  # 数据集（仅 README，不放原始图像）
│   └── README.md
├── models/                # 模型架构（AttentionUNet）
│   └── attention_unet.py
├── utils/                 # 数据集、指标、损失、后处理
│   ├── dataset.py
│   ├── metrics.py         # Dice / HD95 / 边界 Dice
│   ├── losses.py
│   └── postprocess.py
├── experiments/           # 实验结果（仅 summary，不放 .pth）
├── inference.py           # 单图推理脚本
├── ensemble_tta.py        # 5 折集成 + TTA 评估
├── kfold_train.py         # 5 折训练
├── test_metrics.py        # HD95 单元测试
├── requirements.txt
├── README.md
└── .gitignore
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据：参考 data/README.md 下载 Montgomery Set

# 3. 5 折训练（CPU 环境，约 2 小时）
python kfold_train.py --augment baseline --folds 0,1,2,3,4

# 4. 集成 + TTA 评估（首次运行会生成 TTA 缓存，后续秒级复用）
python ensemble_tta.py

# 5. 单图推理
python inference.py <your_xray.png> --exp baseline --outdir outputs/
```

## 关键工程特性

- **TTA 概率图缓存**：`ensemble_tta.py` 首次运行后将每个模型的 TTA 概率图保存到 `tta_cache/`，后续调形态学 kernel、换阈值、加消融只需读缓存，无需重跑模型。
- **自适应形态学后处理**：连通域策略考虑单侧肺（1 个连通域）和噪点（面积比 > 10:1）异常情况，不硬取 top-2。
- **HD95 单元测试**：`test_metrics.py` 用手工构造的小掩膜验证 HD95 实现（相同掩膜=0、偏移 1px≈1、嵌套掩膜≠0）。
- **逐样本 CSV 导出**：每个配置的逐样本指标导出为 CSV，方便 pandas 做配对差异、Wilcoxon 检验。
- **路径可配置**：数据/实验路径通过环境变量 `LUNG_SEG_DATA_DIR` / `LUNG_SEG_EXP_DIR` 覆盖，无硬编码个人路径。

## 模型权重

训练好的 `.pth` 权重（~120MB/折）不放入 Git 仓库。复现时运行 `kfold_train.py` 重新训练，或联系作者获取。
