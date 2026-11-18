"""
Montgomery 肺部分割 —— Attention U-Net（CPU 可跑）
单文件：数据加载 → 训练 → 推理 → 可视化
"""
# ============================================================
# 0. Windows OpenMP 冲突修复（必须在 import torch 之前！）
# ============================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import time
import glob
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 1. 配置区
# ============================================================
from config.paths import (
    IMG_DIR, MASK_LEFT_DIR, MASK_RIGHT_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS, LR,
)
MASK_DIR = None  # 如果是合并版 ManualMask，填路径；否则 None

VAL_SPLIT = 0.15
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUT_DIR, "train_log.txt")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ============================================================
# 2. 数据集
# ============================================================
class MontgomeryDataset(Dataset):
    # 继承 PyTorch 的 Dataset 基类，自定义数据集。必须实现 __len__ 和 __getitem__，否则 DataLoader 无法使用。
    def __init__(self, img_paths, mask_left_dir=None, mask_right_dir=None,
                 mask_dir=None, size=256, augment=False):
        # 初始化数据集。参数依次是：图片路径列表、左肺 mask 目录、右肺 mask 目录、合并 mask 目录（可选）、图片缩放尺寸、是否做数据增强（augment，不是“增长”）。
        self.img_paths = img_paths
        self.mask_left_dir = mask_left_dir
        self.mask_right_dir = mask_right_dir
        self.mask_dir = mask_dir
        self.size = size
        self.augment = augment
        # 把传入的参数保存到 self 上，供 __getitem__ 和其他方法使用。self 是实例本身，相当于“把这个数据集自己的属性记下来”。

    def __len__(self):  # 返回数据集样本总数。DataLoader 靠它知道一个 epoch 有多少个样本。
        return len(self.img_paths)

    def _load_mask(self, name):
        # 根据文件名加载 mask。这是一个内部辅助函数（下划线开头表示“内部使用”）。
        # name 是文件名（如 MCUCXR_0001_0.png），不是“数据名称”这个模糊说法。
        if self.mask_dir:  # 如果配置了合并版 mask 目录，就走合并版分支；否则走左右分开的分支。
            m = Image.open(os.path.join(self.mask_dir, name)).convert("L")
            # 用 os.path.join 拼出完整路径，打开图片，.convert("L") 转成单通道灰度图，赋值给 m。
            return np.array(m)  # 把 PIL 图片转成 numpy 数组返回。
        else:
            ml = np.array(Image.open(os.path.join(self.mask_left_dir, name)).convert("L"))
            # 加载左肺 mask 并转成 numpy 数组。
            mr = np.array(Image.open(os.path.join(self.mask_right_dir, name)).convert("L"))
            # 加载右肺 mask。
            return np.maximum(ml, mr)
            # 逐像素取最大值，把左右肺 mask 合并成一张。因为 mask 是黑底（0）白肺（255），取 max 后两个肺都是白的。
            # ‘不是为了“完整识别”，而是为了合并两张 mask。’

    def __getitem__(self, idx):  # DataLoader 每次调用它取一个样本，返回（图片 tensor, mask tensor, 文件名）。idx 是样本索引。
        img_path = self.img_paths[idx]  # 根据索引 idx 从路径列表中取出当前样本的完整路径。
        name = os.path.basename(img_path)  # os.path.basename 是从完整路径里取出文件名，mask 文件名和原图一样，所以需要这个 name。

        img = Image.open(img_path).convert("L")  # 打开图片，转成单通道灰度图。
        img = img.resize((self.size, self.size), Image.BILINEAR)
        # 把图片缩放到 size × size（这里是 256×256）。Image.BILINEAR 是双线性插值，适合原图这种连续灰度图，能让缩放更平滑。
        img = np.array(img, dtype=np.float32) / 255.0
        # 把 PIL 图片转成 numpy 数组，类型设为 float32，然后除以 255 做归一化，把像素值从 0~255 映射到 0~1。神经网络对 0~1 的输入更稳定，收敛更快。
        # 归一化是深度学习的标准预处理，让输入分布稳定，加速收敛。
        mask = self._load_mask(name)  # 调用 _load_mask，根据文件名加载对应的 mask
        mask = Image.fromarray(mask).resize((self.size, self.size), Image.NEAREST)
        # 把 numpy 数组转回 PIL 图片，缩放到 256×256。Image.NEAREST 是最近邻插值，mask 是 0/1 标签，必须用最近邻，否则会出现 0.5 这种没意义的中间值。
        mask = (np.array(mask) > 127).astype(np.float32)
        # 先转 numpy 数组，然后二值化：大于 127 的像素设为 True（1），否则 False（0），再转成 float32。这样 mask 就只有 0 和 1 两个值。

        if self.augment and np.random.rand() > 0.5:
            # 如果启用了数据增强，并且随机数大于 0.5（50% 概率），就做水平翻转。验证集 augment=False，不会翻转。
            img = np.fliplr(img).copy()  # fliplr 是左右翻转图片。.copy() 是因为 fliplr 返回的是视图，不 copy 可能出问题。
            mask = np.fliplr(mask).copy()  # 同步左右翻转 mask。图片和 mask 必须同步翻转，否则标签就错了。

        img_t = torch.from_numpy(img).unsqueeze(0)
        # 把 numpy 数组转成 PyTorch tensor，然后在第 0 维加一个通道维度，形状从 [256,256] 变成 [1,256,256]（1 表示灰度图单通道）。unsqueeze(0) 是“增加维度”，
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        # 同上，mask 也加通道维度，变成 [1,256,256]
        return img_t, mask_t, name
        # 返回三元组：图片 tensor、mask tensor、文件名。DataLoader 会把它打包成 batch。


# ============================================================
# 3. Attention U-Net
# ============================================================
class ConvBlock(nn.Module):
    # nn.Module：PyTorch 所有神经网络层的基类。继承它，你的类才能被 PyTorch 识别为“模型”，才能自动管理参数、支持 .to(device)、.train()、.eval()。
    # ConvBlock 是 U-Net 的基本积木，做两次「卷积 → 批归一化 → ReLU」
    # U-Net 里编码器、解码器、瓶颈层，全都是重复的 ConvBlock，只是通道数不同。写一次，到处复用。
    def __init__(self, in_ch, out_ch):
        # 通道数（channel）：可以理解为“特征图的厚度”。比如输入灰度图是 1 通道，经过第一层卷积变成 64 通道，意味着网络提取了 64 种不同的特征（边缘、纹理、形状等）。
        # 初始化卷积块。in_ch 是输入通道数，out_ch 是输出通道数
        # in_ch 和 out_ch 的关系：卷积层负责把 in_ch 映射到 out_ch。比如 in_ch=1, out_ch=64，就是把 1 通道的灰度图变成 64 通道的特征图。
        super().__init__()  # 调用父类 nn.Module 的初始化函数。这是必须写的，否则 PyTorch 无法正确注册这个模块的参数。
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            # 2D 卷积层，用 3×3 的卷积核，padding=1 保证输出尺寸不变，bias=False 因为后面有 BatchNorm，不需要偏置。
            # 卷积（Convolution）：拿一个 3×3 的小窗口在图片上滑动，每个位置做加权求和，提取局部特征。可以理解为“用放大镜逐块扫描图片”。
            # 3：卷积核大小 3×3。
            # padding=1：在图片边缘补一圈 0。如果不补，3×3 卷积会让输出尺寸减 2（256→254）。补 1 圈后，输出尺寸不变（256→256）。
            # bias=False：卷积默认会加一个偏置项，但后面紧跟 BatchNorm，BatchNorm 会减掉均值再加缩放，偏置就多余了。省掉能减少参数量。
            nn.BatchNorm2d(out_ch),
            # 对 out_ch 个通道做批归一化，让每个通道的输出分布稳定（均值 0，方差 1），加速训练。
            # 批归一化（Batch Normalization）：在一个 batch 内，对每个通道算均值和方差，然后把数据标准化。相当于“每层输出都做一次归一化”
            # 深层网络里，每层输入分布会随着前层参数变化而漂移，导致训练慢。BatchNorm 把分布拉回稳定范围，允许用更大的学习率，收敛更快。
            nn.ReLU(inplace=True),  # 激活函数，将负数变为0，引入非线性
            # ReLU 公式：f(x) = max(0, x)；inplace=True：直接在原内存上改，不新建 tensor，省显存。但要小心，如果后面还要用原始输入，就不能用 inplace。
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),  # 继续做卷积化
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # sequential将多个层按照顺序穿起来
        # 好处：不用手动写 x = layer1(x); x = layer2(x); ...，直接 self.conv(x) 一次搞定。

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):  # Attention Gate，注意力门控。用解码器特征做“门控信号”，筛选编码器的 skip 特征，让肺部区域权重高、背景权重低。
    def __init__(self, F_g, F_l, F_int):  # F_g 是门控信号通道数，F_l 是 skip 特征通道数，F_int 是中间维度（通常取 F_g // 2）。
        # g（gating）：来自解码器的特征，代表“我目前关注什么”。
        # x（skip）：来自编码器的特征，代表“这里有什么细节”。
        # F_int：中间层通道数，用来降维，减少计算量。
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        # 用 1×1 卷积把 g 从 F_g 通道投影到 F_int 通道，再接 BatchNorm
        # 1×1 卷积：卷积核大小 1×1，不改变空间尺寸，只改变通道数。相当于“逐像素做一次全连接”。
        # 作用：降维/升维，融合通道信息，计算量小。
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        # 为什么 g 和 x 都要投影到同一维度：因为后面要相加（g1 + x1），维度必须一致。
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        # 1×1 卷积把 F_int 通道压成 1 通道，BatchNorm 稳定分布，Sigmoid 把值压到 0~1，得到注意力权重图。
        # Sigmoid 公式：1 / (1 + e^(-x))，输出永远在 0~1。
        # 为什么是 1 通道：注意力图每个像素一个权重，1 通道就够。
        # psi 的含义：每个像素的“重要程度”，0=完全不要，1=完全保留。
        self.relu = nn.ReLU(inplace=True)  # ReLU 激活函数，用在 g1 + x1 之后，引入非线性。

    def forward(self, g, x):  # 前向传播。g 是解码器门控信号，x 是编码器 skip 特征。返回加权后的 x
        g1 = self.W_g(g)  # 把 g 投影到中间维度 F_int
        x1 = self.W_x(x)  # 把 x 投影到中间维度 F_int
        psi = self.relu(g1 + x1)  # 把投影后的 g1 和 x1 逐元素相加，再经过 ReLU 激活。这一步融合了“解码器关注点”和“编码器细节”。
        psi = self.psi(psi)  # 经过 1×1 卷积 + BatchNorm + Sigmoid，输出 0~1 的注意力权重图。
        return x * psi  # 把原始 skip 特征 x 和注意力权重 psi 逐像素相乘。权重大的区域保留，权重小的抑制。
        # 逐像素相乘（element-wise multiplication）：两个形状相同的 tensor，对应位置相乘。
        # Attention 的本质：用学到的权重去“调制”特征，让重要的更突出，不重要的被压低。


class AttentionUNet(nn.Module):  # 完整的 Attention U-Net 模型，包含编码器、瓶颈层、解码器、注意力门控和最终输出层。
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        # in_ch=1 输入灰度图，out_ch=1 输出单通道预测图，features 是每层编码器的输出通道数。
        # 通道为什么逐层翻倍：随着尺寸减小，网络需要更多通道来保留信息。这是 CNN 的经典设计。
        super().__init__()
        self.encoders = nn.ModuleList()  # 用 nn.ModuleList 存编码器的 4 个 ConvBlock
        self.pool = nn.MaxPool2d(2, 2)  # 2×2 最大池化，尺寸减半（256→128），通道数不变。
        # 最大池化（Max Pooling）：在 2×2 窗口里取最大值，相当于“保留最显著的特征，丢掉细节”。
        # 作用：减小尺寸，增大感受野，降低计算量。
        # 通道不变：池化是逐通道独立做的，不混合通道。
        self.decoders = nn.ModuleList()  # 存解码器的 4 个 ConvBlock
        self.upconvs = nn.ModuleList()  # 存解码器的 4 个 ConvTranspose2d 上采样层。
        self.attentions = nn.ModuleList()  # 存解码器的 4 个 AttentionGate。每个解码器层配一个 Attention Gate，用来筛选对应的编码器 skip 特征。

        for f in features:
            # 循环创建 4 个编码器 ConvBlock。第一个 in_ch=1, out_ch=64；第二个 in_ch=64, out_ch=128；以此类推。每次更新 in_ch，让下一层输入通道等于上一层输出通道。
            self.encoders.append(ConvBlock(in_ch, f))
            in_ch = f

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        # 瓶颈层，输入 512 通道，输出 1024 通道。位于编码器和解码器之间，是 U-Net 的最深层。

        for f in reversed(features):
            # 倒序遍历 features，从 512 到 64，逐层构建解码器。
            self.upconvs.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            # 转置卷积，输入 f*2 通道，输出 f 通道，kernel=2，stride=2，尺寸翻倍，通道减半。
            # 转置卷积（ConvTranspose2d）：也叫反卷积，作用是把特征图尺寸放大。
            self.attentions.append(AttentionGate(F_g=f, F_l=f, F_int=f // 2))
            # 创建 Attention Gate。F_g=f 是门控通道，F_l=f 是 skip 通道，F_int=f//2 是中间维度。
            self.decoders.append(ConvBlock(f * 2, f))
            # 解码器 ConvBlock，输入 f*2 通道（上采样输出 f + skip 特征 f），输出 f 通道。

        self.final = nn.Conv2d(features[0], out_ch, 1)  # 1×1卷积，把64通道数压缩成1通道，做出预测图（logits，还没经过 sigmoid）。
        # features[0]=64：最后一层解码器的输出通道数。
        # out_ch=1：二分类任务，1 通道表示“每个像素是肺的概率”。
        # logits：还没经过 sigmoid 的原始输出，范围是 -∞~+∞。后面算 loss 时用 binary_cross_entropy_with_logits 内部会做 sigmoid。

    def forward(self, x):  # 前向传播。数据依次流过编码器、瓶颈、解码器，最后输出预测图。
        skips = []  # 空列表，用来存 4 个编码器的输出（skip 特征
        for enc in self.encoders:  # 编码器循环。每层：卷积 → 保存 skip → 池化下采样
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)  # 经过瓶颈层，通道 512→1024，尺寸 16×16。
        skips = skips[::-1]  # 反转列表，解码器从最深处往回走

        for i, (up, att, dec) in enumerate(zip(self.upconvs, self.attentions, self.decoders)):
            # 解码器循环。每次同时取出上采样层、注意力门、解码器 ConvBlock。
            x = up(x)  # 转置卷积，尺寸翻倍，通道减半。
            skip = att(g=x, x=skips[i])  # 用当前解码器特征 x 做门控 g，筛选对应的编码器 skip 特征 skips[i]
            x = torch.cat([x, skip], dim=1)  # 在通道维度拼接上采样结果和加权 skip。通道数从 f 变成 2f。
            x = dec(x)  # 经过解码器 ConvBlock，融合拼接后的特征，通道 2f→f

        return self.final(x)  # 最终 1×1 卷积输出单通道预测图（logits），形状 [B, 1, 256, 256]


# ============================================================
# 4. 损失函数
# ============================================================
def dice_loss(pred, target, smooth=1.0):  # Dice Loss 函数。pred 是模型输出的 logits，target 是真实 mask，smooth=1.0 是平滑项，防止分母为 0。
    pred = torch.sigmoid(pred)  # 把模型输出的 logits（任意实数）通过 Sigmoid 压到 0~1，变成概率。
    pred_flat = pred.view(-1)  # 把 [B, 1, 256, 256] 的预测图展平成 [B*1*256*256] 的一维向量
    target_flat = target.view(-1)  # 把真实 mask 也展平成一维，和预测形状一致，才能逐元素相乘。
    intersection = (pred_flat * target_flat).sum()  # 逐元素相乘后求和，得到预测和真实重叠的像素总数（交集）
    return 1 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)  # 计算 1 - Dice
    # 为什么是 2 倍交集：这是一个归一化技巧。如果预测和真实完全重合，交集 = |A| = |B|，分子 = 2|A|，分母 = 2|A|，Dice = 1


def combined_loss(pred, target):  # 组合损失函数。把 BCE Loss 和 Dice Loss 相加，兼顾像素级和区域级两种衡量角度。
    return F.binary_cross_entropy_with_logits(pred, target) + dice_loss(pred, target)  # 兼顾像素级和区域级


def dice_score(pred, target, threshold=0.5):  # Dice 评估指标函数。和 dice_loss 不同，它用来评估模型好坏，不参与训练。
    # 为什么单独写一个：训练用 dice_loss（要梯度），评估用 dice_score（不要梯度）。两者逻辑相似但用途不同。
    # threshold=0.5：二值化阈值。概率 > 0.5 算预测为肺。
    pred = (torch.sigmoid(pred) > threshold).float()  # 先 sigmoid 变概率，再 > 0.5 二值化成 0/1，最后转 float
    # 为什么这里要二值化，而 dice_loss 不二值化：评估时只关心“预测对没对”，二值化后算 Dice 就是最终指标。训练时需要连续概率来算梯度，二值化会断掉梯度。
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + 1e-6) / (pred_flat.sum() + target_flat.sum() + 1e-6)
    # 返回 Dice 系数。和 dice_loss 的区别：
    # 没有 1 - ...，因为这是指标本身，Dice 越大越好。
    # 1e-6 是平滑项，比 smooth=1.0 小很多，因为评估时不需要那么强的平滑。


# ============================================================
# 5. 训练
# ============================================================
def train():  # 训练函数。包含数据划分、模型创建、训练循环、验证、保存模型、画图
    log(f"Device: {DEVICE}")  # 记录当前使用的设备（CPU 或 GPU）
    log(f"Image dir: {IMG_DIR}")  # 记录图片所在目录

    img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))  # 用 glob 扫描 IMG_DIR 下所有 .png 文件，拿到路径列表，sorted 按字母排序
    log(f"Found {len(img_paths)} images")  # 日志中的图像数量
    if len(img_paths) == 0:
        log("ERROR: No images found! Check DATA_ROOT path.")
        sys.exit(1)  # 如果一张图都没找到，说明路径错了，打印错误信息并退出程序。

    np.random.seed(SEED)  # 固定随机种子，保证每次运行划分的数据集一样
    np.random.shuffle(img_paths)  # 随机打乱图片的顺序
    n_val = int(len(img_paths) * VAL_SPLIT)  # 计算验证集大小。len(img_paths) * 0.15，取整
    val_paths = img_paths[:n_val]  # 从第一张到15%张
    train_paths = img_paths[n_val:]  # 训练后85%张图片
    log(f"Train: {len(train_paths)}, Val: {len(val_paths)}")  # 日志记录训练和验证的图像数量

    use_combined = MASK_DIR is not None  # 判断是否配置了合并版 mask 目录
    train_ds = MontgomeryDataset(train_paths, MASK_LEFT_DIR, MASK_RIGHT_DIR,
                                 mask_dir=MASK_DIR if use_combined else None,
                                 size=IMG_SIZE, augment=True)  # 创建训练集 Dataset，augment=True 启用数据增强（随机翻转）
    val_ds = MontgomeryDataset(val_paths, MASK_LEFT_DIR, MASK_RIGHT_DIR,
                               mask_dir=MASK_DIR if use_combined else None,
                               size=IMG_SIZE, augment=False)  # 创建验证集 Dataset，augment=False 不增强

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # 训练集 DataLoader，batch_size=4，打乱顺序，单线程加载
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)  # 验证集不打乱顺序

    model = AttentionUNet(in_ch=1, out_ch=1).to(DEVICE)  # 创建模型，输入 1 通道（灰度图），输出 1 通道（预测图），搬到 DEVICE
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # Adam 优化器，学习率 LR=1e-3，自适应调整每个参数的学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    # 如果验证 loss 连续 3 个 epoch 不下降，学习率乘 0.5。
    # 为什么需要学习率衰减：训练后期需要更小的学习率来精细调参

    train_losses, val_losses, val_dices = [], [], []  # 建立训练，验证集的损失函数列表以及验证集的dice系数列表，最后画图用。
    best_dice = 0.0  # 记录历史最佳 Dice，初始为 0

    for epoch in range(1, EPOCHS + 1):  # 外层循环，遍历每个 epoch。range(1, EPOCHS+1) 从 1 到 15
        model.train()  # 切换模型到训练模式。影响 BatchNorm（用当前 batch 统计量）和 Dropout（随机丢弃）
        t0 = time.time()  # 记录 epoch 开始时间，用于计算耗时
        epoch_loss = 0.0  # 累加器，记录这个 epoch 所有 batch 的 loss 总和
        for imgs, masks, _ in train_loader:  # 内层循环，每次取一个 batch。_ 是文件名，这里不用
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)  # 把图片和 mask 搬到 DEVICE（CPU/GPU）
            optimizer.zero_grad()  # 第 1 步：清空上一步的梯度
            preds = model(imgs)  # 第 2 步：前向传播，得到预测
            loss = combined_loss(preds, masks)   # 第 3 步：计算损失
            loss.backward()  # 第 4 步：反向传播，计算每个参数的梯度
            # 梯度 = 方向 + 大小。
            # 方向：参数往哪边调能让 loss 变小。
            # 大小：调的幅度该多大。
            # backward() 只算梯度，不改参数。改参数是下一步 step() 的事。
            optimizer.step()  # 第 5 步：根据梯度更新参数 w_new = w_old - lr（学习率） * gradient
            epoch_loss += loss.item()  # 把 tensor 转成 Python 数字，方便打印和累加。loss 本身是 tensor，loss.item() 是 float。
        epoch_loss /= len(train_loader)  # 把这个 epoch 所有 batch 的 loss 总和除以 batch 数，得到平均 loss
        train_losses.append(epoch_loss)  # 把平均 train_loss 加入列表，用于画图

        model.eval()  # 切换到评估模式。影响 BatchNorm（用全局统计量）和 Dropout（不丢弃）
        v_loss, v_dice = 0.0, 0.0  # 初始化验证 loss 和 dice 累加器
        with torch.no_grad():  # 验证时，关闭梯度计算，省内存、加速
            for imgs, masks, _ in val_loader:  # 验证集循环
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs)
                v_loss += combined_loss(preds, masks).item()  # 累加验证 loss
                v_dice += dice_score(preds, masks).item()  # 累加验证 dice
        v_loss /= len(val_loader)  # 求平均
        v_dice /= len(val_loader)  # 求平均
        val_losses.append(v_loss)
        val_dices.append(v_dice)  # 加入列表，用于画图
        scheduler.step(v_loss)  # 把验证 loss 喂给学习率调度器

        dt = time.time() - t0  # 计算所花费的时间
        log(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={epoch_loss:.4f} | "
            f"val_loss={v_loss:.4f} | val_dice={v_dice:.4f} | {dt:.1f}s")
        # 日志中记录epoch的轮数，训练所得损失函数，验证的损失，验证的dice和花费时间

        if v_dice > best_dice:  # 比较验证的dice与best_dice
            best_dice = v_dice
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pth"))
            log(f"  -> saved best model (dice={best_dice:.4f})")  # 只保存验证dice最高的模型
    # matplotlib画两个子图：
    # 左：train_loss 和 val_loss 曲线。
    # 右：val_dice 曲线。
    # 保存到 loss_curve.png
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss Curve")
    plt.subplot(1, 2, 2)
    plt.plot(val_dices, "g-", label="val dice")
    plt.xlabel("Epoch"); plt.ylabel("Dice"); plt.legend(); plt.title("Validation Dice")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=120)
    plt.close()
    log(f"Loss curve saved to {OUT_DIR}/loss_curve.png")
    log(f"Best val dice: {best_dice:.4f}")
    return model

    # model.train() / model.eval() 主要影响 BatchNorm 和 Dropout。
    #
    # 验证时必须 model.eval()，否则 BatchNorm 行为不对，指标不准。
    #
    # 验证时必须 torch.no_grad()，否则浪费内存和时间。


# ============================================================
# 6. 推理 + 可视化
# ============================================================
def infer(model=None):  # 推理函数。model=None 表示“如果没传模型进来，就自己加载”
    if model is None:  # 如果没传模型，就走加载分支
        model = AttentionUNet(in_ch=1, out_ch=1).to(DEVICE)  # 创建新的 Attention U-Net，输入 1 通道，输出 1 通道，搬到 DEVICE
        model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_model.pth"), map_location=DEVICE))
        # 从 best_model.pth 加载训练好的参数，map_location=DEVICE 确保加载到当前设备（CPU/GPU）
    model.eval()  # 切换到评估模式。影响 BatchNorm（用全局统计量）和 Dropout（不丢弃）

    img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))  # 扫描所有 png 文件，排序
    use_combined = MASK_DIR is not None  # 判断是否配置了合并版 mask 目录
    ds = MontgomeryDataset(img_paths, MASK_LEFT_DIR, MASK_RIGHT_DIR,
                           mask_dir=MASK_DIR if use_combined else None,
                           size=IMG_SIZE, augment=False)  # 创建 Dataset，augment=False 不增强（推理时不需要增强）

    indices = np.linspace(0, len(ds) - 1, 3, dtype=int)  # 在 0 到 len-1 之间均匀取 3 个整数索引。dtype=int 保证是整数
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))  # 创建 3×3 的子图网格，每个子图 12/3=4 英寸
    cols = ["Original X-Ray", "Prediction", "Ground Truth"]  # 列标题
    for ax, col in zip(axes[0], cols):  # 给第一行的 3 个子图设置标题
        ax.set_title(col, fontsize=12)

    for row, idx in enumerate(indices):  # 遍历选中的 3 个索引
        img, mask, name = ds[idx]  # 从 Dataset 取第 idx 个样本
        with torch.no_grad():  # 关闭梯度计算，省内存加速
            pred = torch.sigmoid(model(img.unsqueeze(0).to(DEVICE))).cpu().numpy()[0, 0]
            # img.unsqueeze(0)：加 batch 维，[1,256,256] → [1,1,256,256]。
            # .to(DEVICE)：搬到 CPU/GPU。
            # model(...)：前向传播，得到 logits。
            # torch.sigmoid(...)：把 logits 变成 0~1 概率。
            # .cpu().numpy()[0, 0]：转 numpy，取第 0 个 batch、第 0 个通道，得到 [256,256]
        pred_bin = (pred > 0.5).astype(np.uint8)  # 概率 > 0.5 算肺，转成 0/1 的 uint8 图

        axes[row, 0].imshow(img[0], cmap="gray")  # 第 row 行第 0 列画原图
        axes[row, 0].set_ylabel(name, fontsize=9)
        axes[row, 1].imshow(pred_bin, cmap="gray")  # 第 row 行第 1 列画预测 mask
        axes[row, 2].imshow(mask[0], cmap="gray")
        for ax in axes[row]:  # 隐藏坐标轴
            ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "comparison.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    # tight_layout：自动调整子图间距。
    # savefig：保存到 comparison.png。
    # close：关闭图形，释放内存
    log(f"Comparison saved to {save_path}")  # 记录日志

    # infer() = 加载模型 + 取 3 张图 + 预测 + 画图。
    # unsqueeze(0) 加 batch 维，.cpu().numpy()[0, 0] 去掉 batch 和通道维。
    # 二值化是为了得到干净的 0/1 mask，方便可视化和评估。


# ============================================================
# 7. 入口
# ============================================================
if __name__ == "__main__":
    log("=" * 60)
    log("Montgomery Lung Segmentation - Attention U-Net")
    log("=" * 60)
    model = train()
    infer(model)
    log("ALL DONE! Check outputs/ folder for loss_curve.png and comparison.png")
