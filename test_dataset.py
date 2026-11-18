import numpy as np
from PIL import Image
import torch

img_path = r"D:\vibe_practice\lung_seg\data\MontgomerySet\CXR_png\MCUCXR_0001_0.png"
mask_left_path = r"D:\vibe_practice\lung_seg\data\MontgomerySet\ManualMask\leftMask\MCUCXR_0001_0.png"
mask_right_path = r"D:\vibe_practice\lung_seg\data\MontgomerySet\ManualMask\rightMask\MCUCXR_0001_0.png"

# 图片管线
img = Image.open(img_path).convert("L")
print("1. 打开:", type(img), img.size)
img = img.resize((256, 256), Image.BILINEAR)
print("2. resize:", img.size)
img = np.array(img, dtype=np.float32) / 255.0
print("3. 归一化:", img.shape, img.dtype, img.min(), img.max())
img_t = torch.from_numpy(img).unsqueeze(0)
print("4. tensor:", img_t.shape)

# mask 管线
ml = np.array(Image.open(mask_left_path).convert("L"))
mr = np.array(Image.open(mask_right_path).convert("L"))
mask = np.maximum(ml, mr)
print("5. 合并 mask:", mask.shape, np.unique(mask))
mask = Image.fromarray(mask).resize((256, 256), Image.NEAREST)
mask = (np.array(mask) > 127).astype(np.float32)
print("6. 二值化:", mask.shape, np.unique(mask))
mask_t = torch.from_numpy(mask).unsqueeze(0)
# 那个 1 是通道数（channel）。[256,256] 是一张灰度图的二维数组，.unsqueeze(0) 在最前面加了一维，变成 [1,256,256]，表示“1 个通道，256 高，256 宽”。
# PyTorch 的卷积层要求输入必须是 [batch, channel, height, width] 四维。这里 [1,256,256] 还缺一个 batch 维，等 DataLoader 打包时，会变成 [4,1,256,256]（4 张图，每张 1 通道）。
print("7. mask tensor:", mask_t.shape)
# 图片用归一化，mask 用二值化。这是分割任务的标准做法。