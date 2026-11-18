import torch
from train import AttentionUNet

model = AttentionUNet(in_ch=1, out_ch=1)
x = torch.randn(1, 1, 256, 256)
print("Input:", x.shape)

skips = []
for i, enc in enumerate(model.encoders):
    x = enc(x)
    print(f"Encoder {i+1}: {x.shape}")
    skips.append(x)
    x = model.pool(x)
# 通道翻倍发生在 ConvBlock：1→64→128→256→512。
#
# 尺寸减半发生在 MaxPool：256→128→64→32→16。
#
# 两者是分开的！不是同一层同时做两件事。

x = model.bottleneck(x)
print("Bottleneck:", x.shape)

skips = skips[::-1]
for i, (up, att, dec) in enumerate(zip(model.upconvs, model.attentions, model.decoders)):
    x = up(x)
    print(f"Upconv {i+1}: {x.shape}")
    skip = att(g=x, x=skips[i])
    x = torch.cat([x, skip], dim=1)
    print(f"Concat {i+1}: {x.shape}")
    x = dec(x)
    print(f"Decoder {i+1}: {x.shape}")
# 尺寸翻倍发生在 ConvTranspose2d：16→32→64→128→256。
#
# 通道减半发生在 ConvTranspose2d：1024→512→256→128→64。
#
# 拼接后通道会暂时翻倍：因为 torch.cat([x, skip], dim=1) 把两路拼起来。
#
# 解码器 ConvBlock 再把通道压回去：2f→f。

x = model.final(x)
print("Final:", x.shape)
# 编码器：通道翻倍，尺寸减半；解码器：通道减半，尺寸翻倍。
#
# 最终输出和输入尺寸、通道数相同，但数值含义不同（logits vs 像素值）。
#
# torch.cat 的 dim=1 是通道拼接，dim=2 是高度拼接，别搞混。