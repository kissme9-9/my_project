"""Montgomery 数据集加载。"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    HAS_ALBU = True
except ImportError:
    HAS_ALBU = False


class MontgomeryDataset(Dataset):
    """
    img_paths: 图片路径列表
    augment: None=不增强；'flip'=仅水平翻转(基线)；'combo'=组合增强
    """
    def __init__(self, img_paths, mask_left_dir, mask_right_dir, size=256, augment=None):
        self.img_paths = img_paths
        self.mask_left_dir = mask_left_dir
        self.mask_right_dir = mask_right_dir
        self.size = size
        self.augment = augment
        self.transform = self._build_transform(augment)

    def _build_transform(self, augment):
        if augment is None:
            return None
        if augment == 'flip':
            return None
        if augment == 'combo' and HAS_ALBU:
            return A.Compose([
                A.Rotate(limit=5, p=0.5, border_mode=0),
                A.Affine(scale=(0.95, 1.05), translate_percent=(-0.03, 0.03), p=0.5, mode=0),
                A.ElasticTransform(alpha=15, sigma=10, p=0.3, border_mode=0),
                A.HorizontalFlip(p=0.5),
            ])
        return None

    def __len__(self):
        return len(self.img_paths)

    def _load_mask(self, name):
        ml = np.array(Image.open(os.path.join(self.mask_left_dir, name)).convert("L"))
        mr = np.array(Image.open(os.path.join(self.mask_right_dir, name)).convert("L"))
        return np.maximum(ml, mr)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        name = os.path.basename(img_path)
        img = Image.open(img_path).convert("L")
        img = img.resize((self.size, self.size), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        mask = self._load_mask(name)
        mask = Image.fromarray(mask).resize((self.size, self.size), Image.NEAREST)
        mask = (np.array(mask) > 127).astype(np.float32)

        if self.augment == 'flip' and np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        elif self.augment == 'combo' and self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']

        img_t = torch.from_numpy(img).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        return img_t, mask_t, name
