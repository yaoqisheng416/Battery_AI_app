# dataset.py

import glob
import os
from dataclasses import dataclass


import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class REVSpec:
    """
    REV尺寸定义
    """
    y: int = 258
    z: int = 456
    x: int = 253


class ElectrodeREVDataset(Dataset):
    """
    电极数字孪生3D数据集

    输入：
        连续2D切片

    输出：
        3D patch

    当前训练：
        无条件VAE

    未来条件生成：
        - middle slice
        - porosity
        - anisotropy_hint
    """

    def __init__(
        self,
        image_dir: str,
        patch_size: int = 128,
        samples_per_epoch: int = 2000,
        ext: str = "*.tif",
        rev_spec: REVSpec = REVSpec(),
        threshold: int = 0,
        return_condition: bool = False,
        augment_flip: bool = True,
    ):

        super().__init__()

        self.patch_size = patch_size
        self.samples = samples_per_epoch
        self.return_condition = return_condition
        self.augment_flip = augment_flip

        volume = self._load_image_sequence(image_dir, ext)
        self.rev_volume = self._extract_rev(volume, rev_spec)

        # 二值化
        self.rev_volume = (self.rev_volume > threshold).astype(np.float32)

        print("REV volume:", self.rev_volume.shape)

    def __len__(self):
        return self.samples

    def _load_image_sequence(self, image_dir, ext):

        files = sorted(glob.glob(os.path.join(image_dir, ext)))

        if len(files) == 0:
            raise ValueError("未找到切片")

        slices = []

        for f in files:
            img = Image.open(f).convert("L")
            slices.append(np.array(img))

        volume = np.stack(slices, axis=0)

        return volume

    def _extract_rev(self, volume, spec):

        y0, z0, x0 = volume.shape

        sy = (y0 - spec.y) // 2
        sz = (z0 - spec.z) // 2
        sx = (x0 - spec.x) // 2

        return volume[
            sy:sy + spec.y,
            sz:sz + spec.z,
            sx:sx + spec.x
        ]

    def _random_start(self, shape):

        y, z, x = shape

        ys = np.random.randint(0, y - self.patch_size)
        zs = np.random.randint(0, z - self.patch_size)
        xs = np.random.randint(0, x - self.patch_size)

        return ys, zs, xs

    def _augment(self, patch):

        if not self.augment_flip:
            return patch

        if np.random.rand() < 0.5:
            patch = patch[::-1]

        if np.random.rand() < 0.5:
            patch = patch[:, ::-1]

        if np.random.rand() < 0.5:
            patch = patch[:, :, ::-1]

        return patch.copy()

    def _anisotropy_hint(self, patch):

        dy = np.abs(np.diff(patch, axis=0)).mean()
        dz = np.abs(np.diff(patch, axis=1)).mean()
        dx = np.abs(np.diff(patch, axis=2)).mean()

        return np.array([dy, dz, dx], dtype=np.float32)

    def __getitem__(self, idx):

        ys, zs, xs = self._random_start(self.rev_volume.shape)

        patch = self.rev_volume[
            ys:ys + self.patch_size,
            zs:zs + self.patch_size,
            xs:xs + self.patch_size
        ]

        patch = self._augment(patch)

        # [P,P,P] -> [1,P,P,P]
        patch_t = torch.from_numpy(patch).unsqueeze(0).float()

        if not self.return_condition:

            return {"x": patch_t}

        # 中间条件切片
        mid = self.patch_size // 2
        cond_slice = patch[mid:mid + 1]

        cond_slice = torch.from_numpy(cond_slice).float()

        # 孔隙率
        solid_fraction = patch.mean()
        porosity = 1 - solid_fraction

        porosity = torch.tensor([porosity], dtype=torch.float32)

        # 各向异性提示
        anisotropy = self._anisotropy_hint(patch)
        anisotropy = torch.from_numpy(anisotropy)

        return {
            "x": patch_t,
            "y": {
                "cond_slice": cond_slice,
                "porosity": porosity,
                "anisotropy_hint": anisotropy
            }
        }