# latent_dataset.py
"""将 build_latent_dataset.py 产生的 latent_dataset 转成扩散模型训练数据集"""

from __future__ import annotations

import os
import json
import glob
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class LatentConditionDataset(Dataset):
    """
    读取 build_latent_dataset.py 生成的 .npz 数据。

    每个样本至少包含：
        z: [C, Dz, Hz, Wz]

    当前使用的主条件变量（4个）：
        - porosity
        - surface_area
        - tau_z
        - deff_z

    数据返回：
        {
            "z": Tensor [C, Dz, Hz, Wz],
            "cond": Tensor [4],            # 已归一化到 [0,1]
            "cond_raw": Tensor [4],        # 原始值
        }
    """

    CONDITION_KEYS = [
        "porosity",
        "surface_area",
        "tau_z",
        "deff_z",
    ]

    def __init__(
        self,
        latent_dir: str,
        summary_json: str | None = None,
        normalize_condition: bool = True,
        clip_condition: bool = True,
        skip_non_percolating: bool = False,
    ):
        """
        Args:
            latent_dir: sample_*.npz 所在目录
            summary_json: dataset_summary.json 路径
            normalize_condition: 是否把条件归一化到 [0,1]
            clip_condition: 归一化后是否 clip 到 [0,1]
            skip_non_percolating: 是否跳过 is_percolating_z == 0 的样本
        """
        super().__init__()

        self.latent_dir = latent_dir
        self.normalize_condition = normalize_condition
        self.clip_condition = clip_condition
        self.skip_non_percolating = skip_non_percolating

        files_all: List[str] = sorted(glob.glob(os.path.join(latent_dir, "sample_*.npz")))
        if len(files_all) == 0:
            raise ValueError(f"在 {latent_dir} 中未找到 sample_*.npz")

        if summary_json is None:
            summary_json = os.path.join(latent_dir, "dataset_summary.json")

        if not os.path.exists(summary_json):
            raise ValueError(f"找不到 summary 文件: {summary_json}")

        with open(summary_json, "r", encoding="utf-8") as f:
            self.summary = json.load(f)

        # summary 里的字段名映射
        self.summary_key_map = {
            "porosity": "porosity",
            "surface_area": "surface",
            "tau_z": "tau_z",
            "deff_z": "deff_z",
        }

        # 可选：过滤掉 Z 方向不贯通的 patch
        if self.skip_non_percolating:
            filtered_files = []
            for path in files_all:
                data = np.load(path)
                is_perc = int(data.get("is_percolating_z", 1))
                if is_perc == 1:
                    filtered_files.append(path)
            self.files = filtered_files
        else:
            self.files = files_all

        if len(self.files) == 0:
            raise ValueError("过滤后没有可用样本，请检查 skip_non_percolating 设置或数据集内容。")

        # 预取 condition 的 min / max
        self.cond_mins = []
        self.cond_maxs = []

        for key in self.CONDITION_KEYS:
            skey = self.summary_key_map[key]
            self.cond_mins.append(float(self.summary[skey]["min"]))
            self.cond_maxs.append(float(self.summary[skey]["max"]))

        self.cond_mins = np.array(self.cond_mins, dtype=np.float32)
        self.cond_maxs = np.array(self.cond_maxs, dtype=np.float32)

        # 读第一个样本，检查 z 形状
        sample0 = np.load(self.files[0])
        if "z" not in sample0:
            raise ValueError(f"{self.files[0]} 中没有字段 z")
        self.latent_shape = tuple(sample0["z"].shape)  # (C, D, H, W)

    def __len__(self) -> int:
        return len(self.files)

    def _normalize_condition(self, cond: np.ndarray) -> np.ndarray:
        denom = np.maximum(self.cond_maxs - self.cond_mins, 1e-8)
        cond_norm = (cond - self.cond_mins) / denom

        if self.clip_condition:
            cond_norm = np.clip(cond_norm, 0.0, 1.0)

        return cond_norm.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.files[idx]
        data = np.load(path)

        if "z" not in data:
            raise ValueError(f"{path} 中没有字段 z")

        z = data["z"].astype(np.float32)

        cond_raw = np.array(
            [
                float(data["porosity"]),
                float(data["surface_area"]),
                float(data["tau_z"]),
                float(data["deff_z"]),
            ],
            dtype=np.float32
        )

        if self.normalize_condition:
            cond = self._normalize_condition(cond_raw)
        else:
            cond = cond_raw.copy().astype(np.float32)

        out = {
            "z": torch.from_numpy(z),                 # [C, D, H, W]
            "cond": torch.from_numpy(cond),           # [4]
            "cond_raw": torch.from_numpy(cond_raw),   # [4]
        }

        # 附带返回 origin 和 percolation 信息，后面调试有用
        if "origin" in data:
            out["origin"] = torch.from_numpy(data["origin"].astype(np.int32))

        if "is_percolating_z" in data:
            out["is_percolating_z"] = torch.tensor(int(data["is_percolating_z"]), dtype=torch.int32)

        return out


if __name__ == "__main__":
    # 简单自检
    LATENT_DIR = r"latent_dataset"
    SUMMARY_JSON = os.path.join(LATENT_DIR, "dataset_summary.json")

    dataset = LatentConditionDataset(
        latent_dir=LATENT_DIR,
        summary_json=SUMMARY_JSON,
        normalize_condition=True,
        clip_condition=True,
        skip_non_percolating=False,
    )

    print("Dataset size:", len(dataset))
    print("Latent shape:", dataset.latent_shape)
    print("Condition keys:", dataset.CONDITION_KEYS)

    sample = dataset[0]
    print("z shape:", sample["z"].shape)
    print("cond:", sample["cond"])
    print("cond_raw:", sample["cond_raw"])
    if "origin" in sample:
        print("origin:", sample["origin"])
    if "is_percolating_z" in sample:
        print("is_percolating_z:", sample["is_percolating_z"])