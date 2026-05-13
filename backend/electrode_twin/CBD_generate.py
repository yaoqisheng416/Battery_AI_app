from __future__ import annotations
import os
import json
from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
    generate_binary_structure,
    label,
)
from PIL import Image

# ============================================================
# 路径
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ELECTRODE_TWIN_DIR = os.path.join(BASE_DIR, "electrode_twin")

INPUT_VOLUME_PATH = os.path.join(
    ELECTRODE_TWIN_DIR,
    "generated_results",
    "run_007",
    "best_sample",
    "generated_bin_final.npy"
)

OUT_DIR = os.path.join(
    ELECTRODE_TWIN_DIR,
    "generated_results",
    "run_007",
    "best_sample",
    "three_phase"
)
# INPUT_VOLUME_PATH = r"./generated_results/run_007/best_sample/generated_bin_final.npy"
# OUT_DIR = r"./generated_results/run_007/best_sample/three_phase"

# ============================================================
# 参数
# ============================================================

PORE_VALUE = 0
AM_VALUE = 1
CBD_VALUE = 2

TARGET_CBD_VOL_FRAC = 0.05
W_UM = 0.08

VOXEL_SIZE_Y = 0.0315
VOXEL_SIZE_Z = 0.02791
VOXEL_SIZE_X = 0.02791

MAX_GROWTH_DISTANCE_FACTOR = 4.0
REMOVE_ISOLATED_CBD = True

SEED = 42

# ============================================================
# 工具
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def save_color_slices(volume, out_dir,
                      pore_value, am_value, cbd_value):
    ensure_dir(out_dir)

    for i in range(volume.shape[0]):
        sl = volume[i]

        rgb = np.zeros((sl.shape[0], sl.shape[1], 3), dtype=np.uint8)

        rgb[sl == pore_value] = [255, 255, 255]
        rgb[sl == am_value] = [50, 100, 180]
        rgb[sl == cbd_value] = [0, 180, 150]

        Image.fromarray(rgb).save(
            os.path.join(out_dir, f"slice_{i:04d}.png")
        )


def phase_fraction(volume, val):
    return float((volume == val).mean())


# ============================================================
# 几何函数
# ============================================================

def get_interface_pore_mask(am_mask, pore_mask):
    st = generate_binary_structure(3, 1)
    am_dil = binary_dilation(am_mask, structure=st)
    return pore_mask & am_dil


def compute_distance_to_am(am_mask, voxel_size):
    return distance_transform_edt(
        ~am_mask,
        sampling=voxel_size,
    )


def keep_only_cbd_touching_am(cbd_mask, am_mask):
    labeled, num = label(cbd_mask)
    if num == 0:
        return cbd_mask

    st = generate_binary_structure(3, 1)
    am_dil = binary_dilation(am_mask, structure=st)

    keep = np.zeros_like(cbd_mask)

    for i in range(1, num + 1):
        comp = (labeled == i)
        if np.any(comp & am_dil):
            keep |= comp

    return keep


# ============================================================
# CBD核心生成
# ============================================================

def generate_cbd_phase(
    volume,
    target_cbd_vol_frac,
    w_um,
    voxel_size,
    pore_value=0,
    am_value=1,
    cbd_value=2,
    seed=42,
    max_growth_factor=4.0,
    remove_isolated=True
):
    rng = np.random.default_rng(seed)

    pore_mask = (volume == pore_value)
    am_mask = (volume == am_value)

    total = volume.size
    target = int(target_cbd_vol_frac * total)

    interface = get_interface_pore_mask(am_mask, pore_mask)
    dist = compute_distance_to_am(am_mask, voxel_size)

    sigma = (
        w_um / voxel_size[0],
        w_um / voxel_size[1],
        w_um / voxel_size[2],
    )

    spread = gaussian_filter(interface.astype(float), sigma=sigma)

    decay = np.exp(-dist / (w_um + 1e-6))

    candidate = pore_mask & (dist < w_um * max_growth_factor)

    if np.sum(candidate) < target:
        candidate = pore_mask

    rank = spread * decay + rng.uniform(0, 1e-6, volume.shape)

    idx = np.flatnonzero(candidate)
    score = rank.ravel()[idx]

    order = np.argsort(-score)
    chosen = idx[order[:target]]

    cbd = np.zeros_like(volume, dtype=bool).ravel()
    cbd[chosen] = True
    cbd = cbd.reshape(volume.shape)

    if remove_isolated:
        cbd = keep_only_cbd_touching_am(cbd, am_mask)

    out = volume.copy()
    out[cbd] = cbd_value

    summary = {
        "phi_target": target_cbd_vol_frac,
        "phi_actual": phase_fraction(out, cbd_value),
        "w": w_um,
        "volume_shape": list(volume.shape),
    }

    return out, summary


# ============================================================
# 主程序
# ============================================================

def main():

    ensure_dir(OUT_DIR)

    print("Loading npy volume...")
    volume = np.load(INPUT_VOLUME_PATH).astype(np.uint8)

    print("Original shape:", volume.shape)
    print("Unique values:", np.unique(volume))

    # 🔥 去掉边界（解决红框问题）
    volume = volume[:, 2:-2, 2:-2]

    print("Cropped shape:", volume.shape)

    print("Generating CBD...")
    voxel_size = (VOXEL_SIZE_Y, VOXEL_SIZE_Y, VOXEL_SIZE_Y)
    volume3, summary = generate_cbd_phase(
        volume,
        TARGET_CBD_VOL_FRAC,
        W_UM,
        voxel_size
    )

    # 保存
    np.save(os.path.join(OUT_DIR, "volume_3phase.npy"), volume3)
    save_json(summary, os.path.join(OUT_DIR, "summary.json"))

    print("Saving color slices...")
    save_color_slices(volume3, os.path.join(OUT_DIR, "slices_color"), PORE_VALUE, AM_VALUE, CBD_VALUE)

    print("Done!")
    print("Output folder:", OUT_DIR)


if __name__ == "__main__":
    main()
