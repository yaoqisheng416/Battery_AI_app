#build_large_volume_conditions_from_real.py:
#从原始真实253*456*258中裁剪224*224*224（patch=128,overlap=32,stride=96)一共8个patch，进行后续大体积生成与验证
from __future__ import annotations

import os
import json
import contextlib
from typing import Dict, Tuple, List

import numpy as np
from scipy.ndimage import label
import taufactor as tau


# ============================================================
# 0) 路径与配置
# ============================================================

# 这里改成你的真实原始体积路径（0=pore, 1=solid）
REAL_VOLUME_PATH = r"../electrode_twin/reconstruction_results/original_volume.npy"
OUT_DIR = r"./paper_verify/large_volume_224_real_reference"

# 你要裁的 224^3 真实块在原始体积中的起点 [Y, Z, X]
# 你后面可以改这个起点
CROP_START_Y = 0
CROP_START_Z = 100
CROP_START_X = 0

LARGE_VOL_SIZE = 224
PATCH_SIZE = 128
OVERLAP = 32
STRIDE = PATCH_SIZE - OVERLAP  # 96

# 相定义
PORE_VALUE = 0
SOLID_VALUE = 1

# 体素尺寸 [Y, Z, X]
VOXEL_SIZE_Y = 0.0315
VOXEL_SIZE_Z = 0.02791
VOXEL_SIZE_X = 0.02791

# 小孔隙清理
REMOVE_SMALL_PORE_COMPONENTS = True
MIN_PORE_COMPONENT_SIZE = 10

TAU_NONPERC_VALUE = 1e6
SUPPRESS_TAUFACTOR_OUTPUT = True


# ============================================================
# 1) 基础工具
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def safe_scalar(x, default: float = 0.0) -> float:
    try:
        arr = np.asarray(x).reshape(-1)
        if arr.size == 0:
            return float(default)
        v = float(arr[0])
        if not np.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)


@contextlib.contextmanager
def suppress_stdout_stderr(enabled: bool = True):
    if not enabled:
        yield
        return

    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


# ============================================================
# 2) 指标函数
# ============================================================

def remove_small_pore_components(volume: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1:
        return volume.copy().astype(np.uint8)

    pore_mask = (volume == PORE_VALUE)
    labeled, num = label(pore_mask)

    if num == 0:
        return volume.copy().astype(np.uint8)

    sizes = np.bincount(labeled.ravel())
    keep_pore = np.zeros_like(pore_mask, dtype=bool)

    for comp_id in range(1, len(sizes)):
        if sizes[comp_id] >= min_size:
            keep_pore[labeled == comp_id] = True

    cleaned = np.where(keep_pore, PORE_VALUE, SOLID_VALUE).astype(np.uint8)
    return cleaned


def porosity(volume01: np.ndarray) -> float:
    return float((volume01 == PORE_VALUE).mean())


def surface_area(volume01: np.ndarray) -> float:
    v = volume01.astype(np.uint8)

    n_y = np.abs(v[1:, :, :] - v[:-1, :, :]).sum()
    n_z = np.abs(v[:, 1:, :] - v[:, :-1, :]).sum()
    n_x = np.abs(v[:, :, 1:] - v[:, :, :-1]).sum()

    area_y = n_y * (VOXEL_SIZE_Z * VOXEL_SIZE_X)
    area_z = n_z * (VOXEL_SIZE_Y * VOXEL_SIZE_X)
    area_x = n_x * (VOXEL_SIZE_Y * VOXEL_SIZE_Z)

    total_area = area_y + area_z + area_x

    total_volume = (
        volume01.shape[0] * VOXEL_SIZE_Y *
        volume01.shape[1] * VOXEL_SIZE_Z *
        volume01.shape[2] * VOXEL_SIZE_X
    )

    if total_volume <= 0:
        return 0.0

    return float(total_area / total_volume)


def largest_connected_ratio(volume01: np.ndarray) -> float:
    pore_mask = (volume01 == PORE_VALUE)
    labeled, num = label(pore_mask)

    sizes = np.bincount(labeled.ravel())
    if len(sizes) <= 1:
        return 0.0

    largest = sizes[1:].max()
    total = sizes[1:].sum()

    if total == 0:
        return 0.0

    return float(largest / total)


def is_percolating_along_z(volume: np.ndarray) -> bool:
    pore_mask = (volume == PORE_VALUE)

    labeled, num = label(pore_mask)
    if num == 0:
        return False

    z0_labels = set(np.unique(labeled[:, 0, :]))
    zend_labels = set(np.unique(labeled[:, -1, :]))
    common = z0_labels & zend_labels
    common.discard(0)

    return len(common) > 0


def compute_tau_deff_z(volume: np.ndarray) -> tuple[float, float, int]:
    percolating = is_percolating_along_z(volume)

    if not percolating:
        return float(TAU_NONPERC_VALUE), 0.0, 0

    pore_phase = (volume == PORE_VALUE).astype(np.uint8)   # [Y, Z, X]
    pore_phase_zyx = np.transpose(pore_phase, (1, 0, 2))   # [Z, Y, X]

    try:
        with suppress_stdout_stderr(SUPPRESS_TAUFACTOR_OUTPUT):
            solver = tau.Solver(pore_phase_zyx)
            solver.solve()

        tau_value = safe_scalar(solver.tau, default=TAU_NONPERC_VALUE)
        deff_value = safe_scalar(solver.D_eff, default=0.0)

        if not np.isfinite(tau_value):
            tau_value = float(TAU_NONPERC_VALUE)
        if not np.isfinite(deff_value):
            deff_value = 0.0

        return float(tau_value), float(deff_value), 1

    except Exception:
        return float(TAU_NONPERC_VALUE), 0.0, 0


def estimate_deff_from_porosity_tau(porosity_value: float, tau_value: float) -> float:
    tau_value = max(float(tau_value), 1e-8)
    porosity_value = max(float(porosity_value), 1e-8)
    return float(max(porosity_value / tau_value, 1e-8))


def compute_metrics(volume01: np.ndarray) -> Dict[str, float]:
    tau_z, deff_z, is_perc = compute_tau_deff_z(volume01)
    return {
        "porosity": porosity(volume01),
        "surface_area": surface_area(volume01),
        "tau_z": float(tau_z),
        "deff_z": float(deff_z),
        "largest_connected_ratio": largest_connected_ratio(volume01),
        "is_percolating_z": int(is_perc),
    }


# ============================================================
# 3) 裁剪与切块
# ============================================================

def crop_large_volume(volume: np.ndarray, start_y: int, start_z: int, start_x: int, size: int) -> np.ndarray:
    return volume[start_y:start_y+size, start_z:start_z+size, start_x:start_x+size].copy()


def extract_patch(volume: np.ndarray, y0: int, z0: int, x0: int, patch_size: int) -> np.ndarray:
    return volume[y0:y0+patch_size, z0:z0+patch_size, x0:x0+patch_size].copy()


def main():
    print("=" * 100)
    print("Build 224^3 local conditions from real large volume")
    print("=" * 100)

    ensure_dir(OUT_DIR)
    patch_dir = os.path.join(OUT_DIR, "real_patches")
    ensure_dir(patch_dir)

    real_volume = np.load(REAL_VOLUME_PATH).astype(np.uint8)
    print("Loaded real volume shape:", real_volume.shape)

    # 检查范围
    sy, sz, sx = real_volume.shape
    assert CROP_START_Y + LARGE_VOL_SIZE <= sy, "Y 方向裁剪越界"
    assert CROP_START_Z + LARGE_VOL_SIZE <= sz, "Z 方向裁剪越界"
    assert CROP_START_X + LARGE_VOL_SIZE <= sx, "X 方向裁剪越界"

    real_large = crop_large_volume(
        real_volume,
        CROP_START_Y,
        CROP_START_Z,
        CROP_START_X,
        LARGE_VOL_SIZE,
    )

    if REMOVE_SMALL_PORE_COMPONENTS:
        real_large_clean = remove_small_pore_components(real_large, MIN_PORE_COMPONENT_SIZE)
    else:
        real_large_clean = real_large.copy()

    np.save(os.path.join(OUT_DIR, "real_large_volume_224_raw.npy"), real_large.astype(np.uint8))
    np.save(os.path.join(OUT_DIR, "real_large_volume_224_clean.npy"), real_large_clean.astype(np.uint8))

    large_metrics = compute_metrics(real_large_clean)

    starts = [0, STRIDE]   # [0, 96]
    local_conditions = []

    idx = 0
    for iy, y0 in enumerate(starts):
        for iz, z0 in enumerate(starts):
            for ix, x0 in enumerate(starts):
                patch = extract_patch(real_large_clean, y0, z0, x0, PATCH_SIZE)
                metrics = compute_metrics(patch)

                patch_name = f"real_patch_{idx:03d}_y{iy}_z{iz}_x{ix}"
                np.save(os.path.join(patch_dir, f"{patch_name}.npy"), patch.astype(np.uint8))

                local_conditions.append({
                    "patch_id": idx,
                    "patch_name": patch_name,
                    "grid_index": [iy, iz, ix],
                    "start_in_large_224": [y0, z0, x0],
                    "metrics": metrics,
                })
                idx += 1

    summary = {
        "real_volume_path": REAL_VOLUME_PATH,
        "crop_start_yzx": [CROP_START_Y, CROP_START_Z, CROP_START_X],
        "large_volume_size": LARGE_VOL_SIZE,
        "patch_size": PATCH_SIZE,
        "overlap": OVERLAP,
        "stride": STRIDE,
        "num_patches": len(local_conditions),
        "cleaning": {
            "remove_small_pore_components": REMOVE_SMALL_PORE_COMPONENTS,
            "min_pore_component_size": MIN_PORE_COMPONENT_SIZE,
        },
        "large_volume_metrics_clean": large_metrics,
        "local_conditions": local_conditions,
        "saved_files": {
            "real_large_volume_224_raw": os.path.join(OUT_DIR, "real_large_volume_224_raw.npy"),
            "real_large_volume_224_clean": os.path.join(OUT_DIR, "real_large_volume_224_clean.npy"),
            "real_patches_dir": patch_dir,
        }
    }

    save_json(summary, os.path.join(OUT_DIR, "local_conditions_224.json"))

    print("Saved to:", OUT_DIR)
    print("Large-volume metrics:", large_metrics)
    print("=" * 100)


if __name__ == "__main__":
    main()