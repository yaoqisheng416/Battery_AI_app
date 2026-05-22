# generate_specific_volume_from_user.py
# ============================================================
# 根据用户输入的孔隙率 + 迂曲度条件，生成并拼接大体积 AM-pore 数字孪生结构
# 并保存最终体数据沿 Y 方向的全部 ZX 切片可视化图
# ============================================================
#
# 原始 NPY / 生成体数据轴顺序：
#   volume[y, z, x]
#
# 其中：
#   y = 切片堆积方向
#   z = 电极厚度方向 / 锂离子传输方向
#   x = 横向方向
#
# 每一张 ZX 切片为：
#   volume[y, :, :]
#
# 体素标签：
#   0 = PORE，孔隙
#   1 = AM/SOLID，活性材料骨架
#
# 本版本只额外保存：
#   OUT_DIR/all_Y_slices_ZX_png/slice_y_0000.png
#   OUT_DIR/all_Y_slices_ZX_png/slice_y_0001.png
#   ...
#
# ============================================================

from __future__ import annotations

import sys
sys.path.append("../electrode_twin")

import os
import json
import csv
import contextlib
from typing import Dict, Tuple, List, Any

import numpy as np
import torch
from scipy.ndimage import (
    label,
    binary_erosion,
    binary_dilation,
    binary_opening,
)
import taufactor as tau

# matplotlib 用于保存沿 Y 方向的 ZX 切片图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from latent_diffusion import LatentDiffusionConfig, LatentDiffusionModule
from vaenet import VAENet, VAENetConfig
from vaemodule import VAEModule, VAEModuleConfig


# ============================================================
# 0. 路径与运行配置
# ============================================================

SUMMARY_JSON_PATH = r"../electrode_twin/latent_dataset/dataset_summary.json"

TRAIN_METRICS_TABLE_PATH = r"../electrode_twin/latent_dataset/metrics_table.csv"

# LDM_CKPT_PATH = r"../electrode_twin/ldm_checkpoints/ldm-epoch337-valloss0.109533.ckpt"
# VAE_CKPT_PATH = r"../electrode_twin/checkpoints/vae-epoch074-valloss-1.9715.ckpt"
LDM_CKPT_PATH = r"C:/Users/47574/Documents/My_file/储慧智能/中科大BDA/Battery_AI_app/ldm_checkpoints/ldm-epoch337-valloss0.109533.ckpt"
VAE_CKPT_PATH = r"C:/Users/47574/Documents/My_file/储慧智能/中科大BDA/Battery_AI_app/checkpoints/vae-epoch074-valloss-1.9715.ckpt"
OUT_DIR = r"./paper_verify/large_volume_user_porosity_tau_generated"

DEVICE = "cuda"

PATCH_SIZE = 128
OVERLAP = 32
STRIDE = PATCH_SIZE - OVERLAP

GRID_SHAPE = (2, 2, 2)


# ============================================================
# 2. 用户条件输入模式
# ============================================================
#
# 只保留两个模式：
#
#   "uniform_porosity"
#       所有 patch 使用相同孔隙率和迂曲度。
#
#   "manual_user"
#       用户为每个 patch 手动输入孔隙率和迂曲度。
#
# 当前建议：
#   如果实验电极孔隙率约为 0.30，先使用 uniform_porosity。

CONDITION_INPUT_MODE = "uniform_porosity"


# ------------------------------------------------------------
# 2.1 uniform_porosity 模式
# ------------------------------------------------------------
#
# 所有 patch 都使用下面两个条件：
#
#   TARGET_PATCH_POROSITY:
#       目标孔隙率。
#
#   TARGET_PATCH_TAU_Z:
#       目标 Z 方向迂曲度。
#
# 示例：
#   TARGET_PATCH_POROSITY = 0.30
#   TARGET_PATCH_TAU_Z = 3.30

TARGET_PATCH_POROSITY = 0.30
TARGET_PATCH_TAU_Z = 3.30


# ------------------------------------------------------------
# 2.2 manual_user 模式
# ------------------------------------------------------------
#
# 如果选择：
#
#   CONDITION_INPUT_MODE = "manual_user"
#
# 就在这里手动填写每个 patch 的孔隙率和迂曲度。
#
# 格式：
#
#   MANUAL_PATCH_CONDITIONS = [
#       {"grid_index": [iy, iz, ix], "porosity": 孔隙率, "tau_z": 迂曲度},
#       ...
#   ]
#
# 注意：
#   1. grid_index 顺序是 [Y_patch, Z_patch, X_patch]。
#   2. 如果 GRID_SHAPE = (2, 2, 2)，必须填写 8 个 patch。
#   3. 每个 patch 必须包含：
#        grid_index
#        porosity
#        tau_z
#
# 示例：GRID_SHAPE = (2, 2, 2)
#
# MANUAL_PATCH_CONDITIONS = [
#     {"grid_index": [0, 0, 0], "porosity": 0.30, "tau_z": 3.30},
#     {"grid_index": [0, 0, 1], "porosity": 0.31, "tau_z": 3.20},
#     {"grid_index": [0, 1, 0], "porosity": 0.29, "tau_z": 3.40},
#     {"grid_index": [0, 1, 1], "porosity": 0.30, "tau_z": 3.30},
#     {"grid_index": [1, 0, 0], "porosity": 0.30, "tau_z": 3.30},
#     {"grid_index": [1, 0, 1], "porosity": 0.32, "tau_z": 3.10},
#     {"grid_index": [1, 1, 0], "porosity": 0.28, "tau_z": 3.50},
#     {"grid_index": [1, 1, 1], "porosity": 0.30, "tau_z": 3.30},
# ]

MANUAL_PATCH_CONDITIONS: List[Dict[str, Any]] = []


# ============================================================
# 3. 条件自动补全规则
# ============================================================
#
# 用户输入：
#   porosity
#   tau_z
#
# 自动补全：
#
#   surface_area:
#       从训练集指标表 TRAIN_METRICS_TABLE_PATH 中寻找最近邻。
#       最近邻使用归一化距离：
#
#           distance =
#             ((porosity_i - porosity_target) / std_porosity)^2
#           + ((tau_z_i - tau_z_target) / std_tau_z)^2
#
#       找到距离最小的训练样本后，取该样本 surface_area。
#
#   deff_z:
#       使用：
#
#           deff_z = porosity / tau_z
#
# 如果训练集指标表不存在：
#   surface_area 使用 dataset_summary 中 surface 的 mean。

AUTO_SURFACE_MODE = "nearest_training_porosity_tau"
AUTO_DEFF_MODE = "porosity_over_tau"


# ============================================================
# 4. 生成参数
# ============================================================

# 每个 patch 生成多少个候选，然后筛选最优。
NUM_SAMPLES_PER_PATCH = 32

PORE_VALUE = 0
SOLID_VALUE = 1

VOXEL_SIZE_Y = 0.0315
VOXEL_SIZE_Z = 0.02791
VOXEL_SIZE_X = 0.02791

REMOVE_SMALL_PORE_COMPONENTS = True
MIN_PORE_COMPONENT_SIZE = 10

POSTPROCESS_CONFIGS = [
    {"name": "raw", "mode": "none"},
    {"name": "erode1", "mode": "erode", "iters": 1},
    {"name": "open1", "mode": "open", "iters": 1},
    {"name": "erode1_dilate1", "mode": "erode_dilate", "erode_iters": 1, "dilate_iters": 1},
]

USE_ADAPTIVE_THRESHOLD_FOR_POROSITY = True
ADAPTIVE_THRESHOLD_MAX_ITERS = 25
ADAPTIVE_THRESHOLD_TOL = 1e-4

THRESHOLD_OFFSETS = [-0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]

# 核心筛选指标：孔隙率 + 迂曲度。
# surface_area 只作为模型条件自动补全，不作为主要误差指标。
CHEAP_ERROR_WEIGHTS = {
    "porosity": 4.0,
}

FINAL_ERROR_WEIGHTS = {
    "porosity": 4.0,
    "tau_z": 5.0,
    "deff_z": 0.5,
}

USE_STD_NORMALIZED_ERROR = True

TOPOLOGY_PENALTY_WEIGHT = 1.0
MIN_SOLID_COMPONENT_COUNT_SOFT = 10
EXACT_EVAL_TOPK_PER_CANDIDATE = 3

WARN_IF_TARGET_OOD = True
CLIP_NORMALIZED_CONDITION_TO_TRAIN_RANGE = False

TAU_NONPERC_VALUE = 1e6
SUPPRESS_TAUFACTOR_OUTPUT = True


# ============================================================
# 5. 沿 Y 方向 ZX 切片可视化输出设置
# ============================================================
#
# 本版本只保存沿 Y 方向的每一张 ZX 截面 PNG 图。
#
# 体数据轴顺序：
#   volume[y, z, x]
#
# 固定 y 后：
#   volume[y, :, :] 就是 ZX 截面。
#
# 输出目录：
#   OUT_DIR/all_Y_slices_ZX_png/
#
# 不保存切片 NPY，不保存其他截面。

SAVE_ALL_Y_ZX_SLICE_PNG = True

# 切片颜色风格：
#   "black_yellow" : 0=黄色孔隙，1=黑色 AM
#   "white_blue"   : 0=白色孔隙，1=蓝色 AM
SLICE_COLOR_STYLE = "black_yellow"

# 保存每张切片时是否显示坐标轴。
SLICE_SHOW_AXIS = False

# 每张切片图的 DPI。
SLICE_DPI = 200


# ============================================================
# 6. 条件键
# ============================================================

MODEL_CONDITION_KEYS = [
    "porosity",
    "surface_area",
    "tau_z",
    "deff_z",
]

COND_TO_SUMMARY_KEY = {
    "porosity": "porosity",
    "surface_area": "surface",
    "tau_z": "tau_z",
    "deff_z": "deff_z",
}

COND_TO_METRIC_KEY = {
    "porosity": "porosity",
    "surface_area": "surface_area",
    "tau_z": "tau_z",
    "deff_z": "deff_z",
}


# ============================================================
# 7. 基础工具
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def product_int(values: Tuple[int, int, int]) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return int(out)


def grid_indices(grid_shape: Tuple[int, int, int]):
    gy, gz, gx = grid_shape
    for iy in range(gy):
        for iz in range(gz):
            for ix in range(gx):
                yield iy, iz, ix


def assembled_shape_from_grid(
    grid_shape: Tuple[int, int, int],
    patch_size: int,
    stride: int
) -> Tuple[int, int, int]:
    gy, gz, gx = grid_shape
    sy = patch_size + stride * (gy - 1)
    sz = patch_size + stride * (gz - 1)
    sx = patch_size + stride * (gx - 1)
    return int(sy), int(sz), int(sx)


# ============================================================
# 8. 沿 Y 方向 ZX 切片保存函数
# ============================================================

def get_binary_cmap():
    """
    获取 AM-pore 二值结构颜色表。

    标签：
        0 = PORE
        1 = AM / SOLID
    """
    if SLICE_COLOR_STYLE == "black_yellow":
        return ListedColormap([
            [1.0, 0.78, 0.0],   # 0 = PORE，黄色
            [0.0, 0.0, 0.0],    # 1 = AM/SOLID，黑色
        ])

    if SLICE_COLOR_STYLE == "white_blue":
        return ListedColormap([
            [1.0, 1.0, 1.0],      # 0 = PORE，白色
            [0.1, 0.35, 0.75],    # 1 = AM/SOLID，蓝色
        ])

    raise ValueError('SLICE_COLOR_STYLE 必须是 "black_yellow" 或 "white_blue"。')


def save_all_y_zx_slice_pngs(
    volume_yzx: np.ndarray,
    out_root: str,
) -> Dict[str, str]:
    """
    只保存最终体数据沿 Y 方向的全部 ZX 截面 PNG 图。

    参数：
        volume_yzx:
            三维体数据，轴顺序必须为 [Y, Z, X]。

        out_root:
            输出总文件夹。

    输出：
        返回 PNG 切片目录路径，方便写入 assembly_summary.json。
    """
    if volume_yzx.ndim != 3:
        raise ValueError(f"输入 volume 必须是 3D 数组，当前 shape = {volume_yzx.shape}")

    volume_yzx = volume_yzx.astype(np.uint8, copy=False)

    ny, nz, nx = volume_yzx.shape

    slice_png_dir = os.path.join(out_root, "all_Y_slices_ZX_png")
    ensure_dir(slice_png_dir)

    cmap = get_binary_cmap()

    print("\n开始保存沿 Y 方向的全部 ZX 截面 PNG 图...")
    print(f"  volume shape [Y,Z,X] = {volume_yzx.shape}")
    print(f"  总切片数 Y = {ny}")
    print(f"  每张 ZX 截面 shape [Z,X] = {(nz, nx)}")
    print(f"  输出目录 = {slice_png_dir}")

    for y in range(ny):
        slice_zx = volume_yzx[y, :, :]  # 固定 y，得到 ZX 截面

        png_path = os.path.join(slice_png_dir, f"slice_y_{y:04d}.png")

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)

        ax.imshow(
            slice_zx,
            cmap=cmap,
            interpolation="nearest",
            origin="lower",
            vmin=0,
            vmax=1
        )

        if SLICE_SHOW_AXIS:
            ax.set_title(f"ZX slice at Y = {y}", fontsize=12)
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
        else:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(png_path, dpi=SLICE_DPI, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

        if (y + 1) % 10 == 0 or (y + 1) == ny:
            print(f"  已保存 {y + 1}/{ny} 张 ZX 截面 PNG")

    print("沿 Y 方向全部 ZX 截面 PNG 图保存完成。")

    return {
        "all_y_zx_slice_png_dir": slice_png_dir,
    }


# ============================================================
# 9. 训练集指标表读取与最近邻 surface_area
# ============================================================

def get_summary_stat(summary: Dict, condition_key: str, stat_name: str) -> float:
    skey = COND_TO_SUMMARY_KEY[condition_key]
    return float(summary[skey][stat_name])


def summary_surface_mean(summary: Dict) -> float:
    return float(summary["surface"]["mean"])


def load_training_metrics_table(path: str) -> List[Dict[str, float]]:
    """
    读取训练集逐样本指标表。

    支持：
        csv / json / npz

    必须包含：
        porosity
        tau_z
        surface_area

    如果包含字段名为 surface，也会自动识别为 surface_area。
    """
    if path is None or path == "":
        return []

    if not os.path.exists(path):
        print(f"[提示] 未找到训练集指标表: {path}")
        print("[提示] surface_area 将使用 dataset_summary 中 surface mean。")
        return []

    ext = os.path.splitext(path)[1].lower()
    rows = []

    if ext == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))

    elif ext == ".json":
        data = load_json(path)
        if isinstance(data, dict):
            if "items" in data:
                rows = data["items"]
            elif "samples" in data:
                rows = data["samples"]
            else:
                raise ValueError("JSON 指标表如果是 dict，需要包含 items 或 samples 字段。")
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError("JSON 指标表必须是 list 或包含 items/samples 的 dict。")

    elif ext == ".npz":
        data = np.load(path)
        por_arr = np.asarray(data["porosity"]).reshape(-1)

        if "surface_area" in data:
            surf_arr = np.asarray(data["surface_area"]).reshape(-1)
        elif "surface" in data:
            surf_arr = np.asarray(data["surface"]).reshape(-1)
        else:
            raise ValueError("NPZ 指标表必须包含 surface_area 或 surface。")

        tau_arr = np.asarray(data["tau_z"]).reshape(-1)

        n = len(por_arr)
        if len(surf_arr) != n or len(tau_arr) != n:
            raise ValueError("NPZ 指标表 porosity/surface_area/tau_z 长度不一致。")

        rows = []
        for i in range(n):
            rows.append({
                "porosity": float(por_arr[i]),
                "surface_area": float(surf_arr[i]),
                "tau_z": float(tau_arr[i]),
            })

    else:
        raise ValueError("训练集指标表只支持 .csv / .json / .npz。")

    clean_rows = []

    for r in rows:
        if "surface_area" in r:
            surf_key = "surface_area"
        elif "surface" in r:
            surf_key = "surface"
        else:
            continue

        try:
            item = {
                "porosity": float(r["porosity"]),
                "surface_area": float(r[surf_key]),
                "tau_z": float(r["tau_z"]),
            }

            if np.isfinite(item["porosity"]) and np.isfinite(item["surface_area"]) and np.isfinite(item["tau_z"]):
                clean_rows.append(item)
        except Exception:
            continue

    print(f"训练集指标表读取完成: {path}")
    print(f"  有效样本数 = {len(clean_rows)}")

    return clean_rows


def nearest_surface_area_from_training(
    target_porosity: float,
    target_tau_z: float,
    training_metrics: List[Dict[str, float]],
    summary: Dict,
) -> Tuple[float, Dict[str, float]]:
    """
    根据用户输入的 porosity + tau_z，在训练集指标表中找最近样本，
    返回该样本的 surface_area。

    如果 training_metrics 为空，则返回 summary surface mean。
    """
    if len(training_metrics) == 0:
        surf = summary_surface_mean(summary)
        info = {
            "source": "dataset_summary_surface_mean",
            "surface_area": float(surf),
            "nearest_porosity": None,
            "nearest_tau_z": None,
            "distance": None,
        }
        return float(surf), info

    por_std = max(get_summary_stat(summary, "porosity", "std"), 1e-12)
    tau_std = max(get_summary_stat(summary, "tau_z", "std"), 1e-12)

    best = None

    for item in training_metrics:
        dp = (float(item["porosity"]) - float(target_porosity)) / por_std
        dt = (float(item["tau_z"]) - float(target_tau_z)) / tau_std
        dist = dp * dp + dt * dt

        if best is None or dist < best["distance"]:
            best = {
                "distance": float(dist),
                "nearest_porosity": float(item["porosity"]),
                "nearest_tau_z": float(item["tau_z"]),
                "surface_area": float(item["surface_area"]),
            }

    info = {
        "source": "nearest_training_porosity_tau",
        "surface_area": float(best["surface_area"]),
        "nearest_porosity": float(best["nearest_porosity"]),
        "nearest_tau_z": float(best["nearest_tau_z"]),
        "distance": float(best["distance"]),
    }

    return float(best["surface_area"]), info


def estimate_deff_from_porosity_tau(porosity_value: float, tau_value: float) -> float:
    """
    用 porosity / tau_z 估算有效扩散系数。
    """
    tau_value = max(float(tau_value), 1e-8)
    porosity_value = max(float(porosity_value), 1e-8)
    return float(max(porosity_value / tau_value, 1e-8))


# ============================================================
# 10. 指标函数
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


def solid_component_count(volume01: np.ndarray) -> int:
    solid_mask = (volume01 == SOLID_VALUE)
    _, num = label(solid_mask)
    return int(num)


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


def compute_generated_metrics_cheap(bin_volume: np.ndarray) -> Dict[str, float]:
    return {
        "porosity": porosity(bin_volume),
        "surface_area": surface_area(bin_volume),
        "largest_connected_ratio": largest_connected_ratio(bin_volume),
        "solid_component_count": solid_component_count(bin_volume),
    }


def compute_generated_metrics_exact(bin_volume: np.ndarray) -> Dict[str, float]:
    cheap = compute_generated_metrics_cheap(bin_volume)
    tau_z, deff_z, is_perc = compute_tau_deff_z(bin_volume)

    out = dict(cheap)
    out.update({
        "tau_z": float(tau_z),
        "deff_z": float(deff_z),
        "is_percolating_z": int(is_perc),
    })

    return out


# ============================================================
# 11. 用户条件构造
# ============================================================

def validate_porosity_tau(porosity_value: float, tau_z_value: float):
    if porosity_value <= 0.0 or porosity_value >= 1.0:
        raise ValueError(f"porosity 必须在 (0, 1) 内，当前 porosity={porosity_value}")

    if tau_z_value <= 0.0:
        raise ValueError(f"tau_z 必须大于 0，当前 tau_z={tau_z_value}")


def complete_condition_from_porosity_tau(
    porosity_value: float,
    tau_z_value: float,
    training_metrics: List[Dict[str, float]],
    summary: Dict,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    根据用户输入的 porosity + tau_z 自动补全完整 4 维模型条件。
    """
    validate_porosity_tau(porosity_value, tau_z_value)

    surface_value, surface_info = nearest_surface_area_from_training(
        target_porosity=porosity_value,
        target_tau_z=tau_z_value,
        training_metrics=training_metrics,
        summary=summary,
    )

    deff_value = estimate_deff_from_porosity_tau(
        porosity_value=porosity_value,
        tau_value=tau_z_value,
    )

    condition = {
        "porosity": float(porosity_value),
        "surface_area": float(surface_value),
        "tau_z": float(tau_z_value),
        "deff_z": float(deff_value),
    }

    auto_info = {
        "surface_area_selection": surface_info,
        "deff_z_rule": "deff_z = porosity / tau_z",
    }

    return condition, auto_info


def get_manual_condition_lookup() -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    """
    把 MANUAL_PATCH_CONDITIONS 转成 grid_index -> condition 的字典。
    """
    lookup = {}

    for item in MANUAL_PATCH_CONDITIONS:
        if "grid_index" not in item:
            raise ValueError("manual_user 模式下，每个条目必须包含 grid_index。")

        if "porosity" not in item:
            raise ValueError("manual_user 模式下，每个条目必须包含 porosity。")

        if "tau_z" not in item:
            raise ValueError("manual_user 模式下，每个条目必须包含 tau_z。")

        g = item["grid_index"]

        if len(g) != 3:
            raise ValueError("grid_index 必须是长度为 3 的列表，例如 [0,0,0]。")

        key = (int(g[0]), int(g[1]), int(g[2]))

        if key in lookup:
            raise ValueError(f"manual_user 中 grid_index={key} 重复。")

        lookup[key] = item

    return lookup


def build_user_local_conditions(
    summary: Dict,
    training_metrics: List[Dict[str, float]],
) -> Tuple[List[Dict], Dict]:
    """
    构建 patch 条件表。

    输出结构与原始 local_conditions 逻辑相似，
    但条件来自用户输入而不是从真实体数据裁剪。
    """
    n_patch = product_int(GRID_SHAPE)

    if CONDITION_INPUT_MODE == "manual_user":
        manual_lookup = get_manual_condition_lookup()

        if len(manual_lookup) != n_patch:
            raise ValueError(
                f"manual_user 模式下，MANUAL_PATCH_CONDITIONS 数量必须等于 {n_patch}，"
                f"当前为 {len(manual_lookup)}。"
            )
    else:
        manual_lookup = {}

    local_conditions = []

    idx = 0

    for iy, iz, ix in grid_indices(GRID_SHAPE):
        if CONDITION_INPUT_MODE == "uniform_porosity":
            por = float(TARGET_PATCH_POROSITY)
            tau_value = float(TARGET_PATCH_TAU_Z)
            raw_user_condition = {
                "porosity": por,
                "tau_z": tau_value,
            }

        elif CONDITION_INPUT_MODE == "manual_user":
            key = (iy, iz, ix)

            if key not in manual_lookup:
                raise ValueError(f"manual_user 模式下缺少 grid_index={key} 的条件。")

            raw = manual_lookup[key]
            por = float(raw["porosity"])
            tau_value = float(raw["tau_z"])
            raw_user_condition = {
                "porosity": por,
                "tau_z": tau_value,
                "grid_index": [iy, iz, ix],
            }

        else:
            raise ValueError('CONDITION_INPUT_MODE 必须是 "uniform_porosity" 或 "manual_user"。')

        model_condition, auto_info = complete_condition_from_porosity_tau(
            porosity_value=por,
            tau_z_value=tau_value,
            training_metrics=training_metrics,
            summary=summary,
        )

        y0 = iy * STRIDE
        z0 = iz * STRIDE
        x0 = ix * STRIDE

        patch_name = f"user_patch_{idx:03d}_y{iy}_z{iz}_x{ix}"

        local_conditions.append({
            "patch_id": idx,
            "patch_name": patch_name,
            "grid_index": [iy, iz, ix],
            "start_in_large": [y0, z0, x0],
            "raw_user_condition": raw_user_condition,
            "auto_completed_condition_info": auto_info,
            "metrics": model_condition,
        })

        idx += 1

    out_shape = assembled_shape_from_grid(GRID_SHAPE, PATCH_SIZE, STRIDE)

    target_global_condition = {
        "porosity": float(np.mean([p["metrics"]["porosity"] for p in local_conditions])),
        "surface_area": float(np.mean([p["metrics"]["surface_area"] for p in local_conditions])),
        "tau_z": float(np.mean([p["metrics"]["tau_z"] for p in local_conditions])),
        "deff_z": float(np.mean([p["metrics"]["deff_z"] for p in local_conditions])),
    }

    info = {
        "mode": "user_defined_porosity_tau_conditions",
        "condition_input_mode": CONDITION_INPUT_MODE,
        "grid_shape_yzx": list(GRID_SHAPE),
        "patch_size": PATCH_SIZE,
        "overlap": OVERLAP,
        "stride": STRIDE,
        "assembled_shape_yzx": list(out_shape),
        "num_patches": len(local_conditions),
        "user_input_keys": ["porosity", "tau_z"],
        "model_condition_keys": MODEL_CONDITION_KEYS,
        "surface_area_auto_selection": {
            "mode": AUTO_SURFACE_MODE,
            "training_metrics_table_path": TRAIN_METRICS_TABLE_PATH,
            "fallback": "dataset_summary surface mean if training metrics table is missing",
        },
        "deff_z_auto_rule": AUTO_DEFF_MODE,
        "target_global_condition": target_global_condition,
        "local_conditions": local_conditions,
    }

    return local_conditions, info


def validate_target_condition(cond_raw: Dict[str, float], summary: Dict) -> List[Dict]:
    range_report = []

    for key in MODEL_CONDITION_KEYS:
        raw_val = float(cond_raw[key])
        skey = COND_TO_SUMMARY_KEY[key]

        vmin = float(summary[skey]["min"])
        vmax = float(summary[skey]["max"])

        if raw_val < vmin:
            status = "below_min"
        elif raw_val > vmax:
            status = "above_max"
        else:
            status = "in_range"

        range_report.append({
            "condition_key": key,
            "summary_key": skey,
            "input_value": raw_val,
            "min": vmin,
            "max": vmax,
            "status": status,
        })

    return range_report


def normalize_condition(cond_raw: Dict[str, float], summary: Dict) -> np.ndarray:
    vals = []

    for key in MODEL_CONDITION_KEYS:
        raw_val = float(cond_raw[key])
        skey = COND_TO_SUMMARY_KEY[key]

        vmin = float(summary[skey]["min"])
        vmax = float(summary[skey]["max"])

        if WARN_IF_TARGET_OOD and (raw_val < vmin or raw_val > vmax):
            print(f"[警告] 条件 {key} = {raw_val:.6f} 超出训练范围 [{vmin:.6f}, {vmax:.6f}]")

        denom = max(vmax - vmin, 1e-12)
        norm_val = (raw_val - vmin) / denom

        if CLIP_NORMALIZED_CONDITION_TO_TRAIN_RANGE:
            norm_val = float(np.clip(norm_val, 0.0, 1.0))

        vals.append(norm_val)

    return np.asarray(vals, dtype=np.float32)


# ============================================================
# 12. 阈值与后处理
# ============================================================

def find_threshold_for_target_porosity(
    prob_volume: np.ndarray,
    target_porosity: float,
    max_iters: int = 25,
    tol: float = 1e-4,
) -> Tuple[float, float]:
    """
    对概率体寻找阈值，使二值化后孔隙率接近 target_porosity。

    约定：
        prob_volume >= threshold -> SOLID_VALUE = 1
        prob_volume <  threshold -> PORE_VALUE  = 0
    """
    lo, hi = 0.0, 1.0
    best_t = 0.5
    best_por = None
    best_err = 1e18

    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        bin_volume = (prob_volume >= mid).astype(np.uint8)
        por = porosity(bin_volume)

        err = abs(por - target_porosity)

        if err < best_err:
            best_err = err
            best_t = mid
            best_por = por

        if err <= tol:
            break

        if por < target_porosity:
            lo = mid
        else:
            hi = mid

    return float(best_t), float(best_por if best_por is not None else 0.0)


def threshold_prob_volume(prob_volume: np.ndarray, threshold: float) -> np.ndarray:
    threshold = float(np.clip(threshold, 0.0, 1.0))
    return (prob_volume >= threshold).astype(np.uint8)


def postprocess_solid_topology(bin_volume: np.ndarray, cfg: Dict) -> np.ndarray:
    mode = cfg["mode"]

    solid = (bin_volume == SOLID_VALUE)

    if mode == "none":
        solid2 = solid

    elif mode == "erode":
        iters = int(cfg.get("iters", 1))
        solid2 = binary_erosion(solid, iterations=iters)

    elif mode == "open":
        iters = int(cfg.get("iters", 1))
        solid2 = binary_opening(solid, iterations=iters)

    elif mode == "erode_dilate":
        e = int(cfg.get("erode_iters", 1))
        d = int(cfg.get("dilate_iters", 1))
        solid2 = binary_erosion(solid, iterations=e)
        solid2 = binary_dilation(solid2, iterations=d)

    else:
        raise ValueError(f"未知后处理模式: {mode}")

    return np.where(solid2, SOLID_VALUE, PORE_VALUE).astype(np.uint8)


# ============================================================
# 13. 模型加载
# ============================================================

def load_ldm_model(ckpt_path: str, device: torch.device) -> LatentDiffusionModule:
    config = LatentDiffusionConfig(
        latent_channels=4,
        latent_size=16,
        cond_dim=4,
        cond_embed_dim=128,
        time_embed_dim=128,
        model_channels=64,
        channel_mult=(1, 2, 4),
        dropout=0.0,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        lr=1e-4,
        weight_decay=1e-4,
    )

    model = LatentDiffusionModule.load_from_checkpoint(
        ckpt_path,
        config=config,
        map_location=device,
        weights_only=False,
    )

    model.eval()
    model.to(device)

    return model


def load_vae_model(ckpt_path: str, device: torch.device) -> VAEModule:
    net_config = VAENetConfig(
        dimension=3,
        in_channels=1,
        out_channels=1,
        z_dim=4,
        ch=32,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        resolution=128,
        num_groups=16,
        use_attention=False,
        final_activation="sigmoid",
    )

    vae_config = VAEModuleConfig(
        kl_weight=1e-5,
        kl_start=0.0,
        kl_max=1e-5,
        kl_warmup_steps=5000,
        reconstruction_loss="mse",
        lr=1e-4,
        scheduler_type="none",
    )

    net = VAENet(net_config)

    model = VAEModule.load_from_checkpoint(
        ckpt_path,
        encdec=net,
        config=vae_config,
        conditional=False,
        verbose=False,
        map_location=device,
        weights_only=False,
    )

    model.eval()
    model.to(device)

    return model


# ============================================================
# 14. 单 patch 生成
# ============================================================

@torch.no_grad()
def decode_one_prob_volume(
    ldm_model: LatentDiffusionModule,
    vae_model: VAEModule,
    cond_norm: torch.Tensor,
) -> np.ndarray:
    latent_shape = (4, 16, 16, 16)

    z = ldm_model.sample(cond=cond_norm, shape=latent_shape)

    x_prob = vae_model.decode(z, apply_postdecode=True)
    x_prob = x_prob[:, 0].clamp(0.0, 1.0)

    return x_prob[0].detach().cpu().numpy().astype(np.float32)


def make_error_dict(
    target_condition: Dict[str, float],
    generated_metrics: Dict[str, float],
    summary: Dict,
    keys: List[str],
) -> Dict[str, float]:
    err = {}

    for key in keys:
        metric_key = COND_TO_METRIC_KEY[key]
        summary_key = COND_TO_SUMMARY_KEY[key]

        gv = float(generated_metrics[metric_key])
        tv = float(target_condition[key])

        abs_err = abs(gv - tv)
        rel_err = abs_err / max(abs(tv), 1e-12)

        std_ref = float(summary[summary_key]["std"])
        std_norm_err = abs_err / max(std_ref, 1e-12)

        err[f"{key}_generated"] = gv
        err[f"{key}_target"] = tv
        err[f"{key}_abs_error"] = abs_err
        err[f"{key}_rel_error"] = rel_err
        err[f"{key}_std_normalized_error"] = std_norm_err

    return err


def compute_weighted_score(
    error_dict: Dict[str, float],
    generated_metrics: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    score = 0.0

    for key, w in weights.items():
        if USE_STD_NORMALIZED_ERROR:
            e = float(error_dict[f"{key}_std_normalized_error"])
        else:
            e = float(error_dict[f"{key}_rel_error"])

        score += float(w) * e

    solid_num = int(generated_metrics.get("solid_component_count", 0))

    if solid_num < MIN_SOLID_COMPONENT_COUNT_SOFT:
        score += TOPOLOGY_PENALTY_WEIGHT * float(MIN_SOLID_COMPONENT_COUNT_SOFT - solid_num)

    return float(score)


def generate_best_patch_from_condition(
    model_condition: Dict[str, float],
    summary: Dict,
    ldm_model,
    vae_model,
    num_samples: int,
    save_patch_dir: str | None = None,
) -> Dict:
    """
    根据一个 patch 的 4 维模型条件生成多个候选，并选择最优候选。

    当前筛选重点：
        cheap 阶段：porosity
        exact 阶段：porosity + tau_z + deff_z
    """
    cond_norm_np = normalize_condition(model_condition, summary)

    device = next(ldm_model.parameters()).device
    cond_norm_all = torch.from_numpy(cond_norm_np).unsqueeze(0).repeat(num_samples, 1).to(device)

    best_result = None
    candidate_summaries = []

    for i in range(num_samples):
        candidate_id = f"sample_{i:03d}"
        cond_i = cond_norm_all[i:i + 1]

        prob_volume = decode_one_prob_volume(ldm_model, vae_model, cond_i)

        if USE_ADAPTIVE_THRESHOLD_FOR_POROSITY:
            threshold_base, _ = find_threshold_for_target_porosity(
                prob_volume=prob_volume,
                target_porosity=float(model_condition["porosity"]),
                max_iters=ADAPTIVE_THRESHOLD_MAX_ITERS,
                tol=ADAPTIVE_THRESHOLD_TOL,
            )
        else:
            threshold_base = 0.5

        coarse_pool = []

        for offset in THRESHOLD_OFFSETS:
            t_used = float(np.clip(threshold_base + offset, 0.0, 1.0))
            bin_raw = threshold_prob_volume(prob_volume, t_used)

            for pp_cfg in POSTPROCESS_CONFIGS:
                bin_post = postprocess_solid_topology(bin_raw, pp_cfg)

                if REMOVE_SMALL_PORE_COMPONENTS:
                    bin_final = remove_small_pore_components(bin_post, MIN_PORE_COMPONENT_SIZE)
                else:
                    bin_final = bin_post.copy()

                cheap_metrics = compute_generated_metrics_cheap(bin_final)

                cheap_error_dict = make_error_dict(
                    target_condition=model_condition,
                    generated_metrics=cheap_metrics,
                    summary=summary,
                    keys=["porosity"],
                )

                cheap_score = compute_weighted_score(
                    error_dict=cheap_error_dict,
                    generated_metrics=cheap_metrics,
                    weights=CHEAP_ERROR_WEIGHTS,
                )

                coarse_pool.append({
                    "candidate_id": candidate_id,
                    "prob_volume": prob_volume,
                    "bin_volume_raw": bin_raw,
                    "bin_volume_postprocess": bin_post,
                    "bin_volume_final": bin_final,
                    "threshold_base": float(threshold_base),
                    "threshold_offset": float(offset),
                    "threshold_used": float(t_used),
                    "postprocess_config": pp_cfg,
                    "cheap_metrics": cheap_metrics,
                    "cheap_error_vs_target": cheap_error_dict,
                    "cheap_score": float(cheap_score),
                })

        coarse_pool = sorted(coarse_pool, key=lambda x: x["cheap_score"])
        exact_pool = coarse_pool[:EXACT_EVAL_TOPK_PER_CANDIDATE]

        candidate_best = None

        for item in exact_pool:
            final_metrics = compute_generated_metrics_exact(item["bin_volume_final"])

            final_error_dict = make_error_dict(
                target_condition=model_condition,
                generated_metrics=final_metrics,
                summary=summary,
                keys=["porosity", "tau_z", "deff_z"],
            )

            total_score = compute_weighted_score(
                error_dict=final_error_dict,
                generated_metrics=final_metrics,
                weights=FINAL_ERROR_WEIGHTS,
            )

            result = dict(item)
            result.update({
                "final_metrics": final_metrics,
                "final_error_vs_target": final_error_dict,
                "total_weighted_score": float(total_score),
            })

            if (candidate_best is None) or (result["total_weighted_score"] < candidate_best["total_weighted_score"]):
                candidate_best = result

            if (best_result is None) or (result["total_weighted_score"] < best_result["total_weighted_score"]):
                best_result = result

        if candidate_best is not None:
            candidate_summaries.append({
                "candidate_id": candidate_id,
                "best_score_within_candidate": candidate_best["total_weighted_score"],
                "best_threshold_used": candidate_best["threshold_used"],
                "best_postprocess_config": candidate_best["postprocess_config"],
                "best_final_metrics": candidate_best["final_metrics"],
            })

        print(
            f"    candidate {i + 1}/{num_samples}: "
            f"current best score = {best_result['total_weighted_score']:.6f}"
        )

    if best_result is None:
        raise RuntimeError("没有生成出有效 patch。")

    if save_patch_dir is not None:
        ensure_dir(save_patch_dir)

        np.save(os.path.join(save_patch_dir, "generated_prob.npy"), best_result["prob_volume"].astype(np.float32))
        np.save(os.path.join(save_patch_dir, "generated_bin_raw.npy"), best_result["bin_volume_raw"].astype(np.uint8))
        np.save(os.path.join(save_patch_dir, "generated_bin_postprocess.npy"), best_result["bin_volume_postprocess"].astype(np.uint8))
        np.save(os.path.join(save_patch_dir, "generated_bin_final.npy"), best_result["bin_volume_final"].astype(np.uint8))

        save_json({
            "model_condition": model_condition,
            "range_report": validate_target_condition(model_condition, summary),
            "best_candidate_id": best_result["candidate_id"],
            "best_score": best_result["total_weighted_score"],
            "best_threshold_base": best_result["threshold_base"],
            "best_threshold_offset": best_result["threshold_offset"],
            "best_threshold_used": best_result["threshold_used"],
            "best_postprocess_config": best_result["postprocess_config"],
            "best_final_metrics": best_result["final_metrics"],
            "candidate_summaries": candidate_summaries,
        }, os.path.join(save_patch_dir, "generation_summary.json"))

    return best_result


# ============================================================
# 15. Patch 概率体拼接
# ============================================================

def make_3d_blending_window(size: int) -> np.ndarray:
    """
    生成 3D Hanning blending window。
    保持原始拼接逻辑。
    """
    wy = np.hanning(size)
    wz = np.hanning(size)
    wx = np.hanning(size)

    w3d = wy[:, None, None] * wz[None, :, None] * wx[None, None, :]

    # 避免 patch 边缘权重为 0
    w3d = 0.05 + 0.95 * w3d

    return w3d.astype(np.float32)


def assemble_prob_patches(
    prob_patches: List[np.ndarray],
    patch_size: int,
    stride: int,
    grid_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    根据 GRID_SHAPE 拼接概率体。
    """
    expected_n = product_int(grid_shape)

    if len(prob_patches) != expected_n:
        raise ValueError(f"prob_patches 数量应为 {expected_n}，当前为 {len(prob_patches)}。")

    out_shape = assembled_shape_from_grid(grid_shape, patch_size, stride)

    prob_accum = np.zeros(out_shape, dtype=np.float32)
    weight_accum = np.zeros(out_shape, dtype=np.float32)

    window = make_3d_blending_window(patch_size)

    idx = 0

    for iy, iz, ix in grid_indices(grid_shape):
        y0 = iy * stride
        z0 = iz * stride
        x0 = ix * stride

        y1 = y0 + patch_size
        z1 = z0 + patch_size
        x1 = x0 + patch_size

        prob_accum[y0:y1, z0:z1, x0:x1] += prob_patches[idx] * window
        weight_accum[y0:y1, z0:z1, x0:x1] += window

        idx += 1

    assembled = prob_accum / np.maximum(weight_accum, 1e-8)

    return assembled.astype(np.float32)


# ============================================================
# 16. 拼接体筛选
# ============================================================

def select_best_assembled_binary(
    assembled_prob: np.ndarray,
    target_condition_large: Dict[str, float],
    summary: Dict,
) -> Dict:
    """
    对拼接后的概率体进行全局阈值和后处理筛选。

    当前筛选指标：
        porosity + tau_z + deff_z
    """
    target_porosity_large = float(target_condition_large["porosity"])

    if USE_ADAPTIVE_THRESHOLD_FOR_POROSITY:
        threshold_base, _ = find_threshold_for_target_porosity(
            prob_volume=assembled_prob,
            target_porosity=target_porosity_large,
            max_iters=ADAPTIVE_THRESHOLD_MAX_ITERS,
            tol=ADAPTIVE_THRESHOLD_TOL,
        )
    else:
        threshold_base = 0.5

    best = None
    all_candidate_summaries = []

    for offset in THRESHOLD_OFFSETS:
        t_used = float(np.clip(threshold_base + offset, 0.0, 1.0))
        bin_raw = threshold_prob_volume(assembled_prob, t_used)

        for pp_cfg in POSTPROCESS_CONFIGS:
            bin_post = postprocess_solid_topology(bin_raw, pp_cfg)

            if REMOVE_SMALL_PORE_COMPONENTS:
                bin_final = remove_small_pore_components(bin_post, MIN_PORE_COMPONENT_SIZE)
            else:
                bin_final = bin_post.copy()

            final_metrics = compute_generated_metrics_exact(bin_final)

            final_error_dict = make_error_dict(
                target_condition=target_condition_large,
                generated_metrics=final_metrics,
                summary=summary,
                keys=["porosity", "tau_z", "deff_z"],
            )

            total_score = compute_weighted_score(
                error_dict=final_error_dict,
                generated_metrics=final_metrics,
                weights=FINAL_ERROR_WEIGHTS,
            )

            item = {
                "threshold_base": float(threshold_base),
                "threshold_offset": float(offset),
                "threshold_used": float(t_used),
                "postprocess_config": pp_cfg,
                "bin_raw": bin_raw,
                "bin_post": bin_post,
                "bin_final": bin_final,
                "final_metrics": final_metrics,
                "final_error_vs_target": final_error_dict,
                "total_weighted_score": float(total_score),
            }

            all_candidate_summaries.append({
                "threshold_offset": float(offset),
                "threshold_used": float(t_used),
                "postprocess_config": pp_cfg,
                "final_metrics": final_metrics,
                "total_weighted_score": float(total_score),
            })

            if (best is None) or (item["total_weighted_score"] < best["total_weighted_score"]):
                best = item

    if best is None:
        raise RuntimeError("拼接体没有筛选出有效结果。")

    best["all_candidate_summaries"] = all_candidate_summaries

    return best


# ============================================================
# 17. 主程序
# ============================================================

def main():
    print("=" * 100)
    print("Generate large AM-pore twin volume from porosity + tortuosity user conditions")
    print("=" * 100)

    ensure_dir(OUT_DIR)

    patch_root = os.path.join(OUT_DIR, "generated_patches")
    ensure_dir(patch_root)

    summary = load_json(SUMMARY_JSON_PATH)

    training_metrics = load_training_metrics_table(TRAIN_METRICS_TABLE_PATH)

    local_conditions, user_condition_info = build_user_local_conditions(
        summary=summary,
        training_metrics=training_metrics,
    )

    save_json(user_condition_info, os.path.join(OUT_DIR, "user_local_conditions.json"))

    print("\n用户生成配置：")
    print(f"  CONDITION_INPUT_MODE       = {CONDITION_INPUT_MODE}")
    print(f"  GRID_SHAPE [Y,Z,X]         = {GRID_SHAPE}")
    print(f"  patch 数量                 = {len(local_conditions)}")
    print(f"  PATCH_SIZE                 = {PATCH_SIZE}")
    print(f"  OVERLAP                    = {OVERLAP}")
    print(f"  STRIDE                     = {STRIDE}")
    print(f"  输出体积 shape [Y,Z,X]     = {tuple(user_condition_info['assembled_shape_yzx'])}")
    print(f"  用户输入指标               = porosity + tau_z")
    print(f"  surface_area 补全方式      = {AUTO_SURFACE_MODE}")
    print(f"  deff_z 补全方式            = deff_z = porosity / tau_z")
    print(f"  target global condition    = {user_condition_info['target_global_condition']}")
    print(f"  沿 Y 方向 ZX 截面 PNG 输出 = {SAVE_ALL_Y_ZX_SLICE_PNG}")

    device = torch.device(DEVICE if (DEVICE == "cpu" or torch.cuda.is_available()) else "cpu")
    print("\nUsing device:", device)

    ldm_model = load_ldm_model(LDM_CKPT_PATH, device)
    vae_model = load_vae_model(VAE_CKPT_PATH, device)

    prob_patches = []
    patch_results = []

    for item in local_conditions:
        patch_name = item["patch_name"]
        metrics = item["metrics"]

        model_condition = {
            "porosity": float(metrics["porosity"]),
            "surface_area": float(metrics["surface_area"]),
            "tau_z": float(metrics["tau_z"]),
            "deff_z": float(metrics["deff_z"]),
        }

        patch_dir = os.path.join(patch_root, patch_name)

        print("-" * 100)
        print(f"[Generating] {patch_name}")
        print(f"  grid_index = {item['grid_index']}")
        print(f"  raw_user_condition = {item['raw_user_condition']}")
        print(f"  completed model condition = {model_condition}")
        print(f"  surface_area selection = {item['auto_completed_condition_info']['surface_area_selection']}")

        best_patch = generate_best_patch_from_condition(
            model_condition=model_condition,
            summary=summary,
            ldm_model=ldm_model,
            vae_model=vae_model,
            num_samples=NUM_SAMPLES_PER_PATCH,
            save_patch_dir=patch_dir,
        )

        prob_patches.append(best_patch["prob_volume"])

        patch_results.append({
            "patch_id": item["patch_id"],
            "patch_name": patch_name,
            "grid_index": item["grid_index"],
            "start_in_large": item["start_in_large"],
            "raw_user_condition": item["raw_user_condition"],
            "auto_completed_condition_info": item["auto_completed_condition_info"],
            "target_metrics": metrics,
            "generated_metrics": best_patch["final_metrics"],
            "best_score": best_patch["total_weighted_score"],
            "best_threshold_used": best_patch["threshold_used"],
            "best_postprocess_config": best_patch["postprocess_config"],
        })

    print("-" * 100)
    print("Start assembling probability volume ...")

    assembled_prob = assemble_prob_patches(
        prob_patches=prob_patches,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        grid_shape=GRID_SHAPE,
    )

    shape_str = f"{assembled_prob.shape[0]}x{assembled_prob.shape[1]}x{assembled_prob.shape[2]}"

    target_condition_large = user_condition_info["target_global_condition"]

    best_assembly = select_best_assembled_binary(
        assembled_prob=assembled_prob,
        target_condition_large=target_condition_large,
        summary=summary,
    )

    final_volume = best_assembly["bin_final"].astype(np.uint8)

    # --------------------------------------------------------
    # 保存 NPY 结果
    # --------------------------------------------------------

    assembled_prob_path = os.path.join(OUT_DIR, f"assembled_prob_{shape_str}.npy")
    assembled_bin_raw_path = os.path.join(OUT_DIR, f"assembled_bin_raw_{shape_str}.npy")
    assembled_bin_post_path = os.path.join(OUT_DIR, f"assembled_bin_postprocess_{shape_str}.npy")
    assembled_bin_final_path = os.path.join(OUT_DIR, f"assembled_bin_final_{shape_str}.npy")
    twin_final_path = os.path.join(OUT_DIR, "twin_AM_pore_final.npy")

    np.save(assembled_prob_path, assembled_prob.astype(np.float32))
    np.save(assembled_bin_raw_path, best_assembly["bin_raw"].astype(np.uint8))
    np.save(assembled_bin_post_path, best_assembly["bin_post"].astype(np.uint8))
    np.save(assembled_bin_final_path, final_volume)

    # 固定文件名，方便后续 CBD 生成和 2D slab 电化学仿真脚本读取
    np.save(twin_final_path, final_volume)

    # --------------------------------------------------------
    # 只保存沿 Y 方向的每一张 ZX 截面 PNG 图
    # --------------------------------------------------------

    slice_output_paths = {}

    if SAVE_ALL_Y_ZX_SLICE_PNG:
        slice_output_paths.update(
            save_all_y_zx_slice_pngs(
                volume_yzx=final_volume,
                out_root=OUT_DIR,
            )
        )

    # --------------------------------------------------------
    # 保存汇总 JSON
    # --------------------------------------------------------

    summary_out = {
        "mode": "user_defined_porosity_tau_generation",
        "condition_input_mode": CONDITION_INPUT_MODE,
        "summary_json_path": SUMMARY_JSON_PATH,
        "training_metrics_table_path": TRAIN_METRICS_TABLE_PATH,
        "ldm_ckpt_path": LDM_CKPT_PATH,
        "vae_ckpt_path": VAE_CKPT_PATH,
        "grid_shape_yzx": list(GRID_SHAPE),
        "patch_size": PATCH_SIZE,
        "overlap": OVERLAP,
        "stride": STRIDE,
        "assembled_shape_yzx": list(assembled_prob.shape),
        "num_patches": len(prob_patches),
        "num_samples_per_patch": NUM_SAMPLES_PER_PATCH,
        "user_condition_info": user_condition_info,
        "target_global_condition": target_condition_large,
        "patch_results": patch_results,
        "best_assembly_score": best_assembly["total_weighted_score"],
        "best_threshold_base": best_assembly["threshold_base"],
        "best_threshold_offset": best_assembly["threshold_offset"],
        "best_threshold_used": best_assembly["threshold_used"],
        "best_postprocess_config": best_assembly["postprocess_config"],
        "best_final_metrics": best_assembly["final_metrics"],
        "best_final_error_vs_target": best_assembly["final_error_vs_target"],
        "all_assembly_candidate_summaries": best_assembly["all_candidate_summaries"],
        "slice_visualization_settings": {
            "save_all_y_zx_slice_png": SAVE_ALL_Y_ZX_SLICE_PNG,
            "slice_color_style": SLICE_COLOR_STYLE,
            "slice_show_axis": SLICE_SHOW_AXIS,
            "slice_dpi": SLICE_DPI,
            "note": "只保存沿 Y 方向的 ZX 截面 PNG，不保存切片 NPY，不保存其他方向截面。",
        },
        "saved_files": {
            "assembled_prob": assembled_prob_path,
            "assembled_bin_raw": assembled_bin_raw_path,
            "assembled_bin_postprocess": assembled_bin_post_path,
            "assembled_bin_final": assembled_bin_final_path,
            "twin_AM_pore_final": twin_final_path,
            "user_local_conditions": os.path.join(OUT_DIR, "user_local_conditions.json"),
            "generated_patches_dir": patch_root,
            **slice_output_paths,
        }
    }

    save_json(summary_out, os.path.join(OUT_DIR, "assembly_summary.json"))

    print("\nSaved to:", OUT_DIR)
    print("Assembled probability shape:", assembled_prob.shape)
    print("Best final metrics:", best_assembly["final_metrics"])
    print("Best threshold:", best_assembly["threshold_used"])
    print("Best postprocess:", best_assembly["postprocess_config"])

    print("\n最终 AM-pore 数字孪生体：")
    print("  ", twin_final_path)

    if SAVE_ALL_Y_ZX_SLICE_PNG:
        print("\n最终沿 Y 方向全部 ZX 截面 PNG 已输出到：")
        print("  ", slice_output_paths.get("all_y_zx_slice_png_dir", ""))

    print("=" * 100)


if __name__ == "__main__":
    main()
