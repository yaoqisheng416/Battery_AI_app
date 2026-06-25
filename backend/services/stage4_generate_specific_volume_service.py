# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from dataclasses import asdict
import os
import json

import numpy as np
import torch
import config
from backend.electrode_twin.vaenet import VAENetConfig, VAENet

sys.path.append("../electrode_twin")
import csv
import contextlib
import logging
from scipy.ndimage import (
    label,
    binary_erosion,
    binary_dilation,
    binary_opening,
)
import taufactor as tau
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from latent_diffusion import LatentDiffusionConfig, LatentDiffusionModule
from vaemodule import VAEModule, VAEModuleConfig
from typing import List, Dict, Any, Tuple

# ============================================================
# logger
# ============================================================
logger = logging.getLogger("stage4_generate_specific_volume_service.py")

if not logger.handlers:
    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)

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

def get_binary_cmap(cfg: config.GenerateSpecificVolumeConfig,):
    """
    获取 AM-pore 二值结构颜色表。

    标签：
        0 = PORE
        1 = AM / SOLID
    """
    if cfg.slice_color_style == "black_yellow":
        return ListedColormap([
            [1.0, 0.78, 0.0],  # 0 = PORE，黄色
            [0.0, 0.0, 0.0],  # 1 = AM/SOLID，黑色
        ])

    if cfg.slice_color_style == "white_blue":
        return ListedColormap([
            [1.0, 1.0, 1.0],  # 0 = PORE，白色
            [0.1, 0.35, 0.75],  # 1 = AM/SOLID，蓝色
        ])

    raise ValueError('SLICE_COLOR_STYLE 必须是 "black_yellow" 或 "white_blue"。')


def save_all_y_zx_slice_pngs(
        cfg: config.GenerateSpecificVolumeConfig,
        volume_yzx: np.ndarray,
        out_root: str,
        log_fn=None,
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

    cmap = get_binary_cmap(cfg=cfg)

    log_fn("\n开始保存沿 Y 方向的全部 ZX 截面 PNG 图...")
    log_fn(f"  volume shape [Y,Z,X] = {volume_yzx.shape}")
    log_fn(f"  总切片数 Y = {ny}")
    log_fn(f"  每张 ZX 截面 shape [Z,X] = {(nz, nx)}")
    log_fn(f"  输出目录 = {slice_png_dir}")

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

        if cfg.slice_show_axis:
            ax.set_title(f"ZX slice at Y = {y}", fontsize=12)
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
        else:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(png_path, dpi=cfg.slice_dpi, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

        if (y + 1) % 10 == 0 or (y + 1) == ny:
            log_fn(f"  已保存 {y + 1}/{ny} 张 ZX 截面 PNG")

    log_fn("沿 Y 方向全部 ZX 截面 PNG 图保存完成。")

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


def load_training_metrics_table(path: str, log_fn=None) -> List[Dict[str, float]]:
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
        log_fn(f"[提示] 未找到训练集指标表: {path}")
        log_fn("[提示] surface_area 将使用 dataset_summary 中 surface mean。")
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

    log_fn(f"训练集指标表读取完成: {path}")
    log_fn(f"  有效样本数 = {len(clean_rows)}")

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

def remove_small_pore_components(volume: np.ndarray, min_size: int, cfg: config.GenerateSpecificVolumeConfig,) -> np.ndarray:
    if min_size <= 1:
        return volume.copy().astype(np.uint8)

    pore_mask = (volume == cfg.pore_value)
    labeled, num = label(pore_mask)

    if num == 0:
        return volume.copy().astype(np.uint8)

    sizes = np.bincount(labeled.ravel())
    keep_pore = np.zeros_like(pore_mask, dtype=bool)

    for comp_id in range(1, len(sizes)):
        if sizes[comp_id] >= min_size:
            keep_pore[labeled == comp_id] = True

    cleaned = np.where(keep_pore, cfg.pore_value,
                       cfg.solid_value).astype(np.uint8)
    return cleaned


def porosity(volume01: np.ndarray, cfg: config.GenerateSpecificVolumeConfig,) -> float:
    return float((volume01 == cfg.pore_value).mean())


def surface_area(volume01: np.ndarray, cfg: config.GenerateSpecificVolumeConfig,) -> float:
    v = volume01.astype(np.uint8)

    n_y = np.abs(v[1:, :, :] - v[:-1, :, :]).sum()
    n_z = np.abs(v[:, 1:, :] - v[:, :-1, :]).sum()
    n_x = np.abs(v[:, :, 1:] - v[:, :, :-1]).sum()

    area_y = n_y * (cfg.voxel_size_z * cfg.voxel_size_x)
    area_z = n_z * (cfg.voxel_size_y * cfg.voxel_size_x)
    area_x = n_x * (cfg.voxel_size_y * cfg.voxel_size_z)

    total_area = area_y + area_z + area_x

    total_volume = (
            volume01.shape[0] * cfg.voxel_size_y *
            volume01.shape[1] * cfg.voxel_size_z *
            volume01.shape[2] * cfg.voxel_size_x
    )

    if total_volume <= 0:
        return 0.0

    return float(total_area / total_volume)


def largest_connected_ratio(volume01: np.ndarray, cfg: config.GenerateSpecificVolumeConfig,) -> float:
    pore_mask = (volume01 == cfg.pore_value)
    labeled, num = label(pore_mask)

    sizes = np.bincount(labeled.ravel())
    if len(sizes) <= 1:
        return 0.0

    largest = sizes[1:].max()
    total = sizes[1:].sum()

    if total == 0:
        return 0.0

    return float(largest / total)


def solid_component_count(volume01: np.ndarray, cfg: config.GenerateSpecificVolumeConfig,) -> int:
    solid_mask = (volume01 == cfg.solid_value)
    _, num = label(solid_mask)
    return int(num)


def is_percolating_along_z(volume: np.ndarray, cfg: config.GenerateSpecificVolumeConfig,) -> bool:
    pore_mask = (volume == cfg.pore_value)

    labeled, num = label(pore_mask)
    if num == 0:
        return False

    z0_labels = set(np.unique(labeled[:, 0, :]))
    zend_labels = set(np.unique(labeled[:, -1, :]))

    common = z0_labels & zend_labels
    common.discard(0)

    return len(common) > 0


def compute_tau_deff_z(volume: np.ndarray, cfg: config.GenerateSpecificVolumeConfig,) -> tuple[float, float, int]:
    percolating = is_percolating_along_z(volume, cfg=cfg)

    if not percolating:
        return float(cfg.tau_nonperc_value), 0.0, 0

    pore_phase = (volume == cfg.pore_value).astype(np.uint8)  # [Y, Z, X]
    pore_phase_zyx = np.transpose(pore_phase, (1, 0, 2))  # [Z, Y, X]

    try:
        with suppress_stdout_stderr(cfg.suppress_taufactor_output):
            solver = tau.Solver(pore_phase_zyx)
            solver.solve()

        tau_value = safe_scalar(solver.tau, default=cfg.tau_nonperc_value)
        deff_value = safe_scalar(solver.D_eff, default=0.0)

        if not np.isfinite(tau_value):
            tau_value = float(cfg.tau_nonperc_value)

        if not np.isfinite(deff_value):
            deff_value = 0.0

        return float(tau_value), float(deff_value), 1

    except Exception:
        return float(cfg.tau_nonperc_value), 0.0, 0


def compute_generated_metrics_cheap(bin_volume: np.ndarray, cfg: config.GenerateSpecificVolumeConfig,) -> Dict[str, float]:
    return {
        "porosity": porosity(bin_volume, cfg=cfg),
        "surface_area": surface_area(bin_volume, cfg=cfg),
        "largest_connected_ratio": largest_connected_ratio(bin_volume, cfg=cfg),
        "solid_component_count": solid_component_count(bin_volume, cfg=cfg),
    }


def compute_generated_metrics_exact(bin_volume: np.ndarray, cfg: config.GenerateSpecificVolumeConfig,) -> Dict[str, float]:
    cheap = compute_generated_metrics_cheap(bin_volume, cfg=cfg)
    tau_z, deff_z, is_perc = compute_tau_deff_z(bin_volume, cfg=cfg,)

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


def get_manual_condition_lookup(manual_patch_conditions) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    """
    把 MANUAL_PATCH_CONDITIONS 转成 grid_index -> condition 的字典。
    """
    lookup = {}

    for item in manual_patch_conditions:
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
        cfg: config.GenerateSpecificVolumeConfig,
) -> Tuple[List[Dict], Dict]:
    """
    构建 patch 条件表。

    输出结构与原始 local_conditions 逻辑相似，
    但条件来自用户输入而不是从真实体数据裁剪。
    """
    n_patch = product_int(cfg.grid_shape)

    if cfg.condition_input_mode == "manual_user":
        manual_lookup = get_manual_condition_lookup(cfg.manual_patch_conditions)

        if len(manual_lookup) != n_patch:
            raise ValueError(
                f"manual_user 模式下，MANUAL_PATCH_CONDITIONS 数量必须等于 {n_patch}，"
                f"当前为 {len(manual_lookup)}。"
            )
    else:
        manual_lookup = {}

    local_conditions = []

    idx = 0

    for iy, iz, ix in grid_indices(cfg.grid_shape):
        if cfg.condition_input_mode == "uniform_porosity":
            por = float(cfg.target_patch_porosity)
            tau_value = float(cfg.target_patch_tau_z)
            raw_user_condition = {
                "porosity": por,
                "tau_z": tau_value,
            }

        elif cfg.condition_input_mode == "manual_user":
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

        y0 = iy * cfg.stride
        z0 = iz * cfg.stride
        x0 = ix * cfg.stride

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

    out_shape = assembled_shape_from_grid(cfg.grid_shape,
                                          cfg.patch_size,
                                          cfg.stride)

    target_global_condition = {
        "porosity": float(np.mean([p["metrics"]["porosity"] for p in local_conditions])),
        "surface_area": float(np.mean([p["metrics"]["surface_area"] for p in local_conditions])),
        "tau_z": float(np.mean([p["metrics"]["tau_z"] for p in local_conditions])),
        "deff_z": float(np.mean([p["metrics"]["deff_z"] for p in local_conditions])),
    }

    info = {
        "mode": "user_defined_porosity_tau_conditions",
        "condition_input_mode": cfg.condition_input_mode,
        "grid_shape_yzx": list(cfg.grid_shape),
        "patch_size": cfg.patch_size,
        "overlap": cfg.overlap,
        "stride": cfg.stride,
        "assembled_shape_yzx": list(out_shape),
        "num_patches": len(local_conditions),
        "user_input_keys": ["porosity", "tau_z"],
        "model_condition_keys": MODEL_CONDITION_KEYS,
        "surface_area_auto_selection": {
            "mode": cfg.auto_surface_mode,
            "training_metrics_table_path": cfg.train_metrics_table_path,
            "fallback": "dataset_summary surface mean if training metrics table is missing",
        },
        "deff_z_auto_rule": cfg.auto_deff_mode,
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


def normalize_condition(cond_raw: Dict[str, float], summary: Dict, cfg: config.GenerateSpecificVolumeConfig, log_fn=None) -> np.ndarray:
    vals = []

    for key in MODEL_CONDITION_KEYS:
        raw_val = float(cond_raw[key])
        skey = COND_TO_SUMMARY_KEY[key]

        vmin = float(summary[skey]["min"])
        vmax = float(summary[skey]["max"])

        if cfg.warn_if_target_ood and (raw_val < vmin or raw_val > vmax):
            log_fn(f"[警告] 条件 {key} = {raw_val:.6f} 超出训练范围 [{vmin:.6f}, {vmax:.6f}]")

        denom = max(vmax - vmin, 1e-12)
        norm_val = (raw_val - vmin) / denom

        if cfg.clip_normalized_condition_to_train_range:
            norm_val = float(np.clip(norm_val, 0.0, 1.0))

        vals.append(norm_val)

    return np.asarray(vals, dtype=np.float32)


# ============================================================
# 12. 阈值与后处理
# ============================================================

def find_threshold_for_target_porosity(
        cfg: config.GenerateSpecificVolumeConfig,
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
        por = porosity(bin_volume, cfg=cfg)

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


def postprocess_solid_topology(bin_volume: np.ndarray, cfg: Dict, conf: config.GenerateSpecificVolumeConfig,) -> np.ndarray:
    mode = cfg["mode"]

    solid = (bin_volume == conf.solid_value)

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

    return np.where(solid2, conf.solid_value,
                    conf.pore_value).astype(np.uint8)


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
        cfg: config.GenerateSpecificVolumeConfig,
        error_dict: Dict[str, float],
        generated_metrics: Dict[str, float],
        weights: Dict[str, float],
) -> float:
    score = 0.0

    for key, w in weights.items():
        if cfg.use_std_normalized_error:
            e = float(error_dict[f"{key}_std_normalized_error"])
        else:
            e = float(error_dict[f"{key}_rel_error"])

        score += float(w) * e

    solid_num = int(generated_metrics.get("solid_component_count", 0))

    if solid_num < cfg.min_solid_component_count_soft:
        score += cfg.topology_penalty_weight * float(
            cfg.min_solid_component_count_soft - solid_num)

    return float(score)


def generate_best_patch_from_condition(
        cfg: config.GenerateSpecificVolumeConfig,
        model_condition: Dict[str, float],
        summary: Dict,
        ldm_model,
        vae_model,
        num_samples: int,
        save_patch_dir: str | None = None,
        log_fn=None,
) -> Dict:
    """
    根据一个 patch 的 4 维模型条件生成多个候选，并选择最优候选。

    当前筛选重点：
        cheap 阶段：porosity
        exact 阶段：porosity + tau_z + deff_z
    """
    cond_norm_np = normalize_condition(model_condition, summary, cfg=cfg, log_fn=log_fn)

    device = next(ldm_model.parameters()).device
    cond_norm_all = torch.from_numpy(cond_norm_np).unsqueeze(0).repeat(num_samples, 1).to(device)

    best_result = None
    candidate_summaries = []

    for i in range(num_samples):
        candidate_id = f"sample_{i:03d}"
        cond_i = cond_norm_all[i:i + 1]

        prob_volume = decode_one_prob_volume(ldm_model, vae_model, cond_i)

        if cfg.use_adaptive_threshold_for_porosity:
            threshold_base, _ = find_threshold_for_target_porosity(
                cfg=cfg,
                prob_volume=prob_volume,
                target_porosity=float(model_condition["porosity"]),
                max_iters=cfg.adaptive_threshold_max_iters,
                tol=cfg.adaptive_threshold_tol,
            )
        else:
            threshold_base = 0.5

        coarse_pool = []

        for offset in cfg.threshold_offsets:
            t_used = float(np.clip(threshold_base + offset, 0.0, 1.0))
            bin_raw = threshold_prob_volume(prob_volume, t_used)

            for pp_cfg in cfg.postprocess_configs:
                bin_post = postprocess_solid_topology(bin_raw, pp_cfg, conf=cfg)

                if cfg.remove_small_pore_components:
                    bin_final = remove_small_pore_components(bin_post,
                                                             cfg.min_pore_component_size, cfg=cfg)
                else:
                    bin_final = bin_post.copy()

                cheap_metrics = compute_generated_metrics_cheap(bin_final, cfg=cfg)

                cheap_error_dict = make_error_dict(
                    target_condition=model_condition,
                    generated_metrics=cheap_metrics,
                    summary=summary,
                    keys=["porosity"],
                )

                cheap_score = compute_weighted_score(
                    cfg=cfg,
                    error_dict=cheap_error_dict,
                    generated_metrics=cheap_metrics,
                    weights=cfg.cheap_error_weights,
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
        exact_pool = coarse_pool[:cfg.exact_eval_topk_per_candidate]

        candidate_best = None

        for item in exact_pool:
            final_metrics = compute_generated_metrics_exact(item["bin_volume_final"], cfg=cfg,)

            final_error_dict = make_error_dict(
                target_condition=model_condition,
                generated_metrics=final_metrics,
                summary=summary,
                keys=["porosity", "tau_z", "deff_z"],
            )

            total_score = compute_weighted_score(
                cfg=cfg,
                error_dict=final_error_dict,
                generated_metrics=final_metrics,
                weights=cfg.final_error_weights,
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

        log_fn(
            f"    candidate {i + 1}/{num_samples}: "
            f"current best score = {best_result['total_weighted_score']:.6f}"
        )

    if best_result is None:
        raise RuntimeError("没有生成出有效 patch。")

    if save_patch_dir is not None:
        ensure_dir(save_patch_dir)

        np.save(os.path.join(save_patch_dir, "generated_prob.npy"), best_result["prob_volume"].astype(np.float32))
        np.save(os.path.join(save_patch_dir, "generated_bin_raw.npy"), best_result["bin_volume_raw"].astype(np.uint8))
        np.save(os.path.join(save_patch_dir, "generated_bin_postprocess.npy"),
                best_result["bin_volume_postprocess"].astype(np.uint8))
        np.save(os.path.join(save_patch_dir, "generated_bin_final.npy"),
                best_result["bin_volume_final"].astype(np.uint8))

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
        cfg: config.GenerateSpecificVolumeConfig,
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

    if cfg.use_adaptive_threshold_for_porosity:
        threshold_base, _ = find_threshold_for_target_porosity(
            cfg=cfg,
            prob_volume=assembled_prob,
            target_porosity=target_porosity_large,
            max_iters=cfg.adaptive_threshold_max_iters,
            tol=cfg.adaptive_threshold_tol,
        )
    else:
        threshold_base = 0.5

    best = None
    all_candidate_summaries = []

    for offset in cfg.threshold_offsets:
        t_used = float(np.clip(threshold_base + offset, 0.0, 1.0))
        bin_raw = threshold_prob_volume(assembled_prob, t_used)

        for pp_cfg in cfg.postprocess_configs:
            bin_post = postprocess_solid_topology(bin_raw, pp_cfg, conf=cfg)

            if cfg.remove_small_pore_components:
                bin_final = remove_small_pore_components(bin_post,
                                                         cfg.min_pore_component_size, cfg=cfg)
            else:
                bin_final = bin_post.copy()

            final_metrics = compute_generated_metrics_exact(bin_final, cfg=cfg)

            final_error_dict = make_error_dict(
                target_condition=target_condition_large,
                generated_metrics=final_metrics,
                summary=summary,
                keys=["porosity", "tau_z", "deff_z"],
            )

            total_score = compute_weighted_score(
                cfg=cfg,
                error_dict=final_error_dict,
                generated_metrics=final_metrics,
                weights=cfg.final_error_weights,
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
# 主 service
# ============================================================

def generate_specific_volume_service(
        cfg: config.GenerateSpecificVolumeConfig,
        task_id: str,
        external_logger=None,
) -> Dict[str, Any]:
    # =========================================================
    # logger
    # =========================================================

    ensure_dir(cfg.out_dir)

    # =========================================================
    # logs
    # =========================================================

    def log(msg):

        logger.info(msg)

        if external_logger is not None:
            external_logger(msg)

    log("=" * 100)
    log(
        "Generate large AM-pore twin volume "
        "from porosity + tortuosity user conditions"
    )
    log("=" * 100)
    log(f"[LARGE_VOLUME] task_id = {task_id}")

    # =========================================================
    # derived runtime params
    # =========================================================

    stride = cfg.stride

    patch_root = os.path.join(
        cfg.out_dir,
        "generated_patches"
    )

    ensure_dir(patch_root)

    # =========================================================
    # load summary
    # =========================================================

    log("Loading dataset summary...")

    summary = load_json(
        cfg.summary_json_path
    )

    log("Loading training metrics table...")

    training_metrics = load_training_metrics_table(
        cfg.train_metrics_table_path,
        log_fn=log,
    )

    # =========================================================
    # build user local conditions
    # =========================================================

    log("Building user local conditions...")

    local_conditions, user_condition_info = (
        build_user_local_conditions(
            summary=summary,
            training_metrics=training_metrics,
            cfg=cfg,
        )
    )

    user_local_conditions_path = os.path.join(
        cfg.out_dir,
        "user_local_conditions.json"
    )

    save_json(
        user_condition_info,
        user_local_conditions_path
    )

    # =========================================================
    # print config summary
    # =========================================================

    log("用户生成配置:")

    log(
        f"CONDITION_INPUT_MODE = "
        f"{cfg.condition_input_mode}"
    )

    log(
        f"GRID_SHAPE [Y,Z,X]  = "
        f"{cfg.grid_shape}"
    )

    log(
        f"patch 数量   = "
        f"{len(local_conditions)}"
    )

    log(
        f"PATCH_SIZE = "
        f"{cfg.patch_size}"
    )

    log(
        f"OVERLAP = "
        f"{cfg.overlap}"
    )

    log(
        f"STRIDE = "
        f"{stride}"
    )

    log(
        f"输出体积 shape [Y,Z,X = "
        f"{tuple(user_condition_info['assembled_shape_yzx'])}"
    )
    log(f"用户输入指标    = porosity + tau_z")
    log(
        f"surface_area 补全方式  = "
        f"{cfg.auto_surface_mode}"
    )

    log(
        f"deff_z 补全方式  = "
        f"{cfg.auto_deff_mode}"
    )

    log(
        f"target global condition = "
        f"{user_condition_info['target_global_condition']}"
    )

    log(
        f"沿 Y 方向 ZX 截面 PNG 输出 = "
        f"{cfg.save_all_y_zx_slice_png}"
    )

    # =========================================================
    # device
    # =========================================================

    device = torch.device(
        cfg.device
        if (
                cfg.device == "cpu"
                or torch.cuda.is_available()
        )
        else "cpu"
    )

    log(f"Using device: {device}")

    # =========================================================
    # load model
    # =========================================================

    log("Loading LDM model...")

    ldm_model = load_ldm_model(
        cfg.ldm_ckpt_path,
        device,
    )

    log("Loading VAE model...")

    vae_model = load_vae_model(
        cfg.vae_ckpt_path,
        device,
    )

    # =========================================================
    # patch generation
    # =========================================================

    prob_patches = []

    patch_results = []

    total_patch_num = len(local_conditions)

    for patch_idx, item in enumerate(local_conditions):
        patch_name = item["patch_name"]

        metrics = item["metrics"]

        model_condition = {
            "porosity": float(metrics["porosity"]),
            "surface_area": float(metrics["surface_area"]),
            "tau_z": float(metrics["tau_z"]),
            "deff_z": float(metrics["deff_z"]),
        }

        patch_dir = os.path.join(
            patch_root,
            patch_name
        )

        log("-" * 100)

        log(
            f"[Patch {patch_idx + 1}/{total_patch_num}] "
            f"{patch_name}"
        )

        log(
            f"grid_index = "
            f"{item['grid_index']}"
        )

        log(
            f"raw_user_condition = "
            f"{item['raw_user_condition']}"
        )

        log(
            f"completed_model_condition = "
            f"{model_condition}"
        )

        log(
            f"surface_area_selection = "
            f"{item['auto_completed_condition_info']['surface_area_selection']}"
        )

        best_patch = (
            generate_best_patch_from_condition(
                cfg=cfg,
                model_condition=model_condition,
                summary=summary,
                ldm_model=ldm_model,
                vae_model=vae_model,
                num_samples=cfg.num_samples_per_patch,
                save_patch_dir=patch_dir,
                log_fn=log,
            )
        )

        prob_patches.append(
            best_patch["prob_volume"]
        )

        patch_results.append({
            "patch_id":
                item["patch_id"],

            "patch_name":
                patch_name,

            "grid_index":
                item["grid_index"],

            "start_in_large":
                item["start_in_large"],

            "raw_user_condition":
                item["raw_user_condition"],

            "auto_completed_condition_info":
                item["auto_completed_condition_info"],

            "target_metrics":
                metrics,

            "generated_metrics":
                best_patch["final_metrics"],

            "best_score":
                best_patch["total_weighted_score"],

            "best_threshold_used":
                best_patch["threshold_used"],

            "best_postprocess_config":
                best_patch["postprocess_config"],
        })

    # =========================================================
    # assemble
    # =========================================================

    log("-" * 100)

    log(
        "Start assembling probability volume..."
    )

    assembled_prob = assemble_prob_patches(
        prob_patches=prob_patches,
        patch_size=cfg.patch_size,
        stride=stride,
        grid_shape=cfg.grid_shape,
    )

    shape_str = (
        f"{assembled_prob.shape[0]}x"
        f"{assembled_prob.shape[1]}x"
        f"{assembled_prob.shape[2]}"
    )

    target_condition_large = (
        user_condition_info["target_global_condition"]
    )

    best_assembly = (
        select_best_assembled_binary(
            cfg=cfg,
            assembled_prob=assembled_prob,
            target_condition_large=target_condition_large,
            summary=summary,
        )
    )

    final_volume = (
        best_assembly["bin_final"]
        .astype(np.uint8)
    )

    # =========================================================
    # save npy
    # =========================================================

    log("Saving NPY results...")

    assembled_prob_path = os.path.join(
        cfg.out_dir,
        f"assembled_prob_{shape_str}.npy"
    )

    assembled_bin_raw_path = os.path.join(
        cfg.out_dir,
        f"assembled_bin_raw_{shape_str}.npy"
    )

    assembled_bin_post_path = os.path.join(
        cfg.out_dir,
        f"assembled_bin_postprocess_{shape_str}.npy"
    )

    assembled_bin_final_path = os.path.join(
        cfg.out_dir,
        f"assembled_bin_final_{shape_str}.npy"
    )

    twin_final_path = os.path.join(
        cfg.out_dir,
        "twin_AM_pore_final.npy"
    )

    np.save(
        assembled_prob_path,
        assembled_prob.astype(np.float32)
    )

    np.save(
        assembled_bin_raw_path,
        best_assembly["bin_raw"].astype(np.uint8)
    )

    np.save(
        assembled_bin_post_path,
        best_assembly["bin_post"].astype(np.uint8)
    )

    np.save(
        assembled_bin_final_path,
        final_volume
    )

    np.save(
        twin_final_path,
        final_volume
    )

    # =========================================================
    # slice visualization
    # =========================================================

    slice_output_paths = {}

    if cfg.save_all_y_zx_slice_png:
        log(
            "Saving Y-direction ZX slice PNGs..."
        )

        slice_output_paths.update(
            save_all_y_zx_slice_pngs(
                cfg=cfg,
                volume_yzx=final_volume,
                out_root=cfg.out_dir,
                log_fn=log,
            )
        )

    # =========================================================
    # summary json
    # =========================================================

    log("Saving assembly summary...")

    summary_out = {
        "task_id":
            task_id,

        "service_name":
            "generate_specific_volume_service",

        "config":
            asdict(cfg),

        "assembled_shape_yzx":
            list(assembled_prob.shape),

        "num_patches":
            len(prob_patches),

        "user_condition_info":
            user_condition_info,

        "target_global_condition":
            target_condition_large,

        "patch_results":
            patch_results,

        "best_assembly_score":
            best_assembly["total_weighted_score"],

        "best_threshold_base":
            best_assembly["threshold_base"],

        "best_threshold_offset":
            best_assembly["threshold_offset"],

        "best_threshold_used":
            best_assembly["threshold_used"],

        "best_postprocess_config":
            best_assembly["postprocess_config"],

        "best_final_metrics":
            best_assembly["final_metrics"],

        "best_final_error_vs_target":
            best_assembly["final_error_vs_target"],

        "all_assembly_candidate_summaries":
            best_assembly["all_candidate_summaries"],

        "saved_files": {

            "assembled_prob":
                assembled_prob_path,

            "assembled_bin_raw":
                assembled_bin_raw_path,

            "assembled_bin_postprocess":
                assembled_bin_post_path,

            "assembled_bin_final":
                assembled_bin_final_path,

            "twin_AM_pore_final":
                twin_final_path,

            "user_local_conditions":
                user_local_conditions_path,

            "generated_patches_dir":
                patch_root,

            **slice_output_paths,
        }
    }

    assembly_summary_path = os.path.join(
        cfg.out_dir,
        "assembly_summary.json"
    )

    save_json(
        summary_out,
        assembly_summary_path
    )

    # =========================================================
    # final log
    # =========================================================

    log("=" * 100)

    log(
        f"Generation completed successfully."
    )

    log(
        f"Output directory: "
        f"{cfg.out_dir}"
    )

    log(
        f"Assembled probability shape: "
        f"{assembled_prob.shape}"
    )

    log(
        f"Best final metrics: "
        f"{best_assembly['final_metrics']}"
    )

    log(
        f"Best threshold: "
        f"{best_assembly['threshold_used']}"
    )

    log(
        f"Best postprocess: "
        f"{best_assembly['postprocess_config']}"
    )

    log(
        f"最终 AM-pore 数字孪生体: "
        f"{twin_final_path}"
    )

    if cfg.save_all_y_zx_slice_png:
        log(
            f"最终沿 Y 方向全部 ZX 截面 PNG 已输出到: "
            f"{slice_output_paths.get('all_y_zx_slice_png_dir', '')}"
        )

    log("=" * 100)

    # =========================================================
    # return
    # =========================================================

    return {
        "task_id":
            task_id,

        "success": True,

        "message":
            "generate_specific_volume_service success",

        "out_dir":
            cfg.out_dir,

        "assembly_summary_path":
            assembly_summary_path,

        "twin_final_path":
            twin_final_path,

        "assembled_shape":
            list(assembled_prob.shape),

        "best_final_metrics":
            best_assembly["final_metrics"],

        "best_score":
            best_assembly["total_weighted_score"],

        "saved_files":
            summary_out["saved_files"],
    }
