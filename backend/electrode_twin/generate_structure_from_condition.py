#generate_structure_from_condition.py
from __future__ import annotations

import os
import json
import contextlib
from typing import Dict, Tuple, List

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import (
    label,
    distance_transform_edt,
    binary_erosion,
    binary_dilation,
    binary_opening,
)

import taufactor as tau

from backend.electrode_twin.latent_diffusion import LatentDiffusionConfig, LatentDiffusionModule
from backend.electrode_twin.vaenet import VAENet, VAENetConfig
from backend.electrode_twin.vaemodule import VAEModule, VAEModuleConfig


# ============================================================
# 路径与运行配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ELECTRODE_TWIN_DIR = os.path.join(BASE_DIR, "electrode_twin")

LDM_CKPT_PATH = os.path.join(
    ELECTRODE_TWIN_DIR,
    "ldm_checkpoints",
    "ldm-epoch337-valloss0.109533.ckpt"
)
VAE_CKPT_PATH = os.path.join(
    ELECTRODE_TWIN_DIR,
    "checkpoints",
    "vae-epoch074-valloss-1.9715.ckpt"
)
SUMMARY_JSON_PATH = os.path.join(
    ELECTRODE_TWIN_DIR,
    "latent_dataset",
    "dataset_summary.json"
)
OUT_DIR = os.path.join(
    ELECTRODE_TWIN_DIR,
    "generated_results"
)
# LDM_CKPT_PATH = r"./ldm_checkpoints/ldm-epoch337-valloss0.109533.ckpt"
# VAE_CKPT_PATH = r"./checkpoints/vae-epoch074-valloss-1.9715.ckpt"
# SUMMARY_JSON_PATH = r"./latent_dataset/dataset_summary.json"
# OUT_DIR = r"./generated_results"

# DEVICE = "cpu"
# NUM_SAMPLES = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 128 if torch.cuda.is_available() else 32

# 你现在真正要控制的 3 个目标
TARGET_CONDITION = {
    "porosity": 0.2,
    "surface_area": 1150.0,
    "tau_z": 7,
}

# 是否把归一化条件 clip 到训练范围
CLIP_NORMALIZED_CONDITION_TO_TRAIN_RANGE = False
WARN_IF_TARGET_OOD = True

SAVE_ALL_SLICES = True
SAVE_MID_THREE_VIEWS = True
KEEP_ONLY_BEST = True

PORE_VALUE = 0
SOLID_VALUE = 1

# volume 顺序 [Y, Z, X]
VOXEL_SIZE_Y = 0.0315
VOXEL_SIZE_Z = 0.02791
VOXEL_SIZE_X = 0.02791

REMOVE_SMALL_COMPONENTS_FOR_SURFACE = True
MIN_COMPONENT_SIZE = 10

USE_ADAPTIVE_THRESHOLD_FOR_POROSITY = True
ADAPTIVE_THRESHOLD_MAX_ITERS = 25
ADAPTIVE_THRESHOLD_TOL = 1e-4

THRESHOLD_OFFSETS = [-0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]

POSTPROCESS_CONFIGS = [
    {"name": "raw", "mode": "none"},
    {"name": "erode1", "mode": "erode", "iters": 1},
    {"name": "open1", "mode": "open", "iters": 1},
    {"name": "erode1_dilate1", "mode": "erode_dilate", "erode_iters": 1, "dilate_iters": 1},
]

# 第一阶段：便宜指标粗筛
CHEAP_ERROR_WEIGHTS = {
    "porosity": 3.0,
    "surface_area": 2.0,
}

# 第二阶段：精确评分
FINAL_ERROR_WEIGHTS = {
    "porosity": 3.0,
    "surface_area": 2.0,
    "tau_z": 4.0,
    "deff_z": 0.5,
}

USE_STD_NORMALIZED_ERROR = True

TOPOLOGY_PENALTY_WEIGHT = 1.0
MIN_SOLID_COMPONENT_COUNT_SOFT = 10

# 每个 candidate 先保留多少个粗筛组合，再算 tau/deff
EXACT_EVAL_TOPK_PER_CANDIDATE = 3

# TauFactor
TAU_NONPERC_VALUE = 1e6
SUPPRESS_TAUFACTOR_OUTPUT = True


# ============================================================
# 条件键
# ============================================================

PRIMARY_TARGET_KEYS = [
    "porosity",
    "surface_area",
    "tau_z",
]

# 当前 LDM 仍按 4 条件兼容
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
# 基础工具
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_next_run_dir(base_dir: str) -> str:
    ensure_dir(base_dir)

    runs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")
    ]

    if len(runs) == 0:
        run_id = 1
    else:
        ids = []
        for r in runs:
            try:
                ids.append(int(r.split("_")[1]))
            except Exception:
                pass
        run_id = 1 if len(ids) == 0 else (max(ids) + 1)

    run_dir = os.path.join(base_dir, f"run_{run_id:03d}")
    ensure_dir(run_dir)
    return run_dir


def load_summary(summary_json_path: str) -> Dict:
    with open(summary_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def save_single_slice_tif(slice_2d: np.ndarray, path: str):
    img = (slice_2d * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def save_volume_slices(prob_volume: np.ndarray, bin_volume: np.ndarray, out_dir: str):
    prob_dir = os.path.join(out_dir, "prob_slices_tif")
    bin_dir = os.path.join(out_dir, "bin_slices_tif")
    ensure_dir(prob_dir)
    ensure_dir(bin_dir)

    n_slices = prob_volume.shape[0]
    for i in range(n_slices):
        save_single_slice_tif(prob_volume[i], os.path.join(prob_dir, f"slice_{i:04d}.tif"))
        save_single_slice_tif(bin_volume[i].astype(np.float32), os.path.join(bin_dir, f"slice_{i:04d}.tif"))


def save_mid_views(prob_volume: np.ndarray, bin_volume: np.ndarray, out_dir: str):
    y_mid = prob_volume.shape[0] // 2
    z_mid = prob_volume.shape[1] // 2
    x_mid = prob_volume.shape[2] // 2

    save_single_slice_tif(prob_volume[y_mid], os.path.join(out_dir, "prob_mid_y.tif"))
    save_single_slice_tif(prob_volume[:, z_mid, :], os.path.join(out_dir, "prob_mid_z.tif"))
    save_single_slice_tif(prob_volume[:, :, x_mid], os.path.join(out_dir, "prob_mid_x.tif"))

    save_single_slice_tif(bin_volume[y_mid].astype(np.float32), os.path.join(out_dir, "bin_mid_y.tif"))
    save_single_slice_tif(bin_volume[:, z_mid, :].astype(np.float32), os.path.join(out_dir, "bin_mid_z.tif"))
    save_single_slice_tif(bin_volume[:, :, x_mid].astype(np.float32), os.path.join(out_dir, "bin_mid_x.tif"))


def estimate_deff_from_porosity_tau(
    porosity_value: float,
    tau_value: float,
    mode: str = "simple",
) -> float:
    tau_value = max(float(tau_value), 1e-8)
    porosity_value = max(float(porosity_value), 1e-8)

    if mode == "simple":
        deff = porosity_value / tau_value
    else:
        deff = porosity_value / tau_value

    return float(max(deff, 1e-8))


# ============================================================
# TauFactor 工具
# ============================================================

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


# ============================================================
# 条件处理
# ============================================================

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
            print(
                f"[警告] 条件 {key} = {raw_val:.6f} "
                f"超出训练范围 [{vmin:.6f}, {vmax:.6f}]"
            )

        denom = max(vmax - vmin, 1e-12)
        norm_val = (raw_val - vmin) / denom

        if CLIP_NORMALIZED_CONDITION_TO_TRAIN_RANGE:
            norm_val = float(np.clip(norm_val, 0.0, 1.0))

        vals.append(norm_val)

    return np.asarray(vals, dtype=np.float32)


# ============================================================
# 小连通域清理（用于比表面积统计）
# ============================================================

def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1:
        return mask.copy().astype(np.uint8)

    labeled, num = label(mask.astype(bool))
    if num == 0:
        return mask.copy().astype(np.uint8)

    sizes = np.bincount(labeled.ravel())
    keep = np.zeros_like(mask, dtype=np.uint8)

    for comp_id in range(1, len(sizes)):
        if sizes[comp_id] >= min_size:
            keep[labeled == comp_id] = 1

    return keep


# ============================================================
# 指标计算
# ============================================================

def porosity(volume01: np.ndarray) -> float:
    pore_mask = (volume01 == PORE_VALUE)
    return float(pore_mask.mean())


def surface_area(volume01: np.ndarray) -> float:
    v = (volume01 == PORE_VALUE).astype(np.uint8)

    if REMOVE_SMALL_COMPONENTS_FOR_SURFACE:
        v = remove_small_components(v, MIN_COMPONENT_SIZE)

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


def largest_solid_component_voxels(volume01: np.ndarray) -> int:
    solid_mask = (volume01 == SOLID_VALUE)
    labeled, num = label(solid_mask)
    if num == 0:
        return 0
    binc = np.bincount(labeled.ravel())
    if len(binc) <= 1:
        return 0
    return int(binc[1:].max())


def topk_solid_component_voxels(volume01: np.ndarray, k: int = 5) -> List[int]:
    solid_mask = (volume01 == SOLID_VALUE)
    labeled, num = label(solid_mask)
    if num == 0:
        return []
    binc = np.bincount(labeled.ravel())[1:]
    if len(binc) == 0:
        return []
    vals = np.sort(binc)[::-1][:k]
    return [int(v) for v in vals.tolist()]


def compute_generated_metrics_cheap(bin_volume: np.ndarray) -> Dict[str, float]:
    p = porosity(bin_volume)
    s = surface_area(bin_volume)
    c = largest_connected_ratio(bin_volume)
    solid_num = solid_component_count(bin_volume)

    return {
        "porosity": p,
        "surface_area": s,
        "largest_connected_ratio": c,
        "solid_component_count": solid_num,
    }


def compute_generated_metrics_exact(bin_volume: np.ndarray) -> Dict[str, float]:
    cheap = compute_generated_metrics_cheap(bin_volume)

    tau_z, deff_z, is_perc_z = compute_tau_deff_z(bin_volume)

    largest_solid_vox = largest_solid_component_voxels(bin_volume)
    top5 = topk_solid_component_voxels(bin_volume, k=5)

    out = dict(cheap)
    out.update({
        "tau_z": float(tau_z),
        "deff_z": float(deff_z),
        "is_percolating_z": int(is_perc_z),
        "largest_solid_component_voxels": largest_solid_vox,
        "top5_solid_component_voxels": top5,
    })
    return out


# ============================================================
# 阈值与后处理
# ============================================================

def find_threshold_for_target_porosity(
    prob_volume: np.ndarray,
    target_porosity: float,
    max_iters: int = 25,
    tol: float = 1e-4,
) -> Tuple[float, float]:
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

    out = np.where(solid2, SOLID_VALUE, PORE_VALUE).astype(np.uint8)
    return out


# ============================================================
# 模型加载
# ============================================================

def load_ldm_model(ckpt_path: str, device: torch.device) -> LatentDiffusionModule:
    config = LatentDiffusionConfig(
        latent_channels=4,
        latent_size=16,
        cond_dim=4,   # 这里按你当前真正训练的 4 条件模型来
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
# 生成
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


# ============================================================
# 评分
# ============================================================

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


# ============================================================
# 保存最优结果
# ============================================================

def save_best_outputs(
    best_result: Dict,
    run_dir: str,
):
    best_dir = os.path.join(run_dir, "best_sample")
    ensure_dir(best_dir)

    prob_volume = best_result["prob_volume"]
    bin_volume_raw = best_result["bin_volume_raw"]
    bin_volume_final = best_result["bin_volume_final"]

    np.save(os.path.join(best_dir, "generated_prob.npy"), prob_volume.astype(np.float32))
    np.save(os.path.join(best_dir, "generated_bin_raw.npy"), bin_volume_raw.astype(np.uint8))
    np.save(os.path.join(best_dir, "generated_bin_final.npy"), bin_volume_final.astype(np.uint8))

    if SAVE_ALL_SLICES:
        save_volume_slices(prob_volume, bin_volume_final, best_dir)

    if SAVE_MID_THREE_VIEWS:
        save_mid_views(prob_volume, bin_volume_final, best_dir)

    sample_info = {
        "sample_id": "best_sample",
        "original_candidate_id": best_result["candidate_id"],
        "primary_target_condition": best_result["primary_target_condition"],
        "model_condition": best_result["model_condition"],
        "target_condition_normalized": best_result["target_condition_normalized"],
        "threshold_base": best_result["threshold_base"],
        "threshold_offset": best_result["threshold_offset"],
        "threshold_used": best_result["threshold_used"],
        "postprocess_config": best_result["postprocess_config"],
        "cheap_metrics": best_result["cheap_metrics"],
        "final_metrics": best_result["final_metrics"],
        "cheap_error_vs_target": best_result["cheap_error_vs_target"],
        "final_error_vs_target": best_result["final_error_vs_target"],
        "cheap_score": best_result["cheap_score"],
        "total_weighted_score": best_result["total_weighted_score"],
        "shape": list(bin_volume_final.shape),
        "saved_files": {
            "generated_prob_npy": os.path.join(best_dir, "generated_prob.npy"),
            "generated_bin_raw_npy": os.path.join(best_dir, "generated_bin_raw.npy"),
            "generated_bin_final_npy": os.path.join(best_dir, "generated_bin_final.npy"),
        },
        "units": {
            "porosity": "dimensionless",
            "surface_area": "1/um",
            "tau_z": "dimensionless",
            "deff_z": "relative",
            "largest_connected_ratio": "dimensionless",
        }
    }
    save_json(sample_info, os.path.join(best_dir, "sample_info.json"))


# ============================================================
# 主程序
# ============================================================

# def main():
#     print("=" * 90)
#     print("开始执行按需生成闭环：generate -> coarse filter -> exact evaluate -> select")
#     print("LDM_CKPT_PATH :", LDM_CKPT_PATH)
#     print("VAE_CKPT_PATH :", VAE_CKPT_PATH)
#     print("SUMMARY_JSON_PATH :", SUMMARY_JSON_PATH)
#     print("OUT_DIR :", OUT_DIR)
#     print("DEVICE :", DEVICE)
#     print("NUM_SAMPLES :", NUM_SAMPLES)
#     print("PRIMARY TARGET_CONDITION :", TARGET_CONDITION)
#     print("CLIP_NORMALIZED_CONDITION_TO_TRAIN_RANGE :", CLIP_NORMALIZED_CONDITION_TO_TRAIN_RANGE)
#     print("=" * 90)
#
#     ensure_dir(OUT_DIR)
#     run_dir = get_next_run_dir(OUT_DIR)
#
#     device = torch.device(DEVICE if (DEVICE == "cpu" or torch.cuda.is_available()) else "cpu")
#     print("实际使用设备:", device)
#     print("本次输出目录:", run_dir)
#
#     summary = load_summary(SUMMARY_JSON_PATH)
#
#     # 由 3 主控自动补齐第 4 个条件
#     derived_deff = estimate_deff_from_porosity_tau(
#         porosity_value=float(TARGET_CONDITION["porosity"]),
#         tau_value=float(TARGET_CONDITION["tau_z"]),
#         mode="simple",
#     )
#
#     model_condition = {
#         "porosity": float(TARGET_CONDITION["porosity"]),
#         "surface_area": float(TARGET_CONDITION["surface_area"]),
#         "tau_z": float(TARGET_CONDITION["tau_z"]),
#         "deff_z": float(derived_deff),
#     }
#
#     range_report = validate_target_condition(model_condition, summary)
#
#     if WARN_IF_TARGET_OOD:
#         for item in range_report:
#             if item["status"] != "in_range":
#                 print(
#                     f"[警告] {item['condition_key']} = {item['input_value']:.6f} "
#                     f"超出训练范围 [{item['min']:.6f}, {item['max']:.6f}]"
#                 )
#
#     cond_norm_np = normalize_condition(model_condition, summary)
#     cond_norm_all = torch.from_numpy(cond_norm_np).unsqueeze(0).repeat(NUM_SAMPLES, 1).to(device)
#
#     ldm_model = load_ldm_model(LDM_CKPT_PATH, device)
#     vae_model = load_vae_model(VAE_CKPT_PATH, device)
#
#     run_info = {
#         "primary_target_condition": TARGET_CONDITION,
#         "model_condition": model_condition,
#         "target_condition_normalized": cond_norm_np.tolist(),
#         "target_range_report": range_report,
#         "num_samples": NUM_SAMPLES,
#         "ldm_checkpoint": LDM_CKPT_PATH,
#         "vae_checkpoint": VAE_CKPT_PATH,
#         "summary_json_path": SUMMARY_JSON_PATH,
#         "threshold_offsets": THRESHOLD_OFFSETS,
#         "postprocess_configs": POSTPROCESS_CONFIGS,
#         "clip_normalized_condition_to_train_range": CLIP_NORMALIZED_CONDITION_TO_TRAIN_RANGE,
#         "use_adaptive_threshold_for_porosity": USE_ADAPTIVE_THRESHOLD_FOR_POROSITY,
#         "adaptive_threshold_max_iters": ADAPTIVE_THRESHOLD_MAX_ITERS,
#         "adaptive_threshold_tol": ADAPTIVE_THRESHOLD_TOL,
#         "cheap_error_weights": CHEAP_ERROR_WEIGHTS,
#         "final_error_weights": FINAL_ERROR_WEIGHTS,
#         "exact_eval_topk_per_candidate": EXACT_EVAL_TOPK_PER_CANDIDATE,
#         "topology_penalty_weight": TOPOLOGY_PENALTY_WEIGHT,
#         "min_solid_component_count_soft": MIN_SOLID_COMPONENT_COUNT_SOFT,
#         "use_std_normalized_error": USE_STD_NORMALIZED_ERROR,
#         "voxel_size_um": {
#             "Y": VOXEL_SIZE_Y,
#             "Z": VOXEL_SIZE_Z,
#             "X": VOXEL_SIZE_X,
#         },
#         "note": (
#             "本次运行使用 3 个主目标（porosity/surface_area/tau_z），"
#             "内部自动构造第 4 个模型条件 deff_z=porosity/tau_z，"
#             "先粗筛再对少量候选做 TauFactor 精确评估。"
#         ),
#     }
#     save_json(run_info, os.path.join(run_dir, "run_info.json"))
#
#     all_results = []
#     best_result = None
#
#     print("开始生成与搜索 ...")
#
#     for i in range(NUM_SAMPLES):
#         candidate_id = f"sample_{i:03d}"
#         cond_i = cond_norm_all[i:i + 1]
#
#         prob_volume = decode_one_prob_volume(
#             ldm_model=ldm_model,
#             vae_model=vae_model,
#             cond_norm=cond_i,
#         )
#
#         if USE_ADAPTIVE_THRESHOLD_FOR_POROSITY:
#             threshold_base, _ = find_threshold_for_target_porosity(
#                 prob_volume=prob_volume,
#                 target_porosity=float(TARGET_CONDITION["porosity"]),
#                 max_iters=ADAPTIVE_THRESHOLD_MAX_ITERS,
#                 tol=ADAPTIVE_THRESHOLD_TOL,
#             )
#         else:
#             threshold_base = 0.5
#
#         coarse_pool = []
#
#         for offset in THRESHOLD_OFFSETS:
#             t_used = float(np.clip(threshold_base + offset, 0.0, 1.0))
#             bin_raw = threshold_prob_volume(prob_volume, t_used)
#
#             for pp_cfg in POSTPROCESS_CONFIGS:
#                 bin_final = postprocess_solid_topology(bin_raw, pp_cfg)
#                 cheap_metrics = compute_generated_metrics_cheap(bin_final)
#
#                 cheap_error_dict = make_error_dict(
#                     target_condition=model_condition,
#                     generated_metrics=cheap_metrics,
#                     summary=summary,
#                     keys=["porosity", "surface_area"],
#                 )
#
#                 cheap_score = compute_weighted_score(
#                     error_dict=cheap_error_dict,
#                     generated_metrics=cheap_metrics,
#                     weights=CHEAP_ERROR_WEIGHTS,
#                 )
#
#                 coarse_pool.append({
#                     "candidate_id": candidate_id,
#                     "primary_target_condition": TARGET_CONDITION,
#                     "model_condition": model_condition,
#                     "target_condition_normalized": cond_i[0].detach().cpu().numpy().tolist(),
#                     "prob_volume": prob_volume,
#                     "bin_volume_raw": bin_raw,
#                     "bin_volume_final": bin_final,
#                     "threshold_base": float(threshold_base),
#                     "threshold_offset": float(offset),
#                     "threshold_used": float(t_used),
#                     "postprocess_config": pp_cfg,
#                     "cheap_metrics": cheap_metrics,
#                     "cheap_error_vs_target": cheap_error_dict,
#                     "cheap_score": float(cheap_score),
#                 })
#
#         coarse_pool = sorted(coarse_pool, key=lambda x: x["cheap_score"])
#         exact_pool = coarse_pool[:EXACT_EVAL_TOPK_PER_CANDIDATE]
#
#         candidate_best = None
#
#         for item in exact_pool:
#             final_metrics = compute_generated_metrics_exact(item["bin_volume_final"])
#
#             final_error_dict = make_error_dict(
#                 target_condition=model_condition,
#                 generated_metrics=final_metrics,
#                 summary=summary,
#                 keys=["porosity", "surface_area", "tau_z", "deff_z"],
#             )
#
#             total_score = compute_weighted_score(
#                 error_dict=final_error_dict,
#                 generated_metrics=final_metrics,
#                 weights=FINAL_ERROR_WEIGHTS,
#             )
#
#             result = dict(item)
#             result.update({
#                 "final_metrics": final_metrics,
#                 "final_error_vs_target": final_error_dict,
#                 "total_weighted_score": float(total_score),
#             })
#
#             if (candidate_best is None) or (result["total_weighted_score"] < candidate_best["total_weighted_score"]):
#                 candidate_best = result
#
#             if (best_result is None) or (result["total_weighted_score"] < best_result["total_weighted_score"]):
#                 best_result = result
#
#         if candidate_best is None:
#             continue
#
#         all_results.append({
#             "candidate_id": candidate_id,
#             "best_score_within_candidate": candidate_best["total_weighted_score"],
#             "best_threshold_used": candidate_best["threshold_used"],
#             "best_postprocess_config": candidate_best["postprocess_config"],
#             "best_final_metrics": candidate_best["final_metrics"],
#         })
#
#         fm = candidate_best["final_metrics"]
#         print(
#             f"[候选完成] {candidate_id} | "
#             f"score = {candidate_best['total_weighted_score']:.6f}, "
#             f"thr = {candidate_best['threshold_used']:.6f}, "
#             f"pp = {candidate_best['postprocess_config']['name']}, "
#             f"por = {fm['porosity']:.6f}, "
#             f"surf = {fm['surface_area']:.3f}, "
#             f"tau = {fm['tau_z']:.6f}, "
#             f"deff = {fm['deff_z']:.6f}, "
#             f"solid_num = {fm['solid_component_count']}"
#         )
#
#     if best_result is None:
#         raise RuntimeError("没有生成出任何有效结果。")
#
#     save_best_outputs(best_result, run_dir)
#
#     generation_summary = {
#         "run_dir": run_dir,
#         "num_samples": NUM_SAMPLES,
#         "primary_target_condition": TARGET_CONDITION,
#         "model_condition": model_condition,
#         "target_condition_normalized": cond_norm_np.tolist(),
#         "best_candidate_id": best_result["candidate_id"],
#         "best_score": best_result["total_weighted_score"],
#         "best_threshold_base": best_result["threshold_base"],
#         "best_threshold_offset": best_result["threshold_offset"],
#         "best_threshold_used": best_result["threshold_used"],
#         "best_postprocess_config": best_result["postprocess_config"],
#         "best_cheap_metrics": best_result["cheap_metrics"],
#         "best_final_metrics": best_result["final_metrics"],
#         "best_cheap_error_vs_target": best_result["cheap_error_vs_target"],
#         "best_final_error_vs_target": best_result["final_error_vs_target"],
#         "candidate_summaries": all_results,
#     }
#     save_json(generation_summary, os.path.join(run_dir, "generation_summary.json"))
#
#     print("=" * 90)
#     print("运行完成")
#     print("最佳候选:", best_result["candidate_id"])
#     print("最佳阈值:", f"{best_result['threshold_used']:.6f}")
#     print("最佳后处理:", best_result["postprocess_config"]["name"])
#     print("最佳分数:", f"{best_result['total_weighted_score']:.6f}")
#     print("结果目录:", os.path.join(run_dir, "best_sample"))
#     print("=" * 90)


# if __name__ == "__main__":
#     main()
