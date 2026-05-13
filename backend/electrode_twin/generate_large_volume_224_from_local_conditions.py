#generate_large_volume_224_from_local_conditions.py:
#从上一个build_large_volume_conditions_from_real代码产生的8个patch进行拼接条件生成
from __future__ import annotations

import sys
sys.path.append("../electrode_twin")

import os
import json
import contextlib
from typing import Dict, Tuple, List

import numpy as np
import torch
from scipy.ndimage import label, binary_erosion, binary_dilation, binary_opening
import taufactor as tau

from backend.electrode_twin.latent_diffusion import LatentDiffusionConfig, LatentDiffusionModule
from backend.electrode_twin.vaenet import VAENet, VAENetConfig
from backend.electrode_twin.vaemodule import VAEModule, VAEModuleConfig


# ============================================================
# 0) 路径与运行配置
# ============================================================

LOCAL_CONDITIONS_JSON = r"./large_volume_224_real_reference/local_conditions_224.json"
SUMMARY_JSON_PATH = r"../electrode_twin/latent_dataset/dataset_summary.json"

LDM_CKPT_PATH = r"../electrode_twin/ldm_checkpoints/ldm-epoch337-valloss0.109533.ckpt"
VAE_CKPT_PATH = r"../electrode_twin/checkpoints/vae-epoch074-valloss-1.9715.ckpt"

OUT_DIR = r"./paper_verify/large_volume_224_generated"

DEVICE = "cuda"

PATCH_SIZE = 128
OVERLAP = 32
STRIDE = PATCH_SIZE - OVERLAP  # 96
GRID_SHAPE = (2, 2, 2)

NUM_SAMPLES_PER_PATCH = 64

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

CHEAP_ERROR_WEIGHTS = {
    "porosity": 3.0,
    "surface_area": 2.0,
}

FINAL_ERROR_WEIGHTS = {
    "porosity": 3.0,
    "surface_area": 2.0,
    "tau_z": 4.0,
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
# 1) 条件键
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
# 2) 基础工具
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def load_json(path: str) -> Dict:
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


# ============================================================
# 3) 指标函数
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

    pore_phase = (volume == PORE_VALUE).astype(np.uint8)
    pore_phase_zyx = np.transpose(pore_phase, (1, 0, 2))

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
# 4) 条件处理
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
            print(f"[警告] 条件 {key} = {raw_val:.6f} 超出训练范围 [{vmin:.6f}, {vmax:.6f}]")

        denom = max(vmax - vmin, 1e-12)
        norm_val = (raw_val - vmin) / denom

        if CLIP_NORMALIZED_CONDITION_TO_TRAIN_RANGE:
            norm_val = float(np.clip(norm_val, 0.0, 1.0))

        vals.append(norm_val)

    return np.asarray(vals, dtype=np.float32)


# ============================================================
# 5) 阈值与后处理
# ============================================================

def find_threshold_for_target_porosity(prob_volume: np.ndarray, target_porosity: float, max_iters: int = 25, tol: float = 1e-4) -> Tuple[float, float]:
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
# 6) 模型加载
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
# 7) 单 patch 生成
# ============================================================

@torch.no_grad()
def decode_one_prob_volume(ldm_model: LatentDiffusionModule, vae_model: VAEModule, cond_norm: torch.Tensor) -> np.ndarray:
    latent_shape = (4, 16, 16, 16)
    z = ldm_model.sample(cond=cond_norm, shape=latent_shape)
    x_prob = vae_model.decode(z, apply_postdecode=True)
    x_prob = x_prob[:, 0].clamp(0.0, 1.0)
    return x_prob[0].detach().cpu().numpy().astype(np.float32)


def make_error_dict(target_condition: Dict[str, float], generated_metrics: Dict[str, float], summary: Dict, keys: List[str]) -> Dict[str, float]:
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


def compute_weighted_score(error_dict: Dict[str, float], generated_metrics: Dict[str, float], weights: Dict[str, float]) -> float:
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


def generate_best_patch_from_condition(model_condition: Dict[str, float], summary: Dict, ldm_model, vae_model, num_samples: int, save_patch_dir: str | None = None) -> Dict:
    cond_norm_np = normalize_condition(model_condition, summary)
    cond_norm_all = torch.from_numpy(cond_norm_np).unsqueeze(0).repeat(num_samples, 1).to(next(ldm_model.parameters()).device)

    best_result = None
    candidate_summaries = []

    for i in range(num_samples):
        candidate_id = f"sample_{i:03d}"
        cond_i = cond_norm_all[i:i+1]

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
                    keys=["porosity", "surface_area"],
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
                keys=["porosity", "surface_area", "tau_z", "deff_z"],
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
# 8) 组装
# ============================================================

def make_3d_blending_window(size: int) -> np.ndarray:
    wy = np.hanning(size)
    wz = np.hanning(size)
    wx = np.hanning(size)
    w3d = wy[:, None, None] * wz[None, :, None] * wx[None, None, :]
    w3d = 0.05 + 0.95 * w3d
    return w3d.astype(np.float32)


def assemble_prob_patches(prob_patches: List[np.ndarray], patch_size: int, stride: int) -> np.ndarray:
    out_size = patch_size + stride
    prob_accum = np.zeros((out_size, out_size, out_size), dtype=np.float32)
    weight_accum = np.zeros((out_size, out_size, out_size), dtype=np.float32)

    window = make_3d_blending_window(patch_size)

    starts = [0, stride]
    idx = 0
    for y0 in starts:
        for z0 in starts:
            for x0 in starts:
                y1, z1, x1 = y0 + patch_size, z0 + patch_size, x0 + patch_size
                prob_accum[y0:y1, z0:z1, x0:x1] += prob_patches[idx] * window
                weight_accum[y0:y1, z0:z1, x0:x1] += window
                idx += 1

    assembled = prob_accum / np.maximum(weight_accum, 1e-8)
    return assembled.astype(np.float32)


def main():
    print("=" * 100)
    print("Generate 224^3 large volume from 8 local real conditions")
    print("=" * 100)

    ensure_dir(OUT_DIR)
    patch_root = os.path.join(OUT_DIR, "generated_patches")
    ensure_dir(patch_root)

    local_info = load_json(LOCAL_CONDITIONS_JSON)
    summary = load_json(SUMMARY_JSON_PATH)

    device = torch.device(DEVICE if (DEVICE == "cpu" or torch.cuda.is_available()) else "cpu")
    print("Using device:", device)

    ldm_model = load_ldm_model(LDM_CKPT_PATH, device)
    vae_model = load_vae_model(VAE_CKPT_PATH, device)

    local_conditions = local_info["local_conditions"]
    assert len(local_conditions) == 8, "local_conditions 数量不是 8"

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
        print(f"[Generating] {patch_name} ...")

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
            "patch_name": patch_name,
            "grid_index": item["grid_index"],
            "target_metrics": metrics,
            "generated_metrics": best_patch["final_metrics"],
            "best_score": best_patch["total_weighted_score"],
            "best_threshold_used": best_patch["threshold_used"],
            "best_postprocess_config": best_patch["postprocess_config"],
        })

    print("-" * 100)
    print("Start assembling 224^3 probability volume ...")

    assembled_prob = assemble_prob_patches(
        prob_patches=prob_patches,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
    )

    # 用真实 224³ 的全局 porosity 作为 assembled volume 的全局 threshold 目标
    target_large = local_info["large_volume_metrics_clean"]
    target_porosity_large = float(target_large["porosity"])

    if USE_ADAPTIVE_THRESHOLD_FOR_POROSITY:
        threshold_base, _ = find_threshold_for_target_porosity(
            prob_volume=assembled_prob,
            target_porosity=target_porosity_large,
            max_iters=ADAPTIVE_THRESHOLD_MAX_ITERS,
            tol=ADAPTIVE_THRESHOLD_TOL,
        )
    else:
        threshold_base = 0.5

    assembly_candidates = []

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

            target_condition_large = {
                "porosity": float(target_large["porosity"]),
                "surface_area": float(target_large["surface_area"]),
                "tau_z": float(target_large["tau_z"]),
                "deff_z": float(target_large["deff_z"]),
            }

            final_error_dict = make_error_dict(
                target_condition=target_condition_large,
                generated_metrics=final_metrics,
                summary=summary,
                keys=["porosity", "surface_area", "tau_z", "deff_z"],
            )

            total_score = compute_weighted_score(
                error_dict=final_error_dict,
                generated_metrics=final_metrics,
                weights=FINAL_ERROR_WEIGHTS,
            )

            assembly_candidates.append({
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
            })

    assembly_candidates = sorted(assembly_candidates, key=lambda x: x["total_weighted_score"])
    best_assembly = assembly_candidates[0]

    np.save(os.path.join(OUT_DIR, "assembled_prob_224.npy"), assembled_prob.astype(np.float32))
    np.save(os.path.join(OUT_DIR, "assembled_bin_raw_224.npy"), best_assembly["bin_raw"].astype(np.uint8))
    np.save(os.path.join(OUT_DIR, "assembled_bin_postprocess_224.npy"), best_assembly["bin_post"].astype(np.uint8))
    np.save(os.path.join(OUT_DIR, "assembled_bin_final_224.npy"), best_assembly["bin_final"].astype(np.uint8))

    summary_out = {
        "local_conditions_json": LOCAL_CONDITIONS_JSON,
        "patch_size": PATCH_SIZE,
        "overlap": OVERLAP,
        "stride": STRIDE,
        "assembled_shape": list(assembled_prob.shape),
        "num_patches": len(prob_patches),
        "num_samples_per_patch": NUM_SAMPLES_PER_PATCH,
        "target_large_volume_metrics_clean": target_large,
        "patch_results": patch_results,
        "best_assembly_score": best_assembly["total_weighted_score"],
        "best_threshold_base": best_assembly["threshold_base"],
        "best_threshold_offset": best_assembly["threshold_offset"],
        "best_threshold_used": best_assembly["threshold_used"],
        "best_postprocess_config": best_assembly["postprocess_config"],
        "best_final_metrics": best_assembly["final_metrics"],
        "best_final_error_vs_target": best_assembly["final_error_vs_target"],
    }

    save_json(summary_out, os.path.join(OUT_DIR, "assembly_summary_224.json"))

    print("Saved to:", OUT_DIR)
    print("Assembled shape:", assembled_prob.shape)
    print("Best final metrics:", best_assembly["final_metrics"])
    print("=" * 100)


if __name__ == "__main__":
    main()