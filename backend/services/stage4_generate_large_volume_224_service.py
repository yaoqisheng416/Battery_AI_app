# -*- coding: utf-8 -*-
import os
import logging
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
# logger
# ============================================================
logger = logging.getLogger("stage4_build_real_large_volume_service")

if not logger.handlers:
    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)


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

def remove_small_pore_components(volume: np.ndarray, min_size: int, pore_value, solid_value) -> np.ndarray:
    if min_size <= 1:
        return volume.copy().astype(np.uint8)

    pore_mask = (volume == pore_value)
    labeled, num = label(pore_mask)

    if num == 0:
        return volume.copy().astype(np.uint8)

    sizes = np.bincount(labeled.ravel())
    keep_pore = np.zeros_like(pore_mask, dtype=bool)

    for comp_id in range(1, len(sizes)):
        if sizes[comp_id] >= min_size:
            keep_pore[labeled == comp_id] = True

    cleaned = np.where(keep_pore, pore_value, solid_value).astype(np.uint8)
    return cleaned


def porosity(volume01: np.ndarray, pore_value) -> float:
    return float((volume01 == pore_value).mean())


def surface_area(volume01: np.ndarray, voxel_size_x, voxel_size_y, voxel_size_z) -> float:
    v = volume01.astype(np.uint8)

    n_y = np.abs(v[1:, :, :] - v[:-1, :, :]).sum()
    n_z = np.abs(v[:, 1:, :] - v[:, :-1, :]).sum()
    n_x = np.abs(v[:, :, 1:] - v[:, :, :-1]).sum()

    area_y = n_y * (voxel_size_z * voxel_size_x)
    area_z = n_z * (voxel_size_y * voxel_size_x)
    area_x = n_x * (voxel_size_y * voxel_size_z)

    total_area = area_y + area_z + area_x

    total_volume = (
        volume01.shape[0] * voxel_size_y *
        volume01.shape[1] * voxel_size_z *
        volume01.shape[2] * voxel_size_x
    )

    if total_volume <= 0:
        return 0.0

    return float(total_area / total_volume)


def largest_connected_ratio(volume01: np.ndarray, pore_value) -> float:
    pore_mask = (volume01 == pore_value)
    labeled, num = label(pore_mask)

    sizes = np.bincount(labeled.ravel())
    if len(sizes) <= 1:
        return 0.0

    largest = sizes[1:].max()
    total = sizes[1:].sum()

    if total == 0:
        return 0.0

    return float(largest / total)


def solid_component_count(volume01: np.ndarray, solid_value) -> int:
    solid_mask = (volume01 == solid_value)
    _, num = label(solid_mask)
    return int(num)


def is_percolating_along_z(volume: np.ndarray, pore_value) -> bool:
    pore_mask = (volume == pore_value)

    labeled, num = label(pore_mask)
    if num == 0:
        return False

    z0_labels = set(np.unique(labeled[:, 0, :]))
    zend_labels = set(np.unique(labeled[:, -1, :]))
    common = z0_labels & zend_labels
    common.discard(0)

    return len(common) > 0


def compute_tau_deff_z(volume: np.ndarray, tau_nonperc_value, pore_value, suppress_taufactor_output) -> tuple[float, float, int]:
    percolating = is_percolating_along_z(volume, pore_value)

    if not percolating:
        return float(tau_nonperc_value), 0.0, 0

    pore_phase = (volume == pore_value).astype(np.uint8)
    pore_phase_zyx = np.transpose(pore_phase, (1, 0, 2))

    try:
        with suppress_stdout_stderr(suppress_taufactor_output):
            solver = tau.Solver(pore_phase_zyx)
            solver.solve()

        tau_value = safe_scalar(solver.tau, default=tau_nonperc_value)
        deff_value = safe_scalar(solver.D_eff, default=0.0)

        if not np.isfinite(tau_value):
            tau_value = float(tau_nonperc_value)
        if not np.isfinite(deff_value):
            deff_value = 0.0

        return float(tau_value), float(deff_value), 1

    except Exception:
        return float(tau_nonperc_value), 0.0, 0


def compute_generated_metrics_cheap(bin_volume: np.ndarray, voxel_size_x, voxel_size_y, voxel_size_z, pore_value, solid_value) -> Dict[str, float]:
    return {
        "porosity": porosity(bin_volume, pore_value),
        "surface_area": surface_area(bin_volume, voxel_size_x, voxel_size_y, voxel_size_z),
        "largest_connected_ratio": largest_connected_ratio(bin_volume, pore_value),
        "solid_component_count": solid_component_count(bin_volume, solid_value),
    }


def compute_generated_metrics_exact(bin_volume: np.ndarray, tau_nonperc_value, pore_value, min_solid_component_count_soft, voxel_size_x, voxel_size_y, solid_value, voxel_size_z, suppress_taufactor_output) -> Dict[str, float]:
    cheap = compute_generated_metrics_cheap(bin_volume, voxel_size_x, voxel_size_y, voxel_size_z, pore_value, solid_value)
    tau_z, deff_z, is_perc = compute_tau_deff_z(bin_volume, tau_nonperc_value, pore_value, suppress_taufactor_output)

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


def normalize_condition(cond_raw: Dict[str, float], summary: Dict, warn_if_target_ood, clip_normalized_condition_to_train_range) -> np.ndarray:
    vals = []

    for key in MODEL_CONDITION_KEYS:
        raw_val = float(cond_raw[key])
        skey = COND_TO_SUMMARY_KEY[key]

        vmin = float(summary[skey]["min"])
        vmax = float(summary[skey]["max"])

        if warn_if_target_ood and (raw_val < vmin or raw_val > vmax):
            print(f"[警告] 条件 {key} = {raw_val:.6f} 超出训练范围 [{vmin:.6f}, {vmax:.6f}]")

        denom = max(vmax - vmin, 1e-12)
        norm_val = (raw_val - vmin) / denom

        if clip_normalized_condition_to_train_range:
            norm_val = float(np.clip(norm_val, 0.0, 1.0))

        vals.append(norm_val)

    return np.asarray(vals, dtype=np.float32)


# ============================================================
# 5) 阈值与后处理
# ============================================================

def find_threshold_for_target_porosity(pore_value, prob_volume: np.ndarray, target_porosity: float, max_iters: int = 25, tol: float = 1e-4) -> Tuple[float, float]:
    lo, hi = 0.0, 1.0
    best_t = 0.5
    best_por = None
    best_err = 1e18

    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        bin_volume = (prob_volume >= mid).astype(np.uint8)
        por = porosity(bin_volume, pore_value)
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


def postprocess_solid_topology(bin_volume: np.ndarray, cfg: Dict, pore_value, solid_value) -> np.ndarray:
    mode = cfg["mode"]
    solid = (bin_volume == solid_value)

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

    return np.where(solid2, solid_value, pore_value).astype(np.uint8)


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


def compute_weighted_score(error_dict: Dict[str, float], generated_metrics: Dict[str, float], weights: Dict[str, float], use_std_normalized_error,
                           min_solid_component_count_soft, topology_penalty_weight) -> float:
    score = 0.0

    for key, w in weights.items():
        if use_std_normalized_error:
            e = float(error_dict[f"{key}_std_normalized_error"])
        else:
            e = float(error_dict[f"{key}_rel_error"])
        score += float(w) * e

    solid_num = int(generated_metrics.get("solid_component_count", 0))
    if solid_num < min_solid_component_count_soft:
        score += topology_penalty_weight * float(min_solid_component_count_soft - solid_num)

    return float(score)


def generate_best_patch_from_condition(pore_value, solid_value, warn_if_target_ood, clip_normalized_condition_to_train_range, use_adaptive_threshold_for_porosity,
                                       adaptive_threshold_max_iters, adaptive_threshold_tol, threshold_offsets, postprocess_configs,remove_small_pore_components_flag,
                                       min_pore_component_size, cheap_error_weights, exact_eval_topk_per_candidate, tau_nonperc_value, min_solid_component_count_soft,
                                       final_error_weights, voxel_size_x, voxel_size_y, voxel_size_z,use_std_normalized_error, topology_penalty_weight, suppress_taufactor_output,
                                        model_condition: Dict[str, float], summary: Dict, ldm_model, vae_model, num_samples: int, save_patch_dir: str | None = None
                                      ) -> Dict:
    cond_norm_np = normalize_condition(model_condition, summary, warn_if_target_ood, clip_normalized_condition_to_train_range)
    cond_norm_all = torch.from_numpy(cond_norm_np).unsqueeze(0).repeat(num_samples, 1).to(next(ldm_model.parameters()).device)

    best_result = None
    candidate_summaries = []

    for i in range(num_samples):
        candidate_id = f"sample_{i:03d}"
        cond_i = cond_norm_all[i:i+1]

        prob_volume = decode_one_prob_volume(ldm_model, vae_model, cond_i)

        if use_adaptive_threshold_for_porosity:
            threshold_base, _ = find_threshold_for_target_porosity(
                pore_value=pore_value,
                prob_volume=prob_volume,
                target_porosity=float(model_condition["porosity"]),
                max_iters=adaptive_threshold_max_iters,
                tol=adaptive_threshold_tol,
            )
        else:
            threshold_base = 0.5

        coarse_pool = []

        for offset in threshold_offsets:
            t_used = float(np.clip(threshold_base + offset, 0.0, 1.0))
            bin_raw = threshold_prob_volume(prob_volume, t_used)

            for pp_cfg in postprocess_configs:
                bin_post = postprocess_solid_topology(bin_raw, pp_cfg, pore_value, solid_value)

                if remove_small_pore_components_flag:
                    bin_final = remove_small_pore_components(bin_post, min_pore_component_size, pore_value, solid_value)
                else:
                    bin_final = bin_post.copy()

                cheap_metrics = compute_generated_metrics_cheap(bin_final, voxel_size_x, voxel_size_y, voxel_size_z, pore_value, solid_value)

                cheap_error_dict = make_error_dict(
                    target_condition=model_condition,
                    generated_metrics=cheap_metrics,
                    summary=summary,
                    keys=["porosity", "surface_area"],
                )

                cheap_score = compute_weighted_score(
                    error_dict=cheap_error_dict,
                    generated_metrics=cheap_metrics,
                    weights=cheap_error_weights,
                    use_std_normalized_error=use_std_normalized_error,
                    min_solid_component_count_soft=min_solid_component_count_soft,
                    topology_penalty_weight=topology_penalty_weight
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
        exact_pool = coarse_pool[:exact_eval_topk_per_candidate]

        candidate_best = None

        for item in exact_pool:
            final_metrics = compute_generated_metrics_exact(item["bin_volume_final"], tau_nonperc_value, pore_value, min_solid_component_count_soft, voxel_size_x, voxel_size_y, solid_value, voxel_size_z, suppress_taufactor_output)

            final_error_dict = make_error_dict(
                target_condition=model_condition,
                generated_metrics=final_metrics,
                summary=summary,
                keys=["porosity", "surface_area", "tau_z", "deff_z"],
            )

            total_score = compute_weighted_score(
                error_dict=final_error_dict,
                generated_metrics=final_metrics,
                weights=final_error_weights,
                use_std_normalized_error=use_std_normalized_error,
                min_solid_component_count_soft=min_solid_component_count_soft,
                topology_penalty_weight=topology_penalty_weight
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


def generate_large_volume_service(
    task_id: str,

    local_conditions_json: str,
    summary_json_path: str,

    ldm_ckpt_path: str,
    vae_ckpt_path: str,

    out_dir: str,

    device: str = None,

    patch_size: int = None,
    overlap: int = None,

    num_samples_per_patch: int = None,

    pore_value: int = None,
    solid_value: int = None,

    voxel_size_y: float = None,
    voxel_size_z: float = None,
    voxel_size_x: float = None,

    remove_small_pore_components_flag: bool = None,
    min_pore_component_size: int = None,

    postprocess_configs: list = None,

    use_adaptive_threshold_for_porosity: bool = None,
    adaptive_threshold_max_iters: int = None,
    adaptive_threshold_tol: float = None,

    threshold_offsets: list = None,

    cheap_error_weights: dict = None,
    final_error_weights: dict = None,

    use_std_normalized_error: bool = None,

    topology_penalty_weight: float = None,
    min_solid_component_count_soft: int = None,
    exact_eval_topk_per_candidate: int = None,

    warn_if_target_ood: bool = None,
    clip_normalized_condition_to_train_range: bool = None,

    tau_nonperc_value: float = None,
    suppress_taufactor_output: bool = None,

    external_logger=None,
):
    """
    Phase4:
    从8个local condition生成224³大体积
    """
    # =========================================================
    # 默认参数兜底
    # =========================================================

    device = "cuda" if device is None else device

    patch_size = 128 if patch_size is None else patch_size
    overlap = 32 if overlap is None else overlap

    num_samples_per_patch = (
        64
        if num_samples_per_patch is None
        else num_samples_per_patch
    )
    pore_value = 0 if pore_value is None else pore_value
    solid_value = 1 if solid_value is None else solid_value

    voxel_size_y = 0.0315 if voxel_size_y is None else voxel_size_y
    voxel_size_z = 0.02791 if voxel_size_z is None else voxel_size_z
    voxel_size_x = 0.02791 if voxel_size_x is None else voxel_size_x

    remove_small_pore_components_flag = (
        True
        if remove_small_pore_components_flag is None
        else remove_small_pore_components_flag
    )

    min_pore_component_size = (
        10
        if min_pore_component_size is None
        else min_pore_component_size
    )

    if postprocess_configs is None:
        postprocess_configs = [
            {"name": "raw", "mode": "none"},
            {
                "name": "erode1",
                "mode": "erode",
                "iters": 1
            },
            {
                "name": "open1",
                "mode": "open",
                "iters": 1
            },
            {
                "name": "erode1_dilate1",
                "mode": "erode_dilate",
                "erode_iters": 1,
                "dilate_iters": 1
            },
        ]

    use_adaptive_threshold_for_porosity = (
        True
        if use_adaptive_threshold_for_porosity is None
        else use_adaptive_threshold_for_porosity
    )

    adaptive_threshold_max_iters = (
        25
        if adaptive_threshold_max_iters is None
        else adaptive_threshold_max_iters
    )

    adaptive_threshold_tol = (
        1e-4
        if adaptive_threshold_tol is None
        else adaptive_threshold_tol
    )

    if threshold_offsets is None:
        threshold_offsets = [
            -0.04,
            -0.03,
            -0.02,
            -0.01,
            0.0,
            0.01,
            0.02,
            0.03,
            0.04,
        ]

    if cheap_error_weights is None:

        cheap_error_weights = {
            "porosity": 3.0,
            "surface_area": 2.0,
        }
    if final_error_weights is None:

        final_error_weights = {
            "porosity": 3.0,
            "surface_area": 2.0,
            "tau_z": 4.0,
            "deff_z": 0.5,
        }
    use_std_normalized_error = (
        True
        if use_std_normalized_error is None
        else use_std_normalized_error
    )
    topology_penalty_weight = (
        1.0
        if topology_penalty_weight is None
        else topology_penalty_weight
    )
    min_solid_component_count_soft = (
        10
        if min_solid_component_count_soft is None
        else min_solid_component_count_soft
    )
    exact_eval_topk_per_candidate = (
        3
        if exact_eval_topk_per_candidate is None
        else exact_eval_topk_per_candidate
    )
    warn_if_target_ood = (
        True
        if warn_if_target_ood is None
        else warn_if_target_ood
    )
    clip_normalized_condition_to_train_range = (
        False
        if clip_normalized_condition_to_train_range is None
        else clip_normalized_condition_to_train_range
    )
    tau_nonperc_value = (
        1e6
        if tau_nonperc_value is None
        else tau_nonperc_value
    )
    suppress_taufactor_output = (
        True
        if suppress_taufactor_output is None
        else suppress_taufactor_output
    )

    stride = patch_size - overlap

    # =========================================================
    # logs
    # =========================================================

    def log(msg):

        logger.info(msg)

        if external_logger is not None:
            external_logger(msg)

    # =========================================================
    # start
    # =========================================================

    log("=" * 100)
    log("[LARGE_VOLUME] Generate 224^3 large volume")
    log("=" * 100)

    log(f"[LARGE_VOLUME] task_id = {task_id}")

    log(
        f"[LARGE_VOLUME] "
        f"local_conditions_json = {local_conditions_json}"
    )

    log(
        f"[LARGE_VOLUME] "
        f"summary_json_path = {summary_json_path}"
    )

    log(f"[LARGE_VOLUME] out_dir = {out_dir}")

    # =========================================================
    # path check
    # =========================================================

    for p in [
        local_conditions_json,
        summary_json_path,
        ldm_ckpt_path,
        vae_ckpt_path,
    ]:

        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    ensure_dir(out_dir)

    patch_root = os.path.join(
        out_dir,
        "generated_patches"
    )

    ensure_dir(patch_root)

    # =========================================================
    # load json
    # =========================================================

    log("[LARGE_VOLUME] loading jsons...")

    local_info = load_json(local_conditions_json)

    summary = load_json(summary_json_path)

    # =========================================================
    # device
    # =========================================================

    device_obj = torch.device(
        device
        if (
            device == "cpu"
            or torch.cuda.is_available()
        )
        else "cpu"
    )

    log(f"[LARGE_VOLUME] using device = {device_obj}")

    # =========================================================
    # load models
    # =========================================================

    log("[LARGE_VOLUME] loading ldm model...")

    ldm_model = load_ldm_model(
        ldm_ckpt_path,
        device_obj
    )

    log("[LARGE_VOLUME] loading vae model...")

    vae_model = load_vae_model(
        vae_ckpt_path,
        device_obj
    )

    # =========================================================
    # local conditions
    # =========================================================

    local_conditions = local_info["local_conditions"]

    assert len(local_conditions) == 8, \
        "local_conditions 数量不是8"

    prob_patches = []

    patch_results = []

    # =========================================================
    # generate patches
    # =========================================================

    for item in local_conditions:

        patch_name = item["patch_name"]

        metrics = item["metrics"]

        model_condition = {
            "porosity":
                float(metrics["porosity"]),
            "surface_area":
                float(metrics["surface_area"]),
            "tau_z":
                float(metrics["tau_z"]),
            "deff_z":
                float(metrics["deff_z"]),
        }

        patch_dir = os.path.join(
            patch_root,
            patch_name
        )

        log(
            f"[LARGE_VOLUME] "
            f"generating patch = {patch_name}"
        )

        best_patch = generate_best_patch_from_condition(
            pore_value=pore_value,
            solid_value=solid_value,
            warn_if_target_ood=warn_if_target_ood,
            clip_normalized_condition_to_train_range=clip_normalized_condition_to_train_range,
            use_adaptive_threshold_for_porosity=use_adaptive_threshold_for_porosity,
            adaptive_threshold_max_iters=adaptive_threshold_max_iters,
            adaptive_threshold_tol=adaptive_threshold_tol,
            threshold_offsets=threshold_offsets,
            postprocess_configs=postprocess_configs,
            remove_small_pore_components_flag=remove_small_pore_components_flag,
            min_pore_component_size=min_pore_component_size,
            cheap_error_weights=cheap_error_weights,
            exact_eval_topk_per_candidate=exact_eval_topk_per_candidate,
            tau_nonperc_value=tau_nonperc_value,
            min_solid_component_count_soft=min_solid_component_count_soft,
            final_error_weights=final_error_weights,
            voxel_size_x=voxel_size_x,
            voxel_size_y=voxel_size_y,
            voxel_size_z=voxel_size_z,
            use_std_normalized_error=use_std_normalized_error,
            topology_penalty_weight=topology_penalty_weight,
            suppress_taufactor_output=suppress_taufactor_output,

            model_condition=model_condition,
            summary=summary,
            ldm_model=ldm_model,
            vae_model=vae_model,
            num_samples=num_samples_per_patch,
            save_patch_dir=patch_dir
        )

        prob_patches.append(
            best_patch["prob_volume"]
        )

        patch_results.append({
            "patch_name":
                patch_name,
            "grid_index":
                item["grid_index"],
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

    log("[LARGE_VOLUME] assembling 224^3 volume...")

    assembled_prob = assemble_prob_patches(
        prob_patches=prob_patches,
        patch_size=patch_size,
        stride=stride,
    )

    # =========================================================
    # target large metrics
    # =========================================================

    target_large = local_info[
        "large_volume_metrics_clean"
    ]

    target_porosity_large = float(
        target_large["porosity"]
    )

    # =========================================================
    # adaptive threshold
    # =========================================================

    if use_adaptive_threshold_for_porosity:

        threshold_base, _ = \
            find_threshold_for_target_porosity(
                pore_value=pore_value,
                prob_volume=assembled_prob,
                target_porosity=
                    target_porosity_large,
                max_iters=
                    adaptive_threshold_max_iters,
                tol=
                    adaptive_threshold_tol,
            )

    else:

        threshold_base = 0.5

    # =========================================================
    # assembly candidates
    # =========================================================

    assembly_candidates = []

    for offset in threshold_offsets:

        t_used = float(
            np.clip(
                threshold_base + offset,
                0.0,
                1.0
            )
        )

        bin_raw = threshold_prob_volume(
            assembled_prob,
            t_used
        )

        for pp_cfg in postprocess_configs:

            bin_post = postprocess_solid_topology(
                bin_raw,
                pp_cfg, pore_value, solid_value
            )

            if remove_small_pore_components_flag:

                bin_final = remove_small_pore_components(
                    bin_post, min_pore_component_size, pore_value, solid_value
                )

            else:

                bin_final = bin_post.copy()

            final_metrics = \
                compute_generated_metrics_exact(
                    bin_final, tau_nonperc_value, pore_value, min_solid_component_count_soft, voxel_size_x,
                    voxel_size_y, voxel_size_z, suppress_taufactor_output
                )

            target_condition_large = {
                "porosity":
                    float(target_large["porosity"]),
                "surface_area":
                    float(target_large["surface_area"]),
                "tau_z":
                    float(target_large["tau_z"]),
                "deff_z":
                    float(target_large["deff_z"]),
            }

            final_error_dict = make_error_dict(
                target_condition=
                    target_condition_large,
                generated_metrics=
                    final_metrics,
                summary=summary,
                keys=[
                    "porosity",
                    "surface_area",
                    "tau_z",
                    "deff_z"
                ],
            )

            total_score = compute_weighted_score(
                error_dict=final_error_dict,
                generated_metrics=final_metrics,
                weights=final_error_weights,
                use_std_normalized_error=use_std_normalized_error,
                min_solid_component_count_soft=min_solid_component_count_soft,
                topology_penalty_weight=topology_penalty_weight
            )

            assembly_candidates.append({
                "threshold_base":
                    float(threshold_base),
                "threshold_offset":
                    float(offset),
                "threshold_used":
                    float(t_used),
                "postprocess_config":
                    pp_cfg,
                "bin_raw":
                    bin_raw,
                "bin_post":
                    bin_post,
                "bin_final":
                    bin_final,
                "final_metrics":
                    final_metrics,
                "final_error_vs_target":
                    final_error_dict,
                "total_weighted_score":
                    float(total_score),
            })

    # =========================================================
    # best assembly
    # =========================================================

    assembly_candidates = sorted(
        assembly_candidates,
        key=lambda x: x["total_weighted_score"]
    )

    best_assembly = assembly_candidates[0]

    # =========================================================
    # save
    # =========================================================

    assembled_prob_path = os.path.join(
        out_dir,
        "assembled_prob_224.npy"
    )

    assembled_bin_raw_path = os.path.join(
        out_dir,
        "assembled_bin_raw_224.npy"
    )

    assembled_bin_post_path = os.path.join(
        out_dir,
        "assembled_bin_postprocess_224.npy"
    )

    assembled_bin_final_path = os.path.join(
        out_dir,
        "assembled_bin_final_224.npy"
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
        best_assembly["bin_final"].astype(np.uint8)
    )

    # =========================================================
    # summary
    # =========================================================

    summary_out = {
        "task_id":
            task_id,
        "local_conditions_json":
            local_conditions_json,
        "patch_size":
            patch_size,
        "overlap":
            overlap,
        "stride":
            stride,
        "assembled_shape":
            list(assembled_prob.shape),
        "num_patches":
            len(prob_patches),
        "num_samples_per_patch":
            num_samples_per_patch,
        "target_large_volume_metrics_clean":
            target_large,
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
    }

    summary_out_path = os.path.join(
        out_dir,
        "assembly_summary_224.json"
    )

    save_json(summary_out, summary_out_path)

    # =========================================================
    # finish
    # =========================================================

    log(f"[LARGE_VOLUME] saved to = {out_dir}")

    log(
        f"[LARGE_VOLUME] assembled shape = "
        f"{assembled_prob.shape}"
    )

    log(
        f"[LARGE_VOLUME] "
        f"best final metrics = "
        f"{best_assembly['final_metrics']}"
    )

    log("[LARGE_VOLUME] done")

    return {
        "task_id":
            task_id,
        "output_dir":
            out_dir,
        "assembled_prob_path":
            assembled_prob_path,
        "assembled_bin_raw_path":
            assembled_bin_raw_path,
        "assembled_bin_postprocess_path":
            assembled_bin_post_path,
        "assembled_bin_final_path":
            assembled_bin_final_path,
        "summary_path":
            summary_out_path,
        "summary":
            summary_out,
    }
