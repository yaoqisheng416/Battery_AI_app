# -*- coding: utf-8 -*-
import os
from typing import Dict

import numpy as np
import torch
from backend.electrode_twin.generate_structure_from_condition import OUT_DIR, ensure_dir, get_next_run_dir, DEVICE, \
    load_summary, SUMMARY_JSON_PATH, estimate_deff_from_porosity_tau, normalize_condition, LDM_CKPT_PATH, \
    load_ldm_model, VAE_CKPT_PATH, load_vae_model, NUM_SAMPLES, decode_one_prob_volume, threshold_prob_volume, \
    compute_generated_metrics_exact, CLIP_NORMALIZED_CONDITION_TO_TRAIN_RANGE, validate_target_condition, \
    WARN_IF_TARGET_OOD, THRESHOLD_OFFSETS, POSTPROCESS_CONFIGS, USE_ADAPTIVE_THRESHOLD_FOR_POROSITY, \
    ADAPTIVE_THRESHOLD_MAX_ITERS, ADAPTIVE_THRESHOLD_TOL, CHEAP_ERROR_WEIGHTS, FINAL_ERROR_WEIGHTS, \
    EXACT_EVAL_TOPK_PER_CANDIDATE, TOPOLOGY_PENALTY_WEIGHT, MIN_SOLID_COMPONENT_COUNT_SOFT, USE_STD_NORMALIZED_ERROR, \
    VOXEL_SIZE_Y, VOXEL_SIZE_Z, VOXEL_SIZE_X, save_json, find_threshold_for_target_porosity, postprocess_solid_topology, \
    compute_generated_metrics_cheap, make_error_dict, compute_weighted_score, SAVE_ALL_SLICES, \
    save_volume_slices, SAVE_MID_THREE_VIEWS, save_mid_views


def save_best_outputs(
    best_result: Dict,
    run_dir: str,
):

    best_dir = os.path.join(run_dir, "best_sample")

    ensure_dir(best_dir)

    prob_volume = best_result["prob_volume"]
    bin_volume_raw = best_result["bin_volume_raw"]
    bin_volume_final = best_result["bin_volume_final"]

    generated_prob_path = os.path.join(
        best_dir,
        "generated_prob.npy"
    )

    generated_raw_path = os.path.join(
        best_dir,
        "generated_bin_raw.npy"
    )

    generated_final_path = os.path.join(
        best_dir,
        "generated_bin_final.npy"
    )

    np.save(
        generated_prob_path,
        prob_volume.astype(np.float32)
    )

    np.save(
        generated_raw_path,
        bin_volume_raw.astype(np.uint8)
    )

    np.save(
        generated_final_path,
        bin_volume_final.astype(np.uint8)
    )

    if SAVE_ALL_SLICES:
        save_volume_slices(
            prob_volume,
            bin_volume_final,
            best_dir
        )

    if SAVE_MID_THREE_VIEWS:
        save_mid_views(
            prob_volume,
            bin_volume_final,
            best_dir
        )

    sample_info = {

        "sample_id":
            "best_sample",

        "original_candidate_id":
            best_result["candidate_id"],

        "primary_target_condition":
            best_result["primary_target_condition"],

        "model_condition":
            best_result["model_condition"],

        "target_condition_normalized":
            best_result["target_condition_normalized"],

        "threshold_base":
            best_result["threshold_base"],

        "threshold_offset":
            best_result["threshold_offset"],

        "threshold_used":
            best_result["threshold_used"],

        "postprocess_config":
            best_result["postprocess_config"],

        "cheap_metrics":
            best_result["cheap_metrics"],

        "final_metrics":
            best_result["final_metrics"],

        "cheap_error_vs_target":
            best_result["cheap_error_vs_target"],

        "final_error_vs_target":
            best_result["final_error_vs_target"],

        "cheap_score":
            best_result["cheap_score"],

        "total_weighted_score":
            best_result["total_weighted_score"],

        "shape":
            list(bin_volume_final.shape),

        "saved_files": {

            "generated_prob_npy":
                generated_prob_path,

            "generated_bin_raw_npy":
                generated_raw_path,

            "generated_bin_final_npy":
                generated_final_path,
        },

        "units": {
            "porosity": "dimensionless",
            "surface_area": "1/um",
            "tau_z": "dimensionless",
            "deff_z": "relative",
            "largest_connected_ratio": "dimensionless",
        }
    }

    sample_info_path = os.path.join(
        best_dir,
        "sample_info.json"
    )

    save_json(
        sample_info,
        sample_info_path
    )

    # =====================================================
    # return
    # =====================================================
    return {

        "best_dir":
            best_dir,

        "generated_prob_npy":
            generated_prob_path,

        "generated_bin_raw_npy":
            generated_raw_path,

        "generated_bin_final_npy":
            generated_final_path,

        "sample_info_json":
            sample_info_path,
    }


def run_stage4(
        porosity,
        tau_z,
        surface_area,
        vae_path,
        ldm_path,
        device,
        num_samples,
        external_logger=None,
):
    """
    Stage4:
    条件可控两相结构生成

    Parameters
    ----------
    porosity : float
    tau_z : float
    surface_area : float
    version : str or None

    Returns
    -------
    dict
    :param ldm_path:
    :param vae_path:
    """

    # =========================================================
    # logs
    # =========================================================
    logs = []

    def log(msg):
        print(msg)
        logs.append(str(msg))

        if external_logger is not None:
            external_logger(msg)

    # =========================================================
    # start
    # =========================================================
    log("=" * 90)
    log("开始执行按需生成闭环：generate -> coarse filter -> exact evaluate -> select")
    log("=" * 90)

    ldm_ckpt_path = ldm_path
    vae_ckpt_path = vae_path

    log(f"LDM_CKPT_PATH : {ldm_ckpt_path}")
    log(f"VAE_CKPT_PATH : {vae_ckpt_path}")
    log(f"SUMMARY_JSON_PATH : {SUMMARY_JSON_PATH}")
    log(f"OUT_DIR : {OUT_DIR}")
    log(f"DEVICE : {device}")
    log(f"NUM_SAMPLES : {num_samples}")

    # =========================================================
    # target condition
    # =========================================================
    target_condition = {
        "porosity": float(porosity),
        "surface_area": float(surface_area),
        "tau_z": float(tau_z),
    }

    log(f"PRIMARY TARGET_CONDITION : {target_condition}")

    log(
        f"CLIP_NORMALIZED_CONDITION_TO_TRAIN_RANGE : "
        f"{CLIP_NORMALIZED_CONDITION_TO_TRAIN_RANGE}"
    )

    log("=" * 90)

    # =========================================================
    # output dir
    # =========================================================
    ensure_dir(OUT_DIR)

    run_dir = get_next_run_dir(OUT_DIR)

    # =========================================================
    # device
    # =========================================================
    device = torch.device(
        device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    )

    log(f"实际使用设备: {device}")
    log(f"本次输出目录: {run_dir}")

    # =========================================================
    # summary
    # =========================================================
    summary = load_summary(SUMMARY_JSON_PATH)

    # =========================================================
    # auto derive deff
    # =========================================================
    derived_deff = estimate_deff_from_porosity_tau(
        porosity_value=float(target_condition["porosity"]),
        tau_value=float(target_condition["tau_z"]),
        mode="simple",
    )

    model_condition = {
        "porosity": float(target_condition["porosity"]),
        "surface_area": float(target_condition["surface_area"]),
        "tau_z": float(target_condition["tau_z"]),
        "deff_z": float(derived_deff),
    }

    # =========================================================
    # validate range
    # =========================================================
    range_report = validate_target_condition(
        model_condition,
        summary
    )

    if WARN_IF_TARGET_OOD:

        for item in range_report:

            if item["status"] != "in_range":

                log(
                    f"[警告] "
                    f"{item['condition_key']} = "
                    f"{item['input_value']:.6f} "
                    f"超出训练范围 "
                    f"[{item['min']:.6f}, {item['max']:.6f}]"
                )

    # =========================================================
    # normalize condition
    # =========================================================
    cond_norm_np = normalize_condition(
        model_condition,
        summary
    )

    cond_norm_all = (
        torch.from_numpy(cond_norm_np)
        .unsqueeze(0)
        .repeat(num_samples, 1)
        .to(device)
    )

    # =========================================================
    # load model
    # =========================================================
    log("开始加载 LDM 模型 ...")

    ldm_model = load_ldm_model(
        ldm_ckpt_path,
        device
    )

    log("开始加载 VAE 模型 ...")

    vae_model = load_vae_model(
        vae_ckpt_path,
        device
    )

    log("模型加载完成")

    # =========================================================
    # save run info
    # =========================================================
    run_info = {
        "run_dir": run_dir,

        "primary_target_condition": target_condition,

        "model_condition": model_condition,

        "target_condition_normalized":
            cond_norm_np.tolist(),

        "target_range_report":
            range_report,

        "num_samples":
            num_samples,

        "ldm_checkpoint":
            ldm_ckpt_path,

        "vae_checkpoint":
            vae_ckpt_path,

        "summary_json_path":
            SUMMARY_JSON_PATH,

        "threshold_offsets":
            THRESHOLD_OFFSETS,

        "postprocess_configs":
            POSTPROCESS_CONFIGS,

        "clip_normalized_condition_to_train_range":
            CLIP_NORMALIZED_CONDITION_TO_TRAIN_RANGE,

        "use_adaptive_threshold_for_porosity":
            USE_ADAPTIVE_THRESHOLD_FOR_POROSITY,

        "adaptive_threshold_max_iters":
            ADAPTIVE_THRESHOLD_MAX_ITERS,

        "adaptive_threshold_tol":
            ADAPTIVE_THRESHOLD_TOL,

        "cheap_error_weights":
            CHEAP_ERROR_WEIGHTS,

        "final_error_weights":
            FINAL_ERROR_WEIGHTS,

        "exact_eval_topk_per_candidate":
            EXACT_EVAL_TOPK_PER_CANDIDATE,

        "topology_penalty_weight":
            TOPOLOGY_PENALTY_WEIGHT,

        "min_solid_component_count_soft":
            MIN_SOLID_COMPONENT_COUNT_SOFT,

        "use_std_normalized_error":
            USE_STD_NORMALIZED_ERROR,

        "voxel_size_um": {
            "Y": VOXEL_SIZE_Y,
            "Z": VOXEL_SIZE_Z,
            "X": VOXEL_SIZE_X,
        },

        "note": (
            "本次运行使用 3 个主目标"
            "（porosity/surface_area/tau_z），"
            "内部自动构造第 4 个模型条件 "
            "deff_z=porosity/tau_z，"
            "先粗筛再对少量候选做 TauFactor 精确评估。"
        ),
    }

    save_json(
        run_info,
        os.path.join(run_dir, "run_info.json")
    )

    # =========================================================
    # generate
    # =========================================================
    all_results = []

    best_result = None

    log("开始生成与搜索 ...")

    for i in range(num_samples):

        candidate_id = f"sample_{i:03d}"

        log(f"开始生成 {candidate_id}")

        cond_i = cond_norm_all[i:i + 1]

        # =====================================================
        # decode
        # =====================================================
        prob_volume = decode_one_prob_volume(
            ldm_model=ldm_model,
            vae_model=vae_model,
            cond_norm=cond_i,
        )

        # =====================================================
        # adaptive threshold
        # =====================================================
        if USE_ADAPTIVE_THRESHOLD_FOR_POROSITY:

            threshold_base, _ = find_threshold_for_target_porosity(
                prob_volume=prob_volume,
                target_porosity=float(
                    target_condition["porosity"]
                ),
                max_iters=ADAPTIVE_THRESHOLD_MAX_ITERS,
                tol=ADAPTIVE_THRESHOLD_TOL,
            )

        else:

            threshold_base = 0.5

        coarse_pool = []

        # =====================================================
        # threshold search
        # =====================================================
        for offset in THRESHOLD_OFFSETS:

            t_used = float(
                np.clip(
                    threshold_base + offset,
                    0.0,
                    1.0
                )
            )

            bin_raw = threshold_prob_volume(
                prob_volume,
                t_used
            )

            # =================================================
            # postprocess
            # =================================================
            for pp_cfg in POSTPROCESS_CONFIGS:

                bin_final = postprocess_solid_topology(
                    bin_raw,
                    pp_cfg
                )

                cheap_metrics = compute_generated_metrics_cheap(
                    bin_final
                )

                cheap_error_dict = make_error_dict(
                    target_condition=model_condition,
                    generated_metrics=cheap_metrics,
                    summary=summary,
                    keys=[
                        "porosity",
                        "surface_area"
                    ],
                )

                cheap_score = compute_weighted_score(
                    error_dict=cheap_error_dict,
                    generated_metrics=cheap_metrics,
                    weights=CHEAP_ERROR_WEIGHTS,
                )

                coarse_pool.append({

                    "candidate_id":
                        candidate_id,

                    "primary_target_condition":
                        target_condition,

                    "model_condition":
                        model_condition,

                    "target_condition_normalized":
                        cond_i[0]
                        .detach()
                        .cpu()
                        .numpy()
                        .tolist(),

                    "prob_volume":
                        prob_volume,

                    "bin_volume_raw":
                        bin_raw,

                    "bin_volume_final":
                        bin_final,

                    "threshold_base":
                        float(threshold_base),

                    "threshold_offset":
                        float(offset),

                    "threshold_used":
                        float(t_used),

                    "postprocess_config":
                        pp_cfg,

                    "cheap_metrics":
                        cheap_metrics,

                    "cheap_error_vs_target":
                        cheap_error_dict,

                    "cheap_score":
                        float(cheap_score),
                })

        # =====================================================
        # coarse filter
        # =====================================================
        coarse_pool = sorted(
            coarse_pool,
            key=lambda x: x["cheap_score"]
        )

        exact_pool = coarse_pool[
                     :EXACT_EVAL_TOPK_PER_CANDIDATE
                     ]

        candidate_best = None

        # =====================================================
        # exact evaluate
        # =====================================================
        for item in exact_pool:

            final_metrics = compute_generated_metrics_exact(
                item["bin_volume_final"]
            )

            final_error_dict = make_error_dict(
                target_condition=model_condition,
                generated_metrics=final_metrics,
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
                weights=FINAL_ERROR_WEIGHTS,
            )

            result = dict(item)

            result.update({

                "final_metrics":
                    final_metrics,

                "final_error_vs_target":
                    final_error_dict,

                "total_weighted_score":
                    float(total_score),
            })

            # =============================================
            # candidate best
            # =============================================
            if (
                    candidate_best is None
                    or result["total_weighted_score"]
                    < candidate_best["total_weighted_score"]
            ):
                candidate_best = result

            # =============================================
            # global best
            # =============================================
            if (
                    best_result is None
                    or result["total_weighted_score"]
                    < best_result["total_weighted_score"]
            ):
                best_result = result

        # =====================================================
        # log result
        # =====================================================
        if candidate_best is None:
            continue

        all_results.append({

            "candidate_id":
                candidate_id,

            "best_score_within_candidate":
                candidate_best["total_weighted_score"],

            "best_threshold_used":
                candidate_best["threshold_used"],

            "best_postprocess_config":
                candidate_best["postprocess_config"],

            "best_final_metrics":
                candidate_best["final_metrics"],
        })

        fm = candidate_best["final_metrics"]

        log(
            f"[候选完成] {candidate_id} | "
            f"score = {candidate_best['total_weighted_score']:.6f}, "
            f"thr = {candidate_best['threshold_used']:.6f}, "
            f"pp = {candidate_best['postprocess_config']['name']}, "
            f"por = {fm['porosity']:.6f}, "
            f"surf = {fm['surface_area']:.3f}, "
            f"tau = {fm['tau_z']:.6f}, "
            f"deff = {fm['deff_z']:.6f}, "
            f"solid_num = {fm['solid_component_count']}"
        )

    # =========================================================
    # no result
    # =========================================================
    if best_result is None:
        raise RuntimeError("没有生成出任何有效结果。")

    # =========================================================
    # save best outputs
    # =========================================================
    log("开始保存最佳结果 ...")

    # save_best_outputs(
    #     best_result,
    #     run_dir
    # )
    saved_files = save_best_outputs(
        best_result,
        run_dir
    )

    # =========================================================
    # summary
    # =========================================================
    generation_summary = {

        "run_dir":
            run_dir,

        "num_samples":
            num_samples,

        "primary_target_condition":
            target_condition,

        "model_condition":
            model_condition,

        "target_condition_normalized":
            cond_norm_np.tolist(),

        "best_candidate_id":
            best_result["candidate_id"],

        "best_score":
            best_result["total_weighted_score"],

        "best_threshold_base":
            best_result["threshold_base"],

        "best_threshold_offset":
            best_result["threshold_offset"],

        "best_threshold_used":
            best_result["threshold_used"],

        "best_postprocess_config":
            best_result["postprocess_config"],

        "best_cheap_metrics":
            best_result["cheap_metrics"],

        "best_final_metrics":
            best_result["final_metrics"],

        "best_cheap_error_vs_target":
            best_result["cheap_error_vs_target"],

        "best_final_error_vs_target":
            best_result["final_error_vs_target"],

        "candidate_summaries":
            all_results,
    }

    generation_summary_path = os.path.join(
        run_dir,
        "generation_summary.json"
    )

    save_json(
        generation_summary,
        generation_summary_path
    )

    # =========================================================
    # output path
    # =========================================================
    best_sample_dir = os.path.join(
        run_dir,
        "best_sample"
    )

    # =========================================================
    # finish
    # =========================================================
    log("=" * 90)
    log("运行完成")
    log(f"最佳候选: {best_result['candidate_id']}")
    log(f"最佳阈值: {best_result['threshold_used']:.6f}")
    log(f"最佳后处理: {best_result['postprocess_config']['name']}")
    log(f"最佳分数: {best_result['total_weighted_score']:.6f}")
    log(f"结果目录: {best_sample_dir}")
    log("=" * 90)

    # =========================================================
    # return
    # =========================================================
    return {

        "status": "success",

        "run_dir":
            run_dir,

        "saved_files":
            saved_files,

        "best_sample_dir":
            best_sample_dir,

        "best_candidate_id":
            best_result["candidate_id"],

        "best_score":
            float(best_result["total_weighted_score"]),

        "best_threshold":
            float(best_result["threshold_used"]),

        "best_postprocess":
            best_result["postprocess_config"]["name"],

        "metrics":
            best_result["final_metrics"],

        "summary_json":
            generation_summary_path,

        "logs":
            logs,
    }
