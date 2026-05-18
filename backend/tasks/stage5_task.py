# -*- coding: utf-8 -*-
import traceback

from backend.core.task_manager import update_progress, append_log, set_finished, set_failed, set_running
from backend.services.stage5_build_real_large_volume_service import generate_local_conditions_service
from backend.services.stage5_generate_large_volume_224_service import generate_large_volume_service


# ============================================
# stage5 task build_large_volume_conditions_from_real
# ============================================
def run_local_conditions_generate_task(
    task_id,
    request_data,
):

    try:

        # ============================================
        # task running
        # ============================================
        set_running(task_id)

        append_log(
            task_id,
            "Stage5 build_large_volume_conditions_from_real 开始执行"
        )

        update_progress(task_id, 5)

        # ============================================
        # inject logger
        # ============================================
        def web_log(msg):

            append_log(
                task_id,
                str(msg)
            )

        # ============================================
        # request info
        # ============================================
        append_log(
            task_id,
            f"real_volume_path = "
            f"{request_data.real_volume_path}"
        )

        append_log(
            task_id,
            f"out_dir = "
            f"{request_data.out_dir}"
        )

        append_log(
            task_id,
            f"crop_start_y = "
            f"{request_data.crop_start_y}"
        )

        append_log(
            task_id,
            f"crop_start_z = "
            f"{request_data.crop_start_z}"
        )

        append_log(
            task_id,
            f"crop_start_x = "
            f"{request_data.crop_start_x}"
        )

        update_progress(task_id, 15)

        # ============================================
        # run service
        # ============================================
        result = generate_local_conditions_service(
            task_id=task_id,
            # required
            real_volume_path=
                request_data.real_volume_path,
            out_dir=
                request_data.out_dir,
            # crop
            crop_start_y=
                request_data.crop_start_y,
            crop_start_z=
                request_data.crop_start_z,
            crop_start_x=
                request_data.crop_start_x,
            # volume
            large_vol_size=
                request_data.large_vol_size,
            # patch
            patch_size=
                request_data.patch_size,
            overlap=
                request_data.overlap,
            # phase
            pore_value=
                request_data.pore_value,
            solid_value=
                request_data.solid_value,
            # voxel size
            voxel_size_y=
                request_data.voxel_size_y,
            voxel_size_z=
                request_data.voxel_size_z,
            voxel_size_x=
                request_data.voxel_size_x,
            # clean
            remove_small_pore_components_flag=
                request_data.remove_small_pore_components_flag,
            min_pore_component_size=
                request_data.min_pore_component_size,
            # tau
            tau_nonperc_value=
                request_data.tau_nonperc_value,
            suppress_taufactor_output=
                request_data.suppress_taufactor_output,
            # logger
            external_logger=
                web_log,
        )

        update_progress(task_id, 85)

        # ============================================
        # result log
        # ============================================
        append_log(
            task_id,
            "Stage5 build_large_volume_conditions_from_real 构建完成"
        )

        append_log(
            task_id,
            f"output_dir = "
            f"{result.get('output_dir')}"
        )

        append_log(
            task_id,
            f"summary_json = "
            f"{result.get('summary_json')}"
        )

        append_log(
            task_id,
            f"patch_dir = "
            f"{result.get('patch_dir')}"
        )

        append_log(
            task_id,
            f"num_patches = "
            f"{result.get('num_patches')}"
        )

        update_progress(task_id, 100)

        # ============================================
        # finished
        # ============================================
        set_finished(
            task_id,
            result
        )

    except Exception as e:

        traceback.print_exc()

        append_log(
            task_id,
            str(e)
        )

        set_failed(
            task_id,
            traceback.format_exc()
        )


# ============================================
# stage5 task generate_large_volume_224_from_local_conditions
# ============================================
def run_large_volume_generate_task(
    task_id,
    request_data,
):

    try:

        # ============================================
        # task running
        # ============================================
        set_running(task_id)

        append_log(
            task_id,
            "Stage5 generate_large_volume_224_from_local_conditions 开始执行"
        )

        update_progress(task_id, 5)

        # ============================================
        # inject logger
        # ============================================
        def web_log(msg):

            append_log(
                task_id,
                str(msg)
            )

        # ============================================
        # request info
        # ============================================
        append_log(
            task_id,
            f"local_conditions_json = "
            f"{request_data.local_conditions_json}"
        )

        append_log(
            task_id,
            f"summary_json_path = "
            f"{request_data.summary_json_path}"
        )

        append_log(
            task_id,
            f"ldm_ckpt_path = "
            f"{request_data.ldm_ckpt_path}"
        )

        append_log(
            task_id,
            f"vae_ckpt_path = "
            f"{request_data.vae_ckpt_path}"
        )

        append_log(
            task_id,
            f"out_dir = "
            f"{request_data.out_dir}"
        )

        update_progress(task_id, 10)

        # ============================================
        # run service
        # ============================================
        result = generate_large_volume_service(
            task_id=task_id,
            # required
            local_conditions_json=
                request_data.local_conditions_json,
            summary_json_path=
                request_data.summary_json_path,
            ldm_ckpt_path=
                request_data.ldm_ckpt_path,
            vae_ckpt_path=
                request_data.vae_ckpt_path,
            out_dir=
                request_data.out_dir,
            # device
            device=
                request_data.device,
            # patch
            patch_size=
                request_data.patch_size,
            overlap=
                request_data.overlap,
            num_samples_per_patch=
                request_data.num_samples_per_patch,
            # phase
            pore_value=
                request_data.pore_value,
            solid_value=
                request_data.solid_value,
            # voxel size
            voxel_size_y=
                request_data.voxel_size_y,
            voxel_size_z=
                request_data.voxel_size_z,
            voxel_size_x=
                request_data.voxel_size_x,
            # clean
            remove_small_pore_components_flag=
                request_data.remove_small_pore_components_flag,
            min_pore_component_size=
                request_data.min_pore_component_size,
            # postprocess
            postprocess_configs=
                request_data.postprocess_configs,
            # threshold
            use_adaptive_threshold_for_porosity=
                request_data.use_adaptive_threshold_for_porosity,
            adaptive_threshold_max_iters=
                request_data.adaptive_threshold_max_iters,
            adaptive_threshold_tol=
                request_data.adaptive_threshold_tol,
            threshold_offsets=
                request_data.threshold_offsets,
            # error weights
            cheap_error_weights=
                request_data.cheap_error_weights,
            final_error_weights=
                request_data.final_error_weights,
            use_std_normalized_error=
                request_data.use_std_normalized_error,
            topology_penalty_weight=
                request_data.topology_penalty_weight,
            min_solid_component_count_soft=
                request_data.min_solid_component_count_soft,
            exact_eval_topk_per_candidate=
                request_data.exact_eval_topk_per_candidate,
            # OOD control
            warn_if_target_ood=
                request_data.warn_if_target_ood,
            clip_normalized_condition_to_train_range=
                request_data.clip_normalized_condition_to_train_range,
            # tau
            tau_nonperc_value=
                request_data.tau_nonperc_value,
            suppress_taufactor_output=
                request_data.suppress_taufactor_output,
            # logger
            external_logger=
                web_log,
        )

        update_progress(task_id, 85)

        # ============================================
        # result log
        # ============================================
        append_log(
            task_id,
            "Stage5 generate_large_volume_224_from_local_conditions 生成完成"
        )

        append_log(
            task_id,
            f"output_dir = {result.get('output_dir')}"
        )

        append_log(
            task_id,
            f"assembled_prob_path = {result.get('assembled_prob_path')}"
        )

        append_log(
            task_id,
            f"summary_path = {result.get('summary_path')}"
        )

        append_log(
            task_id,
            f"best_score = "
            f"{result.get('summary', {}).get('best_assembly_score')}"
        )

        update_progress(task_id, 100)

        # ============================================
        # finished
        # ============================================
        set_finished(
            task_id,
            result
        )

    except Exception as e:

        traceback.print_exc()

        append_log(
            task_id,
            str(e)
        )

        set_failed(
            task_id,
            traceback.format_exc()
        )