# -*- coding: utf-8 -*-
import traceback

from backend.core.task_manager import append_log, set_finished, set_failed, update_progress, set_running
from backend.services.stage5_cbd_generate_service import generate_cbd_service
from backend.services.stege5_cbd_fit_service import fit_cbd_w_service


# ============================================
# CBD 生成
# ============================================
def run_stage5_cbd_generate_task(
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
            "Stage5 cbd-generate 开始执行"
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
            f"input_volume_path = {request_data.input_volume_path}"
        )

        append_log(
            task_id,
            f"target_cbd_vol_frac = "
            f"{request_data.target_cbd_vol_frac}"
        )

        append_log(
            task_id,
            f"w_um = {request_data.w_um}"
        )

        update_progress(task_id, 15)

        # ============================================
        # run service
        # ============================================
        result = generate_cbd_service(
            task_id,

            # required
            input_volume_path=
                request_data.input_volume_path,
            out_dir=
            request_data.out_dir,

            target_cbd_vol_frac=
                request_data.target_cbd_vol_frac,

            # cbd
            w_um=
                request_data.w_um,

            # phase label
            pore_value=
                request_data.pore_value,

            am_value=
                request_data.am_value,

            cbd_value=
                request_data.cbd_value,

            # voxel size
            voxel_size_y=
                request_data.voxel_size_y,

            voxel_size_z=
                request_data.voxel_size_z,

            voxel_size_x=
                request_data.voxel_size_x,

            # cbd generation config
            max_growth_distance_factor=
                request_data.max_growth_distance_factor,

            remove_isolated_cbd=
                request_data.remove_isolated_cbd,

            seed=
                request_data.seed,

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
            "CBD 三相结构生成完成"
        )

        append_log(
            task_id,
            f"output_dir = "
            f"{result.get('output_dir')}"
        )

        append_log(
            task_id,
            f"volume_path = "
            f"{result.get('volume_path')}"
        )

        append_log(
            task_id,
            f"summary_path = "
            f"{result.get('summary_path')}"
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
# CBD 参数拟合
# ============================================
def run_stage5_cbd_fit_task(
    task_id,
    request_data,
):

    try:

        # ============================================
        # task running
        # ============================================
        set_running(task_id)

        append_log(task_id, "Stage5 fit_cbd_spreading_parameter 开始执行")

        update_progress(task_id, 5)

        # ============================================
        # inject logger
        # ============================================
        def web_log(msg):

            append_log(task_id, msg)

        # ============================================
        # run service
        # ============================================
        result = fit_cbd_w_service(

            # required
            real_3phase_slice_dir=
                request_data.real_3phase_slice_dir,

            out_dir=
                request_data.out_dir,

            # phase label
            pore_value=
                request_data.pore_value,

            am_value=
                request_data.am_value,

            cbd_value=
                request_data.cbd_value,

            # w scan config
            w_min=
                request_data.w_min,

            w_max=
                request_data.w_max,

            num_w=
                request_data.num_w,

            # cbd generation config
            max_growth_distance_factor=
                request_data.max_growth_distance_factor,

            remove_isolated_cbd=
                request_data.remove_isolated_cbd,

            seed=
                request_data.seed,

            # voxel size
            voxel_size_y=
                request_data.voxel_size_y,

            voxel_size_z=
                request_data.voxel_size_z,

            voxel_size_x=
                request_data.voxel_size_x,

            # logger
            external_logger=web_log,
        )

        # ============================================
        # finished
        # ============================================
        append_log(task_id, "Stage5 fit_cbd_spreading_parameter 执行完成")

        update_progress(task_id, 100)

        set_finished(task_id, result)

    except Exception as e:

        traceback.print_exc()

        append_log(task_id, str(e))

        set_failed(
            task_id,
            traceback.format_exc()
        )
