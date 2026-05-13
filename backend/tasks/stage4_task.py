# -*- coding: utf-8 -*-
# =========================================================
# background worker
# =========================================================
import traceback

from backend.core.task_manager import set_running, append_log, set_finished, set_failed
from backend.services.stage4_service import run_stage4


def run_stage4_task(
        task_id,
        request_data,
):

    try:

        set_running(task_id)

        append_log(task_id, "Stage4 开始执行")

        # ============================================
        # inject logger
        # ============================================
        def web_log(msg):

            append_log(task_id, msg)

        # ============================================
        # run
        # ============================================
        result = run_stage4(
            porosity=request_data.porosity,
            tau_z=request_data.tau_z,
            surface_area=request_data.surface_area,
            version=request_data.version,
            external_logger=web_log,
        )

        append_log(task_id, "Stage4 执行完成")

        set_finished(task_id, result)

    except Exception as e:

        traceback.print_exc()

        set_failed(task_id, traceback.format_exc())