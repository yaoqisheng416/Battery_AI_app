# -*- coding: utf-8 -*-
import os
import logging
import numpy as np

from backend.electrode_twin.CBD_generate import generate_cbd_phase, save_json, save_color_slices
from backend.electrode_twin.generate_structure_from_condition import ensure_dir

# ============================================================
# logger
# ============================================================
logger = logging.getLogger("generate_cbd_service")

if not logger.handlers:
    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)


# ============================================================
# ⭐ SERVICE入口（API直接调用）
# ============================================================

def generate_cbd_service(
    task_id: str,
    input_volume_path: str,
    out_dir: str,
    target_cbd_vol_frac: float,
    w_um: float = None,

    pore_value: int = None,
    am_value: int = None,
    cbd_value: int = None,

    voxel_size_y: float = None,
    voxel_size_z: float = None,
    voxel_size_x: float = None,

    max_growth_distance_factor: float = None,
    remove_isolated_cbd: bool = None,
    seed: int = None,
    external_logger=None,
):
    """
    Phase6: CBD生成服务（基于Phase4 input_volume_path）
    """

    # =========================
    # 默认参数兜底
    # =========================
    pore_value = 0 if pore_value is None else pore_value
    am_value = 1 if am_value is None else am_value
    cbd_value = 2 if cbd_value is None else cbd_value

    w_um = 0.08 if w_um is None else w_um

    voxel_size_y = 0.0315 if voxel_size_y is None else voxel_size_y
    voxel_size_z = 0.02791 if voxel_size_z is None else voxel_size_z
    voxel_size_x = 0.02791 if voxel_size_x is None else voxel_size_x

    max_growth_distance_factor = 4.0 if max_growth_distance_factor is None else max_growth_distance_factor
    remove_isolated_cbd = True if remove_isolated_cbd is None else remove_isolated_cbd
    seed = 42 if seed is None else seed

    # =========================================================
    # logs
    # =========================================================
    def log(msg):

        # 1. 控制台 logging
        logger.info(msg)

        # 2. Web task log
        if external_logger is not None:
            external_logger(msg)
    
    log(f"[CBD] input_volume_path = {input_volume_path}")

    task_root_dir = os.path.dirname(
        os.path.dirname(input_volume_path)
    )

    out_dir = os.path.join(
        out_dir,
        "out_put",
        "cbd_gen_result_best_sample",
        "three_phase"
    )

    ensure_dir(out_dir)

    log(f"[CBD] input = {input_volume_path}")
    log(f"[CBD] output = {out_dir}")

    if not os.path.exists(input_volume_path):
        raise FileNotFoundError(f"Missing: {input_volume_path}")

    volume = np.load(input_volume_path).astype(np.uint8)

    log(f"[CBD] shape = {volume.shape}")

    # 去边界
    volume = volume[:, 2:-2, 2:-2]

    log("[CBD] generating...")

    voxel_size = (voxel_size_y, voxel_size_z, voxel_size_x)

    volume3, summary = generate_cbd_phase(
        volume,
        target_cbd_vol_frac,
        w_um,
        voxel_size=voxel_size,
        pore_value=pore_value,
        am_value=am_value,
        cbd_value=cbd_value,
        seed=seed,
        max_growth_factor=max_growth_distance_factor,
        remove_isolated=remove_isolated_cbd
    )

    out_npy = os.path.join(out_dir, "volume_3phase.npy")
    out_json = os.path.join(out_dir, "summary.json")

    np.save(out_npy, volume3)
    save_json(summary, out_json)

    log("[CBD] saving slices...")

    save_color_slices(
        volume3,
        os.path.join(out_dir, "slices_color"),
        pore_value,
        am_value,
        cbd_value
    )

    log("[CBD] done")

    return {
        "input_volume_path": input_volume_path,
        "output_dir": out_dir,
        "volume_path": out_npy,
        "summary_path": out_json,
        "summary": summary
    }
