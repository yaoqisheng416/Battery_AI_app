# -*- coding: utf-8 -*-
import os
import logging
import numpy as np

from backend.electrode_twin.build_large_volume_conditions_from_real import ensure_dir, crop_large_volume, \
    remove_small_pore_components, compute_metrics, extract_patch, save_json

# ============================================================
# logger
# ============================================================
logger = logging.getLogger("stage5_build_real_large_volume_service")

if not logger.handlers:
    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)


def generate_local_conditions_service(
    task_id: str,
    real_volume_path: str,
    out_dir: str,

    crop_start_y: int = None,
    crop_start_z: int = None,
    crop_start_x: int = None,

    large_vol_size: int = None,

    patch_size: int = None,
    overlap: int = None,

    pore_value: int = None,
    solid_value: int = None,

    voxel_size_y: float = None,
    voxel_size_z: float = None,
    voxel_size_x: float = None,

    remove_small_pore_components_flag: bool = None,
    min_pore_component_size: int = None,

    tau_nonperc_value: float = None,
    suppress_taufactor_output: bool = None,

    external_logger=None,
):
    """
    PhaseX:
    从真实大体积中构建 224^3 local conditions
    """

    # =========================================================
    # 默认参数兜底
    # =========================================================

    crop_start_y = 0 if crop_start_y is None else crop_start_y
    crop_start_z = 100 if crop_start_z is None else crop_start_z
    crop_start_x = 0 if crop_start_x is None else crop_start_x

    large_vol_size = 224 if large_vol_size is None else large_vol_size

    patch_size = 128 if patch_size is None else patch_size
    overlap = 32 if overlap is None else overlap

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

        # 1. 控制台 logging
        logger.info(msg)

        # 2. Web task log
        if external_logger is not None:
            external_logger(msg)

    # =========================================================
    # start
    # =========================================================

    log("=" * 100)
    log("[LOCAL_CONDITION] Build local conditions from real volume")
    log("=" * 100)

    log(f"[LOCAL_CONDITION] task_id = {task_id}")
    log(f"[LOCAL_CONDITION] real_volume_path = {real_volume_path}")
    log(f"[LOCAL_CONDITION] out_dir = {out_dir}")

    # =========================================================
    # path check
    # =========================================================

    if not os.path.exists(real_volume_path):
        raise FileNotFoundError(
            f"Missing: {real_volume_path}"
        )

    ensure_dir(out_dir)

    patch_dir = os.path.join(
        out_dir,
        "real_patches"
    )

    ensure_dir(patch_dir)

    # =========================================================
    # load volume
    # =========================================================

    log("[LOCAL_CONDITION] loading real volume...")

    real_volume = np.load(real_volume_path).astype(np.uint8)

    log(
        f"[LOCAL_CONDITION] loaded shape = "
        f"{real_volume.shape}"
    )

    # =========================================================
    # 越界检查
    # =========================================================

    sy, sz, sx = real_volume.shape

    assert crop_start_y + large_vol_size <= sy, \
        "Y 方向裁剪越界"

    assert crop_start_z + large_vol_size <= sz, \
        "Z 方向裁剪越界"

    assert crop_start_x + large_vol_size <= sx, \
        "X 方向裁剪越界"

    # =========================================================
    # crop large volume
    # =========================================================

    log("[LOCAL_CONDITION] cropping large volume...")

    real_large = crop_large_volume(
        real_volume,
        crop_start_y,
        crop_start_z,
        crop_start_x,
        large_vol_size,
    )

    # =========================================================
    # 小孔清理
    # =========================================================

    if remove_small_pore_components_flag:

        log(
            "[LOCAL_CONDITION] "
            "removing small pore components..."
        )

        real_large_clean = remove_small_pore_components(
            real_large,
            min_pore_component_size
        )

    else:

        real_large_clean = real_large.copy()

    # =========================================================
    # save large volume
    # =========================================================

    raw_large_path = os.path.join(
        out_dir,
        "real_large_volume_224_raw.npy"
    )

    clean_large_path = os.path.join(
        out_dir,
        "real_large_volume_224_clean.npy"
    )

    np.save(
        raw_large_path,
        real_large.astype(np.uint8)
    )

    np.save(
        clean_large_path,
        real_large_clean.astype(np.uint8)
    )

    log(
        f"[LOCAL_CONDITION] "
        f"saved raw large volume = {raw_large_path}"
    )

    log(
        f"[LOCAL_CONDITION] "
        f"saved clean large volume = {clean_large_path}"
    )

    # =========================================================
    # compute large metrics
    # =========================================================

    log(
        "[LOCAL_CONDITION] "
        "computing large volume metrics..."
    )

    large_metrics = compute_metrics(real_large_clean)

    # =========================================================
    # patch extraction
    # =========================================================

    starts = [0, stride]

    local_conditions = []

    idx = 0

    log("[LOCAL_CONDITION] extracting patches...")

    for iy, y0 in enumerate(starts):
        for iz, z0 in enumerate(starts):
            for ix, x0 in enumerate(starts):

                log(
                    f"[LOCAL_CONDITION] "
                    f"processing patch idx={idx}"
                )

                patch = extract_patch(
                    real_large_clean,
                    y0,
                    z0,
                    x0,
                    patch_size
                )

                metrics = compute_metrics(patch)

                patch_name = (
                    f"real_patch_{idx:03d}_"
                    f"y{iy}_z{iz}_x{ix}"
                )

                patch_path = os.path.join(
                    patch_dir,
                    f"{patch_name}.npy"
                )

                np.save(
                    patch_path,
                    patch.astype(np.uint8)
                )

                local_conditions.append({
                    "patch_id": idx,

                    "patch_name": patch_name,

                    "grid_index": [
                        iy,
                        iz,
                        ix
                    ],

                    "start_in_large_224": [
                        y0,
                        z0,
                        x0
                    ],

                    "metrics": metrics,
                })

                idx += 1

    # =========================================================
    # summary
    # =========================================================

    summary = {

        "task_id": task_id,

        "real_volume_path": real_volume_path,

        "crop_start_yzx": [
            crop_start_y,
            crop_start_z,
            crop_start_x
        ],

        "large_volume_size": large_vol_size,

        "patch_size": patch_size,
        "overlap": overlap,
        "stride": stride,

        "phase_definition": {
            "pore_value": pore_value,
            "solid_value": solid_value,
        },

        "voxel_size": {
            "y": voxel_size_y,
            "z": voxel_size_z,
            "x": voxel_size_x,
        },

        "num_patches": len(local_conditions),

        "cleaning": {

            "remove_small_pore_components":
                remove_small_pore_components_flag,

            "min_pore_component_size":
                min_pore_component_size,
        },

        "tau": {

            "tau_nonperc_value":
                tau_nonperc_value,

            "suppress_taufactor_output":
                suppress_taufactor_output,
        },

        "large_volume_metrics_clean":
            large_metrics,

        "local_conditions":
            local_conditions,

        "saved_files": {

            "real_large_volume_224_raw":
                raw_large_path,

            "real_large_volume_224_clean":
                clean_large_path,

            "real_patches_dir":
                patch_dir,
        }
    }

    out_json = os.path.join(
        out_dir,
        "local_conditions_224.json"
    )

    save_json(summary, out_json)

    log(
        f"[LOCAL_CONDITION] "
        f"saved summary json = {out_json}"
    )

    log("[LOCAL_CONDITION] done")

    return {

        "task_id": task_id,

        "real_volume_path":
            real_volume_path,

        "output_dir":
            out_dir,

        "summary_json":
            out_json,

        "patch_dir":
            patch_dir,

        "num_patches":
            len(local_conditions),

        "large_metrics":
            large_metrics,

        "summary":
            summary,
    }
