from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple


@dataclass
class GenerateSpecificVolumeConfig:

    # =========================================================
    # 路径参数
    # =========================================================

    summary_json_path: str
    train_metrics_table_path: str
    ldm_ckpt_path: str
    vae_ckpt_path: str
    out_dir: str

    # =========================================================
    # 2. device
    # =========================================================

    device: str = "cuda"

    # =========================================================
    # 3. patch 参数
    # =========================================================

    patch_size: int = 128

    overlap: int = 32

    stride: int = 96

    grid_shape: Tuple[int, int, int] = (2, 2, 2)

    # =========================================================
    # 4. 用户条件模式
    # =========================================================

    condition_input_mode: str = "uniform_porosity"

    # =========================================================
    # 4.1 uniform_porosity
    # =========================================================

    target_patch_porosity: float = 0.30

    target_patch_tau_z: float = 3.30

    # =========================================================
    # 4.2 manual_user
    # =========================================================

    manual_patch_conditions: List[Dict[str, Any]] = field(
        default_factory=list
    )

    # =========================================================
    # 5. 条件自动补全
    # =========================================================

    auto_surface_mode: str = (
        "nearest_training_porosity_tau"
    )

    auto_deff_mode: str = (
        "porosity_over_tau"
    )

    # =========================================================
    # 6. 生成参数
    # =========================================================

    num_samples_per_patch: int = 32

    pore_value: int = 0

    solid_value: int = 1

    # =========================================================
    # 7. voxel size
    # =========================================================

    voxel_size_y: float = 0.0315

    voxel_size_z: float = 0.02791

    voxel_size_x: float = 0.02791

    # =========================================================
    # 8. 小孔移除
    # =========================================================

    remove_small_pore_components: bool = True

    min_pore_component_size: int = 10

    # =========================================================
    # 9. 后处理
    # =========================================================

    postprocess_configs: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "name": "raw",
                "mode": "none",
            },

            {
                "name": "erode1",
                "mode": "erode",
                "iters": 1,
            },

            {
                "name": "open1",
                "mode": "open",
                "iters": 1,
            },

            {
                "name": "erode1_dilate1",
                "mode": "erode_dilate",
                "erode_iters": 1,
                "dilate_iters": 1,
            },
        ]
    )

    # =========================================================
    # 10. adaptive threshold
    # =========================================================

    use_adaptive_threshold_for_porosity: bool = True

    adaptive_threshold_max_iters: int = 25

    adaptive_threshold_tol: float = 1e-4

    threshold_offsets: List[float] = field(
        default_factory=lambda: [
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
    )

    # =========================================================
    # 11. cheap score 权重
    # =========================================================

    cheap_error_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "porosity": 4.0,
        }
    )

    # =========================================================
    # 12. final score 权重
    # =========================================================

    final_error_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "porosity": 4.0,
            "tau_z": 5.0,
            "deff_z": 0.5,
        }
    )

    # =========================================================
    # 13. std error
    # =========================================================

    use_std_normalized_error: bool = True

    # =========================================================
    # 14. topology penalty
    # =========================================================

    topology_penalty_weight: float = 1.0

    min_solid_component_count_soft: int = 10

    exact_eval_topk_per_candidate: int = 3

    # =========================================================
    # 15. OOD
    # =========================================================

    warn_if_target_ood: bool = True

    clip_normalized_condition_to_train_range: bool = False

    # =========================================================
    # 16. taufactor
    # =========================================================

    tau_nonperc_value: float = 1e6

    suppress_taufactor_output: bool = True

    # =========================================================
    # 17. slice visualization
    # =========================================================

    save_all_y_zx_slice_png: bool = True

    # black_yellow / white_blue
    slice_color_style: str = "black_yellow"

    slice_show_axis: bool = False

    slice_dpi: int = 200

    # =========================================================
    # 18. logger
    # =========================================================

    logger_name: str = (
        "generate_specific_volume_service"
    )

    # =========================================================
    # 19. runtime
    # =========================================================

    save_generated_patch_intermediate: bool = True

    save_assembled_intermediate: bool = True
