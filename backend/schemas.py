from pydantic import BaseModel
from typing import Optional


# =========================================================
# stage4 generate_structure_from_condition
# =========================================================
class Stage4Request(BaseModel):
    task_id: Optional[str] = None
    #  ex: 0.2
    porosity: float
    #  ex: 7
    tau_z: float
    #    ex: 1150.0
    surface_area: float
    #  新增：模型路径（前端传递）
    vae_path: Optional[str] = None
    ldm_path: Optional[str] = None
    # 新增：设备和采样数
    device: str = "cuda"       # 默认 cuda
    num_samples: int = 32      # 默认 32


# =========================================================
# stage5 large-volume-generate
# =========================================================
class largeVolumeGenerateRequest(BaseModel):

    task_id: str | None = None

    local_conditions_json: str
    summary_json_path: str | None = None
    ldm_ckpt_path: str
    vae_ckpt_path: str
    out_dir: str

    device: str | None = None

    patch_size: int | None = None
    overlap: int | None = None
    num_samples_per_patch: int | None = None

    pore_value: int | None = None
    solid_value: int | None = None

    voxel_size_y: float | None = None
    voxel_size_z: float | None = None
    voxel_size_x: float | None = None

    remove_small_pore_components_flag: bool | None = None
    min_pore_component_size: int | None = None

    postprocess_configs: list | None = None
    use_adaptive_threshold_for_porosity: bool | None = None
    adaptive_threshold_max_iters: int | None = None
    adaptive_threshold_tol: float | None = None

    threshold_offsets: list | None = None
    cheap_error_weights: dict | None = None
    final_error_weights: dict | None = None

    use_std_normalized_error: bool | None = None
    topology_penalty_weight: float | None = None
    min_solid_component_count_soft: int | None = None
    exact_eval_topk_per_candidate: int | None = None

    warn_if_target_ood: bool | None = None
    clip_normalized_condition_to_train_range: bool | None = None

    tau_nonperc_value: float | None = None
    suppress_taufactor_output: bool | None = None


# =========================================================
# stage5 build_large_volume_conditions_from_real
# =========================================================
from pydantic import BaseModel
from typing import Optional


class localConditionsGenerateRequest(BaseModel):

    task_id: Optional[str] = None

    # required
    real_volume_path: str
    out_dir: str

    # crop
    crop_start_y: Optional[int] = None
    crop_start_z: Optional[int] = None
    crop_start_x: Optional[int] = None

    # volume
    large_vol_size: Optional[int] = None

    # patch
    patch_size: Optional[int] = None
    overlap: Optional[int] = None

    # phase
    pore_value: Optional[int] = None
    solid_value: Optional[int] = None

    # voxel size
    voxel_size_y: Optional[float] = None
    voxel_size_z: Optional[float] = None
    voxel_size_x: Optional[float] = None

    # cleaning
    remove_small_pore_components_flag: Optional[bool] = None
    min_pore_component_size: Optional[int] = None

    # tau
    tau_nonperc_value: Optional[float] = None
    suppress_taufactor_output: Optional[bool] = None


# =========================================================
# stage6 CBD_generate
# =========================================================
class cbdGenerateRequest(BaseModel):
    # ex: electrode_twin/generated_results/run_007
    task_id: Optional[str] = None
    input_volume_path: str


    # ex: 0.05
    target_cbd_vol_frac: float
    w_um: Optional[float] = None

    pore_value: Optional[int] = None
    am_value: Optional[int] = None
    cbd_value: Optional[int] = None

    voxel_size_y: Optional[float] = None
    voxel_size_z: Optional[float] = None
    voxel_size_x: Optional[float] = None

    max_growth_distance_factor: Optional[float] = None
    remove_isolated_cbd: Optional[bool] = None
    seed: Optional[int] = None


# =========================================================
# stage6 fit_cbd_spreading_parameter
# =========================================================
class fitParameterRequest(BaseModel):
    task_id: Optional[str] = None

    # required
    # ex:electrode_twin/real_3phase
    real_3phase_slice_dir: str

    # ex:electrode_twin/cbd_w_fitting
    out_dir: str

    # 相定义
    pore_value: Optional[int] = 0

    am_value: Optional[int] = 1

    cbd_value: Optional[int] = 2

    # 扫描参数
    w_min: Optional[float] = 0.02

    w_max: Optional[float] = 0.30

    num_w: Optional[int] = 20

    max_growth_distance_factor: Optional[float] = 4.0

    remove_isolated_cbd: Optional[bool] = True

    seed: Optional[int] = 42

    # # 真实体素尺寸（单位 um） volume 顺序：[Y, Z, X]
    voxel_size_y: Optional[float] = 0.0315

    voxel_size_z: Optional[float] = 0.02791

    voxel_size_x: Optional[float] = 0.02791
