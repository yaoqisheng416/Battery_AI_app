import os
import logging

from backend.electrode_twin.metrics_3d import plot_two_point_correlation, save_linear_path_csv, save_two_point_csv, \
    save_scalar_report, compute_all_metrics, load_volume, ensure_dir, MetricsConfig, plot_scalar_comparison, \
    plot_linear_path_function

logger = logging.getLogger(
    "metrics_service"
)

if not logger.handlers:

    handler = logging.StreamHandler()

    formatter = logging.Formatter(

        "%(asctime)s "
        "[%(levelname)s] "
        "[%(name)s] "
        "%(message)s"
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)

logger.setLevel(logging.INFO)


def compute_metrics_service(

    # required
    real_volume_path: str,

    recon_volume_path: str,

    out_dir: str,

    # optional
    use_pore_phase_only: bool = True,

    pore_value: int = 0,

    remove_small_components: bool = True,

    min_component_size: int = 20,

    clean_only_for_surface: bool = True,

    max_lag: int = 50,

    max_path_length: int = 50,

    # web logger
    external_logger=None,
):
    """
    3D structure metrics service
    """

    # ========================================================
    # logger
    # ========================================================
    def log(msg):

        logger.info(msg)

        if external_logger is not None:

            external_logger(msg)

    # ========================================================
    # config
    # ========================================================
    cfg = MetricsConfig()

    cfg.real_volume_path = real_volume_path

    cfg.recon_volume_path = recon_volume_path

    cfg.out_dir = out_dir

    cfg.use_pore_phase_only = \
        use_pore_phase_only

    cfg.pore_value = pore_value

    cfg.remove_small_components = \
        remove_small_components

    cfg.min_component_size = \
        min_component_size

    cfg.clean_only_for_surface = \
        clean_only_for_surface

    cfg.max_lag = max_lag

    cfg.max_path_length = \
        max_path_length

    ensure_dir(cfg.out_dir)

    # ========================================================
    # config info
    # ========================================================
    log("=" * 60)

    log("开始计算 3D 结构指标")

    log(f"REAL_VOLUME_PATH        : {cfg.real_volume_path}")

    log(f"RECON_VOLUME_PATH       : {cfg.recon_volume_path}")

    log(f"OUT_DIR                 : {cfg.out_dir}")

    log(f"USE_PORE_PHASE_ONLY     : {cfg.use_pore_phase_only}")

    log(f"PORE_VALUE              : {cfg.pore_value}")

    log(f"REMOVE_SMALL_COMPONENTS : {cfg.remove_small_components}")

    log(f"MIN_COMPONENT_SIZE      : {cfg.min_component_size}")

    log(f"CLEAN_ONLY_FOR_SURFACE  : {cfg.clean_only_for_surface}")

    log(f"MAX_LAG                 : {cfg.max_lag}")

    log(f"MAX_PATH_LENGTH         : {cfg.max_path_length}")

    log("=" * 60)

    # ========================================================
    # load
    # ========================================================
    log("1) 读取真实体数据 ...")

    real_vol = load_volume(
        cfg.real_volume_path
    )

    log(
        f"真实体 shape: {real_vol.shape}"
    )

    log("2) 读取重建体数据 ...")

    recon_vol = load_volume(
        cfg.recon_volume_path
    )

    log(
        f"重建体 shape: {recon_vol.shape}"
    )

    # ========================================================
    # shape check
    # ========================================================
    if real_vol.shape != recon_vol.shape:

        raise ValueError(

            f"shape 不一致: "
            f"real={real_vol.shape}, "
            f"recon={recon_vol.shape}"
        )

    # ========================================================
    # real metrics
    # ========================================================
    log("3) 计算真实体指标 ...")

    real_metrics = compute_all_metrics(
        real_vol,
        cfg
    )

    # ========================================================
    # recon metrics
    # ========================================================
    log("4) 计算重建体指标 ...")

    original_clean_for_surface = \
        cfg.clean_only_for_surface

    cfg.clean_only_for_surface = False

    recon_metrics = compute_all_metrics(
        recon_vol,
        cfg
    )

    cfg.clean_only_for_surface = \
        original_clean_for_surface

    # ========================================================
    # save report
    # ========================================================
    log("5) 保存总报告 ...")

    save_scalar_report(

        real_metrics,

        recon_metrics,

        cfg
    )

    # ========================================================
    # save csv
    # ========================================================
    log("6) 保存两点相关函数 CSV ...")

    save_two_point_csv(

        real_metrics,

        recon_metrics,

        cfg.out_dir
    )

    log("7) 保存线性路径函数 CSV ...")

    save_linear_path_csv(

        real_metrics,

        recon_metrics,

        cfg.out_dir
    )

    # ========================================================
    # plots
    # ========================================================
    log("8) 绘图 ...")

    plot_two_point_correlation(

        real_metrics,

        recon_metrics,

        cfg.out_dir
    )

    plot_linear_path_function(

        real_metrics,

        recon_metrics,

        cfg.out_dir
    )

    plot_scalar_comparison(

        real_metrics,

        recon_metrics,

        cfg
    )

    # ========================================================
    # metrics info
    # ========================================================
    log("9) 输出核心指标 ...")

    if cfg.use_pore_phase_only:

        log(
            f"真实孔隙率: "
            f"{real_metrics['volume_fraction']:.6f}"
        )

        log(
            f"重建孔隙率: "
            f"{recon_metrics['volume_fraction']:.6f}"
        )

        log(
            f"真实最大连通孔隙占比: "
            f"{real_metrics['largest_connected_component_ratio']:.6f}"
        )

        log(
            f"重建最大连通孔隙占比: "
            f"{recon_metrics['largest_connected_component_ratio']:.6f}"
        )

    else:

        log(
            f"真实固相率: "
            f"{real_metrics['volume_fraction']:.6f}"
        )

        log(
            f"重建固相率: "
            f"{recon_metrics['volume_fraction']:.6f}"
        )

        log(
            f"真实最大连通固相占比: "
            f"{real_metrics['largest_connected_component_ratio']:.6f}"
        )

        log(
            f"重建最大连通固相占比: "
            f"{recon_metrics['largest_connected_component_ratio']:.6f}"
        )

    log(

        f"真实比表面积近似(1/um): "

        f"{real_metrics['specific_surface_area_physical']:.6f}"
    )

    log(

        f"重建比表面积近似(1/um): "

        f"{recon_metrics['specific_surface_area_physical']:.6f}"
    )

    log("完成")

    log(f"结果保存在: {cfg.out_dir}")

    # ========================================================
    # return
    # ========================================================
    return {

        "success": True,

        "real_volume_path":
            real_volume_path,

        "recon_volume_path":
            recon_volume_path,

        "out_dir":
            cfg.out_dir,

        "real_metrics":
            real_metrics,

        "recon_metrics":
            recon_metrics,

        "scalar_report":
            os.path.join(
                cfg.out_dir,
                "scalar_metrics_report.json"
            ),

        "two_point_csv":
            os.path.join(
                cfg.out_dir,
                "two_point_correlation.csv"
            ),

        "linear_path_csv":
            os.path.join(
                cfg.out_dir,
                "linear_path_function.csv"
            ),

        "plots": [

            os.path.join(
                cfg.out_dir,
                "two_point_correlation.png"
            ),

            os.path.join(
                cfg.out_dir,
                "linear_path_function.png"
            ),

            os.path.join(
                cfg.out_dir,
                "scalar_comparison.png"
            ),
        ]
    }
