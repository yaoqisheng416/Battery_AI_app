# -*- coding: utf-8 -*-
import os
import json
import logging
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from backend.electrode_twin.fit_cbd_spreading_parameter import phase_fraction, compute_cluster_size_histogram, \
    compute_block_cbd_fraction_distribution, compute_interface_coverage_on_backbone, compute_cbd_am_contact_ratio, \
    compute_cbd_connectivity_metrics, compute_distance_pdf, compute_distance_to_am, get_interface_pore_mask, \
    load_volume_from_slices, ensure_dir, evaluate_single_w, save_json, save_side_by_side_slices

# ============================================================
# 拟合损失权重
# 这组是偏保守的起始版本
# ============================================================

ALPHA_DIST = 1.0
BETA_CONN = 0.6
GAMMA_COVER = 1.2
DELTA_CONTACT = 1.2
EPS_BLOCK = 2.0
ZETA_CLUSTER = 2.0

# ============================================================
# 统计参数
# ============================================================

DIST_BINS = 60

BLOCK_SIZE = 16
BLOCK_BINS = 40

MIN_CBD_COMPONENT_SIZE = 8
MIN_CBD_COMPONENT_SIZE_FOR_CLUSTER_HIST = 8
CLUSTER_BINS = 24

# 为了防止 cluster histogram 各自范围不同，这里用固定 log range
# 单位是体素数
CLUSTER_LOG_MIN = np.log10(1.0)
CLUSTER_LOG_MAX = np.log10(1e6)

# ============================================================
# logger
# ============================================================
logger = logging.getLogger("cbd_w_fitting_service")

if not logger.handlers:
    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)


# ============================================================
# Service
# ============================================================
def fit_cbd_w_service(
    real_3phase_slice_dir: str,
    out_dir: str,

    # phase label
    pore_value: int = 0,
    am_value: int = 1,
    cbd_value: int = 2,

    # w scan config
    w_min: float = 0.02,
    w_max: float = 0.30,
    num_w: int = 20,

    # cbd generation config
    max_growth_distance_factor: float = 4.0,
    remove_isolated_cbd: bool = True,
    seed: int = 42,

    # voxel size
    voxel_size_y: float = 0.0315,
    voxel_size_z: float = 0.02791,
    voxel_size_x: float = 0.02791,

    external_logger=None,
):
    """
    CBD spreading parameter fitting service.

    Parameters
    ----------
    real_3phase_slice_dir : str
        Input real 3-phase slice directory.

    out_dir : str
        Output directory.

    pore_value : int
        Label value of pore phase.

    am_value : int
        Label value of AM phase.

    cbd_value : int
        Label value of CBD phase.

    w_min : float
        Minimum w value.

    w_max : float
        Maximum w value.

    num_w : int
        Number of w candidates.

    max_growth_distance_factor : float
        CBD growth distance factor.

    remove_isolated_cbd : bool
        Whether remove isolated CBD clusters.

    seed : int
        Random seed.

    voxel_size_y : float
        Voxel size along Y axis.

    voxel_size_z : float
        Voxel size along Z axis.

    voxel_size_x : float
        Voxel size along X axis.
    """
    # =========================================================
    # logs
    # =========================================================
    def log(msg):

        # 1. 控制台 logging
        logger.info(msg)

        # 2. Web task log
        if external_logger is not None:
            external_logger(msg)

    np.random.seed(seed)

    ensure_dir(out_dir)

    log("=" * 80)
    log("Loading real 3-phase slices ...")

    real_3phase = load_volume_from_slices(real_3phase_slice_dir)

    log(f"Real shape: {real_3phase.shape}")
    log(f"Unique labels: {np.unique(real_3phase)}")
    log("=" * 80)

    # ========================================================
    # 两相骨架：把 CBD 去掉后，位置视为 pore
    # ========================================================
    volume_2phase = real_3phase.copy()
    volume_2phase[volume_2phase == cbd_value] = pore_value

    # ========================================================
    # 基于同一个 backbone 定义固定界面
    # ========================================================
    backbone_am_mask = (volume_2phase == am_value)
    backbone_pore_mask = (volume_2phase == pore_value)

    backbone_interface_pore_mask = get_interface_pore_mask(
        backbone_am_mask,
        backbone_pore_mask,
    )

    # ========================================================
    # 真实 CBD 体积分数
    # ========================================================
    phi_cbd_real = phase_fraction(real_3phase, cbd_value)

    real_am_mask = (real_3phase == am_value)
    real_cbd_mask = (real_3phase == cbd_value)

    dist_to_am_real = compute_distance_to_am(real_am_mask)

    d_real = dist_to_am_real[real_cbd_mask]

    if len(d_real) == 0:
        raise RuntimeError("No CBD voxels found in real structure.")

    real_dmax = float(d_real.max())

    dist_x, pdf_real = compute_distance_pdf(
        d_real,
        bins=DIST_BINS,
        dmax=real_dmax,
    )

    real_conn = compute_cbd_connectivity_metrics(real_cbd_mask)

    real_cover = compute_interface_coverage_on_backbone(
        cbd_mask=real_cbd_mask,
        backbone_interface_pore_mask=backbone_interface_pore_mask,
    )

    real_contact_ratio = compute_cbd_am_contact_ratio(
        cbd_mask=real_cbd_mask,
        am_mask=real_am_mask,
    )

    real_block_hist = compute_block_cbd_fraction_distribution(
        cbd_mask=real_cbd_mask,
        block_size=BLOCK_SIZE,
        bins=BLOCK_BINS,
    )

    real_cluster_hist = compute_cluster_size_histogram(
        cbd_mask=real_cbd_mask,
        bins=CLUSTER_BINS,
        min_size=MIN_CBD_COMPONENT_SIZE_FOR_CLUSTER_HIST,
    )

    log(f"Real CBD volume fraction: {phi_cbd_real}")
    log(f"Real CBD connectivity: {real_conn}")
    log(f"Real CBD interface coverage on backbone: {real_cover}")
    log(f"Real CBD-AM contact ratio: {real_contact_ratio}")

    # ========================================================
    # 扫描 w
    # ========================================================
    w_candidates = np.linspace(w_min, w_max, num_w)

    results: List[Dict] = []

    log("=" * 80)
    log("Scanning w candidates ...")

    for w in tqdm(w_candidates, desc="Fitting w", unit="w"):

        res = evaluate_single_w(
            volume_2phase=volume_2phase,
            target_phi_cbd=phi_cbd_real,
            w_um=float(w),

            real_dist_pdf=pdf_real,
            real_dmax=real_dmax,

            real_conn=real_conn,
            real_cover=real_cover,
            real_contact_ratio=real_contact_ratio,

            real_block_hist=real_block_hist,
            real_cluster_hist=real_cluster_hist,

            backbone_interface_pore_mask=backbone_interface_pore_mask,

        )

        results.append(res)

        log(
            f"w={w:.4f} | "
            f"loss={res['total_loss']:.6f} | "
            f"dist={res['loss_dist']:.6f} | "
            f"conn={res['loss_conn']:.6f} | "
            f"cover={res['loss_cover']:.6f} | "
            f"contact={res['loss_contact']:.6f} | "
            f"block={res['loss_block']:.6f} | "
            f"cluster={res['loss_cluster']:.6f}"
        )

    # ========================================================
    # 找最优 w
    # ========================================================
    losses = np.array(
        [r["total_loss"] for r in results],
        dtype=np.float64
    )

    best_idx = int(np.argmin(losses))

    best = results[best_idx]

    best_w = float(best["w_um"])

    log("=" * 80)
    log(f"Best w found: {best_w}")
    log(f"Best total loss: {best['total_loss']}")
    log("=" * 80)

    # ========================================================
    # 保存最优三相重建
    # ========================================================
    best_3phase = best["generated_3phase"]

    np.save(
        os.path.join(out_dir, "best_w_reconstructed_3phase.npy"),
        best_3phase
    )

    # ========================================================
    # 保存 summary
    # ========================================================
    summary = {
        "real_3phase_slice_dir": real_3phase_slice_dir,

        "real_phase_fractions": {
            "pore": phase_fraction(real_3phase, pore_value),
            "AM": phase_fraction(real_3phase, am_value),
            "CBD": phase_fraction(real_3phase, cbd_value),
        },

        "real_connectivity": real_conn,

        "real_interface_coverage_on_backbone": real_cover,

        "real_contact_ratio": real_contact_ratio,

        "scan_config": {
            "W_MIN": w_min,
            "W_MAX": w_max,
            "NUM_W": num_w,

            "MAX_GROWTH_DISTANCE_FACTOR": max_growth_distance_factor,
            "REMOVE_ISOLATED_CBD": remove_isolated_cbd,

            "SEED": seed,

            "VOXEL_SIZE_Y": voxel_size_y,
            "VOXEL_SIZE_Z": voxel_size_z,
            "VOXEL_SIZE_X": voxel_size_x,

            "ALPHA_DIST": ALPHA_DIST,
            "BETA_CONN": BETA_CONN,
            "GAMMA_COVER": GAMMA_COVER,
            "DELTA_CONTACT": DELTA_CONTACT,
            "EPS_BLOCK": EPS_BLOCK,
            "ZETA_CLUSTER": ZETA_CLUSTER,

            "DIST_BINS": DIST_BINS,

            "BLOCK_SIZE": BLOCK_SIZE,
            "BLOCK_BINS": BLOCK_BINS,

            "MIN_CBD_COMPONENT_SIZE": MIN_CBD_COMPONENT_SIZE,

            "MIN_CBD_COMPONENT_SIZE_FOR_CLUSTER_HIST":
                MIN_CBD_COMPONENT_SIZE_FOR_CLUSTER_HIST,

            "CLUSTER_BINS": CLUSTER_BINS,

            "CLUSTER_LOG_MIN": float(CLUSTER_LOG_MIN),
            "CLUSTER_LOG_MAX": float(CLUSTER_LOG_MAX),
        },

        "best_w_um": best_w,

        "best_loss": float(best["total_loss"]),

        "best_loss_components": {
            "loss_dist": best["loss_dist"],
            "loss_conn": best["loss_conn"],
            "loss_cover": best["loss_cover"],
            "loss_contact": best["loss_contact"],
            "loss_block": best["loss_block"],
            "loss_cluster": best["loss_cluster"],
        },

        "best_generated_summary": best["generated_summary"],

        "all_results": [
            {
                "w_um": float(r["w_um"]),
                "total_loss": float(r["total_loss"]),

                "loss_dist": float(r["loss_dist"]),
                "loss_conn": float(r["loss_conn"]),
                "loss_cover": float(r["loss_cover"]),
                "loss_contact": float(r["loss_contact"]),
                "loss_block": float(r["loss_block"]),
                "loss_cluster": float(r["loss_cluster"]),

                "generated_connectivity": r["connectivity"],

                "generated_interface_coverage_on_backbone":
                    float(r["interface_coverage"]),

                "generated_contact_ratio":
                    float(r["contact_ratio"]),

                "generated_phi_cbd":
                    float(r["generated_summary"]["actual_cbd_vol_frac"]),
            }
            for r in results
        ],
    }

    save_json(
        summary,
        os.path.join(out_dir, "w_fitting_summary.json")
    )

    # ========================================================
    # 图 1：w 扫描总损失曲线
    # ========================================================
    fig = plt.figure(figsize=(5.8, 4.5))

    plt.plot(
        [r["w_um"] for r in results],
        [r["total_loss"] for r in results],
        marker="o",
        linewidth=1.8,
    )

    plt.axvline(
        best_w,
        linestyle="--",
        linewidth=1.2,
        label=f"Best w = {best_w:.3f} um"
    )

    plt.xlabel("w (um)")
    plt.ylabel("Total fitting loss")
    plt.title("CBD spreading parameter fitting")

    plt.legend(frameon=False)

    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, "w_scan_total_loss_curve.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # 图 2：真实 vs 最优 w 的距离分布
    # ========================================================
    fig = plt.figure(figsize=(5.8, 4.5))

    plt.plot(
        dist_x,
        pdf_real,
        linewidth=2.0,
        label="Real CBD"
    )

    plt.plot(
        dist_x,
        best["distance_pdf"],
        linewidth=2.0,
        label=f"Generated CBD (w={best_w:.3f} um)"
    )

    plt.xlabel("Distance to AM surface (um)")
    plt.ylabel("Probability density")
    plt.title("CBD distance-to-AM distribution")

    plt.legend(frameon=False)

    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, "distance_distribution_real_vs_best.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # 图 3：loss 分量 vs w
    # ========================================================
    fig = plt.figure(figsize=(6.4, 5.0))

    plt.plot(
        [r["w_um"] for r in results],
        [r["loss_dist"] for r in results],
        marker="o",
        label="Distance loss"
    )

    plt.plot(
        [r["w_um"] for r in results],
        [r["loss_conn"] for r in results],
        marker="o",
        label="Connectivity loss"
    )

    plt.plot(
        [r["w_um"] for r in results],
        [r["loss_cover"] for r in results],
        marker="o",
        label="Coverage loss"
    )

    plt.plot(
        [r["w_um"] for r in results],
        [r["loss_contact"] for r in results],
        marker="o",
        label="Contact loss"
    )

    plt.plot(
        [r["w_um"] for r in results],
        [r["loss_block"] for r in results],
        marker="o",
        label="Block loss"
    )

    plt.plot(
        [r["w_um"] for r in results],
        [r["loss_cluster"] for r in results],
        marker="o",
        label="Cluster loss"
    )

    plt.axvline(
        best_w,
        linestyle="--",
        linewidth=1.2,
        color="black"
    )

    plt.xlabel("w (um)")
    plt.ylabel("Loss component")
    plt.title("Loss components during w fitting")

    plt.legend(frameon=False, ncol=2)

    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, "loss_components_vs_w.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # 图 4：真实 vs 两相骨架 vs 最优重建
    # ========================================================
    save_side_by_side_slices(
        real_3phase=real_3phase,
        backbone_2phase=volume_2phase,
        recon_3phase=best_3phase,
        out_path=os.path.join(
            out_dir,
            "real_vs_backbone_vs_reconstructed.png"
        ),
    )

    # ========================================================
    # 图 5：block 分布对比
    # Block distribution comparison
    # ========================================================
    fig = plt.figure(figsize=(5.8, 4.5))

    x_block = np.arange(len(real_block_hist))

    plt.plot(
        x_block,
        real_block_hist,
        linewidth=2.0,
        label="Real CBD"
    )

    plt.plot(
        x_block,
        best["block_hist"],
        linewidth=2.0,
        label=f"Generated CBD (w={best_w:.3f} um)"
    )

    plt.xlabel("Block histogram bin")
    plt.ylabel("Probability density")
    plt.title("Block-wise CBD fraction distribution")

    plt.legend(frameon=False)

    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, "block_hist_real_vs_best.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # 图 6：coverage 和 contact vs w
    # Coverage and contact ratio vs w
    # ========================================================
    fig = plt.figure(figsize=(6.0, 4.6))

    plt.plot(
        [r["w_um"] for r in results],
        [r["interface_coverage"] for r in results],
        marker="o",
        label="Generated interface coverage on backbone",
    )

    plt.plot(
        [r["w_um"] for r in results],
        [r["contact_ratio"] for r in results],
        marker="o",
        label="Generated CBD-AM contact ratio",
    )

    plt.axhline(
        real_cover,
        linestyle="--",
        linewidth=1.2,
        label="Real interface coverage on backbone"
    )

    plt.axhline(
        real_contact_ratio,
        linestyle="--",
        linewidth=1.2,
        label="Real CBD-AM contact ratio"
    )

    plt.axvline(
        best_w,
        linestyle="--",
        linewidth=1.2,
        color="black"
    )

    plt.xlabel("w (um)")
    plt.ylabel("Metric value")

    plt.title("Interface-related metrics during w fitting")

    plt.legend(frameon=False)

    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, "contact_ratio_and_cover_vs_w.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # 图 7：cluster histogram 对比
    # Cluster histogram comparison
    # ========================================================
    fig = plt.figure(figsize=(5.8, 4.5))

    x_cluster = np.arange(len(real_cluster_hist))

    plt.plot(
        x_cluster,
        real_cluster_hist,
        linewidth=2.0,
        label="Real CBD"
    )

    plt.plot(
        x_cluster,
        best["cluster_hist"],
        linewidth=2.0,
        label=f"Generated CBD (w={best_w:.3f} um)"
    )

    plt.xlabel("Cluster histogram bin")
    plt.ylabel("Probability density")

    plt.title("CBD cluster-size distribution")

    plt.legend(frameon=False)

    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, "cluster_hist_real_vs_best.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # 图 8：加权损失分量 vs w
    # Weighted loss components vs w
    # ========================================================
    fig = plt.figure(figsize=(6.4, 5.0))

    plt.plot(
        [r["w_um"] for r in results],
        [ALPHA_DIST * r["loss_dist"] for r in results],
        marker="o",
        label="Weighted distance"
    )

    plt.plot(
        [r["w_um"] for r in results],
        [BETA_CONN * r["loss_conn"] for r in results],
        marker="o",
        label="Weighted connectivity"
    )

    plt.plot(
        [r["w_um"] for r in results],
        [GAMMA_COVER * r["loss_cover"] for r in results],
        marker="o",
        label="Weighted coverage"
    )

    plt.plot(
        [r["w_um"] for r in results],
        [DELTA_CONTACT * r["loss_contact"] for r in results],
        marker="o",
        label="Weighted contact"
    )

    plt.plot(
        [r["w_um"] for r in results],
        [EPS_BLOCK * r["loss_block"] for r in results],
        marker="o",
        label="Weighted block"
    )

    plt.plot(
        [r["w_um"] for r in results],
        [ZETA_CLUSTER * r["loss_cluster"] for r in results],
        marker="o",
        label="Weighted cluster"
    )

    plt.axvline(
        best_w,
        linestyle="--",
        linewidth=1.2,
        color="black"
    )

    plt.xlabel("w (um)")
    plt.ylabel("Weighted loss contribution")

    plt.title("Weighted loss contributions during w fitting")

    plt.legend(frameon=False, ncol=2)

    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, "weighted_loss_components_vs_w.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    log("Done.")
    log(f"Best w: {best_w}")
    log(f"Outputs saved to: {out_dir}")

    return {
        "success": True,
        "best_w": best_w,
        "best_loss": float(best["total_loss"]),
        "summary_json": os.path.join(out_dir, "w_fitting_summary.json"),
        "best_3phase_npy": os.path.join(
            out_dir,
            "best_w_reconstructed_3phase.npy"
        ),
        "output_dir": out_dir,
    }
