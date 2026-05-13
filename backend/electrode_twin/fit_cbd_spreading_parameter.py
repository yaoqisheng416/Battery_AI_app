from __future__ import annotations

import os
import json
import glob
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
    generate_binary_structure,
    label,
)
from tqdm import tqdm

# ============================================================
# 路径设置
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ELECTRODE_TWIN_DIR = os.path.join(BASE_DIR, "electrode_twin")

# 真实三相结构切片文件夹：切片标签 0=pore, 1=AM, 2=CBD
REAL_3PHASE_SLICE_DIR = os.path.join(
    ELECTRODE_TWIN_DIR,
    "real_3phase"
)
# REAL_3PHASE_SLICE_DIR = r"./real_3phase"

# 输出目录
OUT_DIR = os.path.join(
    ELECTRODE_TWIN_DIR,
    "cbd_w_fitting"
)
# OUT_DIR = r"./cbd_w_fitting"

# ============================================================
# 相定义
# ============================================================

PORE_VALUE = 0
AM_VALUE = 1
CBD_VALUE = 2

# ============================================================
# 扫描参数
# ============================================================

W_MIN = 0.02
W_MAX = 0.30
NUM_W = 20

MAX_GROWTH_DISTANCE_FACTOR = 4.0
REMOVE_ISOLATED_CBD = True
SEED = 42

# ============================================================
# 真实体素尺寸（单位 um）
# volume 顺序：[Y, Z, X]
# ============================================================

VOXEL_SIZE_Y = 0.0315
VOXEL_SIZE_Z = 0.02791
VOXEL_SIZE_X = 0.02791

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
# 颜色可视化
# ============================================================

COLOR_PORE = [255, 255, 255]
COLOR_AM = [50, 100, 180]
COLOR_CBD = [0, 180, 150]

# ============================================================
# 工具函数
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def load_volume_from_slices(folder: str) -> np.ndarray:
    """
    从切片文件夹读取三相标签体
    支持 .tif / .tiff / .png
    假设切片本身已经是整数标签图：
        0 = pore
        1 = AM
        2 = CBD
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"找不到切片文件夹: {folder}")

    files = []
    for ext in ("*.tif", "*.tiff", "*.png"):
        files.extend(glob.glob(os.path.join(folder, ext)))
    files = sorted(files)

    if len(files) == 0:
        raise RuntimeError(f"在 {folder} 中未找到切片文件")

    slices = []
    for f in tqdm(files, desc="Loading real 3-phase slices", unit="slice"):
        img = np.array(Image.open(f))
        if img.ndim == 3:
            img = img[..., 0]
        sl = img.astype(np.uint8)
        slices.append(sl)

    volume = np.stack(slices, axis=0)

    print(f"Loaded {len(files)} slices from: {folder}")
    print("Volume shape:", volume.shape)
    print("Unique labels:", np.unique(volume))

    return volume


def phase_fraction(volume: np.ndarray, value: int) -> float:
    return float((volume == value).mean())


def save_phase_slice_rgb(slice_2d: np.ndarray, path: str):
    rgb = np.zeros((slice_2d.shape[0], slice_2d.shape[1], 3), dtype=np.uint8)
    rgb[slice_2d == PORE_VALUE] = COLOR_PORE
    rgb[slice_2d == AM_VALUE] = COLOR_AM
    rgb[slice_2d == CBD_VALUE] = COLOR_CBD
    Image.fromarray(rgb).save(path)


def save_side_by_side_slices(
    real_3phase: np.ndarray,
    backbone_2phase: np.ndarray,
    recon_3phase: np.ndarray,
    out_path: str,
):
    """
    左：真实三相
    中：两相骨架
    右：最优 w 重建三相
    """
    y_mid = real_3phase.shape[0] // 2

    real_sl = real_3phase[y_mid]
    back_sl = backbone_2phase[y_mid]
    recon_sl = recon_3phase[y_mid]

    def to_rgb(sl: np.ndarray, is_backbone: bool = False) -> np.ndarray:
        rgb = np.zeros((sl.shape[0], sl.shape[1], 3), dtype=np.uint8)
        rgb[sl == PORE_VALUE] = COLOR_PORE
        rgb[sl == AM_VALUE] = COLOR_AM
        if not is_backbone:
            rgb[sl == CBD_VALUE] = COLOR_CBD
        return rgb

    img_real = to_rgb(real_sl, is_backbone=False)
    img_back = to_rgb(back_sl, is_backbone=True)
    img_recon = to_rgb(recon_sl, is_backbone=False)

    pad = 12
    h, w, _ = img_real.shape
    canvas = np.ones((h, 3 * w + 2 * pad, 3), dtype=np.uint8) * 255
    canvas[:, 0:w] = img_real
    canvas[:, w + pad:w + pad + w] = img_back
    canvas[:, 2 * w + 2 * pad:2 * w + 2 * pad + w] = img_recon

    Image.fromarray(canvas).save(out_path)


# ============================================================
# 几何函数
# ============================================================

def get_structuring_element() -> np.ndarray:
    return generate_binary_structure(rank=3, connectivity=1)


def get_interface_pore_mask(am_mask: np.ndarray, pore_mask: np.ndarray) -> np.ndarray:
    st = get_structuring_element()
    am_dil = binary_dilation(am_mask, structure=st, iterations=1)
    return pore_mask & am_dil


def compute_distance_to_am(am_mask: np.ndarray) -> np.ndarray:
    dist = distance_transform_edt(
        ~am_mask,
        sampling=(VOXEL_SIZE_Y, VOXEL_SIZE_Z, VOXEL_SIZE_X),
    )
    return dist.astype(np.float32)


def keep_only_cbd_touching_am(cbd_mask: np.ndarray, am_mask: np.ndarray) -> np.ndarray:
    labeled, num = label(cbd_mask)
    if num == 0:
        return cbd_mask

    st = get_structuring_element()
    am_dil = binary_dilation(am_mask, structure=st, iterations=1)

    keep = np.zeros_like(cbd_mask, dtype=bool)

    for comp_id in range(1, num + 1):
        comp = (labeled == comp_id)
        if np.any(comp & am_dil):
            keep |= comp

    return keep


def refill_to_target(
    current_cbd_mask: np.ndarray,
    candidate_mask: np.ndarray,
    rank_field: np.ndarray,
    target_voxels: int,
) -> np.ndarray:
    current_count = int(current_cbd_mask.sum())
    need = target_voxels - current_count
    if need <= 0:
        return current_cbd_mask

    remain = candidate_mask & (~current_cbd_mask)
    remain_idx = np.flatnonzero(remain)
    if len(remain_idx) == 0:
        return current_cbd_mask

    remain_scores = rank_field.ravel()[remain_idx]
    order = np.argsort(-remain_scores)
    chosen = remain_idx[order[:need]]

    out = current_cbd_mask.copy().ravel()
    out[chosen] = True
    return out.reshape(current_cbd_mask.shape)


# ============================================================
# 统计函数
# ============================================================

def compute_distance_pdf(
    dist_values: np.ndarray,
    bins: int = DIST_BINS,
    dmax: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(dist_values) == 0:
        raise ValueError("距离样本为空，无法计算 PDF")

    if dmax is None:
        dmax = float(dist_values.max())

    hist, edges = np.histogram(
        dist_values,
        bins=bins,
        range=(0.0, dmax),
        density=True,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers.astype(np.float32), hist.astype(np.float32)


def get_component_sizes(mask: np.ndarray, min_size: int = 1) -> np.ndarray:
    labeled, num = label(mask)
    if num == 0:
        return np.zeros((0,), dtype=np.int64)

    sizes = np.bincount(labeled.ravel())[1:]
    sizes = sizes[sizes >= min_size]
    return sizes.astype(np.int64)


def compute_cbd_connectivity_metrics(cbd_mask: np.ndarray) -> Dict[str, float]:
    sizes = get_component_sizes(cbd_mask, min_size=MIN_CBD_COMPONENT_SIZE)

    if len(sizes) == 0:
        return {
            "num_components": 0,
            "largest_component_ratio": 0.0,
            "mean_component_size": 0.0,
            "largest_component_size": 0.0,
        }

    total = int(sizes.sum())
    largest = int(sizes.max())

    return {
        "num_components": int(len(sizes)),
        "largest_component_ratio": float(largest / max(total, 1)),
        "mean_component_size": float(np.mean(sizes)),
        "largest_component_size": float(largest),
    }


def compute_interface_coverage_on_backbone(
    cbd_mask: np.ndarray,
    backbone_interface_pore_mask: np.ndarray,
) -> float:
    """
    在固定的 two-phase backbone interface 上，
    有多少原始界面 pore 位置被 CBD 覆盖或紧邻 CBD
    """
    if not np.any(backbone_interface_pore_mask):
        return 0.0

    st = get_structuring_element()
    cbd_dil = binary_dilation(cbd_mask, structure=st, iterations=1)
    covered = backbone_interface_pore_mask & cbd_dil
    return float(covered.sum() / max(backbone_interface_pore_mask.sum(), 1))


def compute_cbd_am_contact_ratio(cbd_mask: np.ndarray, am_mask: np.ndarray) -> float:
    """
    CBD 中有多少比例紧邻 AM
    """
    if not np.any(cbd_mask):
        return 0.0

    st = get_structuring_element()
    am_dil = binary_dilation(am_mask, structure=st, iterations=1)

    contact = cbd_mask & am_dil
    return float(contact.sum() / max(cbd_mask.sum(), 1))


def compute_block_cbd_fraction_distribution(
    cbd_mask: np.ndarray,
    block_size: int = BLOCK_SIZE,
    bins: int = BLOCK_BINS,
) -> np.ndarray:
    """
    关键修复：
    所有 real / gen 一律用固定 range=(0,1)
    因为 block fraction 本来就在 [0,1]，这样 histogram 可严格比较
    """
    cbd_field = cbd_mask.astype(np.float32)

    ys, zs, xs = cbd_field.shape
    vals = []

    for y in range(0, ys - block_size + 1, block_size):
        for z in range(0, zs - block_size + 1, block_size):
            for x in range(0, xs - block_size + 1, block_size):
                block = cbd_field[y:y + block_size, z:z + block_size, x:x + block_size]
                vals.append(float(block.mean()))

    vals = np.asarray(vals, dtype=np.float32)

    hist, _ = np.histogram(
        vals,
        bins=bins,
        range=(0.0, 1.0),
        density=True,
    )
    return hist.astype(np.float32)


def compute_cluster_size_histogram(
    cbd_mask: np.ndarray,
    bins: int = CLUSTER_BINS,
    min_size: int = MIN_CBD_COMPONENT_SIZE_FOR_CLUSTER_HIST,
) -> np.ndarray:
    """
    用 log10(component size) 的固定区间做 histogram，
    保证 real / gen cluster histogram 严格可比
    """
    sizes = get_component_sizes(cbd_mask, min_size=min_size)

    if len(sizes) == 0:
        hist = np.zeros((bins,), dtype=np.float32)
        return hist

    log_sizes = np.log10(sizes.astype(np.float64))

    hist, _ = np.histogram(
        log_sizes,
        bins=bins,
        range=(CLUSTER_LOG_MIN, CLUSTER_LOG_MAX),
        density=True,
    )
    return hist.astype(np.float32)


# ============================================================
# CBD 生成器
# ============================================================

def generate_cbd_phase(
    volume_2phase: np.ndarray,
    target_cbd_vol_frac: float,
    w_um: float,
) -> Tuple[np.ndarray, Dict]:
    rng = np.random.default_rng(SEED)

    volume = volume_2phase.astype(np.uint8)
    pore_mask = (volume == PORE_VALUE)
    am_mask = (volume == AM_VALUE)

    total_voxels = int(volume.size)
    target_cbd_voxels = int(round(target_cbd_vol_frac * total_voxels))

    if target_cbd_voxels <= 0:
        raise ValueError("target_cbd_vol_frac 太小，目标 CBD 体素数 <= 0")

    pore_voxels = int(np.count_nonzero(pore_mask))
    if target_cbd_voxels > pore_voxels:
        raise ValueError("目标 CBD 体积分数超过 pore 空间容量")

    interface_pore = get_interface_pore_mask(am_mask, pore_mask)
    if not np.any(interface_pore):
        raise RuntimeError("没有找到 AM/pore 界面，无法生长 CBD")

    dist_to_am = compute_distance_to_am(am_mask)

    sigma_vox = (
        max(w_um / VOXEL_SIZE_Y, 1e-6),
        max(w_um / VOXEL_SIZE_Z, 1e-6),
        max(w_um / VOXEL_SIZE_X, 1e-6),
    )

    seed_field = interface_pore.astype(np.float32)
    spread_field = gaussian_filter(seed_field, sigma=sigma_vox, mode="nearest")
    distance_decay = np.exp(-dist_to_am / max(w_um, 1e-8)).astype(np.float32)

    max_dist = MAX_GROWTH_DISTANCE_FACTOR * w_um
    candidate_mask = pore_mask & (dist_to_am <= max_dist)

    if int(candidate_mask.sum()) < target_cbd_voxels:
        candidate_mask = pore_mask

    noise = rng.uniform(0.0, 1e-6, size=volume.shape).astype(np.float32)
    rank_field = (spread_field * distance_decay + noise).astype(np.float32)

    candidate_idx = np.flatnonzero(candidate_mask)
    candidate_scores = rank_field.ravel()[candidate_idx]

    order = np.argsort(-candidate_scores)
    chosen_idx = candidate_idx[order[:target_cbd_voxels]]

    cbd_mask = np.zeros_like(volume, dtype=bool).ravel()
    cbd_mask[chosen_idx] = True
    cbd_mask = cbd_mask.reshape(volume.shape)

    if REMOVE_ISOLATED_CBD:
        cbd_mask = keep_only_cbd_touching_am(cbd_mask, am_mask)
        cbd_mask = refill_to_target(
            current_cbd_mask=cbd_mask,
            candidate_mask=candidate_mask,
            rank_field=rank_field,
            target_voxels=target_cbd_voxels,
        )

    volume_3phase = volume.copy()
    volume_3phase[cbd_mask] = CBD_VALUE

    summary = {
        "target_cbd_vol_frac": float(target_cbd_vol_frac),
        "actual_cbd_vol_frac": phase_fraction(volume_3phase, CBD_VALUE),
        "w_um": float(w_um),
    }

    return volume_3phase.astype(np.uint8), summary


# ============================================================
# 单个 w 的评估
# ============================================================
def evaluate_single_w(
    volume_2phase: np.ndarray,
    target_phi_cbd: float,
    w_um: float,
    real_dist_pdf: np.ndarray,
    real_dmax: float,
    real_conn: Dict[str, float],
    real_cover: float,
    real_contact_ratio: float,
    real_block_hist: np.ndarray,
    real_cluster_hist: np.ndarray,
    backbone_interface_pore_mask: np.ndarray,
) -> Dict:
    gen_3phase, gen_summary = generate_cbd_phase(
        volume_2phase=volume_2phase,
        target_cbd_vol_frac=target_phi_cbd,
        w_um=w_um,
    )

    am_mask = (gen_3phase == AM_VALUE)
    cbd_mask = (gen_3phase == CBD_VALUE)

    # 1) 距离分布误差
    dist_to_am = compute_distance_to_am(am_mask)
    d_gen = dist_to_am[cbd_mask]
    _, pdf_gen = compute_distance_pdf(
        d_gen,
        bins=DIST_BINS,
        dmax=real_dmax,
    )
    loss_dist = float(np.mean((real_dist_pdf - pdf_gen) ** 2))

    # 2) 连通性统计误差
    gen_conn = compute_cbd_connectivity_metrics(cbd_mask)
    loss_conn = (
        abs(gen_conn["largest_component_ratio"] - real_conn["largest_component_ratio"])
        + abs(gen_conn["num_components"] - real_conn["num_components"]) / max(real_conn["num_components"], 1)
        + abs(gen_conn["mean_component_size"] - real_conn["mean_component_size"]) / max(real_conn["mean_component_size"], 1e-6)
        + abs(gen_conn["largest_component_size"] - real_conn["largest_component_size"]) / max(real_conn["largest_component_size"], 1e-6)
    )
    loss_conn = float(loss_conn)

    # 3) 固定骨架界面上的 coverage 误差
    gen_cover = compute_interface_coverage_on_backbone(
        cbd_mask=cbd_mask,
        backbone_interface_pore_mask=backbone_interface_pore_mask,
    )
    loss_cover = float(abs(gen_cover - real_cover))

    # 4) CBD-AM 接触率误差
    gen_contact_ratio = compute_cbd_am_contact_ratio(cbd_mask, am_mask)
    loss_contact = float(abs(gen_contact_ratio - real_contact_ratio))

    # 5) block histogram 误差
    gen_block_hist = compute_block_cbd_fraction_distribution(
        cbd_mask=cbd_mask,
        block_size=BLOCK_SIZE,
        bins=BLOCK_BINS,
    )
    loss_block = float(np.mean((real_block_hist - gen_block_hist) ** 2))

    # 6) cluster size histogram 误差
    gen_cluster_hist = compute_cluster_size_histogram(
        cbd_mask=cbd_mask,
        bins=CLUSTER_BINS,
        min_size=MIN_CBD_COMPONENT_SIZE_FOR_CLUSTER_HIST,
    )
    loss_cluster = float(np.mean((real_cluster_hist - gen_cluster_hist) ** 2))

    total_loss = (
        ALPHA_DIST * loss_dist
        + BETA_CONN * loss_conn
        + GAMMA_COVER * loss_cover
        + DELTA_CONTACT * loss_contact
        + EPS_BLOCK * loss_block
        + ZETA_CLUSTER * loss_cluster
    )

    return {
        "w_um": float(w_um),
        "generated_3phase": gen_3phase,
        "generated_summary": gen_summary,
        "distance_pdf": pdf_gen,
        "connectivity": gen_conn,
        "interface_coverage": gen_cover,
        "contact_ratio": gen_contact_ratio,
        "block_hist": gen_block_hist,
        "cluster_hist": gen_cluster_hist,
        "loss_dist": loss_dist,
        "loss_conn": loss_conn,
        "loss_cover": loss_cover,
        "loss_contact": loss_contact,
        "loss_block": loss_block,
        "loss_cluster": loss_cluster,
        "total_loss": float(total_loss),
    }


# ============================================================
# 主程序
# ============================================================

def main():
    ensure_dir(OUT_DIR)

    print("=" * 80)
    print("Loading real 3-phase slices ...")
    real_3phase = load_volume_from_slices(REAL_3PHASE_SLICE_DIR)
    print("Real shape:", real_3phase.shape)
    print("Unique labels:", np.unique(real_3phase))
    print("=" * 80)

    # 两相骨架：把 CBD 去掉后，位置视为 pore
    volume_2phase = real_3phase.copy()
    volume_2phase[volume_2phase == CBD_VALUE] = PORE_VALUE

    # 基于同一个 backbone 定义固定界面
    backbone_am_mask = (volume_2phase == AM_VALUE)
    backbone_pore_mask = (volume_2phase == PORE_VALUE)
    backbone_interface_pore_mask = get_interface_pore_mask(
        backbone_am_mask,
        backbone_pore_mask,
    )

    # 真实 CBD 体积分数
    phi_cbd_real = phase_fraction(real_3phase, CBD_VALUE)

    # 真实 CBD 距离分布
    real_am_mask = (real_3phase == AM_VALUE)
    real_cbd_mask = (real_3phase == CBD_VALUE)
    dist_to_am_real = compute_distance_to_am(real_am_mask)
    d_real = dist_to_am_real[real_cbd_mask]

    if len(d_real) == 0:
        raise RuntimeError("真实结构中没有 CBD 体素，无法拟合 w")

    real_dmax = float(d_real.max())
    dist_x, pdf_real = compute_distance_pdf(
        d_real,
        bins=DIST_BINS,
        dmax=real_dmax,
    )

    # 真实连通性
    real_conn = compute_cbd_connectivity_metrics(real_cbd_mask)

    # 真实 coverage：也必须基于同一个 backbone interface 来计算
    real_cover = compute_interface_coverage_on_backbone(
        cbd_mask=real_cbd_mask,
        backbone_interface_pore_mask=backbone_interface_pore_mask,
    )

    # 真实 CBD-AM 接触率
    real_contact_ratio = compute_cbd_am_contact_ratio(
        cbd_mask=real_cbd_mask,
        am_mask=real_am_mask,
    )

    # 真实 block histogram
    real_block_hist = compute_block_cbd_fraction_distribution(
        cbd_mask=real_cbd_mask,
        block_size=BLOCK_SIZE,
        bins=BLOCK_BINS,
    )

    # 真实 cluster histogram
    real_cluster_hist = compute_cluster_size_histogram(
        cbd_mask=real_cbd_mask,
        bins=CLUSTER_BINS,
        min_size=MIN_CBD_COMPONENT_SIZE_FOR_CLUSTER_HIST,
    )

    print("Real CBD volume fraction:", phi_cbd_real)
    print("Real CBD connectivity:", real_conn)
    print("Real CBD interface coverage on backbone:", real_cover)
    print("Real CBD-AM contact ratio:", real_contact_ratio)

    # 扫描 w
    w_candidates = np.linspace(W_MIN, W_MAX, NUM_W)
    results: List[Dict] = []

    print("=" * 80)
    print("Scanning w candidates ...")
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

        tqdm.write(
            f"w={w:.4f} | "
            f"loss={res['total_loss']:.6f} | "
            f"dist={res['loss_dist']:.6f} | "
            f"conn={res['loss_conn']:.6f} | "
            f"cover={res['loss_cover']:.6f} | "
            f"contact={res['loss_contact']:.6f} | "
            f"block={res['loss_block']:.6f} | "
            f"cluster={res['loss_cluster']:.6f}"
        )

    # 找最优 w
    losses = np.array([r["total_loss"] for r in results], dtype=np.float64)
    best_idx = int(np.argmin(losses))
    best = results[best_idx]
    best_w = float(best["w_um"])

    print("=" * 80)
    print("Best w found:", best_w)
    print("Best total loss:", best["total_loss"])
    print("=" * 80)

    # 保存最优三相重建
    best_3phase = best["generated_3phase"]
    np.save(os.path.join(OUT_DIR, "best_w_reconstructed_3phase.npy"), best_3phase)

    # 保存 summary
    summary = {
        "real_3phase_slice_dir": REAL_3PHASE_SLICE_DIR,
        "real_phase_fractions": {
            "pore": phase_fraction(real_3phase, PORE_VALUE),
            "AM": phase_fraction(real_3phase, AM_VALUE),
            "CBD": phase_fraction(real_3phase, CBD_VALUE),
        },
        "real_connectivity": real_conn,
        "real_interface_coverage_on_backbone": real_cover,
        "real_contact_ratio": real_contact_ratio,
        "scan_config": {
            "W_MIN": W_MIN,
            "W_MAX": W_MAX,
            "NUM_W": NUM_W,
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
            "MIN_CBD_COMPONENT_SIZE_FOR_CLUSTER_HIST": MIN_CBD_COMPONENT_SIZE_FOR_CLUSTER_HIST,
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
                "generated_interface_coverage_on_backbone": float(r["interface_coverage"]),
                "generated_contact_ratio": float(r["contact_ratio"]),
                "generated_phi_cbd": float(r["generated_summary"]["actual_cbd_vol_frac"]),
            }
            for r in results
        ],
    }
    save_json(summary, os.path.join(OUT_DIR, "w_fitting_summary.json"))

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
    plt.axvline(best_w, linestyle="--", linewidth=1.2, label=f"Best w = {best_w:.3f} um")
    plt.xlabel("w (um)")
    plt.ylabel("Total fitting loss")
    plt.title("CBD spreading parameter fitting")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "w_scan_total_loss_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ========================================================
    # 图 2：真实 vs 最优 w 的距离分布
    # ========================================================
    fig = plt.figure(figsize=(5.8, 4.5))
    plt.plot(dist_x, pdf_real, linewidth=2.0, label="Real CBD")
    plt.plot(dist_x, best["distance_pdf"], linewidth=2.0, label=f"Generated CBD (w={best_w:.3f} um)")
    plt.xlabel("Distance to AM surface (um)")
    plt.ylabel("Probability density")
    plt.title("CBD distance-to-AM distribution")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "distance_distribution_real_vs_best.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ========================================================
    # 图 3：loss 分量 vs w
    # ========================================================
    fig = plt.figure(figsize=(6.4, 5.0))
    plt.plot([r["w_um"] for r in results], [r["loss_dist"] for r in results], marker="o", label="Distance loss")
    plt.plot([r["w_um"] for r in results], [r["loss_conn"] for r in results], marker="o", label="Connectivity loss")
    plt.plot([r["w_um"] for r in results], [r["loss_cover"] for r in results], marker="o", label="Coverage loss")
    plt.plot([r["w_um"] for r in results], [r["loss_contact"] for r in results], marker="o", label="Contact loss")
    plt.plot([r["w_um"] for r in results], [r["loss_block"] for r in results], marker="o", label="Block loss")
    plt.plot([r["w_um"] for r in results], [r["loss_cluster"] for r in results], marker="o", label="Cluster loss")
    plt.axvline(best_w, linestyle="--", linewidth=1.2, color="black")
    plt.xlabel("w (um)")
    plt.ylabel("Loss component")
    plt.title("Loss components during w fitting")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_components_vs_w.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ========================================================
    # 图 4：真实 vs 两相骨架 vs 最优重建
    # ========================================================
    save_side_by_side_slices(
        real_3phase=real_3phase,
        backbone_2phase=volume_2phase,
        recon_3phase=best_3phase,
        out_path=os.path.join(OUT_DIR, "real_vs_backbone_vs_reconstructed.png"),
    )

    # ========================================================
    # 图 5：block 分布对比
    # ========================================================
    fig = plt.figure(figsize=(5.8, 4.5))
    x_block = np.arange(len(real_block_hist))
    plt.plot(x_block, real_block_hist, linewidth=2.0, label="Real CBD")
    plt.plot(x_block, best["block_hist"], linewidth=2.0, label=f"Generated CBD (w={best_w:.3f} um)")
    plt.xlabel("Block histogram bin")
    plt.ylabel("Probability density")
    plt.title("Block-wise CBD fraction distribution")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "block_hist_real_vs_best.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ========================================================
    # 图 6：coverage 和 contact vs w
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
    plt.axhline(real_cover, linestyle="--", linewidth=1.2, label="Real interface coverage on backbone")
    plt.axhline(real_contact_ratio, linestyle="--", linewidth=1.2, label="Real CBD-AM contact ratio")
    plt.axvline(best_w, linestyle="--", linewidth=1.2, color="black")
    plt.xlabel("w (um)")
    plt.ylabel("Metric value")
    plt.title("Interface-related metrics during w fitting")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "contact_ratio_and_cover_vs_w.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ========================================================
    # 图 7：cluster histogram 对比
    # ========================================================
    fig = plt.figure(figsize=(5.8, 4.5))
    x_cluster = np.arange(len(real_cluster_hist))
    plt.plot(x_cluster, real_cluster_hist, linewidth=2.0, label="Real CBD")
    plt.plot(x_cluster, best["cluster_hist"], linewidth=2.0, label=f"Generated CBD (w={best_w:.3f} um)")
    plt.xlabel("Cluster histogram bin")
    plt.ylabel("Probability density")
    plt.title("CBD cluster-size distribution")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cluster_hist_real_vs_best.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ========================================================
    # 图 8：加权损失分量 vs w
    # 这个图很适合论文分析谁在主导最优点
    # ========================================================
    fig = plt.figure(figsize=(6.4, 5.0))
    plt.plot([r["w_um"] for r in results], [ALPHA_DIST * r["loss_dist"] for r in results], marker="o", label="Weighted distance")
    plt.plot([r["w_um"] for r in results], [BETA_CONN * r["loss_conn"] for r in results], marker="o", label="Weighted connectivity")
    plt.plot([r["w_um"] for r in results], [GAMMA_COVER * r["loss_cover"] for r in results], marker="o", label="Weighted coverage")
    plt.plot([r["w_um"] for r in results], [DELTA_CONTACT * r["loss_contact"] for r in results], marker="o", label="Weighted contact")
    plt.plot([r["w_um"] for r in results], [EPS_BLOCK * r["loss_block"] for r in results], marker="o", label="Weighted block")
    plt.plot([r["w_um"] for r in results], [ZETA_CLUSTER * r["loss_cluster"] for r in results], marker="o", label="Weighted cluster")
    plt.axvline(best_w, linestyle="--", linewidth=1.2, color="black")
    plt.xlabel("w (um)")
    plt.ylabel("Weighted loss contribution")
    plt.title("Weighted loss contributions during w fitting")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "weighted_loss_components_vs_w.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("Done.")
    print("Best w:", best_w)
    print("Outputs saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
