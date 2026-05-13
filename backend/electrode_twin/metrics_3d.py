# metrics_3d.py
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 这里是需要修改的参数区域
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ELECTRODE_TWIN_DIR = os.path.join(BASE_DIR, "electrode_twin")

INPUT_VOLUME_PATH = os.path.join(
    ELECTRODE_TWIN_DIR,
    "generated_results",
    "run_007",
    "best_sample",
    "generated_bin_final.npy"
)
# 1. 这里改：真实体数据路径（得换成 128*128*128 结构）！！
# 一般填 reconstruct_volume_from_ckpt.py 输出的 original_volume.npy
REAL_VOLUME_PATH = os.path.join(
    ELECTRODE_TWIN_DIR,
    "reconstruction_results",
    "original_volume.npy"
)
# REAL_VOLUME_PATH = r"reconstruction_results\original_volume.npy"

# 2. 这里改：重建二值体路径
# 一般填 reconstruct_volume_from_ckpt.py 输出的 recon_bin_volume.npy
RECON_VOLUME_PATH = os.path.join(
    ELECTRODE_TWIN_DIR,
    "reconstruction_results",
    "recon_bin_volume.npy"
)
# RECON_VOLUME_PATH = r"reconstruction_results\recon_bin_volume.npy"

# 3. 这里改：输出文件夹
OUT_DIR = os.path.join(
    ELECTRODE_TWIN_DIR,
    "metrics_results"
)
# OUT_DIR = r"metrics_results"

# 4. True: 统计孔隙相；False: 统计固相
USE_PORE_PHASE_ONLY = True

# 5. 孔隙相数值
# 你当前数据里 solid=1, pore=0，所以这里填 0
PORE_VALUE = 0

# 6. 是否移除特别小的连通域
REMOVE_SMALL_COMPONENTS = True

# 7. 小连通域最小体素数
# 小于这个体素数的连通域会被删掉
MIN_COMPONENT_SIZE = 10

# 8. 是否只在比表面积计算时清理小连通域
CLEAN_ONLY_FOR_SURFACE = True

# 9. 两点相关函数最大距离
MAX_LAG = 120

# 10. 线性路径函数最大长度
MAX_PATH_LENGTH = 120

#添加真实物理分辨率（单位：um），体积的顺序是[Y,Z,X]
VOXEL_SIZE_Y = 0.0315
VOXEL_SIZE_Z = 0.02791
VOXEL_SIZE_X = 0.02791


# ============================================================
# 配置
# ============================================================

@dataclass
class MetricsConfig:
    real_volume_path: str = REAL_VOLUME_PATH
    recon_volume_path: str = RECON_VOLUME_PATH
    out_dir: str = OUT_DIR

    use_pore_phase_only: bool = USE_PORE_PHASE_ONLY
    pore_value: int = PORE_VALUE

    remove_small_components: bool = REMOVE_SMALL_COMPONENTS
    min_component_size: int = MIN_COMPONENT_SIZE
    clean_only_for_surface: bool = CLEAN_ONLY_FOR_SURFACE

    max_lag: int = MAX_LAG
    max_path_length: int = MAX_PATH_LENGTH

    voxel_size_y: float = VOXEL_SIZE_Y
    voxel_size_z: float = VOXEL_SIZE_Z
    voxel_size_x: float = VOXEL_SIZE_X


# ============================================================
# 基础工具
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_volume(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    vol = np.load(path)

    if vol.ndim != 3:
        raise ValueError(f"体数据应为 3D [Y,Z,X]，当前 shape={vol.shape}")

    return vol


def to_phase_mask(volume: np.ndarray, use_pore_phase_only: bool, pore_value: int) -> np.ndarray:
    if use_pore_phase_only:
        return (volume == pore_value).astype(np.uint8)
    else:
        return (volume != pore_value).astype(np.uint8)


# ============================================================
# 小连通域清理
# ============================================================

def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1:
        return mask.copy()

    visited = np.zeros_like(mask, dtype=np.uint8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)

    def neighbors(y: int, z: int, x: int):
        if y > 0:
            yield y - 1, z, x
        if y < mask.shape[0] - 1:
            yield y + 1, z, x
        if z > 0:
            yield y, z - 1, x
        if z < mask.shape[1] - 1:
            yield y, z + 1, x
        if x > 0:
            yield y, z, x - 1
        if x < mask.shape[2] - 1:
            yield y, z, x + 1

    coords = np.argwhere(mask == 1)

    for y, z, x in coords:
        y, z, x = int(y), int(z), int(x)

        if visited[y, z, x]:
            continue

        stack = [(y, z, x)]
        visited[y, z, x] = 1
        component = []

        while stack:
            cy, cz, cx = stack.pop()
            component.append((cy, cz, cx))

            for ny, nz, nx in neighbors(cy, cz, cx):
                if mask[ny, nz, nx] == 1 and visited[ny, nz, nx] == 0:
                    visited[ny, nz, nx] = 1
                    stack.append((ny, nz, nx))

        if len(component) >= min_size:
            for cy, cz, cx in component:
                cleaned[cy, cz, cx] = 1

    return cleaned


# ============================================================
# 1. 体积分数
# ============================================================

def volume_fraction(mask: np.ndarray) -> float:
    return float(mask.mean())


# ============================================================
# 2. 最大连通簇占比
# ============================================================

def largest_connected_component_ratio(mask: np.ndarray) -> float:
    coords = np.argwhere(mask == 1)
    total_count = len(coords)

    if total_count == 0:
        return 0.0

    visited = np.zeros_like(mask, dtype=np.uint8)

    def neighbors(y: int, z: int, x: int):
        if y > 0:
            yield y - 1, z, x
        if y < mask.shape[0] - 1:
            yield y + 1, z, x
        if z > 0:
            yield y, z - 1, x
        if z < mask.shape[1] - 1:
            yield y, z + 1, x
        if x > 0:
            yield y, z, x - 1
        if x < mask.shape[2] - 1:
            yield y, z, x + 1

    max_size = 0

    for y, z, x in coords:
        y, z, x = int(y), int(z), int(x)

        if visited[y, z, x]:
            continue

        stack = [(y, z, x)]
        visited[y, z, x] = 1
        comp_size = 0

        while stack:
            cy, cz, cx = stack.pop()
            comp_size += 1

            for ny, nz, nx in neighbors(cy, cz, cx):
                if mask[ny, nz, nx] == 1 and visited[ny, nz, nx] == 0:
                    visited[ny, nz, nx] = 1
                    stack.append((ny, nz, nx))

        if comp_size > max_size:
            max_size = comp_size

    return float(max_size) / float(total_count)


# ============================================================
# 3. 比表面积近似（真实物理尺寸）
# ============================================================

def specific_surface_area_physical(mask: np.ndarray, cfg: MetricsConfig) -> float:
    """
    用相邻体素相变界面近似比表面积（考虑真实物理分辨率）
    返回单位：1 / um
    """
    y_diff = np.abs(mask[1:, :, :] - mask[:-1, :, :]).sum()
    z_diff = np.abs(mask[:, 1:, :] - mask[:, :-1, :]).sum()
    x_diff = np.abs(mask[:, :, 1:] - mask[:, :, :-1]).sum()

    area_y = float(y_diff) * (cfg.voxel_size_z * cfg.voxel_size_x)  # ZX面
    area_z = float(z_diff) * (cfg.voxel_size_y * cfg.voxel_size_x)  # YX面
    area_x = float(x_diff) * (cfg.voxel_size_y * cfg.voxel_size_z)  # YZ面

    total_interface_area = area_y + area_z + area_x  # um^2

    total_volume = (
        mask.shape[0] * cfg.voxel_size_y *
        mask.shape[1] * cfg.voxel_size_z *
        mask.shape[2] * cfg.voxel_size_x
    )  # um^3

    return total_interface_area / max(total_volume, 1e-12)


# ============================================================
# 4. 两点相关函数
# ============================================================

def two_point_correlation_along_axis(mask: np.ndarray, axis: int, max_lag: int) -> np.ndarray:
    values = []

    for r in range(max_lag + 1):
        if r == 0:
            values.append(float((mask * mask).mean()))
            continue

        if axis == 0:
            a = mask[:-r, :, :]
            b = mask[r:, :, :]
        elif axis == 1:
            a = mask[:, :-r, :]
            b = mask[:, r:, :]
        elif axis == 2:
            a = mask[:, :, :-r]
            b = mask[:, :, r:]
        else:
            raise ValueError("axis must be 0/1/2")

        values.append(float((a * b).mean()))

    return np.asarray(values, dtype=np.float64)


# ============================================================
# 5. 线性路径函数
# ============================================================

def linear_path_function_along_axis(mask: np.ndarray, axis: int, max_len: int) -> np.ndarray:
    values = []

    for r in range(1, max_len + 1):
        valid_count = 0
        pass_count = 0

        if axis == 0:
            for y0 in range(mask.shape[0] - r + 1):
                seg = mask[y0:y0 + r, :, :]
                all_phase = np.all(seg == 1, axis=0)
                pass_count += int(all_phase.sum())
                valid_count += all_phase.size

        elif axis == 1:
            for z0 in range(mask.shape[1] - r + 1):
                seg = mask[:, z0:z0 + r, :]
                all_phase = np.all(seg == 1, axis=1)
                pass_count += int(all_phase.sum())
                valid_count += all_phase.size

        elif axis == 2:
            for x0 in range(mask.shape[2] - r + 1):
                seg = mask[:, :, x0:x0 + r]
                all_phase = np.all(seg == 1, axis=2)
                pass_count += int(all_phase.sum())
                valid_count += all_phase.size

        else:
            raise ValueError("axis must be 0/1/2")

        values.append(pass_count / max(valid_count, 1))

    return np.asarray(values, dtype=np.float64)


# ============================================================
# 6. 计算全部指标
# ============================================================

def compute_all_metrics(volume: np.ndarray, cfg: MetricsConfig) -> dict:
    raw_mask = to_phase_mask(
        volume=volume,
        use_pore_phase_only=cfg.use_pore_phase_only,
        pore_value=cfg.pore_value,
    )

    metrics_mask = raw_mask
    metrics = {}

    metrics["volume_fraction"] = volume_fraction(metrics_mask)
    metrics["largest_connected_component_ratio"] = largest_connected_component_ratio(metrics_mask)

    if cfg.remove_small_components and cfg.clean_only_for_surface:
        surface_mask = remove_small_components(raw_mask, cfg.min_component_size)
    elif cfg.remove_small_components and not cfg.clean_only_for_surface:
        surface_mask = remove_small_components(raw_mask, cfg.min_component_size)
        metrics_mask = surface_mask
    else:
        surface_mask = raw_mask

    metrics["specific_surface_area_physical"] = specific_surface_area_physical(surface_mask, cfg)

    metrics["two_point_corr_y"] = two_point_correlation_along_axis(metrics_mask, axis=0, max_lag=cfg.max_lag)
    metrics["two_point_corr_z"] = two_point_correlation_along_axis(metrics_mask, axis=1, max_lag=cfg.max_lag)
    metrics["two_point_corr_x"] = two_point_correlation_along_axis(metrics_mask, axis=2, max_lag=cfg.max_lag)

    metrics["linear_path_y"] = linear_path_function_along_axis(metrics_mask, axis=0, max_len=cfg.max_path_length)
    metrics["linear_path_z"] = linear_path_function_along_axis(metrics_mask, axis=1, max_len=cfg.max_path_length)
    metrics["linear_path_x"] = linear_path_function_along_axis(metrics_mask, axis=2, max_len=cfg.max_path_length)

    return metrics


# ============================================================
# 7. 保存总报告
# ============================================================

def save_scalar_report(real_metrics: dict, recon_metrics: dict, cfg: MetricsConfig):
    report_path = os.path.join(cfg.out_dir, "metrics_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("3D Electrode Metrics Report\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"REAL_VOLUME_PATH         : {cfg.real_volume_path}\n")
        f.write(f"RECON_VOLUME_PATH        : {cfg.recon_volume_path}\n")
        f.write(f"USE_PORE_PHASE_ONLY      : {cfg.use_pore_phase_only}\n")
        f.write(f"PORE_VALUE               : {cfg.pore_value}\n")
        f.write(f"REMOVE_SMALL_COMPONENTS  : {cfg.remove_small_components}\n")
        f.write(f"MIN_COMPONENT_SIZE       : {cfg.min_component_size}\n")
        f.write(f"CLEAN_ONLY_FOR_SURFACE   : {cfg.clean_only_for_surface}\n")
        f.write(f"MAX_LAG                  : {cfg.max_lag}\n")
        f.write(f"MAX_PATH_LENGTH          : {cfg.max_path_length}\n")
        f.write(f"VOXEL_SIZE_Y             : {cfg.voxel_size_y} um\n")
        f.write(f"VOXEL_SIZE_Z             : {cfg.voxel_size_z} um\n")
        f.write(f"VOXEL_SIZE_X             : {cfg.voxel_size_x} um\n\n")

        scalar_keys = [
            "volume_fraction",
            "largest_connected_component_ratio",
            "specific_surface_area_physical",
        ]

        for k in scalar_keys:
            rv = float(real_metrics[k])
            gv = float(recon_metrics[k])
            diff = gv - rv
            rel = diff / (abs(rv) + 1e-12)

            f.write(f"{k}\n")
            f.write(f"  real   : {rv:.8f}\n")
            f.write(f"  recon  : {gv:.8f}\n")
            f.write(f"  diff   : {diff:.8f}\n")
            f.write(f"  relerr : {rel:.8%}\n\n")


# ============================================================
# 8. 保存 CSV
# ============================================================

def save_two_point_csv(real_metrics: dict, recon_metrics: dict, out_dir: str):
    lag = np.arange(len(real_metrics["two_point_corr_x"]))

    table = np.stack([
        lag,
        real_metrics["two_point_corr_y"],
        recon_metrics["two_point_corr_y"],
        real_metrics["two_point_corr_z"],
        recon_metrics["two_point_corr_z"],
        real_metrics["two_point_corr_x"],
        recon_metrics["two_point_corr_x"],
    ], axis=1)

    out_path = os.path.join(out_dir, "two_point_correlation.csv")
    np.savetxt(
        out_path,
        table,
        delimiter=",",
        header="lag,real_y,recon_y,real_z,recon_z,real_x,recon_x",
        comments="",
    )


def save_linear_path_csv(real_metrics: dict, recon_metrics: dict, out_dir: str):
    length = np.arange(1, len(real_metrics["linear_path_x"]) + 1)

    table = np.stack([
        length,
        real_metrics["linear_path_y"],
        recon_metrics["linear_path_y"],
        real_metrics["linear_path_z"],
        recon_metrics["linear_path_z"],
        real_metrics["linear_path_x"],
        recon_metrics["linear_path_x"],
    ], axis=1)

    out_path = os.path.join(out_dir, "linear_path_function.csv")
    np.savetxt(
        out_path,
        table,
        delimiter=",",
        header="length,real_y,recon_y,real_z,recon_z,real_x,recon_x",
        comments="",
    )


# ============================================================
# 9. 画图
# ============================================================

def plot_two_point_correlation(real_metrics: dict, recon_metrics: dict, out_dir: str):
    lag = np.arange(len(real_metrics["two_point_corr_x"]))

    plt.figure(figsize=(8, 6))
    plt.plot(lag, real_metrics["two_point_corr_y"], label="Real-Y")
    plt.plot(lag, recon_metrics["two_point_corr_y"], label="Recon-Y")
    plt.plot(lag, real_metrics["two_point_corr_z"], label="Real-Z")
    plt.plot(lag, recon_metrics["two_point_corr_z"], label="Recon-Z")
    plt.plot(lag, real_metrics["two_point_corr_x"], label="Real-X")
    plt.plot(lag, recon_metrics["two_point_corr_x"], label="Recon-X")
    plt.xlabel("Lag")
    plt.ylabel("Two-point correlation")
    plt.title("Two-point Correlation Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "two_point_correlation.tif"), dpi=300, format="tif")
    plt.close()


def plot_linear_path_function(real_metrics: dict, recon_metrics: dict, out_dir: str):
    length = np.arange(1, len(real_metrics["linear_path_x"]) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(length, real_metrics["linear_path_y"], label="Real-Y")
    plt.plot(length, recon_metrics["linear_path_y"], label="Recon-Y")
    plt.plot(length, real_metrics["linear_path_z"], label="Real-Z")
    plt.plot(length, recon_metrics["linear_path_z"], label="Recon-Z")
    plt.plot(length, real_metrics["linear_path_x"], label="Real-X")
    plt.plot(length, recon_metrics["linear_path_x"], label="Recon-X")
    plt.xlabel("Path length")
    plt.ylabel("Linear path probability")
    plt.title("Linear Path Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "linear_path_function.tif"), dpi=300, format="tiff")
    plt.close()


def plot_scalar_comparison(real_metrics: dict, recon_metrics: dict, cfg: MetricsConfig):
    """
    双Y轴画标量指标：
    - 左轴：体积分数、最大连通簇占比
    - 右轴：比表面积
    """
    if cfg.use_pore_phase_only:
        left_names = [
            "Pore fraction",
            "Largest pore cluster ratio",
        ]
    else:
        left_names = [
            "Solid fraction",
            "Largest solid cluster ratio",
        ]

    right_name = "Specific surface area"

    left_real = [
        real_metrics["volume_fraction"],
        real_metrics["largest_connected_component_ratio"],
    ]
    left_recon = [
        recon_metrics["volume_fraction"],
        recon_metrics["largest_connected_component_ratio"],
    ]

    right_real = real_metrics["specific_surface_area_physical"]
    right_recon = recon_metrics["specific_surface_area_physical"]

    # x 位置
    x_left = np.array([0, 1], dtype=np.float64)
    x_right = np.array([2], dtype=np.float64)

    width = 0.32

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    # 左轴柱子：孔隙率 / 连通占比
    bars1 = ax1.bar(x_left - width / 2, left_real, width, label="Real", color="#1f77b4")
    bars2 = ax1.bar(x_left + width / 2, left_recon, width, label="Recon", color="#ff7f0e")

    # 右轴柱子：比表面积
    bars3 = ax2.bar(x_right - width / 2, [right_real], width, color="#1f77b4", alpha=0.85)
    bars4 = ax2.bar(x_right + width / 2, [right_recon], width, color="#ff7f0e", alpha=0.85)

    # x轴标签
    xticks = np.concatenate([x_left, x_right])
    xticklabels = left_names + [right_name]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels, rotation=15)

    # 左右轴标签
    ax1.set_ylabel("Fraction / Connectivity")
    ax2.set_ylabel("Specific surface area (1/um)")

    ax1.set_title("Scalar Metrics Comparison")

    # 左轴范围稍微留白
    left_all = left_real + left_recon
    left_min = min(left_all)
    left_max = max(left_all)
    left_margin = max(0.02, 0.1 * (left_max - left_min + 1e-8))
    ax1.set_ylim(max(0.0, left_min - left_margin), min(1.1, left_max + left_margin))

    # 右轴范围留白
    right_all = [right_real, right_recon]
    right_min = min(right_all)
    right_max = max(right_all)
    right_margin = max(20.0, 0.1 * (right_max - right_min + 1e-8))
    ax2.set_ylim(0,1200)

    #ax2.set_ylim(max(0.0, right_min - right_margin), right_max + right_margin)

    # 网格只画左轴
    ax1.grid(True, axis="y", alpha=0.3)

    # 图例只保留一份
    ax1.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "scalar_metrics_comparison.tif"), dpi=300, format="tiff")
    plt.close()


# ============================================================
# 10. 主函数
# ============================================================

def main():
    cfg = MetricsConfig()
    ensure_dir(cfg.out_dir)

    print("=" * 60)
    print("开始计算 3D 结构指标")
    print(f"REAL_VOLUME_PATH        : {cfg.real_volume_path}")
    print(f"RECON_VOLUME_PATH       : {cfg.recon_volume_path}")
    print(f"OUT_DIR                 : {cfg.out_dir}")
    print(f"USE_PORE_PHASE_ONLY     : {cfg.use_pore_phase_only}")
    print(f"PORE_VALUE              : {cfg.pore_value}")
    print(f"REMOVE_SMALL_COMPONENTS : {cfg.remove_small_components}")
    print(f"MIN_COMPONENT_SIZE      : {cfg.min_component_size}")
    print(f"CLEAN_ONLY_FOR_SURFACE  : {cfg.clean_only_for_surface}")
    print(f"MAX_LAG                 : {cfg.max_lag}")
    print(f"MAX_PATH_LENGTH         : {cfg.max_path_length}")
    print("=" * 60)

    print("1) 读取真实体数据 ...")
    real_vol = load_volume(cfg.real_volume_path)

    print("2) 读取重建体数据 ...")
    recon_vol = load_volume(cfg.recon_volume_path)

    if real_vol.shape != recon_vol.shape:
        raise ValueError(f"shape 不一致: real={real_vol.shape}, recon={recon_vol.shape}")

    print("3) 计算真实体指标 ...")
    real_metrics = compute_all_metrics(real_vol, cfg)

    print("4) 计算重建体指标 ...")
    original_clean_for_surface = cfg.clean_only_for_surface
    cfg.clean_only_for_surface = False
    recon_metrics = compute_all_metrics(recon_vol, cfg)
    cfg.clean_only_for_surface = original_clean_for_surface

    print("5) 保存总报告 ...")
    save_scalar_report(real_metrics, recon_metrics, cfg)

    print("6) 保存两点相关函数 CSV ...")
    save_two_point_csv(real_metrics, recon_metrics, cfg.out_dir)

    print("7) 保存线性路径函数 CSV ...")
    save_linear_path_csv(real_metrics, recon_metrics, cfg.out_dir)

    print("8) 画图 ...")
    plot_two_point_correlation(real_metrics, recon_metrics, cfg.out_dir)
    plot_linear_path_function(real_metrics, recon_metrics, cfg.out_dir)
    plot_scalar_comparison(real_metrics, recon_metrics, cfg)

    print("9) 控制台打印核心指标 ...")
    if cfg.use_pore_phase_only:
        print(f"真实孔隙率: {real_metrics['volume_fraction']:.6f}")
        print(f"重建孔隙率: {recon_metrics['volume_fraction']:.6f}")
        print(f"真实最大连通孔隙占比: {real_metrics['largest_connected_component_ratio']:.6f}")
        print(f"重建最大连通孔隙占比: {recon_metrics['largest_connected_component_ratio']:.6f}")
    else:
        print(f"真实固相率: {real_metrics['volume_fraction']:.6f}")
        print(f"重建固相率: {recon_metrics['volume_fraction']:.6f}")
        print(f"真实最大连通固相占比: {real_metrics['largest_connected_component_ratio']:.6f}")
        print(f"重建最大连通固相占比: {recon_metrics['largest_connected_component_ratio']:.6f}")

    print(f"真实比表面积近似(1/um): {real_metrics['specific_surface_area_physical']:.6f}")
    print(f"重建比表面积近似(1/um): {recon_metrics['specific_surface_area_physical']:.6f}")

    print("完成，结果保存在:", cfg.out_dir)


if __name__ == "__main__":
    main()
