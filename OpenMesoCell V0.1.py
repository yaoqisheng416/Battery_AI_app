"""
OpenMesoCell V0.9
制造参数驱动的电池电极三维结构生成器 +  voxelization mesher

1. segmented image stack / label volume -> nodes
2. precision 参数控制各轴降采样，即每隔多少个体素保留一个节点
3. 相邻两层的 4+4 个节点形成一个 voxel/cubic element
4. 每个 voxel 分解为 5 或 6 个四面体
5. 每个 CTETRA 通过局部节点相标签多数投票获得 PID
6. 输出 COMSOL 可导入的 NASTRAN/BDF 文件


安装：
    conda create -n openmesocell python=3.10 -y
    conda activate openmesocell
    pip install numpy scipy matplotlib tifffile PySide6 scikit-image pyvista pyvistaqt openpyxl

运行：
    python openmesocell_v09.py
"""

import os
import sys
import math
import csv
import json
import socket
import time
import subprocess
import multiprocessing
import numpy as np
from scipy import ndimage as ndi
import tifffile

# ============================================
# 单实例锁 — 防止重复启动
# 锁 socket 会在整个应用生命周期内保持存活
# ============================================

LOCK_PORT = 54321


def acquire_lock():
    """获取单实例锁，成功返回 socket（调用方必须保持引用），失败返回 None"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", LOCK_PORT))
        return s
    except socket.error:
        return None


from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QSlider, QTextEdit, QGroupBox, QMessageBox, QCheckBox, QDialog, QScrollArea,
    QStackedWidget, QTabWidget, QProgressBar
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

try:
    import openpyxl
    HAS_OPENPYXL = True
except Exception:
    HAS_OPENPYXL = False

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    HAS_3D = True
except Exception:
    HAS_3D = False

try:
    from skimage import measure
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False


# ============================================================
# Phase definitions
# ============================================================
PHASE = {
    "pore": 0,
    "AM": 1,
    "CBD": 2,
    "separator": 3,
    "current_collector": 4,
    "li_foil": 5,
}

PHASE_NAME = {
    0: "Pore / Electrolyte",
    1: "Active Material",
    2: "CBD",
    3: "Separator",
    4: "Current Collector",
    5: "Li Foil",
}

PHASE_SHORT_NAME = {
    0: "Pore",
    1: "AM",
    2: "CBD",
    3: "Sep",
    4: "CC",
    5: "Li",
}

PHASE_COLOR = {
    0: "#214ee8",
    1: "#5bcf72",
    2: "#f5ec1b",
    3: "#b83ee6",
    4: "#f4a333",
    5: "#9db7d5",
}

CMAP = ListedColormap([PHASE_COLOR[i] for i in range(6)])
NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], CMAP.N)


# ============================================================
# Manufacturing formulas
# ============================================================
def normalize_weights(am_wt, carbon_wt, binder_wt):
    total = am_wt + carbon_wt + binder_wt
    if total <= 0:
        raise ValueError("The total weight ratio must be positive.")
    return am_wt / total, carbon_wt / total, binder_wt / total


def compute_electrode_parameters(
    areal_loading_mg_cm2,
    thickness_um,
    am_wt_percent,
    carbon_wt_percent,
    binder_wt_percent,
    am_density,
    carbon_density,
    binder_density,
    porosity_mode="auto",
    manual_porosity_percent=30.0,
):
    if thickness_um <= 0:
        raise ValueError("Electrode thickness must be positive.")
    if areal_loading_mg_cm2 <= 0:
        raise ValueError("Areal loading must be positive.")
    for name, rho in [
        ("AM density", am_density),
        ("Carbon density", carbon_density),
        ("Binder density", binder_density),
    ]:
        if rho <= 0:
            raise ValueError(f"{name} must be positive.")

    w_am, w_c, w_b = normalize_weights(am_wt_percent, carbon_wt_percent, binder_wt_percent)

    v_am_raw = w_am / am_density
    v_c_raw = w_c / carbon_density
    v_b_raw = w_b / binder_density
    v_sum = v_am_raw + v_c_raw + v_b_raw

    rho_solid = 1.0 / v_sum
    rho_coating = 10.0 * areal_loading_mg_cm2 / thickness_um

    phi_am_solid = v_am_raw / v_sum
    phi_cbd_solid = (v_c_raw + v_b_raw) / v_sum

    auto_porosity = 1.0 - rho_coating / rho_solid
    if porosity_mode == "auto":
        eps = auto_porosity
    else:
        eps = manual_porosity_percent / 100.0
    eps_used = float(np.clip(eps, 0.01, 0.95))

    return {
        "rho_solid": rho_solid,
        "rho_coating": rho_coating,
        "auto_porosity": auto_porosity,
        "porosity_raw": eps,
        "porosity_used": eps_used,
        "phi_am_solid": phi_am_solid,
        "phi_cbd_solid": phi_cbd_solid,
        "phi_am_total": (1.0 - eps_used) * phi_am_solid,
        "phi_cbd_total": (1.0 - eps_used) * phi_cbd_solid,
        "phi_pore_total": eps_used,
    }


# ============================================================
# PSD
# ============================================================
def gaussian_pdf(x, mean, std):
    std = max(std, 1e-12)
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def load_psd_file(path):
    rows = []
    lower = path.lower()
    if lower.endswith(".xlsx"):
        if not HAS_OPENPYXL:
            raise RuntimeError("openpyxl is required for .xlsx PSD files. Please run: pip install openpyxl")
        wb = openpyxl.load_workbook(path, data_only=True)
        ws = wb.active
        for row in ws.iter_rows(values_only=True):
            if row and len(row) >= 2:
                rows.append((row[0], row[1]))
    else:
        with open(path, "r", encoding="utf-8-sig") as f:
            sample = f.read(2048)
            f.seek(0)
            if "," in sample:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        rows.append((row[0], row[1]))
            else:
                for line in f:
                    parts = line.strip().replace("\t", " ").split()
                    if len(parts) >= 2:
                        rows.append((parts[0], parts[1]))

    diameters = []
    weights = []
    for a, b in rows:
        try:
            d = float(a)
            w = float(b)
        except Exception:
            continue
        if d > 0 and w >= 0 and np.isfinite(d) and np.isfinite(w):
            diameters.append(d)
            weights.append(w)

    if len(diameters) < 2:
        raise ValueError("PSD file needs at least two numeric rows: column 1 = diameter_um, column 2 = probability/frequency.")
    diameters = np.asarray(diameters, dtype=float)
    weights = np.asarray(weights, dtype=float)
    idx = np.argsort(diameters)
    diameters = diameters[idx]
    weights = weights[idx]
    if weights.sum() <= 0:
        raise ValueError("PSD probabilities/frequencies must have a positive sum.")
    return diameters, weights / weights.sum()


def sample_particle_diameter_um(rng, psd_mode, d_min, d50, d_max, d_std, imported_d=None, imported_p=None):
    if psd_mode == "Imported PSD File":
        if imported_d is None or imported_p is None:
            raise ValueError("Imported PSD mode requires a valid PSD file.")
        return float(imported_d[rng.choice(len(imported_d), p=imported_p)])

    d_min = max(0.01, d_min)
    d_max = max(d_min, d_max)
    d50 = float(np.clip(d50, d_min, d_max))
    d_std = max(0.001, d_std)

    if psd_mode == "Uniform Dmin-Dmax":
        return float(rng.uniform(d_min, d_max))

    for _ in range(100):
        d = float(rng.normal(d50, d_std))
        if d_min <= d <= d_max:
            return d
    return float(np.clip(rng.normal(d50, d_std), d_min, d_max))


# ============================================================
# Structure generation
# ============================================================
def sphere_indices_periodic(shape, center_zyx, radius_vox):
    nz, ny, nx = shape
    cz, cy, cx = center_zyx
    r = int(math.ceil(radius_vox))
    offsets = np.arange(-r, r + 1)
    dz, dy, dx = np.meshgrid(offsets, offsets, offsets, indexing="ij")
    mask = (dx * dx + dy * dy + dz * dz) <= radius_vox * radius_vox
    z = (cz + dz[mask]) % nz
    y = (cy + dy[mask]) % ny
    x = (cx + dx[mask]) % nx
    return z.astype(np.int64), y.astype(np.int64), x.astype(np.int64)


def generate_am_structure(
    dimension_x_um,
    dimension_y_um,
    thickness_um,
    resolution_um,
    target_am_fraction,
    psd_mode,
    d_min_um,
    d50_um,
    d_max_um,
    d_std_um,
    max_overlap_percent=20.0,
    boundary_margin=0.0,
    seed=1,
    max_attempts=300000,
    imported_diameters=None,
    imported_probabilities=None,
):
    rng = np.random.default_rng(seed)
    nx = max(4, int(round(dimension_x_um / resolution_um)))
    ny = max(4, int(round(dimension_y_um / resolution_um)))
    nz = max(4, int(round(thickness_um / resolution_um)))
    volume = np.zeros((nz, ny, nx), dtype=np.uint8)
    total = volume.size
    target_vox = int(np.clip(target_am_fraction, 0.001, 0.95) * total)
    max_overlap = float(np.clip(max_overlap_percent / 100.0, 0.0, 1.0))
    margin = max(0, int(round(boundary_margin)))
    placed = 0
    attempts = 0
    rejected = 0

    while np.count_nonzero(volume == PHASE["AM"]) < target_vox and attempts < max_attempts:
        attempts += 1
        d_um = sample_particle_diameter_um(
            rng, psd_mode, d_min_um, d50_um, d_max_um, d_std_um,
            imported_diameters, imported_probabilities,
        )
        r_vox = max(0.75, (d_um / 2.0) / resolution_um)
        if margin > 0 and nx > 2 * margin and ny > 2 * margin and nz > 2 * margin:
            cx = rng.integers(margin, nx - margin)
            cy = rng.integers(margin, ny - margin)
            cz = rng.integers(margin, nz - margin)
        else:
            cx = rng.integers(0, nx)
            cy = rng.integers(0, ny)
            cz = rng.integers(0, nz)

        z, y, x = sphere_indices_periodic(volume.shape, (cz, cy, cx), r_vox)
        vals = volume[z, y, x]
        overlap = np.count_nonzero(vals == PHASE["AM"]) / max(1, len(vals))
        current = np.count_nonzero(volume == PHASE["AM"]) / total
        allowed = max_overlap
        if current > 0.8 * target_am_fraction:
            allowed = min(1.0, max_overlap + 0.20)
        if overlap > allowed:
            rejected += 1
            continue
        before = np.count_nonzero(volume == PHASE["AM"])
        volume[z, y, x] = PHASE["AM"]
        after = np.count_nonzero(volume == PHASE["AM"])
        if after > before:
            placed += 1

    actual = np.count_nonzero(volume == PHASE["AM"]) / total
    return volume, {
        "shape": volume.shape,
        "target_am_fraction": target_am_fraction,
        "actual_am_fraction": actual,
        "am_fraction_error": actual - target_am_fraction,
        "placed_spheres": placed,
        "attempts": attempts,
        "rejected_overlap": rejected,
    }


def remove_cbd(volume):
    out = volume.copy()
    out[out == PHASE["CBD"]] = PHASE["pore"]
    return out


def add_cbd_by_target_fraction(volume, target_cbd_fraction, w=0.5, seed=1):
    rng = np.random.default_rng(seed)
    out = remove_cbd(volume)
    total = out.size
    target = int(np.clip(target_cbd_fraction, 0.0, 0.8) * total)
    if target <= 0:
        return out, {"target_cbd_voxels": 0, "actual_cbd_voxels": 0, "actual_cbd_fraction": 0.0}

    pore = out == PHASE["pore"]
    am = out == PHASE["AM"]
    target = min(target, int(np.count_nonzero(pore)))
    w = float(np.clip(w, 0.0, 1.0))
    film_target = int(round((1.0 - w) * target))
    cluster_target = target - film_target
    cbd = np.zeros(out.shape, dtype=bool)

    if film_target > 0 and np.any(am):
        dist = ndi.distance_transform_edt(~am)
        cand = pore & (dist <= 3.5)
        if np.count_nonzero(cand) < film_target:
            cand = pore & (dist <= 6.0)
        coords = np.argwhere(cand)
        if len(coords) > 0:
            weights = 1.0 / (dist[cand] + 1e-6)
            weights /= weights.sum()
            n_pick = min(film_target, len(coords))
            sel = coords[rng.choice(len(coords), size=n_pick, replace=False, p=weights)]
            cbd[sel[:, 0], sel[:, 1], sel[:, 2]] = True

    if cluster_target > 0:
        available = pore & (~cbd)
        coords = np.argwhere(available)
        if len(coords) > 0:
            base = max(1, int(math.sqrt(cluster_target) / 2.5))
            n_seeds = max(1, int(base * (1.2 - 0.7 * w)))
            n_seeds = min(n_seeds, len(coords))
            seed_coords = coords[rng.choice(len(coords), size=n_seeds, replace=False)]
            cluster = np.zeros(out.shape, dtype=bool)
            cluster[seed_coords[:, 0], seed_coords[:, 1], seed_coords[:, 2]] = True
            structure = ndi.generate_binary_structure(3, 2)
            for _ in range(500):
                if np.count_nonzero(cluster) >= cluster_target:
                    break
                grown = ndi.binary_dilation(cluster, structure=structure) & available
                shell = grown & (~cluster)
                shell_coords = np.argwhere(shell)
                remain = cluster_target - np.count_nonzero(cluster)
                if len(shell_coords) == 0:
                    break
                if len(shell_coords) > remain:
                    sel = shell_coords[rng.choice(len(shell_coords), size=remain, replace=False)]
                    cluster[sel[:, 0], sel[:, 1], sel[:, 2]] = True
                else:
                    cluster |= shell
            cbd |= cluster

    out[cbd & (out == PHASE["pore"])] = PHASE["CBD"]
    actual = int(np.count_nonzero(out == PHASE["CBD"]))
    return out, {
        "target_cbd_voxels": int(target),
        "actual_cbd_voxels": actual,
        "actual_cbd_fraction": actual / out.size,
    }


def add_layer_zmax(volume, phase_label, thickness_um, resolution_um):
    n = int(round(thickness_um / resolution_um))
    if n <= 0:
        return volume
    _, ny, nx = volume.shape
    return np.concatenate([volume, np.full((n, ny, nx), phase_label, dtype=np.uint8)], axis=0)


def add_layer_zmin(volume, phase_label, thickness_um, resolution_um):
    n = int(round(thickness_um / resolution_um))
    if n <= 0:
        return volume
    _, ny, nx = volume.shape
    return np.concatenate([np.full((n, ny, nx), phase_label, dtype=np.uint8), volume], axis=0)


def volume_statistics(volume):
    total = volume.size
    lines = []
    for label in sorted(np.unique(volume)):
        count = int(np.count_nonzero(volume == label))
        name = PHASE_NAME.get(int(label), f"phase_{label}")
        lines.append(f"{name}: {count} voxels, {100*count/total:.2f}%")
    return "\n".join(lines)


# ============================================================
# MESH
# ============================================================
def make_precision_samples(n, precision):
    """返回节点采样索引。确保包含 0 和 n-1。"""
    precision = int(max(1, precision))
    idx = list(range(0, n, precision))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return np.asarray(idx, dtype=np.int64)


def node_grid_from_label_volume(volume, precision_zyx):
    """

    每个保留节点从 segmented image/label stack 采样一个相标签。
    precision_zyx = (pz, py, px)
    """
    nz, ny, nx = volume.shape
    pz, py, px = [int(max(1, v)) for v in precision_zyx]
    z_idx = make_precision_samples(nz, pz)
    y_idx = make_precision_samples(ny, py)
    x_idx = make_precision_samples(nx, px)
    node_labels = volume[np.ix_(z_idx, y_idx, x_idx)].astype(np.uint8)
    return node_labels, z_idx, y_idx, x_idx


def build_points_from_samples(z_idx, y_idx, x_idx, spacing_um):
    zz, yy, xx = np.meshgrid(z_idx * spacing_um, y_idx * spacing_um, x_idx * spacing_um, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float64)
    return points


def signed_tet_volume(p0, p1, p2, p3):
    return float(np.dot(np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0)


def tet_phase_from_node_labels(labels4):
    """
    根据四面体 4 个节点标签决定 PID。
    - 如果 4 个节点同相，则单元属于该相；
    - 如果多相，则用多数投票；
    - 若 2:2 平局，优先选择较小相标签，保证可复现。
    """
    labels4 = np.asarray(labels4, dtype=np.int32)
    vals, counts = np.unique(labels4, return_counts=True)
    max_count = counts.max()
    candidates = vals[counts == max_count]
    return int(candidates.min())


def cube_tet_patterns(decomposition="6 tetra / voxel"):
    """
    节点局部编号：
      0=(z0,y0,x0), 1=(z0,y0,x1), 2=(z0,y1,x1), 3=(z0,y1,x0)
      4=(z1,y0,x0), 5=(z1,y0,x1), 6=(z1,y1,x1), 7=(z1,y1,x0)
    """
    if decomposition.startswith("5"):
        return [
            (0, 1, 3, 4),
            (1, 2, 3, 6),
            (1, 3, 4, 6),
            (1, 4, 5, 6),
            (3, 4, 6, 7),
        ]
    return [
        (0, 1, 2, 6),
        (0, 2, 3, 6),
        (0, 3, 7, 6),
        (0, 7, 4, 6),
        (0, 4, 5, 6),
        (0, 5, 1, 6),
    ]



def block_partition_indices(n, precision):
    """返回 block 边界索引，例如 [0, p, 2p, ..., n]。"""
    precision = int(max(1, precision))
    edges = list(range(0, n, precision))
    if edges[-1] != n:
        edges.append(n)
    return np.asarray(edges, dtype=np.int64)


def coarsen_label_volume_quota(volume, precision_zyx=(4, 4, 4), preserve_quota=True):
    """
    将原始标签体按照 precision 分块，生成 coarse cell labels。

    这个函数是 V0.8.1 的关键修正：
    - V0.8 只在节点上抽样相标签，CBD/薄层相容易被 precision 抽样漏掉；
    - V0.8.1 改为每个 coarse cell 统计 block 内所有原始体素；
    - preserve_quota=True 时，根据原始体积分数给每个相分配 coarse cell 数量，尽量保留 AM/CBD/pore/隔膜/集流体/Li foil。

    返回：
        coarse_labels: shape = (Nz_c, Ny_c, Nx_c)
        z_edges, y_edges, x_edges: block 边界索引，用于生成节点坐标
        info: 降采样统计
    """
    nz, ny, nx = volume.shape
    pz, py, px = [int(max(1, v)) for v in precision_zyx]
    z_edges = block_partition_indices(nz, pz)
    y_edges = block_partition_indices(ny, py)
    x_edges = block_partition_indices(nx, px)

    ncz = len(z_edges) - 1
    ncy = len(y_edges) - 1
    ncx = len(x_edges) - 1
    labels_all = list(range(6))
    n_blocks = ncz * ncy * ncx

    # 每个 coarse block 内各相体素数
    counts = np.zeros((6, ncz, ncy, ncx), dtype=np.int32)
    block_sizes = np.zeros((ncz, ncy, ncx), dtype=np.int32)
    for iz in range(ncz):
        z0, z1 = z_edges[iz], z_edges[iz + 1]
        for iy in range(ncy):
            y0, y1 = y_edges[iy], y_edges[iy + 1]
            for ix in range(ncx):
                x0, x1 = x_edges[ix], x_edges[ix + 1]
                block = volume[z0:z1, y0:y1, x0:x1]
                block_sizes[iz, iy, ix] = block.size
                for lab in labels_all:
                    counts[lab, iz, iy, ix] = int(np.count_nonzero(block == lab))

    original_counts = {lab: int(np.count_nonzero(volume == lab)) for lab in labels_all}
    original_present = [lab for lab in labels_all if original_counts[lab] > 0]

    if not preserve_quota:
        coarse = np.argmax(counts, axis=0).astype(np.uint8)
    else:
        # 原始体积分数 -> coarse cell 目标数量
        total_vox = int(volume.size)
        raw_targets = {lab: original_counts[lab] / total_vox * n_blocks for lab in original_present}
        targets = {lab: int(round(raw_targets[lab])) for lab in original_present}

        # 调整 round 后总数，使其等于 n_blocks
        diff = n_blocks - sum(targets.values())
        if diff != 0 and original_present:
            # 按小数部分修正
            residual = sorted(
                original_present,
                key=lambda lab: raw_targets[lab] - math.floor(raw_targets[lab]),
                reverse=(diff > 0),
            )
            k = 0
            while diff != 0:
                lab = residual[k % len(residual)]
                if diff > 0:
                    targets[lab] += 1
                    diff -= 1
                else:
                    if targets[lab] > 0:
                        targets[lab] -= 1
                        diff += 1
                k += 1

        coarse_flat = np.full(n_blocks, -1, dtype=np.int16)
        assigned = np.zeros(n_blocks, dtype=bool)
        counts_flat = counts.reshape(6, -1).astype(np.float64)
        block_sizes_flat = block_sizes.ravel().astype(np.float64)
        eps = 1e-12

        # 小相先分配，避免 CBD / thin layers 被 AM 或 pore 吞掉
        order = sorted(original_present, key=lambda lab: targets.get(lab, 0))
        for lab in order:
            target = targets.get(lab, 0)
            if target <= 0:
                continue
            available = np.flatnonzero(~assigned)
            if available.size == 0:
                break

            # 候选优先：block 内包含该相。若不足，再允许所有未分配 block。
            candidates = available[counts_flat[lab, available] > 0]
            if candidates.size < target:
                candidates = available
                need = min(target, candidates.size)
            else:
                need = target

            if need <= 0:
                continue

            # 分数：该相占比越高越优先；小相先分配保证体积分数守恒
            frac_score = counts_flat[lab, candidates] / (block_sizes_flat[candidates] + eps)
            # 加一个极小扰动，保证排序稳定但不随机
            score = frac_score + 1e-9 * counts_flat[lab, candidates]
            pick = candidates[np.argsort(score)[-need:]]
            coarse_flat[pick] = lab
            assigned[pick] = True

        # 剩余 block 用 majority
        remaining = np.flatnonzero(~assigned)
        if remaining.size > 0:
            majority = np.argmax(counts_flat[:, remaining], axis=0)
            coarse_flat[remaining] = majority

        coarse = coarse_flat.reshape((ncz, ncy, ncx)).astype(np.uint8)

    info = {
        "precision_zyx": tuple(int(v) for v in precision_zyx),
        "coarse_shape_zyx": tuple(int(v) for v in coarse.shape),
        "original_counts": {int(k): int(v) for k, v in original_counts.items() if v > 0},
        "coarse_counts": {int(k): int(np.count_nonzero(coarse == k)) for k in labels_all if np.count_nonzero(coarse == k) > 0},
        "preserve_quota": bool(preserve_quota),
    }
    return coarse, z_edges, y_edges, x_edges, info


def build_points_from_block_edges(z_edges, y_edges, x_edges, spacing_um):
    zz, yy, xx = np.meshgrid(z_edges * spacing_um, y_edges * spacing_um, x_edges * spacing_um, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float64)
    return points




def sample_phase_from_original_volume(volume, spacing_um, xyz):
    """按物理坐标 x,y,z 在原始标签体 volume[Z,Y,X] 中采样相标签。"""
    x, y, z = xyz
    nz, ny, nx = volume.shape
    ix = int(np.clip(math.floor(x / spacing_um), 0, nx - 1))
    iy = int(np.clip(math.floor(y / spacing_um), 0, ny - 1))
    iz = int(np.clip(math.floor(z / spacing_um), 0, nz - 1))
    return int(volume[iz, iy, ix])


def get_or_create_midpoint(points_list, edge_cache, a, b):
    """全局缓存边中点，保证相邻四面体共享同一个中点节点。"""
    key = tuple(sorted((int(a), int(b))))
    if key in edge_cache:
        return edge_cache[key]
    pa = points_list[key[0]]
    pb = points_list[key[1]]
    idx = len(points_list)
    points_list.append(0.5 * (pa + pb))
    edge_cache[key] = idx
    return idx


def get_or_create_face_centroid(points_list, face_cache, a, b, c):
    """全局缓存三角面中心，保证共享面的细分节点一致。"""
    key = tuple(sorted((int(a), int(b), int(c))))
    if key in face_cache:
        return face_cache[key]
    pa = points_list[key[0]]
    pb = points_list[key[1]]
    pc = points_list[key[2]]
    idx = len(points_list)
    points_list.append((pa + pb + pc) / 3.0)
    face_cache[key] = idx
    return idx


def create_tet_centroid(points_list, a, b, c, d):
    """四面体中心是单元内部点，不需要跨单元共享。"""
    idx = len(points_list)
    points_list.append((points_list[a] + points_list[b] + points_list[c] + points_list[d]) / 4.0)
    return idx


def barycentric_subdivide_tet(points_list, tet, edge_cache, face_cache):
    """
    四面体重心细分。

    对一个四面体生成 24 个子四面体。这个分解的好处是：
    - 共享边中点和共享面中心使用全局缓存；
    - 相邻四面体在公共面上的细分是一致的；
    - 混合相区域会自然产生更多小单元
    """
    import itertools
    a, b, c, d = [int(v) for v in tet]
    verts = [a, b, c, d]
    center = create_tet_centroid(points_list, a, b, c, d)
    sub_tets = []
    for perm in itertools.permutations(verts, 4):
        v0 = perm[0]
        e01 = get_or_create_midpoint(points_list, edge_cache, perm[0], perm[1])
        f012 = get_or_create_face_centroid(points_list, face_cache, perm[0], perm[1], perm[2])
        sub_tets.append([v0, e01, f012, center])
    return sub_tets


def block_is_mixed(volume, z0, z1, y0, y1, x0, x1):
    block = volume[z0:z1, y0:y1, x0:x1]
    return len(np.unique(block)) > 1

def create_voxel_tetra_mesh(
    volume,
    spacing_um,
    precision_zyx=(4, 4, 4),
    decomposition="6 tetra / voxel",
    phase_method="quota_cell",
):
    """

    phase_method:
        quota_cell
            V0.8.1 的稳定模式。每个 precision block 用 quota-preserving 标签，
            再分解为 5/6 个四面体。速度快、COMSOL 导入稳定。

        mixed_centroid
            新增模式。对于原始 block 内含多个相的 mixed voxel，将基础四面体进行
            barycentric subdivision，每个子四面体按重心采样原始标签体赋 PID。

        global_centroid
            对所有 coarse voxels 都进行 barycentric subdivision。最接近“所有位置一致细分”，
            但单元数会大很多，建议只在较大 precision 下测试。
    """
    coarse_labels, z_edges, y_edges, x_edges, info = coarsen_label_volume_quota(
        volume,
        precision_zyx=precision_zyx,
        preserve_quota=True,
    )

    base_points = build_points_from_block_edges(z_edges, y_edges, x_edges, spacing_um)
    points_list = [base_points[i].copy() for i in range(base_points.shape[0])]
    edge_cache = {}
    face_cache = {}

    ncz, ncy, ncx = coarse_labels.shape
    nz_n = ncz + 1
    ny_n = ncy + 1
    nx_n = ncx + 1

    def nid(iz, iy, ix):
        return iz * ny_n * nx_n + iy * nx_n + ix

    patterns = cube_tet_patterns(decomposition)
    tets = []
    phases = []
    fixed_orientation = 0
    removed_degenerate = 0
    mixed_blocks = 0
    subdivided_base_tets = 0

    phase_method = str(phase_method).strip()

    for iz in range(ncz):
        z0, z1 = int(z_edges[iz]), int(z_edges[iz + 1])
        for iy in range(ncy):
            y0, y1 = int(y_edges[iy]), int(y_edges[iy + 1])
            for ix in range(ncx):
                x0, x1 = int(x_edges[ix]), int(x_edges[ix + 1])
                cell_phase = int(coarse_labels[iz, iy, ix])
                is_mixed = block_is_mixed(volume, z0, z1, y0, y1, x0, x1)
                if is_mixed:
                    mixed_blocks += 1

                local_nodes = [
                    nid(iz, iy, ix),
                    nid(iz, iy, ix + 1),
                    nid(iz, iy + 1, ix + 1),
                    nid(iz, iy + 1, ix),
                    nid(iz + 1, iy, ix),
                    nid(iz + 1, iy, ix + 1),
                    nid(iz + 1, iy + 1, ix + 1),
                    nid(iz + 1, iy + 1, ix),
                ]

                refine_this_cell = (phase_method == "global_centroid") or (phase_method == "mixed_centroid" and is_mixed)

                for pat in patterns:
                    base_tet = [local_nodes[i] for i in pat]
                    if refine_this_cell:
                        subtets = barycentric_subdivide_tet(points_list, base_tet, edge_cache, face_cache)
                        subdivided_base_tets += 1
                        for tet in subtets:
                            pts = np.asarray([points_list[v] for v in tet], dtype=np.float64)
                            vol = signed_tet_volume(pts[0], pts[1], pts[2], pts[3])
                            if abs(vol) < 1e-18:
                                removed_degenerate += 1
                                continue
                            if vol < 0:
                                tet[1], tet[2] = tet[2], tet[1]
                                fixed_orientation += 1
                            centroid = pts.mean(axis=0)
                            ph = sample_phase_from_original_volume(volume, spacing_um, centroid)
                            tets.append(tet)
                            phases.append(ph)
                    else:
                        tet = list(base_tet)
                        pts = np.asarray([points_list[v] for v in tet], dtype=np.float64)
                        vol = signed_tet_volume(pts[0], pts[1], pts[2], pts[3])
                        if abs(vol) < 1e-18:
                            removed_degenerate += 1
                            continue
                        if vol < 0:
                            tet[1], tet[2] = tet[2], tet[1]
                            fixed_orientation += 1
                        tets.append(tet)
                        phases.append(cell_phase)

    points = np.asarray(points_list, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    phases = np.asarray(phases, dtype=np.int32)
    report = {
        "node_grid_shape_zyx": (int(nz_n), int(ny_n), int(nx_n)),
        "coarse_cell_shape_zyx": tuple(int(v) for v in coarse_labels.shape),
        "precision_zyx": tuple(int(v) for v in precision_zyx),
        "phase_method": phase_method,
        "points": int(points.shape[0]),
        "tetrahedra": int(tets.shape[0]),
        "fixed_orientation": int(fixed_orientation),
        "removed_degenerate": int(removed_degenerate),
        "mixed_blocks": int(mixed_blocks),
        "subdivided_base_tets": int(subdivided_base_tets),
        "phases": [int(v) for v in np.unique(phases)],
        "original_phase_counts": info["original_counts"],
        "coarse_phase_counts": info["coarse_counts"],
        "preserve_quota": True,
    }
    return points, tets, phases, report

def write_comsol_bdf(points, tets, phases, path):
    """
    写出 COMSOL 可导入 NASTRAN/BDF。
    PID = phase + 1。
    """
    pids = phases.astype(np.int32) + 1
    used_node_ids = np.unique(tets.ravel())
    used_node_ids = np.asarray(sorted([int(i) for i in used_node_ids]), dtype=np.int64)
    old_to_new = {old: new for new, old in enumerate(used_node_ids, start=1)}
    used_pids = sorted([int(v) for v in np.unique(pids)])

    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write("$ ============================================================\n")
        f.write("$ OpenMesoCell V0.9 \n")
        f.write("$ Coordinates are in micrometers (um).\n")
        f.write("$ PID 1 = PORE / ELECTROLYTE\n")
        f.write("$ PID 2 = AM\n")
        f.write("$ PID 3 = CBD\n")
        f.write("$ PID 4 = SEPARATOR\n")
        f.write("$ PID 5 = CURRENT_COLLECTOR\n")
        f.write("$ PID 6 = LI_FOIL\n")
        f.write("$ ============================================================\n")
        f.write("BEGIN BULK\n")
        f.write("$ Materials are placeholders. Set real parameters in COMSOL.\n")
        for pid in used_pids:
            f.write(f"MAT1,{pid},1.0,,0.30\n")
        for pid in used_pids:
            f.write(f"PSOLID,{pid},{pid}\n")
        f.write("$ GRID nodes, unit = um\n")
        for old in used_node_ids:
            nid = old_to_new[int(old)]
            x, y, z = points[int(old)]
            f.write(f"GRID,{nid},,{x:.12g},{y:.12g},{z:.12g}\n")
        f.write("$ CTETRA elements\n")
        for eid, (tet, pid) in enumerate(zip(tets, pids), start=1):
            n0, n1, n2, n3 = [old_to_new[int(v)] for v in tet]
            f.write(f"CTETRA,{eid},{int(pid)},{n0},{n1},{n2},{n3}\n")
        f.write("ENDDATA\n")

    return {
        "used_nodes": int(len(used_node_ids)),
        "tetrahedra": int(tets.shape[0]),
        "property_ids": used_pids,
        "path": path,
    }


def export_voxel_bdf(volume, spacing_um, path, precision_zyx=(4, 4, 4), decomposition="6 tetra / voxel", phase_method="quota_cell"):
    points, tets, phases, report = create_voxel_tetra_mesh(
        volume, spacing_um, precision_zyx=precision_zyx, decomposition=decomposition, phase_method=phase_method
    )
    bdf_report = write_comsol_bdf(points, tets, phases, path)
    report.update(bdf_report)

    mapping_path = os.path.splitext(path)[0] + "_phase_mapping.txt"
    with open(mapping_path, "w", encoding="utf-8") as f:
        f.write("OpenMesoCell V0.9 voxelization mesh mapping\n")
        f.write("======================================================\n")
        f.write("COMSOL import: Mesh > Import > NASTRAN file (.bdf), unit = um\n\n")
        for label, name in PHASE_NAME.items():
            if (label + 1) in report["property_ids"]:
                f.write(f"PID {label + 1} = {name}\n")
        f.write("\nMesh settings:\n")
        f.write(f"precision_zyx = {report['precision_zyx']}\n")
        f.write(f"node_grid_shape_zyx = {report['node_grid_shape_zyx']}\n")
        f.write(f"decomposition = {decomposition}\n")
        f.write(f"phase_method = {phase_method}\n")
        f.write(f"spacing_um = {spacing_um}\n")

    json_path = os.path.splitext(path)[0] + "_mesh_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    report["mapping_path"] = mapping_path
    report["json_path"] = json_path
    return report


# ============================================================
# 3D helpers
# ============================================================
def create_polydata_from_phase_mask(mask, spacing_um):
    if not HAS_SKIMAGE:
        raise RuntimeError("scikit-image is not installed.")
    if np.count_nonzero(mask) == 0:
        return None
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5, spacing=(spacing_um, spacing_um, spacing_um))
    verts_xyz = np.column_stack([verts[:, 2], verts[:, 1], verts[:, 0]])
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)]).ravel()
    poly = pv.PolyData(verts_xyz, faces_pv)
    poly.clean(inplace=True)
    return poly


# ============================================================
# Matplotlib canvases
# ============================================================
class SliceCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

    def show_slice(self, volume, plane="XY", index=0):
        self.ax.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        if volume is None:
            self.ax.text(0.5, 0.5, "No structure", ha="center", va="center")
            self.draw()
            return
        nz, ny, nx = volume.shape
        if plane == "XY":
            index = int(np.clip(index, 0, nz - 1))
            img = volume[index]
            title = f"XY slice: {index + 1}/{nz}"
        elif plane == "XZ":
            index = int(np.clip(index, 0, ny - 1))
            img = volume[:, index, :]
            title = f"XZ slice: {index + 1}/{ny}"
        else:
            index = int(np.clip(index, 0, nx - 1))
            img = volume[:, :, index]
            title = f"YZ slice: {index + 1}/{nx}"
        self.ax.imshow(img, cmap=CMAP, norm=NORM, interpolation="nearest", origin="lower")
        self.ax.set_title(title, fontsize=10)
        handles = []
        unique = set(int(v) for v in np.unique(volume))
        for label in [1, 2, 0, 3, 4, 5]:
            if label in unique:
                handles.append(Patch(facecolor=PHASE_COLOR[label], edgecolor="black", label=PHASE_SHORT_NAME[label]))
        if handles:
            self.ax.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.82,
                           borderpad=0.25, handlelength=0.8, handletextpad=0.3, labelspacing=0.2)
        self.fig.tight_layout()
        self.draw()


class PSDCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(3.2, 1.8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setMinimumHeight(170)
        self.setMinimumWidth(300)
        self.update_plot("Gaussian D50", 2.0, 5.0, 8.0, 1.0, None, None)

    def update_plot(self, mode, d_min, d50, d_max, d_std, imported_d=None, imported_p=None):
        self.ax.clear()
        self.ax.set_title("PSD", fontsize=9)
        self.ax.set_xlabel("Diameter (um)", fontsize=8)
        self.ax.set_ylabel("Prob.", fontsize=8)
        self.ax.tick_params(axis="both", labelsize=7)
        if mode == "Imported PSD File":
            if imported_d is None or imported_p is None:
                self.ax.text(0.5, 0.5, "Load PSD file", ha="center", va="center", transform=self.ax.transAxes, fontsize=8)
            else:
                self.ax.plot(imported_d, imported_p, marker="o", linewidth=1)
                self.ax.fill_between(imported_d, imported_p, alpha=0.25)
        elif mode == "Uniform Dmin-Dmax":
            d_min = max(0.01, d_min)
            d_max = max(d_min, d_max)
            xs = np.array([d_min, d_min, d_max, d_max])
            ys = np.array([0, 1, 1, 0])
            self.ax.plot(xs, ys)
            self.ax.fill_between(xs, ys, alpha=0.25)
            self.ax.set_ylim(0, 1.2)
        else:
            d_min = max(0.01, d_min)
            d_max = max(d_min, d_max)
            d50 = float(np.clip(d50, d_min, d_max))
            d_std = max(0.001, d_std)
            xs = np.linspace(d_min, d_max, 300)
            ys = gaussian_pdf(xs, d50, d_std)
            if ys.max() > 0:
                ys = ys / ys.max()
            self.ax.plot(xs, ys)
            self.ax.fill_between(xs, ys, alpha=0.25)
            self.ax.set_ylim(0, 1.1)
        self.fig.tight_layout(pad=0.6)
        self.draw()


class Volume3DDialog(QDialog):
    def __init__(self, volume, resolution_um, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OpenMesoCell 3D Viewer")
        self.resize(1100, 800)
        self.volume = volume
        self.resolution_um = resolution_um
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        layout.addLayout(controls)
        phase_group = QGroupBox("Display Phases")
        phase_layout = QGridLayout(phase_group)
        self.check_phase = {}
        defaults = {0: False, 1: True, 2: True, 3: True, 4: True, 5: True}
        row = 0
        for label in [0, 1, 2, 3, 4, 5]:
            cb = QCheckBox(PHASE_NAME[label])
            cb.setChecked(defaults[label])
            self.check_phase[label] = cb
            phase_layout.addWidget(cb, row, 0)
            row += 1
        self.btn_render = QPushButton("Render Selected Phases")
        self.btn_reset = QPushButton("Reset View")
        self.btn_close = QPushButton("Close")
        phase_layout.addWidget(self.btn_render, row, 0); row += 1
        phase_layout.addWidget(self.btn_reset, row, 0); row += 1
        phase_layout.addWidget(self.btn_close, row, 0)
        controls.addWidget(phase_group)
        if not HAS_3D:
            controls.addWidget(QLabel("PyVista / PyVistaQt is not installed."))
            self.plotter = None
            return
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter, stretch=1)
        self.btn_render.clicked.connect(self.render_volume)
        self.btn_reset.clicked.connect(self.reset_view)
        self.btn_close.clicked.connect(self.close)
        self.render_volume()

    def render_volume(self):
        if self.plotter is None:
            return
        self.plotter.clear()
        self.plotter.set_background("white")
        anything = False
        for label, cb in self.check_phase.items():
            if not cb.isChecked():
                continue
            mask = self.volume == label
            if np.count_nonzero(mask) == 0:
                continue
            poly = create_polydata_from_phase_mask(mask, self.resolution_um)
            if poly is None or poly.n_points == 0:
                continue
            opacity = 0.08 if label == 0 else (0.75 if label == 3 else 0.90)
            if label in [1, 2]:
                opacity = 1.0 if label == 1 else 0.88
            self.plotter.add_mesh(poly, color=PHASE_COLOR[label], opacity=opacity, show_edges=False, smooth_shading=True)
            anything = True
        if anything:
            self.plotter.show_bounds(grid="back", location="outer", all_edges=True,
                                     xtitle="X (um)", ytitle="Y (um)", ztitle="Z (um)",
                                     fmt="%.0f", font_size=8)
            self.plotter.add_axes(line_width=1, labels_off=False)
            self.plotter.reset_camera()
        else:
            self.plotter.add_text("No selected phase to display.", font_size=10)

    def reset_view(self):
        if self.plotter is not None:
            self.plotter.reset_camera()


# ============================================================
# Main GUI
# ============================================================
class _MenuProxy:
    """模拟旧版导航菜单，将 setCurrentRow(5) 转译为切换到历史任务 tab。"""

    def __init__(self, window):
        self._window = window

    def setCurrentRow(self, row):
        if row == 5:  # 历史任务中心
            # 切换到深度学习面板
            self._window._switch_right_panel(1)
            # 选中"任务历史" tab
            if hasattr(self._window, '_dl_tabs') and self._window._dl_tabs is not None:
                for i in range(self._window._dl_tabs.count()):
                    if self._window._dl_tabs.tabText(i).startswith("任务历史"):
                        self._window._dl_tabs.setCurrentIndex(i)
                        break


class OpenMesoCellWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenMesoCell V0.9")
        self.resize(1500, 900)
        self.volume = None
        self.current_plane = "XY"
        self.resolution_um = 0.33
        self.imported_psd_path = None
        self.imported_diameters = None
        self.imported_probabilities = None
        self.api_process = None  # 后端 API 子进程

        # 兼容 stage3/4/5 页面对 main_window.menu 和 main_window.history_page 的调用
        self.menu = _MenuProxy(self)
        self.history_page = None  # 在 _init_dl_tabs 中赋值

        self._build_ui()
        self._connect_signals()
        self.update_porosity_input_state()
        self.update_psd_input_state()
        self.update_parameter_preview()
        self.update_psd_preview()
        self.update_view()

        # 检测后端 API 服务器状态（由 main() 预先启动），然后加载深度学习页面
        self._start_api_server()
        self.dl_loading = True
        QTimer.singleShot(500, self._init_dl_tabs)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(0)

        # ---- 主内容区域（左-中-右三栏） ----
        main = QHBoxLayout()
        root_layout.addLayout(main, stretch=1)

        left = QVBoxLayout()
        main.addLayout(left, stretch=3)
        top = QHBoxLayout()
        self.btn_xy = QPushButton("XY")
        self.btn_xz = QPushButton("XZ")
        self.btn_yz = QPushButton("YZ")
        self.btn_3d = QPushButton("3D")
        self.btn_clear = QPushButton("Clear")
        for b in [self.btn_xy, self.btn_xz, self.btn_yz, self.btn_3d, self.btn_clear]:
            top.addWidget(b)
        left.addLayout(top)
        self.slice_canvas = SliceCanvas()
        left.addWidget(self.slice_canvas, stretch=1)
        slider = QHBoxLayout()
        slider.addWidget(QLabel("Slice"))
        self.slice_slider = QSlider(Qt.Horizontal)
        slider.addWidget(self.slice_slider)
        self.slice_label = QLabel("0/0")
        slider.addWidget(self.slice_label)
        left.addLayout(slider)
        self.structure_label = QLabel("Structure: none")
        left.addWidget(self.structure_label)

        mid = QVBoxLayout()
        main.addLayout(mid, stretch=2)
        io_group = QGroupBox("Import / Save Data")
        io = QGridLayout(io_group)
        self.btn_load_npy = QPushButton("Load 3D Matrix (.npy)")
        self.btn_save_npy = QPushButton("Save Matrix (.npy)")
        self.btn_save_tif = QPushButton("Save Images (.tif)")
        io.addWidget(self.btn_load_npy, 0, 0, 1, 2)
        io.addWidget(self.btn_save_npy, 1, 0)
        io.addWidget(self.btn_save_tif, 1, 1)
        mid.addWidget(io_group)

        layer_group = QGroupBox("Separator / Current Collector / Li Foil")
        layer = QGridLayout(layer_group)
        self.spin_separator = QDoubleSpinBox(); self.spin_separator.setRange(0, 1000); self.spin_separator.setDecimals(3); self.spin_separator.setValue(5.0)
        self.spin_cc = QDoubleSpinBox(); self.spin_cc.setRange(0, 1000); self.spin_cc.setDecimals(3); self.spin_cc.setValue(3.0)
        self.spin_li = QDoubleSpinBox(); self.spin_li.setRange(0, 1000); self.spin_li.setDecimals(3); self.spin_li.setValue(5.0)
        self.btn_add_separator = QPushButton("Add Separator at Z max")
        self.btn_add_cc = QPushButton("Add CC at Z min")
        self.btn_add_li = QPushButton("Add Li Foil at Z max")
        layer.addWidget(QLabel("Separator thickness (um)"), 0, 0); layer.addWidget(self.spin_separator, 0, 1); layer.addWidget(self.btn_add_separator, 1, 0, 1, 2)
        layer.addWidget(QLabel("CC thickness (um)"), 2, 0); layer.addWidget(self.spin_cc, 2, 1); layer.addWidget(self.btn_add_cc, 3, 0, 1, 2)
        layer.addWidget(QLabel("Li foil thickness (um)"), 4, 0); layer.addWidget(self.spin_li, 4, 1); layer.addWidget(self.btn_add_li, 5, 0, 1, 2)
        mid.addWidget(layer_group)

        mesh_group = QGroupBox("Meshing Options - voxelization")
        mesh = QGridLayout(mesh_group)
        self.combo_tet_decomposition = QComboBox(); self.combo_tet_decomposition.addItems(["6 tetra / voxel", "5 tetra / voxel"])
        self.combo_phase_method = QComboBox()
        self.combo_phase_method.addItems([
            "quota_cell",
            "mixed_centroid",
            "global_centroid",
        ])
        self.combo_phase_method.setToolTip(
            "quota_cell: 稳定快速；mixed_centroid: 混合相 voxel 内部细分并用重心采样原始相；global_centroid: 全局细分，最细但很大。"
        )
        self.spin_precision_z = QSpinBox(); self.spin_precision_z.setRange(1, 50); self.spin_precision_z.setValue(4)
        self.spin_precision_y = QSpinBox(); self.spin_precision_y.setRange(1, 50); self.spin_precision_y.setValue(4)
        self.spin_precision_x = QSpinBox(); self.spin_precision_x.setRange(1, 50); self.spin_precision_x.setValue(4)
        self.btn_save_bdf = QPushButton("Save COMSOL BDF Mesh")
        mesh.addWidget(QLabel("Tet decomposition"), 0, 0); mesh.addWidget(self.combo_tet_decomposition, 0, 1, 1, 3)
        mesh.addWidget(QLabel("Voxel phase method"), 1, 0); mesh.addWidget(self.combo_phase_method, 1, 1, 1, 3)
        mesh.addWidget(QLabel("Precision Z / Y / X"), 2, 0)
        mesh.addWidget(self.spin_precision_z, 2, 1); mesh.addWidget(self.spin_precision_y, 2, 2); mesh.addWidget(self.spin_precision_x, 2, 3)
        mesh.addWidget(self.btn_save_bdf, 3, 0, 1, 4)
        mid.addWidget(mesh_group)

        calc_group = QGroupBox("Calculated Parameters")
        calc = QVBoxLayout(calc_group)
        self.preview_box = QTextEdit(); self.preview_box.setReadOnly(True); self.preview_box.setMaximumHeight(180)
        calc.addWidget(self.preview_box)
        mid.addWidget(calc_group)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
        log_layout.addWidget(self.log_box)
        mid.addWidget(log_group, stretch=1)

        # ---- 右侧切换面板（按钮 + 内容） ----
        right_panel = QVBoxLayout()
        right_panel.setSpacing(4)

        # 顶部按钮
        nav = QHBoxLayout()
        nav.setSpacing(8)
        self.btn_dl = QPushButton("深度学习")
        self.btn_mfg = QPushButton("制造参数")
        btn_style = (
            "QPushButton {"
            "background-color: #1a73e8; color: white; font-weight: bold;"
            "padding: 8px 20px; border-radius: 6px; font-size: 13px;"
            "}"
            "QPushButton:hover { background-color: #1557b0; }"
            "QPushButton:checked { background-color: #0d47a1; }"
        )
        for btn in [self.btn_dl, self.btn_mfg]:
            btn.setStyleSheet(btn_style)
            btn.setCheckable(True)
            nav.addWidget(btn)
        nav.addStretch()
        right_panel.addLayout(nav)

        self.right_stack = QStackedWidget()
        self.right_stack.setMinimumWidth(430)
        right_panel.addWidget(self.right_stack, stretch=1)
        main.addLayout(right_panel, stretch=2)

        # -- 第 0 页：制造参数 --
        right_scroll = QScrollArea(); right_scroll.setWidgetResizable(True)
        right_container = QWidget(); right = QVBoxLayout(right_container)
        right.setContentsMargins(6, 6, 6, 6); right.setSpacing(8)
        right_scroll.setWidget(right_container)
        self.right_stack.addWidget(right_scroll)

        manuf_group = QGroupBox("Electrode Manufacturing Parameters")
        manuf = QGridLayout(manuf_group)
        self.spin_areal_loading = QDoubleSpinBox(); self.spin_areal_loading.setRange(0.01, 200); self.spin_areal_loading.setDecimals(3); self.spin_areal_loading.setValue(10.0)
        self.spin_thickness = QDoubleSpinBox(); self.spin_thickness.setRange(1, 1000); self.spin_thickness.setDecimals(3); self.spin_thickness.setValue(80.0)
        self.combo_porosity_mode = QComboBox(); self.combo_porosity_mode.addItems(["Auto from loading/thickness", "Manual porosity"])
        self.spin_manual_porosity = QDoubleSpinBox(); self.spin_manual_porosity.setRange(1, 95); self.spin_manual_porosity.setDecimals(2); self.spin_manual_porosity.setValue(35.0)
        r = 0
        manuf.addWidget(QLabel("Areal loading (mg/cm²)"), r, 0); manuf.addWidget(self.spin_areal_loading, r, 1); r += 1
        manuf.addWidget(QLabel("Electrode thickness (um)"), r, 0); manuf.addWidget(self.spin_thickness, r, 1); r += 1
        manuf.addWidget(QLabel("Porosity mode"), r, 0); manuf.addWidget(self.combo_porosity_mode, r, 1); r += 1
        manuf.addWidget(QLabel("Manual porosity (%)"), r, 0); manuf.addWidget(self.spin_manual_porosity, r, 1)
        right.addWidget(manuf_group)

        comp_group = QGroupBox("Composition and Density")
        comp = QGridLayout(comp_group)
        self.spin_am_wt = QDoubleSpinBox(); self.spin_am_wt.setRange(0, 100); self.spin_am_wt.setDecimals(3); self.spin_am_wt.setValue(90.0)
        self.spin_c_wt = QDoubleSpinBox(); self.spin_c_wt.setRange(0, 100); self.spin_c_wt.setDecimals(3); self.spin_c_wt.setValue(5.0)
        self.spin_b_wt = QDoubleSpinBox(); self.spin_b_wt.setRange(0, 100); self.spin_b_wt.setDecimals(3); self.spin_b_wt.setValue(5.0)
        self.spin_am_density = QDoubleSpinBox(); self.spin_am_density.setRange(0.1, 20); self.spin_am_density.setDecimals(3); self.spin_am_density.setValue(3.60)
        self.spin_c_density = QDoubleSpinBox(); self.spin_c_density.setRange(0.1, 20); self.spin_c_density.setDecimals(3); self.spin_c_density.setValue(2.00)
        self.spin_b_density = QDoubleSpinBox(); self.spin_b_density.setRange(0.1, 20); self.spin_b_density.setDecimals(3); self.spin_b_density.setValue(1.78)
        r = 0
        for label, widget in [("AM weight ratio (%)", self.spin_am_wt), ("Carbon weight ratio (%)", self.spin_c_wt), ("Binder weight ratio (%)", self.spin_b_wt), ("AM density (g/cm³)", self.spin_am_density), ("Carbon density (g/cm³)", self.spin_c_density), ("Binder density (g/cm³)", self.spin_b_density)]:
            comp.addWidget(QLabel(label), r, 0); comp.addWidget(widget, r, 1); r += 1
        right.addWidget(comp_group)

        geom_group = QGroupBox("Geometry / PSD / CBD")
        geom = QGridLayout(geom_group)
        self.spin_dim_x = QDoubleSpinBox(); self.spin_dim_x.setRange(1, 500); self.spin_dim_x.setDecimals(3); self.spin_dim_x.setValue(35.0)
        self.spin_dim_y = QDoubleSpinBox(); self.spin_dim_y.setRange(1, 500); self.spin_dim_y.setDecimals(3); self.spin_dim_y.setValue(35.0)
        self.spin_resolution = QDoubleSpinBox(); self.spin_resolution.setRange(0.01, 10); self.spin_resolution.setDecimals(5); self.spin_resolution.setValue(0.33)
        self.combo_psd = QComboBox(); self.combo_psd.addItems(["Gaussian D50", "Uniform Dmin-Dmax", "Imported PSD File"])
        self.spin_d_min = QDoubleSpinBox(); self.spin_d_min.setRange(0.01, 100); self.spin_d_min.setDecimals(3); self.spin_d_min.setValue(2.0)
        self.spin_d50 = QDoubleSpinBox(); self.spin_d50.setRange(0.01, 100); self.spin_d50.setDecimals(3); self.spin_d50.setValue(5.0)
        self.spin_d_max = QDoubleSpinBox(); self.spin_d_max.setRange(0.01, 100); self.spin_d_max.setDecimals(3); self.spin_d_max.setValue(8.0)
        self.spin_d_std = QDoubleSpinBox(); self.spin_d_std.setRange(0.01, 100); self.spin_d_std.setDecimals(3); self.spin_d_std.setValue(1.0)
        self.spin_w = QDoubleSpinBox(); self.spin_w.setRange(0, 1); self.spin_w.setDecimals(3); self.spin_w.setSingleStep(0.05); self.spin_w.setValue(0.50)
        self.btn_load_psd = QPushButton("Load PSD File")
        self.lbl_psd_file = QLabel("No PSD file loaded"); self.lbl_psd_file.setWordWrap(True)
        self.psd_canvas = PSDCanvas(); self.psd_canvas.setMinimumHeight(180); self.psd_canvas.setMaximumHeight(230)
        r = 0
        for label, widget in [("Dimension X (um)", self.spin_dim_x), ("Dimension Y (um)", self.spin_dim_y), ("Resolution (um/voxel)", self.spin_resolution)]:
            geom.addWidget(QLabel(label), r, 0); geom.addWidget(widget, r, 1); r += 1
        geom.addWidget(QLabel("PSD mode"), r, 0); geom.addWidget(self.combo_psd, r, 1); r += 1
        for label, widget in [("AM Dmin (um)", self.spin_d_min), ("AM D50 / mean (um)", self.spin_d50), ("AM Dmax (um)", self.spin_d_max), ("AM PSD std (um)", self.spin_d_std), ("CBD cluster degree w", self.spin_w)]:
            geom.addWidget(QLabel(label), r, 0); geom.addWidget(widget, r, 1); r += 1
        geom.addWidget(self.btn_load_psd, r, 0); geom.addWidget(self.lbl_psd_file, r, 1); r += 1
        geom.addWidget(QLabel("PSD preview"), r, 0, 1, 2); r += 1
        geom.addWidget(self.psd_canvas, r, 0, 1, 2)
        right.addWidget(geom_group)

        adv_group = QGroupBox("Advanced Generation Settings")
        adv = QGridLayout(adv_group)
        self.spin_overlap = QDoubleSpinBox(); self.spin_overlap.setRange(0, 100); self.spin_overlap.setDecimals(2); self.spin_overlap.setValue(20.0)
        self.spin_boundary_margin = QDoubleSpinBox(); self.spin_boundary_margin.setRange(0, 50); self.spin_boundary_margin.setDecimals(2); self.spin_boundary_margin.setValue(0.0)
        self.spin_seed = QSpinBox(); self.spin_seed.setRange(0, 999999); self.spin_seed.setValue(1)
        self.check_auto_cbd = QCheckBox("Generate CBD automatically"); self.check_auto_cbd.setChecked(True)
        self.btn_generate = QPushButton("Generate Electrode")
        self.btn_regenerate_cbd = QPushButton("Regenerate CBD only")
        adv.addWidget(QLabel("Max AM-AM overlap (%)"), 0, 0); adv.addWidget(self.spin_overlap, 0, 1)
        adv.addWidget(QLabel("Boundary margin"), 1, 0); adv.addWidget(self.spin_boundary_margin, 1, 1)
        adv.addWidget(QLabel("Structure random seed"), 2, 0); adv.addWidget(self.spin_seed, 2, 1)
        adv.addWidget(self.check_auto_cbd, 3, 0, 1, 2)
        adv.addWidget(self.btn_generate, 4, 0, 1, 2)
        adv.addWidget(self.btn_regenerate_cbd, 5, 0, 1, 2)
        right.addWidget(adv_group)
        right.addStretch(1)

        # -- 第 1 页：深度学习（lazy init，首次点击"深度学习"时加载） --
        self.dl_container = QWidget()
        self.dl_container_layout = QVBoxLayout(self.dl_container)
        self.dl_container_layout.setContentsMargins(0, 0, 0, 0)
        self.right_stack.addWidget(self.dl_container)
        self.dl_initialized = False

    def _connect_signals(self):
        self.btn_xy.clicked.connect(lambda: self.set_plane("XY"))
        self.btn_xz.clicked.connect(lambda: self.set_plane("XZ"))
        self.btn_yz.clicked.connect(lambda: self.set_plane("YZ"))
        self.btn_3d.clicked.connect(self.open_3d_view)
        self.btn_clear.clicked.connect(self.clear_volume)
        # 顶部导航切换
        self.btn_mfg.clicked.connect(lambda: self._switch_right_panel(0))
        self.btn_dl.clicked.connect(lambda: self._switch_right_panel(1))
        self.btn_mfg.setChecked(True)
        self.right_stack.setCurrentIndex(0)
        self.slice_slider.valueChanged.connect(self.update_view)
        self.btn_generate.clicked.connect(self.generate_electrode)
        self.btn_regenerate_cbd.clicked.connect(self.regenerate_cbd)
        self.btn_add_separator.clicked.connect(self.add_separator)
        self.btn_add_cc.clicked.connect(self.add_current_collector)
        self.btn_add_li.clicked.connect(self.add_li_foil)
        self.btn_load_npy.clicked.connect(self.load_npy)
        self.btn_save_npy.clicked.connect(self.save_npy)
        self.btn_save_tif.clicked.connect(self.save_tif)
        self.btn_load_psd.clicked.connect(self.load_psd_file)
        self.btn_save_bdf.clicked.connect(self.save_bdf_mesh)
        for widget in [self.spin_areal_loading, self.spin_thickness, self.spin_manual_porosity, self.spin_am_wt, self.spin_c_wt, self.spin_b_wt, self.spin_am_density, self.spin_c_density, self.spin_b_density, self.spin_dim_x, self.spin_dim_y, self.spin_resolution, self.spin_d_min, self.spin_d50, self.spin_d_max, self.spin_d_std, self.spin_w]:
            widget.valueChanged.connect(self.update_parameter_preview)
        for widget in [self.spin_d_min, self.spin_d50, self.spin_d_max, self.spin_d_std]:
            widget.valueChanged.connect(self.update_psd_preview)
        self.combo_porosity_mode.currentIndexChanged.connect(self.update_porosity_input_state)
        self.combo_porosity_mode.currentIndexChanged.connect(self.update_parameter_preview)
        self.combo_psd.currentIndexChanged.connect(self.update_psd_input_state)
        self.combo_psd.currentIndexChanged.connect(self.update_psd_preview)

    # ======================== 后端 API 服务器管理 ========================

    def _start_api_server(self):
        """启动后端 API 服务器（如未运行）。
        使用 multiprocessing.Process 而非 subprocess.Popen，
        确保 PyInstaller 打包后正常工作。"""
        self.api_ready = False  # 标志位：API 是否可用

        # 先检查端口是否已被占用（服务器可能已在运行）
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', 8001))
        sock.close()
        if result == 0:
            self.log("后端 API 服务器已在运行 (port 8001)")
            self.api_ready = True
            return

        try:
            self.api_process = start_api_process()
            # 等待服务器就绪（最多 30 秒）
            for _ in range(60):
                time.sleep(0.5)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                r = sock.connect_ex(('127.0.0.1', 8001))
                sock.close()
                if r == 0:
                    self.log("后端 API 服务器已启动 (port 8001)")
                    self.api_ready = True
                    return

            # 超时：提示用户
            self.log("⚠ 后端 API 服务器启动超时")
            QMessageBox.critical(
                self,
                "后端服务启动失败",
                "API 服务器启动超时（30秒）。\n\n"
                "深度学习功能将无法使用。\n"
                "请关闭本应用后重新启动。\n\n"
                "如问题持续，请检查后台是否已有残留进程占用端口 8001。"
            )
        except Exception as e:
            self.log(f"⚠ 后端 API 服务器启动失败: {e}")
            QMessageBox.critical(
                self,
                "后端服务启动失败",
                f"API 服务器启动出错：{e}\n\n"
                "深度学习功能将无法使用。\n"
                "请关闭本应用后重新启动。"
            )

    def _stop_api_server(self):
        """关闭后端 API 子进程。"""
        if self.api_process is not None:
            kill_api_process(self.api_process)
            self.log("后端 API 服务器已停止")
            self.api_process = None

    def closeEvent(self, event):
        """窗口关闭时清理后端子进程。"""
        self._stop_api_server()
        super().closeEvent(event)

    def cleanup_on_exit(self):
        """应用退出时清理所有后台进程（aboutToQuit 兜底）。"""
        self._stop_api_server()

    # ======================== 右侧面板切换 ========================

    def _switch_right_panel(self, index):
        """切换右侧面板：0=制造参数, 1=深度学习"""
        if index == 1:
            if self.dl_loading:
                QMessageBox.information(self, "请稍等", "后台正在加载'深度学习'")
                self.btn_mfg.setChecked(True)
                self.btn_dl.setChecked(False)
                return
            if not getattr(self, 'api_ready', True):
                QMessageBox.warning(
                    self,
                    "深度学习不可用",
                    "后端 API 服务器未启动，深度学习功能无法使用。\n"
                    "请关闭应用后重新启动。"
                )
                self.btn_mfg.setChecked(True)
                self.btn_dl.setChecked(False)
                return
        self.right_stack.setCurrentIndex(index)
        self.btn_mfg.setChecked(index == 0)
        self.btn_dl.setChecked(index == 1)

    def _init_dl_tabs(self):
        """后台加载深度学习 Stage 页面（单个失败不影响其余）。"""
        # 清空占位
        while self.dl_container_layout.count():
            item = self.dl_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        tabs = QTabWidget()

        # 每个 Stage 独立 try，一个失败不影响其他
        _stages = [
            ("Stage1: VAE",    lambda: __import__('pages.stage1_page', fromlist=['Stage1Page']).Stage1Page()),
            ("Stage2: LDM",    lambda: __import__('pages.stage2_page', fromlist=['Stage2Page']).Stage2Page()),
            ("Stage3: 条件生成", lambda: __import__('pages.stage3_page', fromlist=['Stage3Page']).Stage3Page(self)),
            ("Stage4: 特定体积", lambda: __import__('pages.stage4_page', fromlist=['Stage4Page']).Stage4Page(self)),
            ("Stage5: CBD/参数", lambda: __import__('pages.stage5_page', fromlist=['Stage5Page']).Stage5Page(self)),
            ("任务历史",        lambda: __import__('pages.history_page', fromlist=['HistoryPage']).HistoryPage()),
        ]

        self._stage_pages = {}  # 保存引用，用于切 tab 时自动刷新模型列表
        self._dl_tabs = tabs

        for name, factory in _stages:
            try:
                page = factory()
                self._stage_pages[name] = page
                tabs.addTab(self._wrap_scroll(page), name)
                # 保存 history_page 引用，供 stage3/4/5 的 refresh_task_list 调用
                if name == "任务历史":
                    self.history_page = page
            except Exception as e:
                print(f"[DL加载] {name} 加载失败: {e}")
                err_label = QLabel(f"{name}\n加载失败：{e}")
                err_label.setStyleSheet("color: #c00; padding: 20px;")
                tabs.addTab(self._wrap_scroll(err_label), name)

        # 切换到 Stage3 / Stage4 时自动重新扫描模型目录
        tabs.currentChanged.connect(self._on_dl_tab_changed)

        self.dl_container_layout.addWidget(tabs)
        self.dl_initialized = True
        self.dl_loading = False

    def _on_dl_tab_changed(self, index):
        """切换到 Stage3/Stage4 时自动刷新模型列表。"""
        tab_text = self._dl_tabs.tabText(index)
        # Stage3 和 Stage4 都有 load_models() 方法
        refresh_keys = ["Stage3: 条件生成", "Stage4: 特定体积"]
        for key in refresh_keys:
            if tab_text.startswith(key[:6]) and key in self._stage_pages:
                page = self._stage_pages[key]
                if hasattr(page, 'load_models'):
                    try:
                        page.load_models()
                    except Exception as e:
                        print(f"[模型刷新] {key} 失败: {e}")

    @staticmethod
    def _wrap_scroll(widget):
        """将 widget 包裹在可滚动的 QScrollArea 中。"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        return scroll

    def show_message(self, title, text):
        """兼容 stage3/4/5 页面的 show_message 调用。"""
        QMessageBox.information(self, title, text)

    def log(self, text):
        self.log_box.append(str(text))

    def get_porosity_mode(self):
        return "auto" if self.combo_porosity_mode.currentText().startswith("Auto") else "manual"

    def get_psd_mode(self):
        return self.combo_psd.currentText()

    def update_porosity_input_state(self):
        self.spin_manual_porosity.setEnabled(self.get_porosity_mode() == "manual")

    def update_psd_input_state(self):
        mode = self.get_psd_mode()
        imported = mode == "Imported PSD File"
        gaussian = mode == "Gaussian D50"
        self.spin_d_min.setEnabled(not imported)
        self.spin_d_max.setEnabled(not imported)
        self.spin_d50.setEnabled(gaussian)
        self.spin_d_std.setEnabled(gaussian)
        self.btn_load_psd.setEnabled(imported)
        self.lbl_psd_file.setEnabled(imported)

    def compute_current_params(self):
        return compute_electrode_parameters(
            self.spin_areal_loading.value(), self.spin_thickness.value(),
            self.spin_am_wt.value(), self.spin_c_wt.value(), self.spin_b_wt.value(),
            self.spin_am_density.value(), self.spin_c_density.value(), self.spin_b_density.value(),
            self.get_porosity_mode(), self.spin_manual_porosity.value()
        )

    def update_parameter_preview(self):
        try:
            p = self.compute_current_params()
            nx = int(round(self.spin_dim_x.value() / self.spin_resolution.value()))
            ny = int(round(self.spin_dim_y.value() / self.spin_resolution.value()))
            nz = int(round(self.spin_thickness.value() / self.spin_resolution.value()))
            text = (
                f"rho_solid = {p['rho_solid']:.4f} g/cm³\n"
                f"rho_coating = {p['rho_coating']:.4f} g/cm³\n"
                f"auto porosity = {100*p['auto_porosity']:.2f}%\n"
                f"used porosity = {100*p['porosity_used']:.2f}%\n"
                f"AM volume fraction = {100*p['phi_am_total']:.2f}%\n"
                f"CBD volume fraction = {100*p['phi_cbd_total']:.2f}%\n"
                f"Pore volume fraction = {100*p['phi_pore_total']:.2f}%\n"
                f"voxel shape ≈ {nx} x {ny} x {nz}\n"
                f"total voxels ≈ {nx*ny*nz:,}"
            )
            self.preview_box.setText(text)
        except Exception as e:
            self.preview_box.setText(f"Parameter error: {e}")

    def update_psd_preview(self):
        self.psd_canvas.update_plot(
            self.get_psd_mode(), self.spin_d_min.value(), self.spin_d50.value(),
            self.spin_d_max.value(), self.spin_d_std.value(),
            self.imported_diameters, self.imported_probabilities,
        )

    def load_psd_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load PSD file", "", "PSD files (*.csv *.txt *.xlsx);;CSV (*.csv);;Text (*.txt);;Excel (*.xlsx)")
        if not path:
            return
        try:
            d, p = load_psd_file(path)
        except Exception as e:
            QMessageBox.warning(self, "PSD file error", str(e))
            return
        self.imported_psd_path = path
        self.imported_diameters = d
        self.imported_probabilities = p
        self.lbl_psd_file.setText(os.path.basename(path))
        self.log(f"Loaded PSD file: {path}")
        self.update_psd_preview()

    def set_plane(self, plane):
        self.current_plane = plane
        self.reset_slider_range()
        self.update_view()

    def reset_slider_range(self):
        if self.volume is None:
            self.slice_slider.setMinimum(0); self.slice_slider.setMaximum(0); self.slice_slider.setValue(0)
            return
        nz, ny, nx = self.volume.shape
        max_idx = nz - 1 if self.current_plane == "XY" else (ny - 1 if self.current_plane == "XZ" else nx - 1)
        old = self.slice_slider.value()
        self.slice_slider.setMinimum(0); self.slice_slider.setMaximum(max_idx); self.slice_slider.setValue(min(old, max_idx))

    def update_view(self):
        idx = self.slice_slider.value()
        self.slice_canvas.show_slice(self.volume, self.current_plane, idx)
        if self.volume is None:
            self.slice_label.setText("0/0")
            self.structure_label.setText("Structure: none")
            return
        nz, ny, nx = self.volume.shape
        total = nz if self.current_plane == "XY" else (ny if self.current_plane == "XZ" else nx)
        self.slice_label.setText(f"{idx+1}/{total}")
        self.structure_label.setText(
            f"Structure: {nx} x {ny} x {nz} voxels | "
            f"{nx*self.resolution_um:.2f} x {ny*self.resolution_um:.2f} x {nz*self.resolution_um:.2f} um | "
            f"{len(np.unique(self.volume))} phases detected"
        )

    def clear_volume(self):
        self.volume = None
        self.log("Structure cleared.")
        self.reset_slider_range(); self.update_view()

    def generate_electrode(self):
        try:
            params = self.compute_current_params()
        except Exception as e:
            QMessageBox.warning(self, "Parameter error", str(e))
            return
        if self.get_psd_mode() == "Imported PSD File" and self.imported_diameters is None:
            QMessageBox.warning(self, "No PSD file", "Please load a PSD file first.")
            return
        self.resolution_um = self.spin_resolution.value()
        self.log("Generating electrode from manufacturing parameters ...")
        self.log(f"rho_solid={params['rho_solid']:.4f} g/cm3, rho_coating={params['rho_coating']:.4f} g/cm3, porosity_used={100*params['porosity_used']:.2f}%")
        self.log(f"Target volume fractions: AM={100*params['phi_am_total']:.2f}%, CBD={100*params['phi_cbd_total']:.2f}%, pore={100*params['phi_pore_total']:.2f}%")
        QApplication.processEvents()
        vol, info = generate_am_structure(
            self.spin_dim_x.value(), self.spin_dim_y.value(), self.spin_thickness.value(), self.resolution_um,
            params["phi_am_total"], self.get_psd_mode(), self.spin_d_min.value(), self.spin_d50.value(),
            self.spin_d_max.value(), self.spin_d_std.value(), self.spin_overlap.value(), self.spin_boundary_margin.value(),
            self.spin_seed.value(), imported_diameters=self.imported_diameters, imported_probabilities=self.imported_probabilities
        )
        self.volume = vol
        self.log(f"AM generated. shape={info['shape']}, particles={info['placed_spheres']}, attempts={info['attempts']}, rejected_by_overlap={info['rejected_overlap']}")
        self.log(f"AM target={100*info['target_am_fraction']:.2f}%, AM actual={100*info['actual_am_fraction']:.2f}%, error={100*info['am_fraction_error']:.3f}%")
        if self.check_auto_cbd.isChecked():
            self._apply_cbd(params)
        self.log(volume_statistics(self.volume))
        self.reset_slider_range(); self.update_view()

    def _apply_cbd(self, params):
        self.log("Generating CBD from carbon + binder volume fraction ...")
        QApplication.processEvents()
        self.volume, info = add_cbd_by_target_fraction(self.volume, params["phi_cbd_total"], self.spin_w.value(), self.spin_seed.value() + 1000)
        self.log(f"CBD generated. w={self.spin_w.value():.3f}, target={info['target_cbd_voxels']}, actual={info['actual_cbd_voxels']}, fraction={100*info['actual_cbd_fraction']:.2f}%")

    def regenerate_cbd(self):
        if self.volume is None:
            QMessageBox.warning(self, "No structure", "Please generate or load an electrode first.")
            return
        try:
            params = self.compute_current_params()
        except Exception as e:
            QMessageBox.warning(self, "Parameter error", str(e)); return
        self._apply_cbd(params)
        self.log(volume_statistics(self.volume))
        self.update_view()

    def add_separator(self):
        if self.volume is None:
            QMessageBox.warning(self, "No structure", "Please generate or load an electrode first."); return
        self.volume = add_layer_zmax(self.volume, PHASE["separator"], self.spin_separator.value(), self.resolution_um)
        self.log(f"Separator added at Z max. thickness={self.spin_separator.value()} um")
        self.reset_slider_range(); self.update_view()

    def add_current_collector(self):
        if self.volume is None:
            QMessageBox.warning(self, "No structure", "Please generate or load an electrode first."); return
        self.volume = add_layer_zmin(self.volume, PHASE["current_collector"], self.spin_cc.value(), self.resolution_um)
        self.log(f"Current collector added at Z min. thickness={self.spin_cc.value()} um")
        self.reset_slider_range(); self.update_view()

    def add_li_foil(self):
        if self.volume is None:
            QMessageBox.warning(self, "No structure", "Please generate or load an electrode first."); return
        self.volume = add_layer_zmax(self.volume, PHASE["li_foil"], self.spin_li.value(), self.resolution_um)
        self.log(f"Li foil added at Z max. thickness={self.spin_li.value()} um")
        self.reset_slider_range(); self.update_view()

    def save_npy(self):
        if self.volume is None:
            QMessageBox.warning(self, "No structure", "Nothing to save."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save .npy", "structure.npy", "NumPy Matrix (*.npy)")
        if not path:
            return
        if not path.lower().endswith(".npy"):
            path += ".npy"
        np.save(path, self.volume)
        self.log(f"Saved matrix: {path}")

    def save_tif(self):
        if self.volume is None:
            QMessageBox.warning(self, "No structure", "Nothing to save."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save .tif", "structure.tif", "TIFF Stack (*.tif *.tiff)")
        if not path:
            return
        if not path.lower().endswith((".tif", ".tiff")):
            path += ".tif"
        tifffile.imwrite(path, self.volume.astype(np.uint8))
        self.log(f"Saved image stack: {path}")

    def load_npy(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load .npy", "", "NumPy Matrix (*.npy)")
        if not path:
            return
        arr = np.load(path)
        if arr.ndim != 3:
            QMessageBox.warning(self, "Invalid file", "The .npy file must be a 3D matrix."); return
        self.volume = arr.astype(np.uint8)
        self.resolution_um = self.spin_resolution.value()
        self.log(f"Loaded matrix: {path}, shape={self.volume.shape}")
        self.log(volume_statistics(self.volume))
        self.reset_slider_range(); self.update_view()

    def open_3d_view(self):
        if self.volume is None:
            QMessageBox.warning(self, "No structure", "Please generate or load an electrode first."); return
        if not HAS_3D or not HAS_SKIMAGE:
            QMessageBox.warning(self, "Missing package", "3D viewer requires pyvista, pyvistaqt and scikit-image.")
            return
        dlg = Volume3DDialog(self.volume, self.resolution_um, self)
        dlg.exec()

    def save_bdf_mesh(self):
        if self.volume is None:
            QMessageBox.warning(self, "No structure", "Please generate or load an electrode first."); return
        pz = self.spin_precision_z.value()
        py = self.spin_precision_y.value()
        px = self.spin_precision_x.value()
        decomp = self.combo_tet_decomposition.currentText()
        phase_method = self.combo_phase_method.currentText()

        node_labels, _, _, _ = node_grid_from_label_volume(self.volume, (pz, py, px))
        nz_n, ny_n, nx_n = node_labels.shape
        n_cubes = max(0, (nz_n - 1) * (ny_n - 1) * (nx_n - 1))
        tets_per = 6 if decomp.startswith("6") else 5
        if phase_method == "global_centroid":
            est_tets = n_cubes * tets_per * 24
        elif phase_method == "mixed_centroid":
            # 只是保守提示。实际数量取决于 mixed block 数。
            est_tets = n_cubes * tets_per * 8
        else:
            est_tets = n_cubes * tets_per
        est_nodes = nz_n * ny_n * nx_n

        if est_tets > 5_000_000:
            reply = QMessageBox.question(
                self, "Large mesh",
                f"Estimated tetrahedra: {est_tets:,}\nEstimated nodes: {est_nodes:,}\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        default = f"openmesocell_precision_{pz}_{py}_{px}.bdf"
        path, _ = QFileDialog.getSaveFileName(self, "Save COMSOL BDF mesh", default, "NASTRAN BDF (*.bdf);;All files (*)")
        if not path:
            return
        if not path.lower().endswith(".bdf"):
            path += ".bdf"

        try:
            self.log(f"Exporting voxelization BDF mesh... precision=[{pz},{py},{px}], decomposition={decomp}, method={phase_method}, estimated_tets={est_tets:,}")
            QApplication.processEvents()
            report = export_voxel_bdf(self.volume, self.resolution_um, path, (pz, py, px), decomp, phase_method=phase_method)
            self.log(
                f"BDF mesh saved: {path}\n"
                f"  node_grid_shape={report['node_grid_shape_zyx']}\n"
                f"  used_nodes={report['used_nodes']:,}, tetrahedra={report['tetrahedra']:,}\n"
                f"  phases={report['phases']}, property_ids={report['property_ids']}\n"
                f"  method={report.get('phase_method', phase_method)}, mixed_blocks={report.get('mixed_blocks', 0)}, subdivided_base_tets={report.get('subdivided_base_tets', 0)}\n"
                f"  fixed_orientation={report['fixed_orientation']}, removed_degenerate={report['removed_degenerate']}\n"
                f"  mapping={report['mapping_path']}"
            )
            QMessageBox.information(self, "Mesh saved", f"COMSOL BDF mesh saved.\n\nTetrahedra: {report['tetrahedra']:,}\nFile: {path}")
        except Exception as e:
            QMessageBox.warning(self, "Mesh export error", str(e))
            self.log(f"Mesh export failed: {e}")


# ============================================================
# 启动 Splash — 与 main.py 保持一致
# ============================================================

class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenMesoCell")
        self.setFixedSize(420, 220)
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.CustomizeWindowHint
        )
        layout = QVBoxLayout()
        self.label = QLabel("正在启动 OpenMesoCell...")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
        """)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addStretch()
        layout.addWidget(self.label)
        layout.addWidget(self.progress)
        layout.addStretch()
        self.setLayout(layout)
        self.timer = QTimer()
        self.timer.timeout.connect(self.fake_progress)
        self.current = 0
        self.timer.start(120)

    def fake_progress(self):
        if self.current < 90:
            self.current += 1
            self.progress.setValue(self.current)

    def finish(self):
        self.timer.stop()
        self.progress.setValue(100)
        self.label.setText("启动完成")


# ============================================================
# API 服务器管理（模块级辅助函数）
# ============================================================

def run_api_server():
    """在子进程中启动 FastAPI 后端（供 multiprocessing.Process 调用）"""
    from backend.api_server import start_server
    start_server()


def start_api_process():
    """启动 API 子进程（multiprocessing），返回 Process 对象。

    使用 multiprocessing.Process 而非 subprocess.Popen 的原因：
    PyInstaller 打包后 sys.executable 是 .exe 本身而非 python，
    subprocess.Popen([sys.executable, script]) 会重新启动 GUI 主程序，
    而非 API 服务器。multiprocessing.Process 直接 import 模块运行，
    不依赖外部 python 解释器或脚本文件路径。
    """
    try:
        proc = multiprocessing.Process(
            target=run_api_server,
            daemon=True,
            name="api-server",
        )
        proc.start()
        print(f"[OK] API 进程已启动 (pid={proc.pid})")
        return proc
    except Exception as e:
        print(f"启动 API 进程失败: {e}")
        return None


def wait_for_api_port(timeout=30, app=None):
    """轮询等待 API 端口就绪，过程中保持 UI 响应。

    app: 可选的 QApplication 实例，传入则会在等待期间处理 UI 事件
         （保持 splash 进度条动画）。
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            r = sock.connect_ex(('127.0.0.1', 8001))
            sock.close()
            if r == 0:
                print("[OK] API 端口已就绪")
                return True
        except Exception as e:
            print("等待 API:", e)
        # 保持 UI 响应（splash 进度条动画）
        if app is not None:
            app.processEvents()
        time.sleep(0.5)
    return False


def kill_api_process(proc):
    """终止 API 子进程（适配 multiprocessing.Process）"""
    if proc is None:
        return
    try:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2)
        print("API 进程已关闭")
    except Exception as e:
        print(f"关闭 API 进程时出错: {e}")


# ============================================================
# 主入口 — 与 main.py 保持一致的启动流程
# ============================================================

def main():
    # ---- PyInstaller 打包支持（Windows 必须） ----
    multiprocessing.freeze_support()

    # ---- 单实例锁（socket 保持存活直到进程退出） ----
    lock_socket = acquire_lock()
    if lock_socket is None:
        app = QApplication(sys.argv)
        QMessageBox.warning(
            None,
            "提示",
            "OpenMesoCell 已经在运行中，请勿重复启动"
        )
        sys.exit(0)

    print("=" * 50)
    print("开始启动应用...")
    print("=" * 50)

    app = QApplication(sys.argv)

    # ---- Splash 屏幕 ----
    splash = SplashScreen()
    splash.show()
    app.processEvents()

    # ---- 启动 API ----
    splash.label.setText("正在启动后端 API...")
    splash.repaint()
    app.processEvents()

    api_process = start_api_process()

    # ---- 等待 API 就绪（传入 app 以保持进度条动画） ----
    splash.label.setText("程序启动中（请稍等）...")
    splash.repaint()
    app.processEvents()

    print("等待 API 完全初始化...")

    if not wait_for_api_port(timeout=300, app=app):
        # 1. 关闭 Splash 进度条窗口
        splash.close()
        app.processEvents()

        # 2. 终止 API 子进程
        kill_api_process(api_process)

        # 3. 弹窗提示用户
        QMessageBox.critical(
            None,
            "错误",
            "后端 API 初始化失败\n\n"
            "深度学习功能将无法使用。\n"
            "请关闭本应用后重新启动。\n\n"
            "如问题持续，请检查后台是否已有残留进程占用端口 8001。"
        )

        # 4. 彻底退出：先尝试 Qt 正常退出，再用 os._exit() 兜底
        #    （PyInstaller 打包后 sys.exit() 可能被 bootloader 吞掉）
        try:
            app.quit()
            sys.exit(1)
        finally:
            os._exit(1)

    # ---- API 就绪，创建主窗口 ----
    splash.label.setText("启动界面...")
    splash.progress.setValue(95)
    splash.repaint()
    app.processEvents()

    window = OpenMesoCellWindow()
    window.show()
    splash.finish()
    splash.close()

    # ---- 清理 ----
    def cleanup():
        kill_api_process(api_process)
        window._stop_api_server()

    app.aboutToQuit.connect(cleanup)

    print("=" * 50)
    print("GUI 已启动")
    print("=" * 50)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
