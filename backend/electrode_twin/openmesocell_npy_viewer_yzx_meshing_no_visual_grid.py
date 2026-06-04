"""
OpenMesoCell NPY Viewer YZX
============================================================
外部 NPY 导入、切片/3D 可视化、手动分辨率、隔膜/集流体/Li foil 添加、保存图片，以及 COMSOL BDF 四面体体网格导出。

重要约定：
    外部导入的 .npy 默认轴顺序固定为 volume[Y, Z, X]
        Y = 切片堆积方向
        Z = 电极厚度方向 / 传输方向
        X = 横向方向

    程序内部统一转换为 volume[Z, Y, X] 以便显示 XY/XZ/YZ 三个标准平面。

相标签：
    0 = Pore / Electrolyte
    1 = Active Material, AM
    2 = CBD
    3 = Separator
    4 = Current Collector
    5 = Li Foil

安装：
    conda activate openmesocell
    pip install numpy matplotlib PySide6 scikit-image pyvista pyvistaqt tifffile

运行：
    python openmesocell_npy_viewer_yzx_meshing_no_visual_grid.py
"""

import os
import sys
import math
import json
import numpy as np
import tifffile

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QSlider,
    QTextEdit,
    QGroupBox,
    QMessageBox,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QSpinBox,
    QComboBox,
    QScrollArea,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

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
    0: "#214ee8",   # pore: blue
    1: "#5bcf72",   # AM: green
    2: "#f5ec1b",   # CBD: yellow
    3: "#b83ee6",   # separator: purple
    4: "#f4a333",   # current collector: orange
    5: "#9db7d5",   # Li foil: gray-blue
}

CMAP = ListedColormap([PHASE_COLOR[i] for i in range(6)])
NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], CMAP.N)


# ============================================================
# Utility functions
# ============================================================
def yzx_to_zyx(volume_yzx: np.ndarray) -> np.ndarray:
    """外部默认 [Y,Z,X] -> 内部 [Z,Y,X]。"""
    return np.transpose(volume_yzx, (1, 0, 2)).astype(np.uint8, copy=False)


def zyx_to_yzx(volume_zyx: np.ndarray) -> np.ndarray:
    """内部 [Z,Y,X] -> 外部 [Y,Z,X]。"""
    return np.transpose(volume_zyx, (1, 0, 2)).astype(np.uint8, copy=False)


def add_layer_zmax(volume_zyx: np.ndarray, phase_label: int, thickness_um: float, voxel_size_z_um: float) -> np.ndarray:
    n_layer = int(round(float(thickness_um) / float(voxel_size_z_um)))
    if n_layer <= 0:
        return volume_zyx
    _, ny, nx = volume_zyx.shape
    layer = np.full((n_layer, ny, nx), phase_label, dtype=np.uint8)
    return np.concatenate([volume_zyx, layer], axis=0)


def add_layer_zmin(volume_zyx: np.ndarray, phase_label: int, thickness_um: float, voxel_size_z_um: float) -> np.ndarray:
    n_layer = int(round(float(thickness_um) / float(voxel_size_z_um)))
    if n_layer <= 0:
        return volume_zyx
    _, ny, nx = volume_zyx.shape
    layer = np.full((n_layer, ny, nx), phase_label, dtype=np.uint8)
    return np.concatenate([layer, volume_zyx], axis=0)


def volume_statistics(volume_zyx: np.ndarray, voxel_size_xyz) -> str:
    vx, vy, vz = voxel_size_xyz
    nz, ny, nx = volume_zyx.shape
    total = int(volume_zyx.size)
    lines = []
    lines.append(f"Internal shape [Z,Y,X] = {volume_zyx.shape}")
    lines.append(f"External shape [Y,Z,X] = {(ny, nz, nx)}")
    lines.append(f"Voxel size X/Y/Z = {vx:.6g}, {vy:.6g}, {vz:.6g} um")
    lines.append(f"Physical size X/Y/Z = {nx*vx:.4f}, {ny*vy:.4f}, {nz*vz:.4f} um")
    lines.append(f"Total voxels = {total:,}")
    lines.append("")
    for label in sorted(int(v) for v in np.unique(volume_zyx)):
        count = int(np.count_nonzero(volume_zyx == label))
        name = PHASE_NAME.get(label, f"Phase {label}")
        lines.append(f"{name}: {count:,} voxels, {100.0 * count / total:.4f}%")
    return "\n".join(lines)


def get_slice_image_and_extent(volume_zyx, plane, index, voxel_size_xyz):
    """
    返回用于显示的二维图像、真实物理范围 extent、标题、坐标轴名称。
    内部 volume_zyx: [Z,Y,X]
    """
    vx, vy, vz = voxel_size_xyz
    nz, ny, nx = volume_zyx.shape

    if plane == "XY":
        index = int(np.clip(index, 0, nz - 1))
        img = volume_zyx[index, :, :]
        extent = [0, nx * vx, 0, ny * vy]
        title = f"XY slice at Z index {index + 1}/{nz}"
        xlabel, ylabel = "X (um)", "Y (um)"
    elif plane == "XZ":
        index = int(np.clip(index, 0, ny - 1))
        img = volume_zyx[:, index, :]
        extent = [0, nx * vx, 0, nz * vz]
        title = f"XZ slice at Y index {index + 1}/{ny}"
        xlabel, ylabel = "X (um)", "Z (um)"
    else:  # YZ
        index = int(np.clip(index, 0, nx - 1))
        img = volume_zyx[:, :, index]
        extent = [0, ny * vy, 0, nz * vz]
        title = f"YZ slice at X index {index + 1}/{nx}"
        xlabel, ylabel = "Y (um)", "Z (um)"

    return img, extent, title, xlabel, ylabel


def add_phase_legend(ax, volume_zyx):
    unique_labels = set(int(v) for v in np.unique(volume_zyx))
    handles = []
    for label in [1, 2, 0, 3, 4, 5]:
        if label in unique_labels:
            handles.append(Patch(facecolor=PHASE_COLOR[label], edgecolor="black", label=PHASE_SHORT_NAME[label]))
    if handles:
        leg = ax.legend(
            handles=handles,
            loc="upper left",
            fontsize=7,
            framealpha=0.82,
            borderpad=0.25,
            handlelength=0.8,
            handletextpad=0.3,
            labelspacing=0.2,
        )
        leg.get_frame().set_linewidth(0.5)


def save_slice_png(volume_zyx, plane, index, voxel_size_xyz, path):
    fig = Figure(figsize=(6, 5), dpi=160)
    ax = fig.add_subplot(111)
    img, extent, title, xlabel, ylabel = get_slice_image_and_extent(
        volume_zyx, plane, index, voxel_size_xyz
    )
    ax.imshow(img, cmap=CMAP, norm=NORM, interpolation="nearest", origin="lower", extent=extent)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    add_phase_legend(ax, volume_zyx)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def create_polydata_from_phase_mask(mask_zyx, voxel_size_xyz):
    if not HAS_SKIMAGE:
        raise RuntimeError("scikit-image is not installed.")
    if np.count_nonzero(mask_zyx) == 0:
        return None

    vx, vy, vz = voxel_size_xyz
    padded = np.pad(mask_zyx.astype(np.uint8), 1, mode="constant", constant_values=0)
    verts, faces, _, _ = measure.marching_cubes(
        padded,
        level=0.5,
        spacing=(vz, vy, vx),  # skimage receives coordinates in Z,Y,X order
    )
    # Remove padding offset, then convert from Z,Y,X to X,Y,Z.
    verts[:, 0] -= vz
    verts[:, 1] -= vy
    verts[:, 2] -= vx
    verts_xyz = np.column_stack([verts[:, 2], verts[:, 1], verts[:, 0]])
    faces_pv = np.hstack([
        np.full((faces.shape[0], 1), 3, dtype=np.int64),
        faces.astype(np.int64),
    ]).ravel()
    poly = pv.PolyData(verts_xyz, faces_pv)
    poly.clean(inplace=True)
    return poly


# ============================================================
# Matplotlib slice canvas
# ============================================================
class SliceCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

    def show_slice(self, volume_zyx, plane="XY", index=0, voxel_size_xyz=(1, 1, 1)):
        self.ax.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        if volume_zyx is None:
            self.ax.text(0.5, 0.5, "No structure loaded", ha="center", va="center", transform=self.ax.transAxes)
            self.draw()
            return

        img, extent, title, xlabel, ylabel = get_slice_image_and_extent(
            volume_zyx, plane, index, voxel_size_xyz
        )
        self.ax.imshow(img, cmap=CMAP, norm=NORM, interpolation="nearest", origin="lower", extent=extent)
        self.ax.set_title(title, fontsize=10)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        add_phase_legend(self.ax, volume_zyx)
        self.fig.tight_layout()
        self.draw()




# ============================================================
# COMSOL BDF tetrahedral volume mesher
# ============================================================
def block_partition_indices(n, precision):
    """Return block boundary indices: [0, p, 2p, ..., n]."""
    precision = int(max(1, precision))
    edges = list(range(0, int(n), precision))
    if edges[-1] != int(n):
        edges.append(int(n))
    return np.asarray(edges, dtype=np.int64)


def coarsen_label_volume_quota(volume_zyx, precision_zyx=(4, 4, 4), preserve_quota=True):
    """
    Coarsen a label volume by precision blocks while preserving phase quotas.

    Internal axis order is [Z,Y,X].
    precision_zyx = (precision_z, precision_y, precision_x).
    """
    nz, ny, nx = volume_zyx.shape
    pz, py, px = [int(max(1, v)) for v in precision_zyx]
    z_edges = block_partition_indices(nz, pz)
    y_edges = block_partition_indices(ny, py)
    x_edges = block_partition_indices(nx, px)

    ncz = len(z_edges) - 1
    ncy = len(y_edges) - 1
    ncx = len(x_edges) - 1
    labels_all = list(range(6))
    n_blocks = ncz * ncy * ncx

    counts = np.zeros((6, ncz, ncy, ncx), dtype=np.int32)
    block_sizes = np.zeros((ncz, ncy, ncx), dtype=np.int32)

    for iz in range(ncz):
        z0, z1 = int(z_edges[iz]), int(z_edges[iz + 1])
        for iy in range(ncy):
            y0, y1 = int(y_edges[iy]), int(y_edges[iy + 1])
            for ix in range(ncx):
                x0, x1 = int(x_edges[ix]), int(x_edges[ix + 1])
                block = volume_zyx[z0:z1, y0:y1, x0:x1]
                block_sizes[iz, iy, ix] = block.size
                for lab in labels_all:
                    counts[lab, iz, iy, ix] = int(np.count_nonzero(block == lab))

    original_counts = {lab: int(np.count_nonzero(volume_zyx == lab)) for lab in labels_all}
    original_present = [lab for lab in labels_all if original_counts[lab] > 0]

    if not preserve_quota:
        coarse = np.argmax(counts, axis=0).astype(np.uint8)
    else:
        total_vox = int(volume_zyx.size)
        raw_targets = {lab: original_counts[lab] / total_vox * n_blocks for lab in original_present}
        targets = {lab: int(round(raw_targets[lab])) for lab in original_present}

        diff = n_blocks - sum(targets.values())
        if diff != 0 and original_present:
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

        # Assign rare/small phases first so CBD or thin layers are not swallowed by AM/pore.
        order = sorted(original_present, key=lambda lab: targets.get(lab, 0))
        for lab in order:
            target = targets.get(lab, 0)
            if target <= 0:
                continue
            available = np.flatnonzero(~assigned)
            if available.size == 0:
                break
            candidates = available[counts_flat[lab, available] > 0]
            if candidates.size < target:
                candidates = available
                need = min(target, candidates.size)
            else:
                need = target
            if need <= 0:
                continue
            frac_score = counts_flat[lab, candidates] / (block_sizes_flat[candidates] + eps)
            score = frac_score + 1e-9 * counts_flat[lab, candidates]
            pick = candidates[np.argsort(score)[-need:]]
            coarse_flat[pick] = lab
            assigned[pick] = True

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


def build_points_from_block_edges(z_edges, y_edges, x_edges, voxel_size_xyz):
    """Create node coordinates in X,Y,Z using anisotropic voxel sizes."""
    vx, vy, vz = voxel_size_xyz
    zz, yy, xx = np.meshgrid(z_edges * vz, y_edges * vy, x_edges * vx, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float64)
    return points


def signed_tet_volume(p0, p1, p2, p3):
    return float(np.dot(np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0)


def cube_tet_patterns(decomposition="6 tetra / voxel"):
    """
    Local cube node order:
      0=(z0,y0,x0), 1=(z0,y0,x1), 2=(z0,y1,x1), 3=(z0,y1,x0)
      4=(z1,y0,x0), 5=(z1,y0,x1), 6=(z1,y1,x1), 7=(z1,y1,x0)
    """
    if str(decomposition).startswith("5"):
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


def sample_phase_from_original_volume(volume_zyx, voxel_size_xyz, xyz):
    """Sample phase label from the original [Z,Y,X] volume using physical X,Y,Z coordinate."""
    vx, vy, vz = voxel_size_xyz
    x, y, z = xyz
    nz, ny, nx = volume_zyx.shape
    ix = int(np.clip(math.floor(x / vx), 0, nx - 1))
    iy = int(np.clip(math.floor(y / vy), 0, ny - 1))
    iz = int(np.clip(math.floor(z / vz), 0, nz - 1))
    return int(volume_zyx[iz, iy, ix])


def get_or_create_midpoint(points_list, edge_cache, a, b):
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
    idx = len(points_list)
    points_list.append((points_list[a] + points_list[b] + points_list[c] + points_list[d]) / 4.0)
    return idx


def barycentric_subdivide_tet(points_list, tet, edge_cache, face_cache):
    """Barycentric subdivision: 1 tetra -> 24 child tetra."""
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


def block_is_mixed(volume_zyx, z0, z1, y0, y1, x0, x1):
    block = volume_zyx[z0:z1, y0:y1, x0:x1]
    return len(np.unique(block)) > 1


def create_voxel_tetra_mesh(
    volume_zyx,
    voxel_size_xyz,
    precision_zyx=(4, 4, 4),
    decomposition="6 tetra / voxel",
    phase_method="quota_cell",
):
    """
    Create tetrahedral mesh directly from imported label volume.

    phase_method:
        quota_cell: each precision block becomes one phase with quota-preserving coarsening.
        mixed_centroid: only mixed blocks are locally refined; child tetra phases use centroid sampling.
        global_centroid: all blocks are refined; child tetra phases use centroid sampling.
    """
    coarse_labels, z_edges, y_edges, x_edges, info = coarsen_label_volume_quota(
        volume_zyx,
        precision_zyx=precision_zyx,
        preserve_quota=True,
    )

    base_points = build_points_from_block_edges(z_edges, y_edges, x_edges, voxel_size_xyz)
    points_list = [base_points[i].copy() for i in range(base_points.shape[0])]
    edge_cache = {}
    face_cache = {}

    ncz, ncy, ncx = coarse_labels.shape
    nz_n, ny_n, nx_n = ncz + 1, ncy + 1, ncx + 1

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
                is_mixed = block_is_mixed(volume_zyx, z0, z1, y0, y1, x0, x1)
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
                            ph = sample_phase_from_original_volume(volume_zyx, voxel_size_xyz, centroid)
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
        "subdivision_rule": "1 base tetra -> 24 child tetra in barycentric subdivision",
        "subdivision_per_base_tet": 24,
        "max_subtet_per_mixed_voxel": int(24 * len(patterns)),
        "phases": [int(v) for v in np.unique(phases)],
        "original_phase_counts": info["original_counts"],
        "coarse_phase_counts": info["coarse_counts"],
        "preserve_quota": True,
        "voxel_size_xyz_um": tuple(float(v) for v in voxel_size_xyz),
    }
    return points, tets, phases, report


def write_comsol_bdf(points, tets, phases, path):
    """Write COMSOL-importable NASTRAN/BDF. PID = phase + 1."""
    pids = phases.astype(np.int32) + 1
    used_node_ids = np.unique(tets.ravel())
    used_node_ids = np.asarray(sorted([int(i) for i in used_node_ids]), dtype=np.int64)
    old_to_new = {old: new for new, old in enumerate(used_node_ids, start=1)}
    used_pids = sorted([int(v) for v in np.unique(pids)])

    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write("$ ============================================================\n")
        f.write("$ OpenMesoCell imported-NPY tetrahedral BDF mesh\n")
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


def export_voxel_bdf(volume_zyx, voxel_size_xyz, path, precision_zyx=(4, 4, 4), decomposition="6 tetra / voxel", phase_method="quota_cell"):
    points, tets, phases, report = create_voxel_tetra_mesh(
        volume_zyx,
        voxel_size_xyz,
        precision_zyx=precision_zyx,
        decomposition=decomposition,
        phase_method=phase_method,
    )
    bdf_report = write_comsol_bdf(points, tets, phases, path)
    report.update(bdf_report)

    mapping_path = os.path.splitext(path)[0] + "_phase_mapping.txt"
    with open(mapping_path, "w", encoding="utf-8") as f:
        f.write("OpenMesoCell imported-NPY mesh mapping\n")
        f.write("======================================================\n")
        f.write("COMSOL import: Mesh > Import > NASTRAN file (.bdf), unit = um\n\n")
        for label, name in PHASE_NAME.items():
            if (label + 1) in report["property_ids"]:
                f.write(f"PID {label + 1} = {name}\n")
        f.write("\nMesh settings:\n")
        f.write(f"precision_zyx = {report['precision_zyx']}\n")
        f.write(f"node_grid_shape_zyx = {report['node_grid_shape_zyx']}\n")
        f.write(f"coarse_cell_shape_zyx = {report['coarse_cell_shape_zyx']}\n")
        f.write(f"decomposition = {decomposition}\n")
        f.write(f"phase_method = {phase_method}\n")
        f.write(f"voxel_size_xyz_um = {tuple(float(v) for v in voxel_size_xyz)}\n")

    json_path = os.path.splitext(path)[0] + "_mesh_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    report["mapping_path"] = mapping_path
    report["json_path"] = json_path
    return report


# ============================================================
# 3D dialog
# ============================================================
class Volume3DDialog(QDialog):
    def __init__(self, volume_zyx, voxel_size_xyz, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OpenMesoCell 3D Viewer")
        self.resize(1100, 800)
        self.volume = volume_zyx
        self.voxel_size_xyz = voxel_size_xyz

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        layout.addLayout(top)

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
        top.addWidget(phase_group, stretch=0)

        if not HAS_3D:
            top.addWidget(QLabel("PyVista / PyVistaQt is not installed."), stretch=1)
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
            poly = create_polydata_from_phase_mask(mask, self.voxel_size_xyz)
            if poly is None or poly.n_points == 0:
                continue
            opacity = 1.0
            if label == PHASE["pore"]:
                opacity = 0.08
            elif label == PHASE["CBD"]:
                opacity = 0.88
            elif label == PHASE["separator"]:
                opacity = 0.70
            elif label in [PHASE["current_collector"], PHASE["li_foil"]]:
                opacity = 0.90
            self.plotter.add_mesh(
                poly,
                color=PHASE_COLOR.get(label, "white"),
                opacity=opacity,
                show_edges=False,
                smooth_shading=True,
            )
            anything = True

        if anything:
            self.plotter.show_bounds(
                grid="back",
                location="outer",
                all_edges=True,
                xtitle="X (um)",
                ytitle="Y (um)",
                ztitle="Z (um)",
                fmt="%.0f",
                font_size=8,
            )
            self.plotter.add_axes(line_width=1, labels_off=False)
            self.plotter.reset_camera()
        else:
            self.plotter.add_text("No selected phase to display.", font_size=10)

    def reset_view(self):
        if self.plotter is not None:
            self.plotter.reset_camera()


# ============================================================
# Main window
# ============================================================
class OpenMesoCellNPYViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenMesoCell NPY Viewer")
        self.resize(1300, 850)
        self.volume = None  # internal [Z,Y,X]
        self.current_plane = "XY"
        self.loaded_path = None

        self._build_ui()
        self._connect_signals()
        self.reset_slider_range()
        self.update_view()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)

        # Left: viewer
        left = QVBoxLayout()
        main.addLayout(left, stretch=3)

        top_buttons = QHBoxLayout()
        self.btn_xy = QPushButton("XY")
        self.btn_xz = QPushButton("XZ")
        self.btn_yz = QPushButton("YZ")
        self.btn_3d = QPushButton("3D")
        self.btn_clear = QPushButton("Clear")
        for b in [self.btn_xy, self.btn_xz, self.btn_yz, self.btn_3d, self.btn_clear]:
            top_buttons.addWidget(b)
        left.addLayout(top_buttons)

        self.slice_canvas = SliceCanvas()
        left.addWidget(self.slice_canvas, stretch=1)

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Slice"))
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        slider_layout.addWidget(self.slice_slider)
        self.slice_label = QLabel("0/0")
        slider_layout.addWidget(self.slice_label)
        left.addLayout(slider_layout)

        self.structure_label = QLabel("Structure: none")
        left.addWidget(self.structure_label)

        # Right: controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(430)
        container = QWidget()
        right = QVBoxLayout(container)
        right.setContentsMargins(6, 6, 6, 6)
        right.setSpacing(8)
        scroll.setWidget(container)
        main.addWidget(scroll, stretch=2)

        # Import / Save
        io_group = QGroupBox("Import / Save Data")
        io = QGridLayout(io_group)
        self.btn_load_npy = QPushButton("Load 3D Matrix (.npy, YZX)")
        self.btn_save_npy = QPushButton("Save Matrix (.npy, YZX)")
        self.btn_save_tif = QPushButton("Save TIFF Stack (.tif, YZX)")
        self.btn_save_current_png = QPushButton("Save Current Slice (.png)")
        self.btn_save_all_images = QPushButton("Save All Slice Images")
        io.addWidget(self.btn_load_npy, 0, 0, 1, 2)
        io.addWidget(self.btn_save_npy, 1, 0)
        io.addWidget(self.btn_save_tif, 1, 1)
        io.addWidget(self.btn_save_current_png, 2, 0)
        io.addWidget(self.btn_save_all_images, 2, 1)
        right.addWidget(io_group)

        # Resolution
        res_group = QGroupBox("Voxel Resolution")
        res = QGridLayout(res_group)
        self.spin_vx = QDoubleSpinBox(); self.spin_vx.setRange(1e-6, 1000); self.spin_vx.setDecimals(6); self.spin_vx.setValue(0.02791)
        self.spin_vy = QDoubleSpinBox(); self.spin_vy.setRange(1e-6, 1000); self.spin_vy.setDecimals(6); self.spin_vy.setValue(0.03150)
        self.spin_vz = QDoubleSpinBox(); self.spin_vz.setRange(1e-6, 1000); self.spin_vz.setDecimals(6); self.spin_vz.setValue(0.02791)
        res.addWidget(QLabel("Voxel size X (um)"), 0, 0); res.addWidget(self.spin_vx, 0, 1)
        res.addWidget(QLabel("Voxel size Y (um)"), 1, 0); res.addWidget(self.spin_vy, 1, 1)
        res.addWidget(QLabel("Voxel size Z (um)"), 2, 0); res.addWidget(self.spin_vz, 2, 1)
        right.addWidget(res_group)

        # Layers
        layer_group = QGroupBox("Separator / Current Collector / Li Foil")
        layer = QGridLayout(layer_group)
        self.spin_separator = QDoubleSpinBox(); self.spin_separator.setRange(0, 1000); self.spin_separator.setDecimals(3); self.spin_separator.setValue(5.0)
        self.spin_cc = QDoubleSpinBox(); self.spin_cc.setRange(0, 1000); self.spin_cc.setDecimals(3); self.spin_cc.setValue(3.0)
        self.spin_li = QDoubleSpinBox(); self.spin_li.setRange(0, 1000); self.spin_li.setDecimals(3); self.spin_li.setValue(5.0)
        self.btn_add_separator = QPushButton("Add Separator at Z max")
        self.btn_add_cc = QPushButton("Add CC at Z min")
        self.btn_add_li = QPushButton("Add Li Foil at Z max")
        layer.addWidget(QLabel("Separator thickness (um)"), 0, 0); layer.addWidget(self.spin_separator, 0, 1)
        layer.addWidget(self.btn_add_separator, 1, 0, 1, 2)
        layer.addWidget(QLabel("CC thickness (um)"), 2, 0); layer.addWidget(self.spin_cc, 2, 1)
        layer.addWidget(self.btn_add_cc, 3, 0, 1, 2)
        layer.addWidget(QLabel("Li foil thickness (um)"), 4, 0); layer.addWidget(self.spin_li, 4, 1)
        layer.addWidget(self.btn_add_li, 5, 0, 1, 2)
        right.addWidget(layer_group)

        # Meshing options
        mesh_group = QGroupBox("Meshing Options")
        mesh = QGridLayout(mesh_group)
        self.combo_tet_decomposition = QComboBox()
        self.combo_tet_decomposition.addItems(["6 tetra / voxel", "5 tetra / voxel"])
        self.combo_phase_method = QComboBox()
        self.combo_phase_method.addItems(["quota_cell", "mixed_centroid", "global_centroid"])
        self.combo_phase_method.setToolTip(
            "quota_cell: fast and stable. mixed_centroid: refine mixed blocks and assign phases by centroid sampling. "
            "global_centroid: refine every block, more accurate but larger."
        )
        self.spin_precision_z = QSpinBox(); self.spin_precision_z.setRange(1, 50); self.spin_precision_z.setValue(4)
        self.spin_precision_y = QSpinBox(); self.spin_precision_y.setRange(1, 50); self.spin_precision_y.setValue(4)
        self.spin_precision_x = QSpinBox(); self.spin_precision_x.setRange(1, 50); self.spin_precision_x.setValue(4)
        self.btn_save_bdf = QPushButton("Save COMSOL BDF Mesh")
        mesh.addWidget(QLabel("Tet decomposition"), 0, 0)
        mesh.addWidget(self.combo_tet_decomposition, 0, 1, 1, 3)
        mesh.addWidget(QLabel("Voxel phase method"), 1, 0)
        mesh.addWidget(self.combo_phase_method, 1, 1, 1, 3)
        mesh.addWidget(QLabel("Precision Z / Y / X"), 2, 0)
        mesh.addWidget(self.spin_precision_z, 2, 1)
        mesh.addWidget(self.spin_precision_y, 2, 2)
        mesh.addWidget(self.spin_precision_x, 2, 3)
        mesh.addWidget(self.btn_save_bdf, 3, 0, 1, 4)
        right.addWidget(mesh_group)


        # Log / stats
        log_group = QGroupBox("Log / Statistics")
        log_layout = QVBoxLayout(log_group)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        log_layout.addWidget(self.log_box)
        right.addWidget(log_group, stretch=1)

    def _connect_signals(self):
        self.btn_xy.clicked.connect(lambda: self.set_plane("XY"))
        self.btn_xz.clicked.connect(lambda: self.set_plane("XZ"))
        self.btn_yz.clicked.connect(lambda: self.set_plane("YZ"))
        self.btn_3d.clicked.connect(self.open_3d_view)
        self.btn_clear.clicked.connect(self.clear_volume)
        self.slice_slider.valueChanged.connect(self.update_view)

        self.btn_load_npy.clicked.connect(self.load_npy)
        self.btn_save_npy.clicked.connect(self.save_npy)
        self.btn_save_tif.clicked.connect(self.save_tif)
        self.btn_save_current_png.clicked.connect(self.save_current_slice_png)
        self.btn_save_all_images.clicked.connect(self.save_all_slice_images)
        self.btn_save_bdf.clicked.connect(self.save_bdf_mesh)

        self.btn_add_separator.clicked.connect(self.add_separator)
        self.btn_add_cc.clicked.connect(self.add_current_collector)
        self.btn_add_li.clicked.connect(self.add_li_foil)

        for spin in [self.spin_vx, self.spin_vy, self.spin_vz]:
            spin.valueChanged.connect(self.update_view)
            spin.valueChanged.connect(self.update_statistics)

    def voxel_size_xyz(self):
        return (self.spin_vx.value(), self.spin_vy.value(), self.spin_vz.value())

    def log(self, text):
        self.log_box.append(str(text))

    def update_statistics(self):
        if self.volume is None:
            return
        self.structure_label.setText(self.structure_summary())

    def structure_summary(self):
        if self.volume is None:
            return "Structure: none"
        vx, vy, vz = self.voxel_size_xyz()
        nz, ny, nx = self.volume.shape
        return (
            f"Structure: external [Y,Z,X]={(ny, nz, nx)} | "
            f"physical X/Y/Z={nx*vx:.3f}/{ny*vy:.3f}/{nz*vz:.3f} um | "
            f"{len(np.unique(self.volume))} phases"
        )

    def set_plane(self, plane):
        self.current_plane = plane
        self.reset_slider_range()
        self.update_view()

    def reset_slider_range(self):
        if self.volume is None:
            self.slice_slider.setMinimum(0)
            self.slice_slider.setMaximum(0)
            self.slice_slider.setValue(0)
            return
        nz, ny, nx = self.volume.shape
        if self.current_plane == "XY":
            max_idx = nz - 1
        elif self.current_plane == "XZ":
            max_idx = ny - 1
        else:
            max_idx = nx - 1
        old = self.slice_slider.value()
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(max_idx)
        self.slice_slider.setValue(min(old, max_idx))

    def update_view(self):
        idx = self.slice_slider.value()
        self.slice_canvas.show_slice(
            self.volume,
            self.current_plane,
            idx,
            self.voxel_size_xyz(),
        )
        if self.volume is None:
            self.slice_label.setText("0/0")
            self.structure_label.setText("Structure: none")
            return
        nz, ny, nx = self.volume.shape
        total = nz if self.current_plane == "XY" else (ny if self.current_plane == "XZ" else nx)
        self.slice_label.setText(f"{idx + 1}/{total}")
        self.structure_label.setText(self.structure_summary())

    def clear_volume(self):
        self.volume = None
        self.loaded_path = None
        self.log("Structure cleared.")
        self.reset_slider_range()
        self.update_view()

    def load_npy(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load .npy", "", "NumPy Matrix (*.npy)")
        if not path:
            return
        try:
            arr = np.load(path)
        except Exception as e:
            QMessageBox.warning(self, "Load error", str(e))
            return
        if arr.ndim != 3:
            QMessageBox.warning(self, "Invalid file", "The .npy file must be a 3D matrix with axis order [Y,Z,X].")
            return
        self.volume = yzx_to_zyx(arr)
        self.loaded_path = path
        self.log(f"Loaded NPY as external [Y,Z,X]: {path}")
        self.log(volume_statistics(self.volume, self.voxel_size_xyz()))
        self.reset_slider_range()
        self.update_view()

    def load_volume_from_array(self, arr, voxel_size_xyz=None):
        """Load volume directly from a numpy array (YZX order), without file dialog."""
        if arr.ndim != 3:
            raise ValueError(f"Array must be 3D [Y,Z,X], got shape {arr.shape}")
        self.volume = yzx_to_zyx(arr)
        self.loaded_path = None
        if voxel_size_xyz is not None:
            self.spin_vx.setValue(float(voxel_size_xyz[0]))
            self.spin_vy.setValue(float(voxel_size_xyz[1]))
            self.spin_vz.setValue(float(voxel_size_xyz[2]))
        self.log(f"Loaded from array, external shape [Y,Z,X]={arr.shape}")
        self.log(volume_statistics(self.volume, self.voxel_size_xyz()))
        self.reset_slider_range()
        self.update_view()

    def save_npy(self):
        if self.volume is None:
            QMessageBox.warning(self, "No structure", "Nothing to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save .npy", "structure_yzx.npy", "NumPy Matrix (*.npy)")
        if not path:
            return
        if not path.lower().endswith(".npy"):
            path += ".npy"
        np.save(path, zyx_to_yzx(self.volume))
        self.log(f"Saved NPY in external [Y,Z,X] order: {path}")

    def save_tif(self):
        if self.volume is None:
            QMessageBox.warning(self, "No structure", "Nothing to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save TIFF stack", "structure_yzx.tif", "TIFF Stack (*.tif *.tiff)")
        if not path:
            return
        if not path.lower().endswith((".tif", ".tiff")):
            path += ".tif"
        tifffile.imwrite(path, zyx_to_yzx(self.volume).astype(np.uint8))
        self.log(f"Saved TIFF stack in external [Y,Z,X] order: {path}")

    def save_current_slice_png(self):
        if self.volume is None:
            QMessageBox.warning(self, "No structure", "Nothing to save.")
            return
        default = f"slice_{self.current_plane}_{self.slice_slider.value()+1:04d}.png"
        path, _ = QFileDialog.getSaveFileName(self, "Save current slice", default, "PNG Image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        save_slice_png(
            self.volume,
            self.current_plane,
            self.slice_slider.value(),
            self.voxel_size_xyz(),
            path,
        )
        self.log(f"Saved current slice: {path}")

    def save_all_slice_images(self):
        if self.volume is None:
            QMessageBox.warning(self, "No structure", "Nothing to save.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Choose output folder for all slice images")
        if not folder:
            return
        nz, ny, nx = self.volume.shape
        total_images = nz + ny + nx
        reply = QMessageBox.question(
            self,
            "Save all slices",
            f"This will save {total_images} PNG images into three folders:\n"
            f"  Z_slices_XY: {nz}\n"
            f"  Y_slices_XZ: {ny}\n"
            f"  X_slices_YZ: {nx}\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        subfolders = {
            "XY": os.path.join(folder, "Z_slices_XY"),
            "XZ": os.path.join(folder, "Y_slices_XZ"),
            "YZ": os.path.join(folder, "X_slices_YZ"),
        }
        for sub in subfolders.values():
            os.makedirs(sub, exist_ok=True)

        self.log(f"Saving all slice images to: {folder}")
        QApplication.processEvents()

        for i in range(nz):
            save_slice_png(self.volume, "XY", i, self.voxel_size_xyz(), os.path.join(subfolders["XY"], f"Z_{i+1:04d}_XY.png"))
            if i % 20 == 0:
                QApplication.processEvents()
        for i in range(ny):
            save_slice_png(self.volume, "XZ", i, self.voxel_size_xyz(), os.path.join(subfolders["XZ"], f"Y_{i+1:04d}_XZ.png"))
            if i % 20 == 0:
                QApplication.processEvents()
        for i in range(nx):
            save_slice_png(self.volume, "YZ", i, self.voxel_size_xyz(), os.path.join(subfolders["YZ"], f"X_{i+1:04d}_YZ.png"))
            if i % 20 == 0:
                QApplication.processEvents()

        self.log(f"All slice images saved. Total PNG files = {total_images}")
        QMessageBox.information(self, "Done", f"Saved {total_images} PNG images.")

    def add_separator(self):
        if self.volume is None:
            QMessageBox.warning(self, "No structure", "Please load a structure first.")
            return
        self.volume = add_layer_zmax(self.volume, PHASE["separator"], self.spin_separator.value(), self.spin_vz.value())
        self.log(f"Separator added at Z max. thickness = {self.spin_separator.value()} um")
        self.reset_slider_range()
        self.update_view()
        self.log(volume_statistics(self.volume, self.voxel_size_xyz()))

    def add_current_collector(self):
        if self.volume is None:
            QMessageBox.warning(self, "No structure", "Please load a structure first.")
            return
        self.volume = add_layer_zmin(self.volume, PHASE["current_collector"], self.spin_cc.value(), self.spin_vz.value())
        self.log(f"Current collector added at Z min. thickness = {self.spin_cc.value()} um")
        self.reset_slider_range()
        self.update_view()
        self.log(volume_statistics(self.volume, self.voxel_size_xyz()))

    def add_li_foil(self):
        if self.volume is None:
            QMessageBox.warning(self, "No structure", "Please load a structure first.")
            return
        self.volume = add_layer_zmax(self.volume, PHASE["li_foil"], self.spin_li.value(), self.spin_vz.value())
        self.log(f"Li foil added at Z max. thickness = {self.spin_li.value()} um")
        self.reset_slider_range()
        self.update_view()
        self.log(volume_statistics(self.volume, self.voxel_size_xyz()))

    def save_bdf_mesh(self):
        if self.volume is None:
            QMessageBox.warning(self, "No structure", "Please load a structure first.")
            return
        pz = self.spin_precision_z.value()
        py = self.spin_precision_y.value()
        px = self.spin_precision_x.value()
        decomposition = self.combo_tet_decomposition.currentText()
        phase_method = self.combo_phase_method.currentText()

        nz, ny, nx = self.volume.shape
        estimated_blocks = int(np.ceil(nz / pz) * np.ceil(ny / py) * np.ceil(nx / px))
        base_tets_per_block = 6 if decomposition.startswith("6") else 5
        rough_tets = estimated_blocks * base_tets_per_block
        if phase_method == "global_centroid":
            rough_tets *= 24

        if rough_tets > 5_000_000:
            reply = QMessageBox.question(
                self,
                "Large mesh warning",
                f"Estimated tetrahedra may exceed {rough_tets:,}.\n"
                "This can be slow and memory-intensive.\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        default_name = f"comsol_bdf_precision_{pz}_{py}_{px}.bdf"
        path, _ = QFileDialog.getSaveFileName(self, "Save COMSOL BDF mesh", default_name, "NASTRAN BDF (*.bdf)")
        if not path:
            return
        if not path.lower().endswith(".bdf"):
            path += ".bdf"

        try:
            self.log(
                f"Exporting BDF mesh... precision=({pz},{py},{px}), "
                f"decomposition={decomposition}, method={phase_method}"
            )
            QApplication.processEvents()
            report = export_voxel_bdf(
                self.volume,
                self.voxel_size_xyz(),
                path,
                precision_zyx=(pz, py, px),
                decomposition=decomposition,
                phase_method=phase_method,
            )
            self.log(
                f"BDF mesh saved: {path}\n"
                f"  nodes={report['used_nodes']:,}\n"
                f"  tetrahedra={report['tetrahedra']:,}\n"
                f"  phases={report['phases']}\n"
                f"  property_ids={report['property_ids']}\n"
                f"  mixed_blocks={report['mixed_blocks']:,}\n"
                f"  mapping={report['mapping_path']}\n"
                f"  report={report['json_path']}"
            )
            QMessageBox.information(
                self,
                "Mesh saved",
                f"BDF mesh saved.\n\nNodes: {report['used_nodes']:,}\n"
                f"Tetrahedra: {report['tetrahedra']:,}\nFile: {path}",
            )
        except Exception as e:
            QMessageBox.warning(self, "Mesh export error", str(e))
            self.log(f"Mesh export failed: {e}")

    def open_3d_view(self):
        if self.volume is None:
            QMessageBox.warning(self, "No structure", "Please load a structure first.")
            return
        if not HAS_3D or not HAS_SKIMAGE:
            QMessageBox.warning(
                self,
                "Missing package",
                "3D viewer requires pyvista, pyvistaqt and scikit-image.\n\n"
                "Please run:\n"
                "pip install pyvista pyvistaqt scikit-image",
            )
            return
        dlg = Volume3DDialog(self.volume, self.voxel_size_xyz(), self)
        dlg.exec()


def main():
    app = QApplication(sys.argv)
    win = OpenMesoCellNPYViewer()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
