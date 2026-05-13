# build_latent_dataset.py

from __future__ import annotations

import os
import json
import random
import warnings
import contextlib
import io

import numpy as np
import torch
from tqdm import tqdm
from scipy.ndimage import label, distance_transform_edt

import taufactor as tau

from backend.electrode_twin.vaenet import VAENet, VAENetConfig
from backend.electrode_twin.vaemodule import VAEModule, VAEModuleConfig


# ==============================
# 路径设置（如果改，记得把vae路径换掉）
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ELECTRODE_TWIN_DIR = os.path.join(BASE_DIR, "electrode_twin")

VOLUME_PATH = os.path.join(
    ELECTRODE_TWIN_DIR,
    "reconstruction_results",
    "original_volume.npy"
)
CKPT_PATH = os.path.join(
    ELECTRODE_TWIN_DIR,
    "checkpoints",
    "vae-epoch045-valloss-1.9100.ckpt"
)
OUTPUT_DIR = os.path.join(
    ELECTRODE_TWIN_DIR,
    "latent_dataset"
)
# VOLUME_PATH = r"reconstruction_results\original_volume.npy"
# CKPT_PATH = r"checkpoints\vae-epoch045-valloss-1.9100.ckpt"
# OUTPUT_DIR = r"latent_dataset"

DEVICE = "cpu"


# ==============================
# patch 设置
# ==============================

PATCH_SIZE = 128
GRID_STRIDE = 64
NUM_RANDOM_PATCHES = 7000

PORE_VALUE = 0
SOLID_VALUE = 1


# ==============================
# 真实物理分辨率（单位：um）
# volume 顺序: [Y, Z, X]
#
# 你已经明确：
# Y = 切片方向
# Z = 电极厚度方向（through-plane）
# X = 面内方向
# ==============================

VOXEL_SIZE_Y = 0.0315
VOXEL_SIZE_Z = 0.02791
VOXEL_SIZE_X = 0.02791


# ==============================
# TauFactor 设置
# ==============================

COMPUTE_TAUFACTOR = True

# 非贯通 patch 给一个大 tau
TAU_NONPERC_VALUE = 1e6

# 是否屏蔽 TauFactor 输出
SUPPRESS_TAUFACTOR_OUTPUT = True


# ==============================
# 随机种子
# ==============================

SEED = 42


# ==============================
# 工具函数
# ==============================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_scalar(x, default: float = 0.0) -> float:
    """
    安全地把 solver.tau / solver.D_eff 这种可能是 shape=(1,) 的数组取成标量
    """
    try:
        arr = np.asarray(x).reshape(-1)
        if arr.size == 0:
            return float(default)
        v = float(arr[0])
        if not np.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)


@contextlib.contextmanager
def suppress_stdout_stderr(enabled: bool = True):
    """
    屏蔽 stdout / stderr，用来压掉 TauFactor 的求解日志
    """
    if not enabled:
        yield
        return

    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


# ==============================
# 模型加载
# ==============================

def load_model():
    net_config = VAENetConfig(
        dimension=3,
        in_channels=1,
        out_channels=1,
        z_dim=4,
        ch=32,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        resolution=128,
        num_groups=16,
        use_attention=False,
        final_activation="sigmoid"
    )

    vae_config = VAEModuleConfig(
        kl_weight=1e-5,
        kl_start=0.0,
        kl_max=1e-5,
        kl_warmup_steps=5000,
        reconstruction_loss="mse",
        lr=1e-4,
        scheduler_type="none"
    )

    net = VAENet(net_config)

    model = VAEModule.load_from_checkpoint(
        CKPT_PATH,
        encdec=net,
        config=vae_config,
        conditional=False,
        verbose=False,
        map_location=DEVICE,
        weights_only=False
    )

    model.eval()
    model.to(DEVICE)

    return model


# ==============================
# patch 采样
# ==============================

def grid_sampling(volume: np.ndarray):
    patches = []
    y, z, x = volume.shape

    ys = list(range(0, y - PATCH_SIZE + 1, GRID_STRIDE))
    zs = list(range(0, z - PATCH_SIZE + 1, GRID_STRIDE))
    xs = list(range(0, x - PATCH_SIZE + 1, GRID_STRIDE))

    for yy in ys:
        for zz in zs:
            for xx in xs:
                patches.append((yy, zz, xx))

    return patches


def random_sampling(volume: np.ndarray, num: int):
    patches = []
    y, z, x = volume.shape

    for _ in range(num):
        yy = random.randint(0, y - PATCH_SIZE)
        zz = random.randint(0, z - PATCH_SIZE)
        xx = random.randint(0, x - PATCH_SIZE)
        patches.append((yy, zz, xx))

    return patches


# ==============================
# 结构指标
# ==============================

def porosity(volume: np.ndarray) -> float:
    pore_mask = (volume == PORE_VALUE)
    return float(pore_mask.mean())


def surface_area(volume: np.ndarray) -> float:
    """
    近似界面面积密度（单位：1/um）
    """
    v = volume.astype(np.uint8)

    n_y = np.abs(v[1:, :, :] - v[:-1, :, :]).sum()
    n_z = np.abs(v[:, 1:, :] - v[:, :-1, :]).sum()
    n_x = np.abs(v[:, :, 1:] - v[:, :, :-1]).sum()

    area_y = n_y * (VOXEL_SIZE_Z * VOXEL_SIZE_X)
    area_z = n_z * (VOXEL_SIZE_Y * VOXEL_SIZE_X)
    area_x = n_x * (VOXEL_SIZE_Y * VOXEL_SIZE_Z)

    total_area = area_y + area_z + area_x

    total_volume = (
        volume.shape[0] * VOXEL_SIZE_Y *
        volume.shape[1] * VOXEL_SIZE_Z *
        volume.shape[2] * VOXEL_SIZE_X
    )

    if total_volume <= 0:
        return 0.0

    return float(total_area / total_volume)


def largest_connected_ratio(volume: np.ndarray) -> float:
    """
    最大孔隙连通占比（孔隙相）
    """
    pore_mask = (volume == PORE_VALUE)

    labeled, num = label(pore_mask)
    sizes = np.bincount(labeled.ravel())

    if len(sizes) <= 1:
        return 0.0

    largest = sizes[1:].max()
    total = sizes[1:].sum()

    if total == 0:
        return 0.0

    return float(largest / total)


def pore_size_stats(volume: np.ndarray):
    """
    返回：
        mean_pore_size, std_pore_size, max_pore_size
    单位：um
    """
    pore_mask = (volume == PORE_VALUE)

    dist = distance_transform_edt(
        pore_mask,
        sampling=(VOXEL_SIZE_Y, VOXEL_SIZE_Z, VOXEL_SIZE_X)
    )

    sizes = dist[pore_mask] * 2.0

    if len(sizes) == 0:
        return 0.0, 0.0, 0.0

    return float(sizes.mean()), float(sizes.std()), float(sizes.max())


def particle_size_stats(volume: np.ndarray):
    """
    返回：
        mean_particle_size, std_particle_size, max_particle_size
    单位：um
    """
    solid_mask = (volume == SOLID_VALUE)

    labeled, num = label(solid_mask)
    if num == 0:
        return 0.0, 0.0, 0.0

    voxel_volume = VOXEL_SIZE_Y * VOXEL_SIZE_Z * VOXEL_SIZE_X
    sizes = []
    binc = np.bincount(labeled.ravel())

    for comp_id in range(1, len(binc)):
        voxel_count = int(binc[comp_id])
        if voxel_count <= 0:
            continue

        real_volume = voxel_count * voxel_volume
        eq_diameter = ((6.0 * real_volume) / np.pi) ** (1.0 / 3.0)
        sizes.append(eq_diameter)

    if len(sizes) == 0:
        return 0.0, 0.0, 0.0

    sizes = np.asarray(sizes, dtype=np.float32)
    return float(sizes.mean()), float(sizes.std()), float(sizes.max())


# ==============================
# TauFactor 相关
# ==============================

def is_percolating_along_z(volume: np.ndarray) -> bool:
    """
    检查孔隙相是否沿 Z 方向贯通
    输入 volume 顺序: [Y, Z, X]
    """
    pore_mask = (volume == PORE_VALUE)

    labeled, num = label(pore_mask)
    if num == 0:
        return False

    z0_labels = set(np.unique(labeled[:, 0, :]))
    zend_labels = set(np.unique(labeled[:, -1, :]))

    common = z0_labels & zend_labels
    common.discard(0)

    return len(common) > 0


def compute_tau_deff_z(volume: np.ndarray) -> tuple[float, float, int]:
    """
    计算沿 Z 方向（through-plane）的 tau 和 deff

    输入:
        volume: [Y, Z, X], 0=孔隙, 1=固相

    TauFactor 要求:
        1 = conductive / transport phase
        0 = blocking phase

    因此这里要把孔隙相转成 1

    同时 TauFactor 默认沿 axis=0 求解，
    所以把 [Y, Z, X] 转成 [Z, Y, X]，
    从而使原 Z 方向变成 axis=0
    """
    percolating = is_percolating_along_z(volume)

    if not percolating:
        return float(TAU_NONPERC_VALUE), 0.0, 0

    pore_phase = (volume == PORE_VALUE).astype(np.uint8)   # [Y, Z, X]
    pore_phase_zyx = np.transpose(pore_phase, (1, 0, 2))   # [Z, Y, X]

    try:
        with suppress_stdout_stderr(SUPPRESS_TAUFACTOR_OUTPUT):
            solver = tau.Solver(pore_phase_zyx)
            solver.solve()

        tau_value = safe_scalar(solver.tau, default=TAU_NONPERC_VALUE)
        deff_value = safe_scalar(solver.D_eff, default=0.0)

        if not np.isfinite(tau_value):
            tau_value = float(TAU_NONPERC_VALUE)
        if not np.isfinite(deff_value):
            deff_value = 0.0

        return tau_value, deff_value, 1

    except Exception:
        return float(TAU_NONPERC_VALUE), 0.0, 0


# ==============================
# 主程序
# ==============================

def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ensure_dir(OUTPUT_DIR)

    print("Loading volume ...")
    volume = np.load(VOLUME_PATH)

    print("Volume shape:", volume.shape)
    print("Physical voxel size (um):", (VOXEL_SIZE_Y, VOXEL_SIZE_Z, VOXEL_SIZE_X))
    print("Transport direction: Z (through-plane)")
    print("Compute TauFactor:", COMPUTE_TAUFACTOR)

    model = load_model()

    print("Sampling patches ...")
    grid_patches = grid_sampling(volume)
    rand_patches = random_sampling(volume, NUM_RANDOM_PATCHES)
    patches = grid_patches + rand_patches

    print("Grid patches  :", len(grid_patches))
    print("Random patches:", len(rand_patches))
    print("Total patches :", len(patches))

    stats = {
        "porosity": [],
        "surface": [],
        "conn": [],
        "mean_pore": [],
        "std_pore": [],
        "max_pore": [],
        "mean_particle": [],
        "std_particle": [],
        "max_particle": [],
        "tau_z": [],
        "deff_z": [],
        "is_percolating_z": [],
    }

    idx = 0

    for (yy, zz, xx) in tqdm(patches):
        patch = volume[
            yy:yy + PATCH_SIZE,
            zz:zz + PATCH_SIZE,
            xx:xx + PATCH_SIZE
        ]

        # 输入给 VAE 的仍然是 0/1 体素值
        patch01 = (patch > 0).astype(np.float32)
        x = torch.from_numpy(patch01).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            enc = model.encode(x, sample=False)
            z = enc["zsample"].detach().cpu().numpy()[0]

        # 基础结构指标
        p = porosity(patch)
        s = surface_area(patch)
        c = largest_connected_ratio(patch)

        mp, sp, xp = pore_size_stats(patch)
        mps, sps, xps = particle_size_stats(patch)

        # TauFactor 指标
        if COMPUTE_TAUFACTOR:
            tau_z, deff_z, is_perc_z = compute_tau_deff_z(patch)
        else:
            tau_z, deff_z, is_perc_z = 0.0, 0.0, 0

        np.savez(
            os.path.join(OUTPUT_DIR, f"sample_{idx:06d}.npz"),

            z=z,

            porosity=p,
            surface_area=s,
            largest_connected_ratio=c,

            mean_pore_size=mp,
            std_pore_size=sp,
            max_pore_size=xp,

            mean_particle_size=mps,
            std_particle_size=sps,
            max_particle_size=xps,

            tau_z=tau_z,
            deff_z=deff_z,
            is_percolating_z=is_perc_z,

            origin=np.array([yy, zz, xx], dtype=np.int32)
        )

        stats["porosity"].append(p)
        stats["surface"].append(s)
        stats["conn"].append(c)

        stats["mean_pore"].append(mp)
        stats["std_pore"].append(sp)
        stats["max_pore"].append(xp)

        stats["mean_particle"].append(mps)
        stats["std_particle"].append(sps)
        stats["max_particle"].append(xps)

        stats["tau_z"].append(tau_z)
        stats["deff_z"].append(deff_z)
        stats["is_percolating_z"].append(is_perc_z)

        idx += 1

    summary = {}

    for k, v in stats.items():
        v = np.asarray(v, dtype=np.float32)
        summary[k] = {
            "min": float(v.min()),
            "max": float(v.max()),
            "mean": float(v.mean()),
            "std": float(v.std())
        }

    summary["_units"] = {
        "porosity": "dimensionless",
        "surface": "1/um",
        "conn": "dimensionless",
        "mean_pore": "um",
        "std_pore": "um",
        "max_pore": "um",
        "mean_particle": "um",
        "std_particle": "um",
        "max_particle": "um",
        "tau_z": "dimensionless",
        "deff_z": "relative",
        "is_percolating_z": "binary",
    }

    summary["_voxel_size_um"] = {
        "Y": VOXEL_SIZE_Y,
        "Z": VOXEL_SIZE_Z,
        "X": VOXEL_SIZE_X
    }

    summary["_transport_definition"] = {
        "transport_phase": "pore phase",
        "transport_direction": "Z (through-plane)",
        "input_volume_order": "[Y, Z, X]",
        "taufactor_solver_order": "[Z, Y, X]",
        "taufactor_phase_convention": "1=conductive(pore), 0=blocking(solid)",
        "non_percolating_tau_value": TAU_NONPERC_VALUE
    }

    summary["_recommended_condition_keys"] = [
        "porosity",
        "surface_area",
        "tau_z",
        "deff_z"
    ]

    with open(os.path.join(OUTPUT_DIR, "dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("Dataset complete")
    print("Total samples:", idx)
    print("Summary saved to:", os.path.join(OUTPUT_DIR, "dataset_summary.json"))


if __name__ == "__main__":
    main()