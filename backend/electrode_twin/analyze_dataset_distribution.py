from __future__ import annotations

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw
from tqdm import tqdm

# ===============================
# 路径
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ELECTRODE_TWIN_DIR = os.path.join(BASE_DIR, "electrode_twin")

DATASET_DIR = os.path.join(
    ELECTRODE_TWIN_DIR,
    "latent_dataset"
)
OUT_DIR = os.path.join(
    ELECTRODE_TWIN_DIR,
    "dataset_analysis"
)
# DATASET_DIR = r"./latent_dataset"
# OUT_DIR = r"./dataset_analysis"

os.makedirs(OUT_DIR, exist_ok=True)

# ===============================
# 绘图风格（尽量接近期刊风格）
# ===============================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# ===============================
# 读取数据（带进度条）
# ===============================
files = sorted(glob.glob(os.path.join(DATASET_DIR, "sample_*.npz")))
if len(files) == 0:
    raise FileNotFoundError(f"未在 {DATASET_DIR} 中找到 sample_*.npz")

porosity_list = []
surface_list = []
tau_list = []
deff_list = []

for f in tqdm(files, desc="Loading samples", unit="file"):
    data = np.load(f)
    porosity_list.append(float(data["porosity"]))
    surface_list.append(float(data["surface_area"]))
    tau_list.append(float(data["tau_z"]))
    deff_list.append(float(data["deff_z"]))

por = np.array(porosity_list, dtype=np.float64)
surf = np.array(surface_list, dtype=np.float64)
tau = np.array(tau_list, dtype=np.float64)
deff = np.array(deff_list, dtype=np.float64)

# ===============================
# 基本统计
# ===============================
summary = {
    "dataset_size": int(len(por)),
    "porosity": {
        "min": float(por.min()),
        "max": float(por.max()),
        "mean": float(por.mean()),
        "std": float(por.std()),
    },
    "tau_z": {
        "min": float(tau.min()),
        "max": float(tau.max()),
        "mean": float(tau.mean()),
        "std": float(tau.std()),
    },
    "surface_area": {
        "min": float(surf.min()),
        "max": float(surf.max()),
        "mean": float(surf.mean()),
        "std": float(surf.std()),
    },
    "deff_z": {
        "min": float(deff.min()),
        "max": float(deff.max()),
        "mean": float(deff.mean()),
        "std": float(deff.std()),
    },
    "corrcoef": {
        "porosity_tau": float(np.corrcoef(por, tau)[0, 1]),
        "porosity_surface": float(np.corrcoef(por, surf)[0, 1]),
        "tau_surface": float(np.corrcoef(tau, surf)[0, 1]),
        "deff_vs_por_div_tau": float(np.corrcoef(deff, por / tau)[0, 1]),
    }
}

with open(os.path.join(OUT_DIR, "dataset_distribution_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)

print("=" * 70)
print("Dataset size:", len(por))
print("Porosity range:", por.min(), por.max())
print("Tau_z range:", tau.min(), tau.max())
print("Surface area range:", surf.min(), surf.max())
print("Correlation porosity-tau_z:", summary["corrcoef"]["porosity_tau"])
print("=" * 70)

# ===============================
# 工具函数
# ===============================
def add_mean_line(ax, x, color="black", label=None):
    m = float(np.mean(x))
    ax.axvline(m, linestyle="--", linewidth=1.2, color=color, alpha=0.9, label=label)
    return m

def binned_stats(x, y, nbins=12):
    bins = np.linspace(np.min(x), np.max(x), nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    means = np.full(nbins, np.nan, dtype=np.float64)
    stds = np.full(nbins, np.nan, dtype=np.float64)
    counts = np.zeros(nbins, dtype=int)

    for i in range(nbins):
        if i < nbins - 1:
            mask = (x >= bins[i]) & (x < bins[i + 1])
        else:
            mask = (x >= bins[i]) & (x <= bins[i + 1])

        counts[i] = int(np.sum(mask))
        if counts[i] > 0:
            means[i] = np.mean(y[mask])
            stds[i] = np.std(y[mask])

    valid = counts > 0
    return centers[valid], means[valid], stds[valid], counts[valid]

# ===============================
# Figure 1: 三个分布图（建议正文可用）
# 改为概率密度 + 均值虚线
# ===============================
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

# Porosity
axes[0].hist(por, bins=40, density=True, alpha=0.9)
m_por = add_mean_line(axes[0], por, label=f"Mean = {por.mean():.3f}")
axes[0].set_title("Porosity Distribution")
axes[0].set_xlabel("Porosity")
axes[0].set_ylabel("Probability density")
axes[0].legend(frameon=False, loc="upper right")

# Tau_z
axes[1].hist(tau, bins=40, density=True, alpha=0.9)
m_tau = add_mean_line(axes[1], tau, label=f"Mean = {tau.mean():.2f}")
axes[1].set_title(r"Tortuosity Distribution ($\tau_z$)")
axes[1].set_xlabel(r"$\tau_z$")
axes[1].set_ylabel("Probability density")
axes[1].legend(frameon=False, loc="upper right")

# Surface
axes[2].hist(surf, bins=40, density=True, alpha=0.9)
m_surf = add_mean_line(axes[2], surf, label=f"Mean = {surf.mean():.1f}")
axes[2].set_title("Surface Area Distribution")
axes[2].set_xlabel(r"Surface area (1/$\mu$m)")
axes[2].set_ylabel("Probability density")
axes[2].legend(frameon=False, loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig1_distributions.png"), bbox_inches="tight")
plt.close()

# ===============================
# Figure 2: porosity vs tau_z，surface_area 着色
# 这是你最核心的一张图
# ===============================
fig, ax = plt.subplots(figsize=(5.8, 4.8))
sc = ax.scatter(
    por, tau,
    c=surf,
    s=12,
    alpha=0.65,
    linewidths=0,
    cmap="viridis"
)
ax.set_xlabel("Porosity")
ax.set_ylabel(r"$\tau_z$")
ax.set_title(r"Structure–transport manifold")
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label(r"Surface area (1/$\mu$m)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig2_porosity_tau_surface.png"), bbox_inches="tight")
plt.close()

# ===============================
# Figure 3: porosity -> tau_z 的趋势图（均值 + std 阴影）
# 更有物理意义，适合论文正文
# ===============================
x_bin, y_mean, y_std, y_n = binned_stats(por, tau, nbins=12)

fig, ax = plt.subplots(figsize=(5.8, 4.8))
ax.scatter(por, tau, s=8, alpha=0.15, color="gray", label="Samples")
ax.plot(x_bin, y_mean, linewidth=2.0, label="Binned mean")
ax.fill_between(x_bin, y_mean - y_std, y_mean + y_std, alpha=0.2, label=r"$\pm 1$ std")
ax.set_xlabel("Porosity")
ax.set_ylabel(r"$\tau_z$")
ax.set_title(r"Porosity-dependent tortuosity trend")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig3_porosity_tau_trend.png"), bbox_inches="tight")
plt.close()

# ===============================
# Figure 4: porosity-tau_z 二维密度图（可行域）
# 很适合说明哪些组合在训练集中常见
# ===============================
fig, ax = plt.subplots(figsize=(5.8, 4.8))
hb = ax.hexbin(
    por, tau,
    gridsize=35,
    cmap="viridis",
    mincnt=1
)
ax.set_xlabel("Porosity")
ax.set_ylabel(r"$\tau_z$")
ax.set_title(r"Feasible region density in transport space")
cbar = plt.colorbar(hb, ax=ax)
cbar.set_label("Counts per bin")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig4_porosity_tau_hexbin.png"), bbox_inches="tight")
plt.close()

# ===============================
# Figure 5: porosity vs surface_area
# 作为补充图或扩展数据图很合适
# ===============================
fig, ax = plt.subplots(figsize=(5.8, 4.8))
ax.scatter(por, surf, s=10, alpha=0.4)
ax.set_xlabel("Porosity")
ax.set_ylabel(r"Surface area (1/$\mu$m)")
ax.set_title("Porosity–surface area relationship")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig5_porosity_surface.png"), bbox_inches="tight")
plt.close()

# ===============================
# Figure 6: tau_z vs surface_area
# 用于说明界面复杂度与输运的关系
# ===============================
fig, ax = plt.subplots(figsize=(5.8, 4.8))
ax.scatter(tau, surf, s=10, alpha=0.4)
ax.set_xlabel(r"$\tau_z$")
ax.set_ylabel(r"Surface area (1/$\mu$m)")
ax.set_title(r"Tortuosity–surface area relationship")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig6_tau_surface.png"), bbox_inches="tight")
plt.close()

# ===============================
# Figure 7: D_eff vs porosity/tau_z
# 物理一致性验证图，非常建议做
# ===============================
proxy = por / tau

fig, ax = plt.subplots(figsize=(5.8, 4.8))
ax.scatter(proxy, deff, s=12, alpha=0.5)
xy_min = min(proxy.min(), deff.min())
xy_max = max(proxy.max(), deff.max())
ax.plot([xy_min, xy_max], [xy_min, xy_max], linestyle="--", linewidth=1.2, color="black", label="y = x")
ax.set_xlabel(r"Porosity / $\tau_z$")
ax.set_ylabel(r"$D_{\mathrm{eff},z}$")
ax.set_title(r"Consistency between $D_{\mathrm{eff},z}$ and porosity/$\tau_z$")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig7_deff_vs_por_over_tau.png"), bbox_inches="tight")
plt.close()

print("分析完成，结果保存在:", OUT_DIR)
print("建议正文优先使用：")
print("  Fig1_distributions.png")
print("  Fig2_porosity_tau_surface.png")
print("  Fig3_porosity_tau_trend.png")
print("  Fig4_porosity_tau_hexbin.png")
print("补充图可考虑：")
print("  Fig5_porosity_surface.png")
print("  Fig6_tau_surface.png")
print("  Fig7_deff_vs_por_over_tau.png")

def load_image_safe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到图片: {path}")
    return Image.open(path).convert("RGB")


def resize_keep_aspect(img, target_w, target_h, bg_color=(255,255,255)):
    w, h = img.size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), bg_color)

    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(img_resized, (paste_x, paste_y))
    return canvas


def add_label(img, text, x=15, y=10):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    draw.text((x, y), text, fill=(0,0,0))
    return img


# ===== 读取图 =====
fig1 = load_image_safe(os.path.join(OUT_DIR, "Fig1_distributions.png"))
fig2 = load_image_safe(os.path.join(OUT_DIR, "Fig2_porosity_tau_surface.png"))
fig3 = load_image_safe(os.path.join(OUT_DIR, "Fig3_porosity_tau_trend.png"))
fig4 = load_image_safe(os.path.join(OUT_DIR, "Fig4_porosity_tau_hexbin.png"))

# ===== 布局参数 =====
MARGIN = 40
GAP_X = 30
GAP_Y = 30

COL_W = 600
ROW_H = 420

TOP_W = COL_W * 3 + GAP_X * 2
TOP_H = 420

CANVAS_W = MARGIN*2 + TOP_W
CANVAS_H = MARGIN*2 + TOP_H + GAP_Y + ROW_H

canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (255,255,255))

# ===== 第一行（Fig1 横跨）=====
img1 = resize_keep_aspect(fig1, TOP_W, TOP_H)
img1 = add_label(img1, "(a)")
canvas.paste(img1, (MARGIN, MARGIN))

# ===== 第二行（三张）=====
y2 = MARGIN + TOP_H + GAP_Y

imgs = [fig2, fig3, fig4]
labels = ["(b)", "(c)", "(d)"]

for i, (im, lab) in enumerate(zip(imgs, labels)):
    x = MARGIN + i * (COL_W + GAP_X)
    im_resized = resize_keep_aspect(im, COL_W, ROW_H)
    im_resized = add_label(im_resized, lab)
    canvas.paste(im_resized, (x, y2))

# ===== 保存 =====
out_path = os.path.join(OUT_DIR, "Fig_main_2x3.png")
canvas.save(out_path, dpi=(300,300))

print(" 已生成论文主图：", out_path)