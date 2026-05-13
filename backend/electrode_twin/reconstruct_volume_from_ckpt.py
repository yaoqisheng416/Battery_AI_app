# reconstruct_volume_from_ckpt.py
from __future__ import annotations

import os
import glob
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from dataset import REVSpec
from vaenet import VAENet, VAENetConfig
from vaemodule import VAEModule, VAEModuleConfig


# ============================================================
# 这里是你需要修改的参数区域
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ELECTRODE_TWIN_DIR = os.path.join(BASE_DIR, "electrode_twin")

# 1. 改成你的 checkpoint 路径
CKPT_PATH = os.path.join(
    ELECTRODE_TWIN_DIR,
    "checkpoints",
    "vae-epoch045-valloss-1.9100.ckpt"
)
# CKPT_PATH = r".\checkpoints\vae-epoch045-valloss-1.9100.ckpt"

# 2. 改成你的原始切片文件夹路径
IMAGE_DIR = r"../image_dir"

# 3. 改成你希望输出结果保存到哪里
OUT_DIR = os.path.join(
    ELECTRODE_TWIN_DIR,
    "reconstruction_results"
)
# OUT_DIR = r"reconstruction_results"

# 4. 切片格式
EXT = "*.tif"

# 5. patch 尺寸（要和训练时一致）
PATCH_SIZE = 128

# 6. 滑窗步长
#    推荐先用 64
#    如果你想更平滑一点，可以改成 32，但会更慢
STRIDE = 64

# 7. 二值化阈值
#    如果你的原图就是0/255二值图，通常 threshold=0 就行
THRESHOLD = 0

# 8. REV 尺寸（要和 dataset.py 里面一致）
REV_Y = 258
REV_Z = 456
REV_X = 253

# 9. 设备
#    有显卡就 "cuda"
#    没显卡就 "cpu"
# DEVICE = "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 推理配置
# ============================================================

@dataclass
class InferenceConfig:
    patch_size: int = PATCH_SIZE
    stride: int = STRIDE
    threshold: int = THRESHOLD
    use_mode: bool = True   # True 表示用 posterior mean 重建，更稳定


def load_image_sequence(image_dir: str, ext: str = "*.tif") -> np.ndarray:
    files = sorted(glob.glob(os.path.join(image_dir, ext)))
    if len(files) == 0:
        raise ValueError(f"在 {image_dir} 中没有找到 {ext} 文件")

    slices = []
    for f in files:
        img = Image.open(f).convert("L")
        slices.append(np.asarray(img, dtype=np.uint8))

    volume = np.stack(slices, axis=0)  # [Y, Z, X]
    return volume


def extract_rev(volume: np.ndarray, spec: REVSpec) -> np.ndarray:
    y0, z0, x0 = volume.shape
    if spec.y > y0 or spec.z > z0 or spec.x > x0:
        raise ValueError(f"REV尺寸 {(spec.y, spec.z, spec.x)} 大于原始体积 {volume.shape}")

    sy = (y0 - spec.y) // 2
    sz = (z0 - spec.z) // 2
    sx = (x0 - spec.x) // 2

    return volume[sy:sy + spec.y, sz:sz + spec.z, sx:sx + spec.x]


def compute_starts(size: int, patch_size: int, stride: int) -> list[int]:
    if size < patch_size:
        raise ValueError(f"size={size} 小于 patch_size={patch_size}")

    starts = list(range(0, size - patch_size + 1, stride))
    if starts[-1] != size - patch_size:
        starts.append(size - patch_size)
    return starts


def make_model(ckpt_path: str, device: torch.device) -> VAEModule:
    """
    这里的网络配置要和训练时保持一致
    """
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
        final_activation="sigmoid",
    )

    vae_config = VAEModuleConfig(
        kl_weight=1e-5,
        kl_start=0.0,
        kl_max=1e-5,
        kl_warmup_steps=5000,
        reconstruction_loss="mse",
        lr=1e-4,
        scheduler_type="none",
    )

    net = VAENet(net_config)

    model = VAEModule.load_from_checkpoint(
        ckpt_path,
        encdec=net,
        config=vae_config,
        conditional=False,
        verbose=False,
        map_location=device,
        weights_only=False
    )

    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def reconstruct_volume(
    model: VAEModule,
    volume01: np.ndarray,
    patch_size: int,
    stride: int,
    device: torch.device,
    use_mode: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    输入:
        volume01: [Y, Z, X], 二值体，取值 0/1

    输出:
        recon_prob: [Y, Z, X], 重建概率体 [0,1]
        recon_bin:  [Y, Z, X], 二值重建体 0/1
    """
    y, z, x = volume01.shape

    ys_list = compute_starts(y, patch_size, stride)
    zs_list = compute_starts(z, patch_size, stride)
    xs_list = compute_starts(x, patch_size, stride)

    recon_sum = np.zeros_like(volume01, dtype=np.float32)
    recon_count = np.zeros_like(volume01, dtype=np.float32)

    total = len(ys_list) * len(zs_list) * len(xs_list)
    pbar = tqdm(total=total, desc="Reconstructing 3D volume")

    for ys in ys_list:
        for zs in zs_list:
            for xs in xs_list:
                patch = volume01[
                    ys:ys + patch_size,
                    zs:zs + patch_size,
                    xs:xs + patch_size,
                ].astype(np.float32)

                patch_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,P,P,P]

                out = model(patch_t, sample_posterior=not use_mode)
                recon_patch = out["x_recon"][0, 0].detach().float().cpu().numpy()

                recon_sum[
                    ys:ys + patch_size,
                    zs:zs + patch_size,
                    xs:xs + patch_size,
                ] += recon_patch

                recon_count[
                    ys:ys + patch_size,
                    zs:zs + patch_size,
                    xs:xs + patch_size,
                ] += 1.0

                pbar.update(1)

    pbar.close()

    recon_prob = recon_sum / np.maximum(recon_count, 1e-8)
    recon_prob = np.clip(recon_prob, 0.0, 1.0)
    recon_bin = (recon_prob >= 0.5).astype(np.uint8)

    return recon_prob, recon_bin


def save_volume_slices(
    original01: np.ndarray,
    recon_prob: np.ndarray,
    recon_bin: np.ndarray,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "compare_png"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "original_png"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "recon_prob_png"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "recon_bin_png"), exist_ok=True)

    y = original01.shape[0]

    for i in tqdm(range(y), desc="Saving slices"):
        orig = (original01[i] * 255).astype(np.uint8)
        prob = (recon_prob[i] * 255).astype(np.uint8)
        rb = (recon_bin[i] * 255).astype(np.uint8)

        Image.fromarray(orig).save(
            os.path.join(out_dir, "original_png", f"slice_{i:04d}.png")
        )
        Image.fromarray(prob).save(
            os.path.join(out_dir, "recon_prob_png", f"slice_{i:04d}.png")
        )
        Image.fromarray(rb).save(
            os.path.join(out_dir, "recon_bin_png", f"slice_{i:04d}.png")
        )

        # 横向拼接：原图 | 重建概率 | 二值重建
        compare = np.concatenate([orig, prob, rb], axis=1)
        Image.fromarray(compare).save(
            os.path.join(out_dir, "compare_png", f"slice_{i:04d}.png")
        )


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    rev_spec = REVSpec(y=REV_Y, z=REV_Z, x=REV_X)
    infer_cfg = InferenceConfig()

    print("==================================================")
    print("开始 VAE 重建")
    print("CKPT_PATH :", CKPT_PATH)
    print("IMAGE_DIR :", IMAGE_DIR)
    print("OUT_DIR   :", OUT_DIR)
    print("DEVICE    :", device)
    print("PATCH_SIZE:", infer_cfg.patch_size)
    print("STRIDE    :", infer_cfg.stride)
    print("==================================================")

    print("1) 读取原始切片 ...")
    volume = load_image_sequence(IMAGE_DIR, EXT)

    print("2) 提取 REV ...")
    rev = extract_rev(volume, rev_spec)

    # 和训练保持一致：solid=1, pore=0
    rev01 = (rev > infer_cfg.threshold).astype(np.uint8)

    print("3) 加载模型 ...")
    model = make_model(CKPT_PATH, device)

    print("4) 重建整个 3D REV ...")
    recon_prob, recon_bin = reconstruct_volume(
        model=model,
        volume01=rev01,
        patch_size=infer_cfg.patch_size,
        stride=infer_cfg.stride,
        device=device,
        use_mode=infer_cfg.use_mode,
    )

    print("5) 保存体数据 ...")
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "original_volume.npy"), rev01.astype(np.uint8))
    np.save(os.path.join(OUT_DIR, "recon_prob_volume.npy"), recon_prob.astype(np.float32))
    np.save(os.path.join(OUT_DIR, "recon_bin_volume.npy"), recon_bin.astype(np.uint8))

    print("6) 保存逐层切片图 ...")
    save_volume_slices(
        original01=rev01,
        recon_prob=recon_prob,
        recon_bin=recon_bin,
        out_dir=OUT_DIR,
    )

    print("完成，结果保存在:", OUT_DIR)


if __name__ == "__main__":
    main()
