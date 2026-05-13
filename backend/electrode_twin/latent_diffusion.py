# latent_diffusion.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.electrode_twin.cond_embedder import ConditionEmbedder


# ============================================================
# 1. 配置
# ============================================================

@dataclass
class LatentDiffusionConfig:
    """
    3D latent diffusion 配置

    默认适配：
        latent shape = [4,16,16,16]
        condition dim = 6
    """
    latent_channels: int = 4
    latent_size: int = 16

    cond_dim: int = 6
    cond_embed_dim: int = 128
    time_embed_dim: int = 128

    model_channels: int = 64
    channel_mult: Tuple[int, ...] = (1, 2, 4)

    dropout: float = 0.0

    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    lr: float = 1e-4
    weight_decay: float = 1e-4

    prediction_type: str = "epsilon"  # 这里只做 epsilon-prediction


# ============================================================
# 2. 时间步嵌入
# ============================================================

def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    标准 DDPM sinusoidal timestep embedding

    timesteps: [B]
    return: [B, dim]
    """
    half = dim // 2
    device = timesteps.device

    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=device, dtype=torch.float32) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)

    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)

    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))

    return emb


class TimeEmbedder(nn.Module):
    def __init__(self, time_embed_dim: int = 128):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = timestep_embedding(t, self.time_embed_dim)
        emb = self.mlp(emb)
        return emb


# ============================================================
# 3. 基础模块
# ============================================================

def make_group_norm(num_channels: int, num_groups: int = 8) -> nn.Module:
    groups = min(num_groups, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ResBlock3D(nn.Module):
    """
    带 time/condition 调制的 3D ResBlock
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = make_group_norm(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, out_channels),
        )

        self.norm2 = make_group_norm(out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        x: [B,C,D,H,W]
        emb: [B,emb_channels]
        """
        h = self.conv1(self.act1(self.norm1(x)))

        emb_out = self.emb_proj(emb).view(emb.shape[0], -1, 1, 1, 1)
        h = h + emb_out

        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class Downsample3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


# ============================================================
# 4. 条件 3D U-Net
# ============================================================

class ConditionalUNet3D(nn.Module):
    """
    输入:
        x_t:  [B, 4, 16, 16, 16]
        t:    [B]
        cond: [B, 6]

    输出:
        pred_noise: [B, 4, 16, 16, 16]
    """

    def __init__(self, config: LatentDiffusionConfig):
        super().__init__()
        self.config = config

        self.time_embedder = TimeEmbedder(config.time_embed_dim)
        self.cond_embedder = ConditionEmbedder(
            in_dim=config.cond_dim,
            embed_dim=config.cond_embed_dim,
            hidden_dim=max(256, config.cond_embed_dim * 2),
            dropout=config.dropout,
        )

        self.global_embed = nn.Sequential(
            nn.Linear(config.time_embed_dim + config.cond_embed_dim, config.time_embed_dim),
            nn.SiLU(),
            nn.Linear(config.time_embed_dim, config.time_embed_dim),
        )

        base = config.model_channels
        ch1 = base * config.channel_mult[0]   # 64
        ch2 = base * config.channel_mult[1]   # 128
        ch3 = base * config.channel_mult[2]   # 256

        self.conv_in = nn.Conv3d(config.latent_channels, ch1, kernel_size=3, padding=1)

        # encoder
        self.down1_block1 = ResBlock3D(ch1, ch1, config.time_embed_dim, config.dropout)
        self.down1_block2 = ResBlock3D(ch1, ch1, config.time_embed_dim, config.dropout)
        self.down1 = Downsample3D(ch1)  # 16 -> 8

        self.down2_block1 = ResBlock3D(ch1, ch2, config.time_embed_dim, config.dropout)
        self.down2_block2 = ResBlock3D(ch2, ch2, config.time_embed_dim, config.dropout)
        self.down2 = Downsample3D(ch2)  # 8 -> 4

        self.down3_block1 = ResBlock3D(ch2, ch3, config.time_embed_dim, config.dropout)
        self.down3_block2 = ResBlock3D(ch3, ch3, config.time_embed_dim, config.dropout)

        # middle
        self.mid1 = ResBlock3D(ch3, ch3, config.time_embed_dim, config.dropout)
        self.mid2 = ResBlock3D(ch3, ch3, config.time_embed_dim, config.dropout)

        # decoder
        self.up2 = Upsample3D(ch3)  # 4 -> 8
        self.up2_block1 = ResBlock3D(ch3 + ch2, ch2, config.time_embed_dim, config.dropout)
        self.up2_block2 = ResBlock3D(ch2, ch2, config.time_embed_dim, config.dropout)

        self.up1 = Upsample3D(ch2)  # 8 -> 16
        self.up1_block1 = ResBlock3D(ch2 + ch1, ch1, config.time_embed_dim, config.dropout)
        self.up1_block2 = ResBlock3D(ch1, ch1, config.time_embed_dim, config.dropout)

        self.norm_out = make_group_norm(ch1)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv3d(ch1, config.latent_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedder(t)         # [B, tdim]
        c_emb = self.cond_embedder(cond)      # [B, cdim]
        emb = self.global_embed(torch.cat([t_emb, c_emb], dim=1))

        h0 = self.conv_in(x)

        h1 = self.down1_block1(h0, emb)
        h1 = self.down1_block2(h1, emb)
        d1 = self.down1(h1)

        h2 = self.down2_block1(d1, emb)
        h2 = self.down2_block2(h2, emb)
        d2 = self.down2(h2)

        h3 = self.down3_block1(d2, emb)
        h3 = self.down3_block2(h3, emb)

        m = self.mid1(h3, emb)
        m = self.mid2(m, emb)

        u2 = self.up2(m)
        u2 = torch.cat([u2, h2], dim=1)
        u2 = self.up2_block1(u2, emb)
        u2 = self.up2_block2(u2, emb)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, h1], dim=1)
        u1 = self.up1_block1(u1, emb)
        u1 = self.up1_block2(u1, emb)

        out = self.conv_out(self.act_out(self.norm_out(u1)))
        return out


# ============================================================
# 5. DDPM 模块
# ============================================================

class LatentDiffusionModule(L.LightningModule):
    def __init__(self, config: LatentDiffusionConfig):
        super().__init__()
        self.config = config
        self.model = ConditionalUNet3D(config)

        # diffusion schedule
        betas = torch.linspace(config.beta_start, config.beta_end, config.num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

        self.save_hyperparameters()

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*noise
        """
        a = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1, 1)
        return a * x0 + b * noise

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.model(x_t, t, cond)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        z = batch["z"]         # [B,4,16,16,16]
        cond = batch["cond"]   # [B,6]

        b = z.shape[0]
        device = z.device

        t = torch.randint(0, self.config.num_timesteps, (b,), device=device, dtype=torch.long)
        noise = torch.randn_like(z)
        x_t = self.q_sample(z, t, noise)

        pred_noise = self.model(x_t, t, cond)

        loss = F.mse_loss(pred_noise, noise)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=False)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        z = batch["z"]
        cond = batch["cond"]

        b = z.shape[0]
        device = z.device

        t = torch.randint(0, self.config.num_timesteps, (b,), device=device, dtype=torch.long)
        noise = torch.randn_like(z)
        x_t = self.q_sample(z, t, noise)

        pred_noise = self.model(x_t, t, cond)
        loss = F.mse_loss(pred_noise, noise)

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        return optimizer

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        shape: Tuple[int, int, int, int],
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        简单 DDPM 采样
        cond: [B,6]
        shape: (C,D,H,W)
        return: [B,C,D,H,W]
        """
        device = cond.device
        b = cond.shape[0]

        x = torch.randn((b, *shape), device=device)

        # classifier-free guidance 这版先不做真正 dropout 训练
        # guidance_scale 先保留接口
        for i in reversed(range(self.config.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)

            pred_noise = self.model(x, t, cond)

            alpha = self.alphas[i]
            alpha_bar = self.alpha_bars[i]
            beta = self.betas[i]

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (
                1.0 / torch.sqrt(alpha)
            ) * (
                x - (beta / torch.sqrt(1.0 - alpha_bar)) * pred_noise
            ) + torch.sqrt(beta) * noise

        return x
