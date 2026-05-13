# vaenet.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 类型定义
# ============================================================

# decoder 最后一层的激活类型
# - "none": 不加激活
# - "sigmoid": 输出压到 [0, 1]，适合二值/灰度已归一化数据
# - "tanh": 输出压到 [-1, 1]
FinalActivationType = Literal["none", "sigmoid", "tanh"]


# ============================================================
# 2. 配置类
# ============================================================

@dataclass
class VAENetConfig:
    """
    适用于 3D 电极体素数据的 VAE 网络配置。

    推荐给 16GB 显存、128^3 patch 的起始配置：
        dimension=3
        in_channels=1
        out_channels=1
        z_dim=4
        ch=32
        ch_mult=[1, 2, 4, 4]
        num_res_blocks=2
        resolution=128
        num_groups=16
        use_attention=False
        final_activation="sigmoid"   # 若输入已归一化到 [0,1]

    参数说明：
        dimension:
            当前版本只支持 3D，因此必须为 3。

        in_channels:
            输入通道数。你的电极体素通常是单通道，所以一般设为 1。

        out_channels:
            输出通道数。重建体素一般也是单通道，所以一般设为 1。

        z_dim:
            真正的 latent 通道数。
            encoder 最终输出 2 * z_dim 个通道，
            前一半是 mean，后一半是 logvar。

        ch:
            基础通道数。越大模型越强，但越吃显存。

        ch_mult:
            每个分辨率层级的通道倍率。
            例如 [1,2,4,4] 表示：
                第1层: 32
                第2层: 64
                第3层: 128
                第4层: 128

        num_res_blocks:
            每个层级有几个 ResBlock。

        dropout:
            3D VAE 第一版一般设为 0.0。

        resolution:
            输入 patch 的边长。你现在目标是 128。

        num_groups:
            GroupNorm 的分组数。对 3D 小 batch 更稳。

        use_attention:
            是否开 attention。

        final_activation:
            decoder 输出激活。
            如果输入数据归一化到 [0,1]，建议用 sigmoid。
    """
    dimension: int = 3
    in_channels: int = 1
    out_channels: int = 1

    # latent 通道数
    z_dim: int = 4

    # base channels
    ch: int = 32

    # 每层通道倍率
    ch_mult: List[int] = None

    # 每层 resblock 数
    num_res_blocks: int = 2

    # dropout
    dropout: float = 0.0

    # 输入 patch 尺寸
    resolution: int = 128

    # GroupNorm 分组数
    num_groups: int = 16

    # 是否使用 attention
    use_attention: bool = False

    # 输出激活函数
    final_activation: FinalActivationType = "sigmoid"

    def __post_init__(self):
        """
        初始化后的配置检查。
        """
        if self.ch_mult is None:
            self.ch_mult = [1, 2, 4, 4]

        if self.dimension != 3:
            raise ValueError("This VAENet currently supports 3D only.")

        if self.final_activation not in {"none", "sigmoid", "tanh"}:
            raise ValueError(f"Unsupported final_activation: {self.final_activation}")

        # 检查 resolution 是否能被下采样层数整除
        # len(ch_mult)=4 时，有 3 次下采样：
        # 128 -> 64 -> 32 -> 16
        down_factor = 2 ** (len(self.ch_mult) - 1)
        if self.resolution % down_factor != 0:
            raise ValueError(
                f"resolution={self.resolution} must be divisible by down_factor={down_factor}"
            )


# ============================================================
# 3. 工具函数与基础模块
# ============================================================

def get_norm(num_channels: int, num_groups: int = 16) -> nn.Module:
    groups = min(num_groups, num_channels)

    # 保证 num_channels 能被 groups 整除
    while num_channels % groups != 0 and groups > 1:
        groups -= 1

    return nn.GroupNorm(groups, num_channels, eps=1e-6, affine=True)


class SiLU(nn.Module):
    """
    自定义 SiLU / Swish 激活。
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResBlock3D(nn.Module):
    """
    3D 残差块。

    结构：
        x
         ├─ norm -> act -> conv
         ├─ norm -> act -> dropout -> conv
         └─ skip
        output = main + skip

    用途：
        作为 encoder / decoder 的基础 building block。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_groups: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = get_norm(in_channels, num_groups)
        self.act1 = SiLU()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.norm2 = get_norm(out_channels, num_groups)
        self.act2 = SiLU()
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # 如果通道数变了，需要一个 1x1x1 的 skip 映射
        if in_channels != out_channels:
            self.skip = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return self.skip(x) + h


class SelfAttention3D(nn.Module):
    """
    3D 自注意力模块。

    注意：
        你当前实验里 use_attention=False，
        所以这部分默认不会参与训练。

    保留它的原因：
        后面你如果想做消融实验，可以直接开。
    """
    def __init__(self, channels: int, num_groups: int = 16):
        super().__init__()
        self.norm = get_norm(channels, num_groups)
        self.q = nn.Conv3d(channels, channels, kernel_size=1)
        self.k = nn.Conv3d(channels, channels, kernel_size=1)
        self.v = nn.Conv3d(channels, channels, kernel_size=1)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.scale = channels ** -0.5

    def forward(self, x):
        """
        输入:
            x: [B, C, D, H, W]

        输出:
            与输入 shape 相同
        """
        b, c, d, h, w = x.shape
        h_ = self.norm(x)

        q = self.q(h_).reshape(b, c, d * h * w).permute(0, 2, 1)  # [B, N, C]
        k = self.k(h_).reshape(b, c, d * h * w)                    # [B, C, N]
        v = self.v(h_).reshape(b, c, d * h * w)                    # [B, C, N]

        attn = torch.bmm(q, k) * self.scale                        # [B, N, N]
        attn = torch.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1))                  # [B, C, N]
        out = out.reshape(b, c, d, h, w)
        out = self.proj(out)

        return x + out


class Downsample3D(nn.Module):
    """
    3D 下采样模块。
    使用 stride=2 的 3D 卷积。
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, x):
        return self.conv(x)


class Upsample3D(nn.Module):
    """
    3D 上采样模块。

    做法：
        先 nearest interpolation，再做 3D conv
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


# ============================================================
# 4. Encoder
# ============================================================

class VAEEncoder(nn.Module):
    """
    3D VAE 编码器。

    输入:
        [B, in_channels, D, H, W]

    输出:
        [B, 2*z_dim, D_lat, H_lat, W_lat]

    其中：
        前一半通道 = mean
        后一半通道 = logvar
    """
    def __init__(self, config: VAENetConfig):
        super().__init__()
        self.config = config

        # 初始卷积，把输入映射到 base channels
        in_ch = config.ch
        self.conv_in = nn.Conv3d(
            config.in_channels,
            in_ch,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        curr_ch = in_ch
        n_levels = len(config.ch_mult)

        # 逐层构建 encoder
        for i, mult in enumerate(config.ch_mult):
            out_ch = config.ch * mult

            blocks = nn.ModuleList()
            for _ in range(config.num_res_blocks):
                blocks.append(
                    ResBlock3D(
                        in_channels=curr_ch,
                        out_channels=out_ch,
                        dropout=config.dropout,
                        num_groups=config.num_groups,
                    )
                )
                curr_ch = out_ch

            # 当前阶段的 attention；默认是 Identity
            attn = SelfAttention3D(curr_ch, config.num_groups) if config.use_attention else nn.Identity()

            self.down_blocks.append(nn.ModuleDict({
                "blocks": blocks,
                "attn": attn,
            }))

            # 除最后一层外都继续下采样
            if i != n_levels - 1:
                self.downsamples.append(Downsample3D(curr_ch))

        # bottleneck / middle blocks
        self.mid_block1 = ResBlock3D(curr_ch, curr_ch, config.dropout, config.num_groups)
        self.mid_attn = SelfAttention3D(curr_ch, config.num_groups) if config.use_attention else nn.Identity()
        self.mid_block2 = ResBlock3D(curr_ch, curr_ch, config.dropout, config.num_groups)

        self.norm_out = get_norm(curr_ch, config.num_groups)
        self.act_out = SiLU()

        # 最终输出 2*z_dim 通道，用于 mean/logvar
        self.conv_out = nn.Conv3d(
            curr_ch,
            2 * config.z_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x, y: Optional[torch.Tensor] = None):
        """
        y 预留给未来条件版，当前先忽略。
        """
        h = self.conv_in(x)

        for i, stage in enumerate(self.down_blocks):
            for block in stage["blocks"]:
                h = block(h)

            h = stage["attn"](h)

            if i < len(self.downsamples):
                h = self.downsamples[i](h)

        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        h = self.act_out(self.norm_out(h))
        h = self.conv_out(h)

        return h


# ============================================================
# 5. Decoder
# ============================================================

class VAEDecoder(nn.Module):
    """
    3D VAE 解码器。

    输入:
        [B, z_dim, D_lat, H_lat, W_lat]

    输出:
        [B, out_channels, D, H, W]
    """
    def __init__(self, config: VAENetConfig):
        super().__init__()
        self.config = config

        # decoder 起始通道数，与 encoder 最底层通道数对应
        curr_ch = config.ch * config.ch_mult[-1]

        # 先把 z 投影到 decoder 起始特征通道
        self.conv_in = nn.Conv3d(
            config.z_dim,
            curr_ch,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # bottleneck / middle blocks
        self.mid_block1 = ResBlock3D(curr_ch, curr_ch, config.dropout, config.num_groups)
        self.mid_attn = SelfAttention3D(curr_ch, config.num_groups) if config.use_attention else nn.Identity()
        self.mid_block2 = ResBlock3D(curr_ch, curr_ch, config.dropout, config.num_groups)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        reversed_mults = list(reversed(config.ch_mult))
        n_levels = len(reversed_mults)

        # 逐层构建 decoder
        for i, mult in enumerate(reversed_mults):
            out_ch = config.ch * mult

            blocks = nn.ModuleList()
            for _ in range(config.num_res_blocks):
                blocks.append(
                    ResBlock3D(
                        in_channels=curr_ch,
                        out_channels=out_ch,
                        dropout=config.dropout,
                        num_groups=config.num_groups,
                    )
                )
                curr_ch = out_ch

            attn = SelfAttention3D(curr_ch, config.num_groups) if config.use_attention else nn.Identity()

            self.up_blocks.append(nn.ModuleDict({
                "blocks": blocks,
                "attn": attn,
            }))

            # 除最后一层外都要继续上采样
            if i != n_levels - 1:
                self.upsamples.append(Upsample3D(curr_ch))

        self.norm_out = get_norm(curr_ch, config.num_groups)
        self.act_out = SiLU()

        # 输出层：恢复到体素通道数
        self.conv_out = nn.Conv3d(
            curr_ch,
            config.out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, z, y: Optional[torch.Tensor] = None):
        """
        y 预留给未来条件版，当前先忽略。
        """
        h = self.conv_in(z)

        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        for i, stage in enumerate(self.up_blocks):
            for block in stage["blocks"]:
                h = block(h)

            h = stage["attn"](h)

            if i < len(self.upsamples):
                h = self.upsamples[i](h)

        h = self.act_out(self.norm_out(h))
        h = self.conv_out(h)

        # 输出激活
        if self.config.final_activation == "sigmoid":
            h = torch.sigmoid(h)
        elif self.config.final_activation == "tanh":
            h = torch.tanh(h)

        return h


# ============================================================
# 6. 顶层网络：作为 VAEModule 的 encdec
# ============================================================

class VAENet(nn.Module):
    """
    作为 VAEModule 的 encdec 使用。

    关键接口：
        - self.encoder(x[, y]) -> z_params
        - self.decoder(z[, y]) -> x_recon

    与你现有 vaemodule.py 兼容：
        vaemodule 会调用:
            encdec.encoder(...)
            encdec.decoder(...)

    注意：
        真正训练 VAE 时，建议通过 vaemodule.py 来统一管理：
            - mean/logvar
            - sampling
            - KL loss
            - reconstruction loss

        这里的 forward() 只是为了方便做一个简单 sanity check。
    """
    def __init__(self, config: VAENetConfig):
        super().__init__()
        self.config = config
        self.encoder = VAEEncoder(config)
        self.decoder = VAEDecoder(config)

    def forward(self, x, y: Optional[torch.Tensor] = None):
        """
        简单前向：
            x -> encoder -> z_params
            z_params -> 取 mean -> decoder -> x_recon

        返回:
            z_params, x_recon

        注意：
            这里只取 mean 做重建，不是完整训练逻辑。
            正式训练时应由 vaemodule.py 调用 encoder / decoder。
        """
        z_params = self.encoder(x, y)

        # z_params 的前半部分是 mean，后半部分是 logvar
        mean, _ = torch.chunk(z_params, 2, dim=1)

        x_recon = self.decoder(mean, y)
        return z_params, x_recon