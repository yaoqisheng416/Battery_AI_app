# vaemodule.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Union

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor

import backend.electrode_twin.batch_norm as batchnorm
from backend.electrode_twin import preprocessors

ReconstructionLossType = Literal["mse", "huber", "bce"]
LossPreprocessorType = Literal["none", "edges"]
SchedulerType = Literal["none", "cosine", "constant"]


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: SchedulerType,
    max_steps: int,
    min_lr_ratio: float = 0.1,
):
    if scheduler_type == "none":
        return None

    if scheduler_type == "constant":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: 1.0,
        )

    if scheduler_type == "cosine":
        eta_min = optimizer.param_groups[0]["lr"] * min_lr_ratio
        t_max = max(1, int(max_steps))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )

    raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")


@dataclass
class VAEModuleConfig:
    kl_weight: float = 1e-5
    recon_weight: float = 1.0

    logvar_init: float = 0.0
    trainable_logvar: bool = True

    reconstruction_loss: ReconstructionLossType = "mse"
    loss_preprocessor: Union[LossPreprocessorType, nn.Module] = "none"
    total_variation_weight: float = 0.0

    initial_norm: bool = False
    num_channels: int = 1

    conditional: bool = False

    optimizer_name: Literal["adamw", "adam"] = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)

    scheduler_type: SchedulerType = "none"
    scheduler_interval: Literal["step", "epoch"] = "step"
    scheduler_max_steps: int = 100000
    scheduler_min_lr_ratio: float = 0.1

    grad_clip_val: float = 0.0

    kl_start: float = 0.0
    kl_max: float = 1e-5
    kl_warmup_steps: int = 5000

    # -------- 新增：验证图像展示 --------
    log_val_slices: bool = True
    val_slice_log_interval: int = 1
    max_log_images: int = 1
    # ----------------------------------

    def __post_init__(self):
        if self.reconstruction_loss not in {"mse", "huber", "bce"}:
            raise ValueError(
                f"reconstruction_loss must be one of ['mse', 'huber', 'bce'], "
                f"got {self.reconstruction_loss}"
            )

        if isinstance(self.loss_preprocessor, str):
            if self.loss_preprocessor not in {"none", "edges"}:
                raise ValueError(
                    f"loss_preprocessor must be 'none' or 'edges', got {self.loss_preprocessor}"
                )

        if self.kl_weight < 0:
            raise ValueError("kl_weight must be >= 0")

        if self.recon_weight <= 0:
            raise ValueError("recon_weight must be > 0")

        if self.total_variation_weight < 0:
            raise ValueError("total_variation_weight must be >= 0")

        if self.kl_start < 0 or self.kl_max < 0:
            raise ValueError("kl_start / kl_max must be >= 0")

        if self.kl_warmup_steps < 1:
            raise ValueError("kl_warmup_steps must be >= 1")

        if self.max_log_images < 1:
            raise ValueError("max_log_images must be >= 1")


class DiagonalGaussianDistribution(nn.Module):
    def __init__(
        self,
        mean_and_logvar: Tensor,
        low_clamp: float = -30.0,
        high_clamp: float = 20.0,
    ):
        super().__init__()
        mean, logvar = torch.chunk(mean_and_logvar, 2, dim=1)
        self.mean = mean
        self.logvar = torch.clamp(logvar, low_clamp, high_clamp)

    @property
    def std(self) -> Tensor:
        return torch.exp(0.5 * self.logvar)

    @property
    def var(self) -> Tensor:
        return torch.exp(self.logvar)

    def sample(self) -> Tensor:
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self) -> Tensor:
        return self.mean

    def kl(self) -> Tensor:
        dims = tuple(range(1, self.mean.dim()))
        return 0.5 * torch.sum(
            self.mean.pow(2) + self.var - 1.0 - self.logvar,
            dim=dims,
        )

    def nll(self, sample: Tensor) -> Tensor:
        dims = tuple(range(1, sample.dim()))
        log2pi = math.log(2.0 * math.pi)
        return 0.5 * torch.sum(
            log2pi + self.logvar + (sample - self.mean).pow(2) / self.var,
            dim=dims,
        )


class TotalVariationLoss(nn.Module):
    def __init__(self, loss_type: Literal["mse", "huber"] = "mse", weight: float = 1.0):
        super().__init__()
        self.weight = weight
        if loss_type == "mse":
            self.loss_fn = F.mse_loss
        elif loss_type == "huber":
            self.loss_fn = F.huber_loss
        else:
            raise ValueError(f"TV loss only supports 'mse' or 'huber', got {loss_type}")

    def total_variation(self, x: Tensor) -> Tensor:
        tv = 0.0
        for dim in range(2, x.dim()):
            front = [slice(None)] * x.dim()
            back = [slice(None)] * x.dim()
            front[dim] = slice(1, None)
            back[dim] = slice(None, -1)
            diff = torch.abs(x[tuple(front)] - x[tuple(back)])
            tv = tv + torch.sum(diff, dim=tuple(range(1, diff.dim())))
        return tv

    def forward(self, x_real: Tensor, x_recon: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        tv_real = self.total_variation(x_real)
        tv_recon = self.total_variation(x_recon)
        tv_loss = self.loss_fn(tv_recon, tv_real, reduction="mean")
        total_loss = self.weight * tv_loss
        logs = {
            "tv_loss": tv_loss.detach(),
            "tv_real_mean": tv_real.mean().detach(),
            "tv_recon_mean": tv_recon.mean().detach(),
        }
        return total_loss, logs


class VAELoss(nn.Module):
    def __init__(self, config: VAEModuleConfig):
        super().__init__()
        self.config = config

        if config.trainable_logvar:
            self.logvar = nn.Parameter(torch.tensor(float(config.logvar_init)))
        else:
            self.register_buffer("logvar", torch.tensor(float(config.logvar_init)))

        if isinstance(config.loss_preprocessor, nn.Module):
            self.loss_preprocessor = config.loss_preprocessor
        else:
            if config.loss_preprocessor == "none":
                self.loss_preprocessor = nn.Identity()
            elif config.loss_preprocessor == "edges":
                self.loss_preprocessor = preprocessors.EdgeDetectionPreprocessor(dim=3)
            else:
                raise ValueError(f"Unsupported loss_preprocessor: {config.loss_preprocessor}")

        if config.total_variation_weight > 0.0:
            tv_base = "mse" if config.reconstruction_loss in {"mse", "bce"} else "huber"
            self.tv_loss_module = TotalVariationLoss(
                loss_type=tv_base,
                weight=config.total_variation_weight,
            )
        else:
            self.tv_loss_module = None

    def get_current_kl_weight(self, global_step: int) -> float:
        if global_step >= self.config.kl_warmup_steps:
            return float(self.config.kl_max)

        ratio = float(global_step) / float(self.config.kl_warmup_steps)
        return float(self.config.kl_start + ratio * (self.config.kl_max - self.config.kl_start))

    def _reconstruction_error(self, x: Tensor, x_recon: Tensor) -> Tensor:
        x_p = self.loss_preprocessor(x.contiguous())
        xr_p = self.loss_preprocessor(x_recon.contiguous())

        if self.config.reconstruction_loss == "mse":
            err = F.mse_loss(xr_p, x_p, reduction="none")
        elif self.config.reconstruction_loss == "huber":
            err = F.huber_loss(xr_p, x_p, reduction="none")
        elif self.config.reconstruction_loss == "bce":
            err = F.binary_cross_entropy(xr_p, x_p, reduction="none")
        else:
            raise ValueError(f"Unsupported reconstruction_loss: {self.config.reconstruction_loss}")

        err = err.mean(dim=tuple(range(1, err.dim())))
        return err

    def forward(
        self,
        inputs: Tensor,
        reconstructions: Tensor,
        posterior: DiagonalGaussianDistribution,
        global_step: int,
        split: str = "train",
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        recon_error = self._reconstruction_error(inputs, reconstructions)
        weighted_recon = recon_error / torch.exp(self.logvar) + self.logvar
        recon_loss = torch.mean(weighted_recon)

        kl_loss = torch.mean(posterior.kl())
        current_kl_weight = self.get_current_kl_weight(global_step)

        total_loss = (
            self.config.recon_weight * recon_loss
            + current_kl_weight * kl_loss
        )

        log_dict: Dict[str, Tensor] = {
            f"{split}/total_loss": total_loss.detach(),
            f"{split}/recon_loss": recon_loss.detach(),
            f"{split}/kl_loss": kl_loss.detach(),
            f"{split}/logvar": self.logvar.detach(),
            f"{split}/kl_weight": torch.tensor(current_kl_weight, device=inputs.device),
        }

        if self.tv_loss_module is not None:
            tv_loss, tv_logs = self.tv_loss_module(inputs, reconstructions)
            total_loss = total_loss + tv_loss
            for k, v in tv_logs.items():
                log_dict[f"{split}/{k}"] = v.detach()

        return total_loss, log_dict


class VAEModule(L.LightningModule):
    def __init__(
        self,
        encdec: nn.Module,
        config: VAEModuleConfig,
        conditional: Optional[bool] = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.encdec = encdec
        self.config = config
        self.verbose = verbose

        if not hasattr(self.encdec, "encoder") or not hasattr(self.encdec, "decoder"):
            raise ValueError("encdec must have both 'encoder' and 'decoder' attributes")

        self.conditional = config.conditional if conditional is None else conditional
        self.loss_module = VAELoss(config)

        if config.initial_norm:
            self.initial_norm = batchnorm.DimensionAgnosticBatchNorm(
                num_channels=config.num_channels,
                eps=1e-5,
                affine=False,
            )
        else:
            self.initial_norm = batchnorm.IdentityBatchNorm()

        self.save_hyperparameters(ignore=["encdec"])
        # 同时保存 encdec 的网络配置，以便推理时自动匹配模型结构
        if hasattr(encdec, 'config'):
            self.hparams['net_config'] = encdec.config

    @property
    def encoder(self):
        return self.encdec.encoder

    @property
    def decoder(self):
        return self.encdec.decoder

    def preencode(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        return self.initial_norm(x)

    def postdecode(self, x_recon: Tensor, y: Optional[Tensor] = None) -> Tensor:
        return self.initial_norm.unnorm(x_recon)

    def encode(
        self,
        x: Tensor,
        y: Optional[Tensor] = None,
        sample: bool = True,
        apply_preencode: bool = True,
    ) -> Dict[str, Tensor]:
        if apply_preencode:
            x = self.preencode(x, y)

        z_params = self.encoder(x, y) if self.conditional else self.encoder(x)
        zdistrib = DiagonalGaussianDistribution(z_params)
        zsample = zdistrib.sample() if sample else zdistrib.mode()

        return {
            "z_params": z_params,
            "zdistrib": zdistrib,
            "zsample": zsample,
        }

    def decode(
        self,
        z: Tensor,
        y: Optional[Tensor] = None,
        apply_postdecode: bool = True,
    ) -> Tensor:
        x_recon = self.decoder(z, y) if self.conditional else self.decoder(z)
        if apply_postdecode:
            x_recon = self.postdecode(x_recon, y)
        return x_recon

    def forward(
        self,
        x: Tensor,
        y: Optional[Tensor] = None,
        sample_posterior: bool = True,
    ) -> Dict[str, Tensor]:
        enc = self.encode(x, y=y, sample=sample_posterior, apply_preencode=True)
        x_recon = self.decode(enc["zsample"], y=y, apply_postdecode=True)
        return {
            **enc,
            "x_recon": x_recon,
        }

    def _unpack_batch(self, batch):
        if isinstance(batch, dict):
            x = batch["x"]
            y = batch.get("y", None)
            return x, y

        if isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                return batch[0], batch[1]
            if len(batch) == 1:
                return batch[0], None

        return batch, None

    def _make_reconstruction_slice_grid(self, x: Tensor, x_recon: Tensor) -> Tensor:
        """
        生成用于 TensorBoard 展示的 2D 拼图。
        每个样本展示三张正交中间切片：
            [真实xy | 重建xy]
            [真实xz | 重建xz]
            [真实yz | 重建yz]
        """
        x = x.detach().float().cpu()
        x_recon = x_recon.detach().float().cpu()

        x = x.clamp(0.0, 1.0)
        x_recon = x_recon.clamp(0.0, 1.0)

        num_items = min(x.shape[0], self.config.max_log_images)
        panels = []

        for i in range(num_items):
            real = x[i, 0]       # [D,H,W]
            recon = x_recon[i, 0]

            d, h, w = real.shape
            md, mh, mw = d // 2, h // 2, w // 2

            real_xy = real[md, :, :]
            real_xz = real[:, mh, :]
            real_yz = real[:, :, mw]

            recon_xy = recon[md, :, :]
            recon_xz = recon[:, mh, :]
            recon_yz = recon[:, :, mw]

            # 每张切片转成 [1,H,W]
            real_xy = real_xy.unsqueeze(0)
            real_xz = real_xz.unsqueeze(0)
            real_yz = real_yz.unsqueeze(0)

            recon_xy = recon_xy.unsqueeze(0)
            recon_xz = recon_xz.unsqueeze(0)
            recon_yz = recon_yz.unsqueeze(0)

            row1 = torch.cat([real_xy, recon_xy], dim=2)
            row2 = torch.cat([real_xz, recon_xz], dim=2)
            row3 = torch.cat([real_yz, recon_yz], dim=2)

            panel = torch.cat([row1, row2, row3], dim=1)  # [1, 3H, 2W]
            panels.append(panel)

        grid = torchvision.utils.make_grid(panels, nrow=1, padding=4)
        return grid

    def _log_val_slices(self, batch, batch_idx: int):
        if not self.config.log_val_slices:
            return
        if batch_idx != 0:
            return
        if (self.current_epoch % self.config.val_slice_log_interval) != 0:
            return

        x, y = self._unpack_batch(batch)

        with torch.no_grad():
            enc = self.encode(
                x,
                y=y,
                sample=False,
                apply_preencode=True,
            )
            x_recon = self.decode(
                enc["zsample"],
                y=y,
                apply_postdecode=True,
            )

        grid = self._make_reconstruction_slice_grid(x, x_recon)

        if hasattr(self.logger, "experiment"):
            self.logger.experiment.add_image(
                "val/reconstruction_slices",
                grid,
                global_step=self.global_step
            )

    def shared_step(self, batch, split: str = "train") -> Tuple[Tensor, Dict[str, Tensor]]:
        x, y = self._unpack_batch(batch)

        enc = self.encode(
            x,
            y=y,
            sample=True if split == "train" else False,
            apply_preencode=True,
        )
        x_recon = self.decode(
            enc["zsample"],
            y=y,
            apply_postdecode=True,
        )

        loss, logs = self.loss_module(
            inputs=x,
            reconstructions=x_recon,
            posterior=enc["zdistrib"],
            global_step=self.global_step,
            split=split,
        )

        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.shared_step(batch, split="train")
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.shared_step(batch, split="val")
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        self._log_val_slices(batch, batch_idx)
        return loss

    def test_step(self, batch, batch_idx):
        loss, logs = self.shared_step(batch, split="test")
        self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def configure_optimizers(self):
        if self.config.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.lr,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.config.lr,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer_name: {self.config.optimizer_name}")

        scheduler = build_scheduler(
            optimizer=optimizer,
            scheduler_type=self.config.scheduler_type,
            max_steps=self.config.scheduler_max_steps,
            min_lr_ratio=self.config.scheduler_min_lr_ratio,
        )

        if scheduler is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.config.scheduler_interval,
                "frequency": 1,
            },
        }