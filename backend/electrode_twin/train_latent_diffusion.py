# train_latent_diffusion.py
from __future__ import annotations

import os

import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import TensorBoardLogger

from backend.electrode_twin.latent_dataset import LatentConditionDataset
from backend.electrode_twin.latent_diffusion import LatentDiffusionConfig, LatentDiffusionModule


class EpochSummaryCallback(Callback):
    """
    每个 epoch 结束后打印一行：
        Epoch xxx | train_loss=... | val_loss=... | best_val=...
    """

    def __init__(self):
        super().__init__()
        self.best_val = None
        self.last_train_loss = None

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train/loss_epoch", None)

        if train_loss is not None:
            self.last_train_loss = float(train_loss.detach().cpu().item())

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        val_loss = metrics.get("val/loss", None)

        train_loss_str = "None"
        val_loss_str = "None"

        if self.last_train_loss is not None:
            train_loss_str = f"{self.last_train_loss:.6f}"

        if val_loss is not None:
            val_loss_value = float(val_loss.detach().cpu().item())
            val_loss_str = f"{val_loss_value:.6f}"

            if self.best_val is None or val_loss_value < self.best_val:
                self.best_val = val_loss_value

        best_val_str = "None" if self.best_val is None else f"{self.best_val:.6f}"

        current_epoch = trainer.current_epoch + 1
        print(
            f"Epoch {current_epoch:03d} | "
            f"train_loss={train_loss_str} | "
            f"val_loss={val_loss_str} | "
            f"best_val={best_val_str}"
        )


def main():
    # ============================================================
    # 1. 路径设置
    # ============================================================
    LATENT_DIR = r"latent_dataset"
    SUMMARY_JSON = os.path.join(LATENT_DIR, "dataset_summary.json")

    # ============================================================
    # 2. 训练参数
    # ============================================================
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    MAX_EPOCHS = 500
    TRAIN_RATIO = 0.9

    # 是否过滤非贯通样本
    # 这里建议直接开 True，语义更清楚
    SKIP_NON_PERCOLATING = True

    # 条件是否归一化 / clip
    NORMALIZE_CONDITION = True
    CLIP_CONDITION = True

    L.seed_everything(42)
    torch.set_float32_matmul_precision("high")

    # ============================================================
    # 3. 数据集
    # ============================================================
    dataset = LatentConditionDataset(
        latent_dir=LATENT_DIR,
        summary_json=SUMMARY_JSON,
        normalize_condition=NORMALIZE_CONDITION,
        clip_condition=CLIP_CONDITION,
        skip_non_percolating=SKIP_NON_PERCOLATING,
    )

    print("=" * 80)
    print("Latent dataset loaded")
    print("Dataset size   :", len(dataset))
    print("Latent shape   :", dataset.latent_shape)
    print("Condition keys :", dataset.CONDITION_KEYS)
    print("Cond dim       :", len(dataset.CONDITION_KEYS))
    print("=" * 80)

    train_size = int(len(dataset) * TRAIN_RATIO)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    # ============================================================
    # 4. 模型配置
    # ============================================================
    cond_dim = len(dataset.CONDITION_KEYS)

    config = LatentDiffusionConfig(
        latent_channels=dataset.latent_shape[0],   # 一般是 4
        latent_size=dataset.latent_shape[-1],      # 一般是 16

        cond_dim=cond_dim,                         # 现在是 4
        cond_embed_dim=128,
        time_embed_dim=128,

        model_channels=64,
        channel_mult=(1, 2, 4),

        dropout=0.0,

        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,

        lr=1e-4,
        weight_decay=1e-4,
    )

    model = LatentDiffusionModule(config)

    # ============================================================
    # 5. logger / checkpoint
    # ============================================================
    logger = TensorBoardLogger(
        save_dir="tb_logs",
        name="latent_diffusion_tau",
    )

    checkpoint = ModelCheckpoint(
        dirpath="ldm_checkpoints",
        filename="ldm-epoch{epoch:03d}-valloss{val/loss:.6f}",
        save_top_k=3,
        monitor="val/loss",
        mode="min",
        auto_insert_metric_name=False,
    )

    epoch_summary = EpochSummaryCallback()

    # ============================================================
    # 6. trainer
    # ============================================================
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        max_epochs=MAX_EPOCHS,
        logger=logger,
        callbacks=[checkpoint, epoch_summary],

        enable_progress_bar=False,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        num_sanity_val_steps=1,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
