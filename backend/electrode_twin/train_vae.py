# train_vae.py

import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from dataset import ElectrodeREVDataset
from vaenet import VAENet, VAENetConfig
from vaemodule import VAEModule, VAEModuleConfig


def main():
    L.seed_everything(42)
    torch.set_float32_matmul_precision("high")

    dataset = ElectrodeREVDataset(
        image_dir="../image_dir",
        patch_size=128,
        samples_per_epoch=2000,
        return_condition=False,
        augment_flip=True,
    )

    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

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

    net = VAENet(net_config)

    vae_config = VAEModuleConfig(
        kl_weight=1e-5,
        kl_start=0.0,
        kl_max=1e-5,
        kl_warmup_steps=5000,

        reconstruction_loss="mse",
        lr=1e-4,
        scheduler_type="none",

        # 新增：验证切片展示
        log_val_slices=True,
        val_slice_log_interval=1,
        max_log_images=1,
    )

    model = VAEModule(
        encdec=net,
        config=vae_config,
        conditional=False,
        verbose=True
    )

    checkpoint = ModelCheckpoint(
        dirpath="checkpoints",
        filename="vae-epoch{epoch:03d}-valloss{val/loss:.4f}",
        save_top_k=3,
        monitor="val/loss",
        mode="min",
        auto_insert_metric_name=False,
    )

    logger = TensorBoardLogger(
        save_dir="tb_logs",
        name="vae_electrode"
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        max_epochs=200,
        callbacks=[checkpoint],
        logger=logger,
        log_every_n_steps=10,

        # 这两个很关键
        gradient_clip_val=1.0,
        num_sanity_val_steps=1,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()