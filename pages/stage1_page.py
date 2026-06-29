# -*- coding: utf-8 -*-
import os
import sys
import shutil
import traceback
import numpy as np

# 确保项目根目录在 sys.path，以便 import backend 模块
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 同时确保 electrode_twin 目录可被 import（供 train_vae 内部相对 import 使用）
_ELECTRODE_TWIN = os.path.join(_PROJECT_ROOT, "backend", "electrode_twin")
if _ELECTRODE_TWIN not in sys.path:
    sys.path.insert(0, _ELECTRODE_TWIN)

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, Callback

from PySide6.QtGui import Qt, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QTextEdit, QSpinBox, QProgressBar, QMessageBox,
    QCheckBox,
)
from PySide6.QtCore import QTimer, QThread, Signal

from backend.electrode_twin.vaenet import VAENet, VAENetConfig
from backend.electrode_twin.vaemodule import VAEModule, VAEModuleConfig
from backend.electrode_twin.dataset import ElectrodeREVDataset
from backend.electrode_twin.CBD_generate import BASE_DIR

# ============================================================
# 合成数据集 —— 无需真实 TIFF 切片即可快速训练
# ============================================================

class Synthetic3DDataset(Dataset):
    """
    生成含有随机球体结构的 3D 二值体素数据。
    用于在无真实数据时快速验证 VAE 训练流程。
    """

    def __init__(
        self,
        volume_shape=(128, 128, 128),
        patch_size=64,
        samples_per_epoch=50,
        seed=42,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.samples = samples_per_epoch

        rng = np.random.RandomState(seed)

        # 生成带结构的合成体积：在空白体积中放置随机球体
        volume = np.zeros(volume_shape, dtype=np.float32)
        for _ in range(8):
            r = rng.randint(8, 25)
            cy = rng.randint(r, volume_shape[0] - r)
            cz = rng.randint(r, volume_shape[1] - r)
            cx = rng.randint(r, volume_shape[2] - r)
            y, z, x = np.ogrid[
                :volume_shape[0], :volume_shape[1], :volume_shape[2]
            ]
            dist = np.sqrt((y - cy) ** 2 + (z - cz) ** 2 + (x - cx) ** 2)
            volume[dist <= r] = 1.0

        self.volume = volume
        print(f"[Synthetic3DDataset] volume={volume_shape}, "
              f"solid_fraction={volume.mean():.3f}")

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        v = self.volume
        ps = self.patch_size
        ys = np.random.randint(0, v.shape[0] - ps)
        zs = np.random.randint(0, v.shape[1] - ps)
        xs = np.random.randint(0, v.shape[2] - ps)

        patch = v[ys:ys + ps, zs:zs + ps, xs:xs + ps].copy()

        # 随机翻转增强（用 np.flip 而非切片，避免负 stride）
        if np.random.rand() < 0.5:
            patch = np.flip(patch, axis=0)
        if np.random.rand() < 0.5:
            patch = np.flip(patch, axis=1)
        if np.random.rand() < 0.5:
            patch = np.flip(patch, axis=2)

        patch_t = torch.from_numpy(patch.copy()).unsqueeze(0).float()
        return {"x": patch_t}


# ============================================================
# 训练工作线程 —— 在后台运行 Lightning 训练
# ============================================================

class TrainWorker(QThread):
    """在后台线程中运行 VAE 训练，通过信号通知 UI。"""

    progress = Signal(int)       # 进度百分比 0~100
    log_msg = Signal(str)        # 日志消息
    finished = Signal(str)       # 训练成功，携带 ckpt 路径
    failed = Signal(str)         # 训练失败，携带错误信息

    def __init__(
        self,
        dataset_dir="",
        output_dir="",
        epochs=3,
        use_synthetic=True,
        parent=None,
    ):
        super().__init__(parent)
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.epochs = epochs
        self.use_synthetic = use_synthetic

    def run(self):
        try:
            ckpt = self._do_training()
            if ckpt and os.path.exists(ckpt):
                self.finished.emit(ckpt)
            else:
                self.failed.emit("训练完成但未找到 checkpoint 文件。")
        except Exception:
            self.failed.emit(f"训练异常:\n{traceback.format_exc()}")

    def _do_training(self):
        """执行训练，返回 checkpoint 路径或 None。"""
        L.seed_everything(42)
        torch.set_float32_matmul_precision("high")

        # ---- 输出目录 ----
        output_dir = self.output_dir
        if not output_dir:
            output_dir = os.path.join(_PROJECT_ROOT, "checkpoints")
        os.makedirs(output_dir, exist_ok=True)

        # ---- 加速器 ----
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        self.log_msg.emit(f"加速器: {accelerator}")

        # ---- 快速训练参数（保证 ~1 分钟内完成） ----
        patch_size = 64
        samples_per_epoch = 50
        batch_size = 1
        z_dim = 2
        ch = 16
        ch_mult = (1, 2, 2)  # 下采样 2 次: 64→32→16
        num_res_blocks = 1

        # ---- 数据集 ----
        if self.dataset_dir and os.path.isdir(self.dataset_dir):
            self.log_msg.emit(f"加载真实数据: {self.dataset_dir}")
            dataset = ElectrodeREVDataset(
                image_dir=self.dataset_dir,
                patch_size=patch_size,
                samples_per_epoch=samples_per_epoch,
                return_condition=False,
                augment_flip=True,
            )
        elif self.use_synthetic:
            self.log_msg.emit("使用合成数据进行快速训练 ...")
            vol_shape = (max(patch_size * 2, 128),) * 3
            dataset = Synthetic3DDataset(
                volume_shape=vol_shape,
                patch_size=patch_size,
                samples_per_epoch=samples_per_epoch,
                seed=42,
            )
        else:
            raise ValueError("无数据来源，请选择训练数据目录或开启快速模式。")

        train_size = int(len(dataset) * 0.9)
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_set, batch_size=batch_size,
            shuffle=True, num_workers=0,
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size,
            shuffle=False, num_workers=0,
        )

        # ---- 模型 ----
        net_config = VAENetConfig(
            dimension=3,
            in_channels=1,
            out_channels=1,
            z_dim=z_dim,
            ch=ch,
            ch_mult=list(ch_mult),
            num_res_blocks=num_res_blocks,
            resolution=patch_size,
            num_groups=min(8, ch // 2),
            use_attention=False,
            final_activation="sigmoid",
        )
        net = VAENet(net_config)

        vae_config = VAEModuleConfig(
            kl_weight=1e-5,
            kl_start=0.0,
            kl_max=1e-5,
            kl_warmup_steps=min(100, samples_per_epoch),
            reconstruction_loss="mse",
            lr=1e-4,
            scheduler_type="none",
            log_val_slices=False,
        )
        model = VAEModule(
            encdec=net,
            config=vae_config,
            conditional=False,
            verbose=True,
        )

        # ---- Checkpoint 回调 ----
        checkpoint_cb = ModelCheckpoint(
            dirpath=output_dir,
            filename="vae-fast-epoch{epoch:03d}-valloss{val/loss:.4f}",
            save_top_k=1,
            monitor="val/loss",
            mode="min",
            auto_insert_metric_name=False,
            save_last=True,
        )

        # ---- 进度回调 ----
        worker_ref = self  # 避免闭包引用问题

        class EpochProgressCB(Callback):
            def on_train_epoch_end(self, trainer, pl_module):
                epoch = trainer.current_epoch
                total = trainer.max_epochs
                # Lightning 在不同版本中 key 可能是 "train/loss" 或 "train/loss_epoch"
                loss = (
                    trainer.callback_metrics.get("train/loss_epoch")
                    or trainer.callback_metrics.get("train/loss")
                )
                loss_val = float(loss.item()) if loss is not None else 0.0
                pct = int((epoch + 1) / total * 100)
                worker_ref.progress.emit(pct)
                worker_ref.log_msg.emit(
                    f"Epoch {epoch + 1}/{total} — loss: {loss_val:.4f}"
                )

        callbacks = [checkpoint_cb, EpochProgressCB()]

        # ---- Trainer ----
        # 快速模式：禁用 logger 和内置进度条（避免 Windows 控制台编码问题），
        # 训练指标通过 UI 进度回调实时展示，checkpoint 文件名包含 loss。
        trainer = L.Trainer(
            accelerator=accelerator,
            devices=1,
            precision="32-true",
            max_epochs=self.epochs,
            callbacks=callbacks,
            logger=False,
            enable_progress_bar=False,
            log_every_n_steps=5,
            gradient_clip_val=1.0,
            num_sanity_val_steps=1,
        )

        self.log_msg.emit(
            f"开始训练: {self.epochs} epochs, "
            f"{samples_per_epoch} 样本/epoch, patch={patch_size}^3"
        )

        trainer.fit(model, train_loader, val_loader)

        # ---- 找到保存的 checkpoint ----
        best = checkpoint_cb.best_model_path
        last = getattr(checkpoint_cb, "last_model_path", "")
        ckpt = best if (best and os.path.exists(best)) else last

        if ckpt and os.path.exists(ckpt):
            return ckpt

        # 兜底：在 output_dir 中找任意 ckpt
        ckpts = sorted(
            [f for f in os.listdir(output_dir) if f.endswith(".ckpt")],
            key=lambda f: os.path.getmtime(os.path.join(output_dir, f)),
            reverse=True,
        )
        if ckpts:
            return os.path.join(output_dir, ckpts[0])

        return None


# ============================================================
# Stage1Page
# ============================================================

class Stage1Page(QWidget):

    def __init__(self):
        super().__init__()

        self.worker = None

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 标题
        title = QLabel("Stage1 3D-VAE：用于将切片数据转换为潜空间向量")
        title.setMaximumHeight(35)
        layout.addWidget(title)

        # 数据目录（可选，留空 = 合成数据）
        self.dataset_edit = QLineEdit()
        self.dataset_edit.setMaximumHeight(35)
        self.dataset_edit.setPlaceholderText("留空则自动使用合成数据")

        btn_dataset = QPushButton("选择训练数据目录（可选）")
        btn_dataset.setMaximumHeight(35)
        btn_dataset.clicked.connect(self.select_dataset_dir)

        layout.addWidget(self.dataset_edit)
        layout.addWidget(btn_dataset)

        # 输出目录（可选，留空 = 默认 checkpoints/）
        self.output_edit = QLineEdit()
        self.output_edit.setMaximumHeight(35)
        self.output_edit.setPlaceholderText("留空默认保存到项目 checkpoints/ 目录")

        btn_output = QPushButton("选择输出目录（可选）")
        btn_output.setMaximumHeight(35)
        btn_output.clicked.connect(self.select_output_dir)

        layout.addWidget(self.output_edit)
        layout.addWidget(btn_output)

        # epoch
        epoch_label = QLabel("训练 Epoch")
        epoch_label.setMaximumHeight(30)
        layout.addWidget(epoch_label)

        self.epoch_input = QSpinBox()
        self.epoch_input.setRange(1, 100000)
        self.epoch_input.setSingleStep(1)
        self.epoch_input.setValue(3)  # 快速模式默认 3
        self.epoch_input.setMaximumHeight(30)
        layout.addWidget(self.epoch_input)

        # 快速模式复选框
        self.fast_mode_cb = QCheckBox("快速模式（合成数据，约 1 分钟完成）")
        self.fast_mode_cb.setChecked(True)
        self.fast_mode_cb.setMaximumHeight(30)
        layout.addWidget(self.fast_mode_cb)

        # 开始按钮
        self.start_btn = QPushButton("开始训练")
        self.start_btn.setMaximumHeight(35)
        self.start_btn.clicked.connect(self.start_train)
        layout.addWidget(self.start_btn)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setMaximumHeight(35)
        layout.addWidget(self.progress)

        # 日志输出
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(120)
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # ========== Mock 展示区 ==========
        mock_container = QWidget()
        mock_layout = QVBoxLayout(mock_container)
        mock_layout.setContentsMargins(0, 10, 0, 0)
        mock_container.setFixedHeight(320)

        # 图片
        self.img_label = QLabel()
        img_path = os.path.join(BASE_DIR, "mock_data", "slice_0000.png")
        if os.path.exists(img_path):
            pixmap = QPixmap(img_path)
            self.img_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("border: 1px solid #ccc;")
        mock_layout.addWidget(QLabel("案例结果:"))
        mock_layout.addWidget(self.img_label)

        # 下载按钮
        btn_original = QPushButton("下载 original_volume.npy")
        btn_original.setMaximumHeight(35)
        btn_original.clicked.connect(
            lambda: self.download_file(
                os.path.join(BASE_DIR, "mock_data", "original_volume.npy"),
                "original_volume.npy"
            )
        )
        mock_layout.addWidget(btn_original)

        btn_recon = QPushButton("下载 recon_bin_volume.npy")
        btn_recon.setMaximumHeight(35)
        btn_recon.clicked.connect(
            lambda: self.download_file(
                os.path.join(BASE_DIR, "mock_data", "recon_bin_volume.npy"),
                "recon_bin_volume.npy"
            )
        )
        mock_layout.addWidget(btn_recon)

        layout.addWidget(mock_container)

        # 保留定时器用于兼容，但不再使用轮询
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_task)

    # ======================== UI 交互 ========================

    def download_file(self, src_path, filename):
        if not os.path.exists(src_path):
            QMessageBox.warning(self, "错误", f"文件不存在: {src_path}")
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "保存文件", filename)
        if save_path:
            shutil.copy(src_path, save_path)
            QMessageBox.information(self, "完成", f"已保存到: {save_path}")

    def select_dataset_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择训练数据目录")
        if path:
            self.dataset_edit.setText(path)

    def select_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_edit.setText(path)

    # ======================== 训练逻辑 ========================

    def start_train(self):
        """启动本地训练（替代原来的 API 调用）。"""
        dataset_dir = self.dataset_edit.text().strip()
        output_dir = self.output_edit.text().strip()
        epoch = self.epoch_input.value()
        use_synthetic = self.fast_mode_cb.isChecked()

        # 禁用按钮防止重复点击
        self.start_btn.setEnabled(False)
        self.progress.setValue(0)
        self.log_text.clear()

        # 创建并启动后台训练线程
        self.worker = TrainWorker(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            epochs=epoch,
            use_synthetic=use_synthetic,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.log_msg.connect(self._on_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, pct):
        self.progress.setValue(pct)

    def _on_log(self, msg):
        self.log_text.append(msg)
        # 自动滚动到底部
        sb = self.log_text.verticalScrollBar()
        if sb:
            sb.setValue(sb.maximum())

    def _on_finished(self, ckpt_path):
        self.start_btn.setEnabled(True)
        self.progress.setValue(100)
        self.log_text.append(f" 训练完成！")
        self.log_text.append(f"Checkpoint: {ckpt_path}")
        QMessageBox.information(
            self, "训练完成",
            f"Checkpoint 已保存到:\n{ckpt_path}"
        )

    def _on_failed(self, error_msg):
        self.start_btn.setEnabled(True)
        self.progress.setValue(0)
        self.log_text.append(f" {error_msg}")
        QMessageBox.warning(self, "训练失败", error_msg)

    def refresh_task(self):
        """保留兼容，不再使用 API 轮询。"""
        pass
