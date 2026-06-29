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

from backend.electrode_twin.latent_diffusion import (
    LatentDiffusionConfig,
    LatentDiffusionModule,
)
from backend.electrode_twin.latent_dataset import LatentConditionDataset
from backend.electrode_twin.CBD_generate import BASE_DIR


# ============================================================
# 合成 Latent 数据集 —— 无需真实 .npz 文件即可快速训练
# ============================================================

class SyntheticLatentDataset(Dataset):
    """
    生成随机潜向量和物理条件，模拟 VAE 编码器的输出。
    每条样本: z ∈ R^{C×D×H×W}, cond ∈ R^4。
    """

    CONDITION_KEYS = ["porosity", "surface_area", "tau_z", "deff_z"]

    def __init__(
        self,
        num_samples=100,
        latent_shape=(4, 16, 16, 16),
        seed=42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.latent_shape = latent_shape

        rng = np.random.RandomState(seed)

        # 预生成所有潜向量（模拟 VAE encoder 输出，标准正态分布）
        self.latents = [
            rng.randn(*latent_shape).astype(np.float32)
            for _ in range(num_samples)
        ]

        # 预生成物理条件
        conditions_raw = []
        for _ in range(num_samples):
            cond = np.array([
                rng.uniform(0.1, 0.9),    # porosity
                rng.uniform(0.005, 0.6),  # surface_area
                rng.uniform(1.0, 8.0),    # tau_z (曲折度)
                rng.uniform(0.005, 1.5),  # deff_z (有效扩散系数)
            ], dtype=np.float32)
            conditions_raw.append(cond)

        all_cond = np.stack(conditions_raw)
        self.cond_mins = all_cond.min(axis=0)
        self.cond_maxs = all_cond.max(axis=0)
        self.conditions_raw = conditions_raw

        print(f"[SyntheticLatentDataset] samples={num_samples}, "
              f"latent_shape={latent_shape}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        z = torch.from_numpy(self.latents[idx])
        cond_raw = self.conditions_raw[idx]

        # 归一化到 [0, 1]
        denom = np.maximum(self.cond_maxs - self.cond_mins, 1e-8)
        cond_norm = np.clip((cond_raw - self.cond_mins) / denom, 0.0, 1.0)
        cond_norm = cond_norm.astype(np.float32)

        return {
            "z": z,
            "cond": torch.from_numpy(cond_norm),
            "cond_raw": torch.from_numpy(cond_raw),
        }


# ============================================================
# 训练工作线程
# ============================================================

class TrainLDMWorker(QThread):
    """在后台线程中运行 Latent Diffusion 训练。"""

    progress = Signal(int)
    log_msg = Signal(str)
    finished = Signal(str)   # ckpt 路径
    failed = Signal(str)

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
        """执行训练，返回 checkpoint 路径。"""
        L.seed_everything(42)
        torch.set_float32_matmul_precision("high")

        # ---- 输出目录 ----
        output_dir = self.output_dir
        if not output_dir:
            output_dir = os.path.join(_PROJECT_ROOT, "ldm_checkpoints")
        os.makedirs(output_dir, exist_ok=True)

        # ---- 加速器 ----
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        self.log_msg.emit(f"加速器: {accelerator}")

        # ---- 快速训练参数 ----
        fast_latent_shape = (4, 16, 16, 16)
        num_synthetic_samples = 100
        batch_size = 4
        num_workers = 0

        # ---- 数据集 ----
        if self.dataset_dir and os.path.isdir(self.dataset_dir):
            summary_json = os.path.join(self.dataset_dir, "dataset_summary.json")
            if not os.path.exists(summary_json):
                raise ValueError(f"数据集目录缺少 dataset_summary.json: {self.dataset_dir}")
            self.log_msg.emit(f"加载真实 latent 数据: {self.dataset_dir}")
            dataset = LatentConditionDataset(
                latent_dir=self.dataset_dir,
                summary_json=summary_json,
                normalize_condition=True,
                clip_condition=True,
                skip_non_percolating=False,
            )
        elif self.use_synthetic:
            self.log_msg.emit("使用合成 latent 数据进行快速训练 ...")
            dataset = SyntheticLatentDataset(
                num_samples=num_synthetic_samples,
                latent_shape=fast_latent_shape,
                seed=42,
            )
        else:
            raise ValueError("无数据来源，请选择训练数据目录或开启快速模式。")

        train_size = int(len(dataset) * 0.9)
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_set, batch_size=batch_size,
            shuffle=True, num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size,
            shuffle=False, num_workers=num_workers,
        )

        # ---- 模型配置（快速版） ----
        latent_channels = dataset.latent_shape[0] if hasattr(dataset, 'latent_shape') else fast_latent_shape[0]
        latent_size = dataset.latent_shape[-1] if hasattr(dataset, 'latent_shape') else fast_latent_shape[-1]
        cond_dim = 4

        config = LatentDiffusionConfig(
            latent_channels=latent_channels,
            latent_size=latent_size,
            cond_dim=cond_dim,
            cond_embed_dim=64,          # 快速版: 64（原版 128）
            time_embed_dim=64,          # 快速版: 64（原版 128）
            model_channels=32,          # 快速版: 32（原版 64）
            channel_mult=(1, 2, 2),     # 快速版: 3 层（原版 (1, 2, 4)）
            dropout=0.0,
            num_timesteps=100,          # 快速版: 100（原版 1000）
            beta_start=1e-4,
            beta_end=2e-2,
            lr=1e-4,
            weight_decay=1e-4,
        )
        model = LatentDiffusionModule(config)

        # ---- Checkpoint 回调 ----
        checkpoint_cb = ModelCheckpoint(
            dirpath=output_dir,
            filename="ldm-fast-epoch{epoch:03d}-valloss{val/loss:.4f}",
            save_top_k=1,
            monitor="val/loss",
            mode="min",
            auto_insert_metric_name=False,
            save_last=True,
        )

        # ---- 进度回调 ----
        worker_ref = self

        class EpochProgressCB(Callback):
            def on_train_epoch_end(self, trainer, pl_module):
                epoch = trainer.current_epoch
                total = trainer.max_epochs
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
            f"{len(dataset)} 样本, latent={latent_channels}×{latent_size}^3"
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
# Stage2Page
# ============================================================

class Stage2Page(QWidget):

    def __init__(self):
        super().__init__()

        self.worker = None

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 标题
        title = QLabel("Stage2 LDM：将Stage1产生的数据用于扩散去噪生成学习")
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

        # 输出目录（可选，留空 = 默认 ldm_checkpoints/）
        self.output_edit = QLineEdit()
        self.output_edit.setMaximumHeight(35)
        self.output_edit.setPlaceholderText("留空默认保存到项目 ldm_checkpoints/ 目录")

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

        mock_layout.addWidget(QLabel("案例结果:"))

        # 下载按钮
        btn_download = QPushButton("下载 dataset_summary.json")
        btn_download.setMaximumHeight(35)
        btn_download.clicked.connect(
            lambda: self.download_file(
                os.path.join(BASE_DIR, "mock_data", "dataset_summary.json"),
                "dataset_summary.json"
            )
        )
        mock_layout.addWidget(btn_download)

        layout.addWidget(mock_container)

        # 定时器（保留兼容但不再使用）
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
        self.worker = TrainLDMWorker(
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
