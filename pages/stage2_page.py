# -*- coding: utf-8 -*-
import os
import shutil

from PySide6.QtGui import Qt, QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QTextEdit,
    QSpinBox,
    QProgressBar,
    QMessageBox,
)

from PySide6.QtCore import (
    QTimer,
)

from api_client import (
    create_task,
    query_task,
)
from backend.electrode_twin.CBD_generate import BASE_DIR


class Stage2Page(QWidget):

    def __init__(self):
        super().__init__()

        self.task_id = None

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 标题
        title = QLabel("Stage2 LDM：将Stage1产生的数据用于扩散去噪生成学习")
        title.setMaximumHeight(35)
        layout.addWidget(title)

        # 数据目录
        self.dataset_edit = QLineEdit()
        self.dataset_edit.setMaximumHeight(35)

        btn_dataset = QPushButton("选择训练数据目录")
        btn_dataset.setMaximumHeight(35)
        btn_dataset.clicked.connect(self.select_dataset_dir)

        layout.addWidget(self.dataset_edit)
        layout.addWidget(btn_dataset)

        # 输出目录
        self.output_edit = QLineEdit()
        self.output_edit.setMaximumHeight(35)

        btn_output = QPushButton("选择输出目录")
        btn_output.setMaximumHeight(35)
        btn_output.clicked.connect(self.select_output_dir)

        layout.addWidget(self.output_edit)
        layout.addWidget(btn_output)

        # epoch
        self.epoch_input = QSpinBox()
        self.epoch_input.setValue(100)
        self.epoch_input.setMaximum(100000)
        self.epoch_input.setMaximumHeight(35)

        epoch_label = QLabel("训练 Epoch")
        epoch_label.setMaximumHeight(35)
        layout.addWidget(epoch_label)

        layout.addWidget(self.epoch_input)

        # 开始按钮
        self.start_btn = QPushButton("开始训练")
        self.start_btn.setMaximumHeight(35)
        self.start_btn.clicked.connect(self.start_train)

        layout.addWidget(self.start_btn)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setMaximumHeight(35)

        layout.addWidget(self.progress)

        # ========== Mock 展示区 ==========
        mock_container = QWidget()
        mock_layout = QVBoxLayout(mock_container)
        mock_layout.setContentsMargins(0, 10, 0, 0)
        mock_container.setFixedHeight(320)

        mock_layout.addWidget(QLabel("案例结果:"))

        # 下载按钮（仅一个文件）
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

        # 定时器轮询
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_task)

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

    def start_train(self):
        dataset_dir = self.dataset_edit.text()
        output_dir = self.output_edit.text()
        epoch = self.epoch_input.value()

        if not os.path.exists(dataset_dir):
            QMessageBox.warning(self, "错误", "训练目录不存在")
            return

        payload = {
            "dataset_dir": dataset_dir,
            "output_dir": output_dir,
            "epoch": epoch,
        }

        result = create_task("/stage2/create-task", payload)

        if not result.get("success", True):
            QMessageBox.warning(self, "错误", result.get("message"))
            return

        self.task_id = result["task_id"]
        self.log_text.append(f"任务创建成功: {self.task_id}")
        self.timer.start(2000)

    def refresh_task(self):
        if not self.task_id:
            return

        task = query_task(self.task_id)
        status = task.get("status")
        progress = task.get("progress", 0)
        logs = task.get("logs", [])

        self.progress.setValue(int(progress))
        self.log_text.setPlainText("\n".join(logs[-200:]))

        if status == "finished":
            self.timer.stop()
            QMessageBox.information(self, "完成", "任务执行完成")
        elif status == "failed":
            self.timer.stop()
            QMessageBox.warning(self, "失败", task.get("error", ""))
