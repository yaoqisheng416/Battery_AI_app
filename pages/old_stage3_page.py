# -*- coding: utf-8 -*-
import os

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


class Stage7Page(QWidget):

    def __init__(self):
        super().__init__()

        self.task_id = None

        layout = QVBoxLayout(self)

        # 标题
        title = QLabel("Stage3 模型训练")

        layout.addWidget(title)

        # 数据目录
        self.dataset_edit = QLineEdit()

        btn_dataset = QPushButton(
            "选择训练数据目录"
        )

        btn_dataset.clicked.connect(
            self.select_dataset_dir
        )

        layout.addWidget(self.dataset_edit)

        layout.addWidget(btn_dataset)

        # 输出目录
        self.output_edit = QLineEdit()

        btn_output = QPushButton(
            "选择输出目录"
        )

        btn_output.clicked.connect(
            self.select_output_dir
        )

        layout.addWidget(self.output_edit)

        layout.addWidget(btn_output)

        # epoch
        self.epoch_input = QSpinBox()

        self.epoch_input.setValue(100)

        self.epoch_input.setMaximum(100000)

        layout.addWidget(
            QLabel("训练 Epoch")
        )

        layout.addWidget(
            self.epoch_input
        )

        # 开始按钮
        self.start_btn = QPushButton(
            "开始训练"
        )

        self.start_btn.clicked.connect(
            self.start_train
        )

        layout.addWidget(
            self.start_btn
        )

        # 进度条
        self.progress = QProgressBar()

        layout.addWidget(
            self.progress
        )

        # 日志
        self.log_text = QTextEdit()

        self.log_text.setReadOnly(True)

        layout.addWidget(
            self.log_text
        )

        # 定时器轮询
        self.timer = QTimer()

        self.timer.timeout.connect(
            self.refresh_task
        )

    def select_dataset_dir(self):

        path = QFileDialog.getExistingDirectory(
            self,
            "选择训练数据目录"
        )

        if path:
            self.dataset_edit.setText(path)

    def select_output_dir(self):

        path = QFileDialog.getExistingDirectory(
            self,
            "选择输出目录"
        )

        if path:
            self.output_edit.setText(path)

    def start_train(self):

        dataset_dir = self.dataset_edit.text()

        output_dir = self.output_edit.text()

        epoch = self.epoch_input.value()

        if not os.path.exists(dataset_dir):

            QMessageBox.warning(
                self,
                "错误",
                "训练目录不存在"
            )

            return

        payload = {
            "dataset_dir": dataset_dir,
            "output_dir": output_dir,
            "epoch": epoch,
        }

        result = create_task(
            "/stage1/create-task",
            payload,
        )

        if not result.get("success", True):

            QMessageBox.warning(
                self,
                "错误",
                result.get("message")
            )

            return

        self.task_id = result["task_id"]

        self.log_text.append(
            f"任务创建成功: {self.task_id}"
        )

        # 开始轮询
        self.timer.start(2000)

    def refresh_task(self):

        if not self.task_id:
            return

        task = query_task(self.task_id)

        status = task.get("status")

        progress = task.get("progress", 0)

        logs = task.get("logs", [])

        self.progress.setValue(
            int(progress)
        )

        self.log_text.setPlainText(
            "\n".join(logs[-200:])
        )

        if status == "finished":

            self.timer.stop()

            QMessageBox.information(
                self,
                "完成",
                "任务执行完成"
            )

        elif status == "failed":

            self.timer.stop()

            QMessageBox.warning(
                self,
                "失败",
                task.get("error", "")
            )