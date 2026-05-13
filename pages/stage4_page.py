# -*- coding: utf-8 -*-

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QProgressBar,
    QMessageBox,
    QFrame,
    QComboBox,
    QDoubleSpinBox,
)

from api_client import (
    create_task,
    query_task,
    get_model_versions,
)


class Stage4Page(QWidget):

    def __init__(self):
        super().__init__()

        self.task_id = None

        root_layout = QVBoxLayout(self)

        # ============================================
        # 模型区域
        # ============================================
        model_frame = QFrame()

        model_layout = QVBoxLayout(model_frame)

        title1 = QLabel("模型版本选择")

        model_layout.addWidget(title1)

        row1 = QHBoxLayout()

        self.version_combo = QComboBox()

        self.ldm_combo = QComboBox()

        self.vae_combo = QComboBox()

        row1.addWidget(self.version_combo)

        row1.addWidget(self.ldm_combo)

        row1.addWidget(self.vae_combo)

        model_layout.addLayout(row1)

        root_layout.addWidget(model_frame)

        # ============================================
        # 参数区域
        # ============================================
        param_frame = QFrame()

        param_layout = QVBoxLayout(param_frame)

        title2 = QLabel("条件参数")

        param_layout.addWidget(title2)

        row2 = QHBoxLayout()

        self.porosity_input = QDoubleSpinBox()

        self.porosity_input.setValue(0.2)

        self.porosity_input.setDecimals(4)

        self.porosity_input.setPrefix(
            "Porosity: "
        )

        self.tau_input = QDoubleSpinBox()

        self.tau_input.setValue(7)

        self.tau_input.setPrefix(
            "Tau Z: "
        )

        self.surface_input = QDoubleSpinBox()

        # ✅ 关键设置（按顺序）
        self.surface_input.setRange(0, 999999)  # 范围 0~999999
        self.surface_input.setDecimals(2)  # 2 位小数
        self.surface_input.setSingleStep(10)  # 步长 10
        self.surface_input.setValue(1150)  # 设置值 1150
        self.surface_input.setPrefix("Surface Area: ")  # 前缀

        # ✅ 强制刷新
        self.surface_input.update()

        # 添加到布局
        row2.addWidget(self.porosity_input)
        row2.addWidget(self.tau_input)
        row2.addWidget(self.surface_input)

        param_layout.addLayout(row2)

        root_layout.addWidget(param_frame)

        # ============================================
        # 启动按钮
        # ============================================
        self.start_btn = QPushButton(
            "开始 Stage4 生成"
        )

        self.start_btn.setMinimumHeight(50)

        self.start_btn.clicked.connect(
            self.start_generate
        )

        root_layout.addWidget(self.start_btn)

        # ============================================
        # 进度条
        # ============================================
        self.progress = QProgressBar()

        root_layout.addWidget(self.progress)

        # ============================================
        # 日志区域
        # ============================================
        self.log_text = QTextEdit()

        self.log_text.setReadOnly(True)

        root_layout.addWidget(self.log_text)

        # ============================================
        # 定时器
        # ============================================
        self.timer = QTimer()

        self.timer.timeout.connect(
            self.refresh_task
        )

        self.load_versions()

    def load_versions(self):

        versions = get_model_versions()

        self.version_data = versions

        self.version_combo.clear()

        for v in versions:

            self.version_combo.addItem(
                f"{v['version']} | "
                f"{v['create_time']}"
            )

        self.version_combo.currentIndexChanged.connect(
            self.update_models
        )

        self.update_models()

    def update_models(self):

        idx = self.version_combo.currentIndex()

        if idx < 0:
            return

        version = self.version_data[idx]

        self.ldm_combo.clear()

        self.vae_combo.clear()

        for m in version["ldm_models"]:
            self.ldm_combo.addItem(
                m["file_name"]
            )

        for m in version["vae_models"]:
            self.vae_combo.addItem(
                m["file_name"]
            )

    def start_generate(self):

        payload = {
            "porosity": self.porosity_input.value(),
            "tau_z": self.tau_input.value(),
            "surface_area": self.surface_input.value(),
        }

        result = create_task(
            "/stage4/generate_structure_from_condition",
            payload,
        )

        if "task_id" not in result:

            QMessageBox.warning(
                self,
                "错误",
                str(result)
            )

            return

        self.task_id = result["task_id"]

        self.log_text.append(
            f"任务创建成功: {self.task_id}"
        )

        self.timer.start(2000)

    def refresh_task(self):

        if not self.task_id:
            return

        task = query_task(
            self.task_id
        )

        status = task.get("status")

        progress = task.get(
            "progress",
            0,
        )

        logs = task.get(
            "logs",
            [],
        )

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
                "Stage4 生成完成"
            )

        elif status == "failed":

            self.timer.stop()

            QMessageBox.warning(
                self,
                "失败",
                task.get("error", "")
            )