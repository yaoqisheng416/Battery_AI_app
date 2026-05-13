# -*- coding: utf-8 -*-

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QMessageBox,
    QComboBox,
    QDoubleSpinBox, QFileDialog, QGroupBox, QLineEdit,
)

from api_client import (
    create_task,
    query_task,
    get_model_versions,
)


class Stage4Page(QWidget):

    def __init__(self, main_window):  # ✅ 接收 MainWindow 引用
        super().__init__()
        self.main_window = main_window  # ✅ 保存引用
        self.task_id = None

        root_layout = QVBoxLayout(self)
        root_layout.setSpacing(15)

        # ============================================
        # 顶部提示
        # ============================================
        tip_label = QLabel(
            "💡 选择模型版本和条件参数，设置输出目录后点击「开始 Stage4 生成，任务将提交到「任务中心」,可前往进行查看状态\n"
            "点解'恢复默认参数'按钮再次输入参数 进行推理。"
        )
        tip_label.setWordWrap(True)
        tip_label.setAlignment(Qt.AlignCenter)
        tip_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #4f8cff;
                padding: 15px;
                background: #25262b;
                border-radius: 10px;
                border: 2px solid #4f8cff;
            }
        """)
        root_layout.addWidget(tip_label)

        # ============================================
        # 模型版本选择（水平排列）
        # ============================================
        model_group = QGroupBox("模型版本选择")
        model_layout = QHBoxLayout(model_group)
        model_layout.setSpacing(15)

        title1 = QLabel("模型版本:")
        self.version_combo = QComboBox()
        self.ldm_combo = QComboBox()
        self.vae_combo = QComboBox()

        model_layout.addWidget(title1)
        model_layout.addWidget(self.version_combo)
        model_layout.addWidget(self.ldm_combo)
        model_layout.addWidget(self.vae_combo)
        model_layout.addStretch()

        root_layout.addWidget(model_group)

        # ============================================
        # 条件参数（水平排列）
        # ============================================
        param_group = QGroupBox("条件参数")
        param_layout = QHBoxLayout(param_group)
        param_layout.setSpacing(15)

        title2 = QLabel("参数:")
        self.porosity_input = QDoubleSpinBox()
        self.porosity_input.setValue(0.2)
        self.porosity_input.setDecimals(4)
        self.porosity_input.setPrefix("Porosity: ")
        self.porosity_input.setMinimumWidth(150)

        self.tau_input = QDoubleSpinBox()
        self.tau_input.setValue(7)
        self.tau_input.setPrefix("Tau Z: ")
        self.tau_input.setMinimumWidth(150)

        self.surface_input = QDoubleSpinBox()
        self.surface_input.setRange(0, 999999)
        self.surface_input.setDecimals(2)
        self.surface_input.setSingleStep(10)
        self.surface_input.setValue(1150)
        self.surface_input.setPrefix("Surface Area: ")
        self.surface_input.setMinimumWidth(150)

        param_layout.addWidget(title2)
        param_layout.addWidget(self.porosity_input)
        param_layout.addWidget(self.tau_input)
        param_layout.addWidget(self.surface_input)
        param_layout.addStretch()

        root_layout.addWidget(param_group)

        # ============================================
        # 输出目录选择（✅ 新增）
        # ============================================
        # output_group = QGroupBox("输出目录")
        # output_layout = QHBoxLayout(output_group)
        # output_layout.setSpacing(15)
        #
        # self.out_dir_edit = QLineEdit()
        # self.out_dir_edit.setPlaceholderText("选择输出目录...")
        # self.out_dir_edit.setMinimumWidth(400)
        #
        # btn_out_dir = QPushButton("选择输出目录")
        # btn_out_dir.clicked.connect(self.select_out_dir)
        #
        # output_layout.addWidget(QLabel("输出目录:"))
        # output_layout.addWidget(self.out_dir_edit)
        # output_layout.addWidget(btn_out_dir)
        # output_layout.addStretch()
        #
        # root_layout.addWidget(output_group)

        # ============================================
        # 按钮区域（✅ 修改）
        # ============================================
        btn_layout = QHBoxLayout()

        btn_start = QPushButton("开始 Stage4 生成")
        btn_start.setMinimumHeight(50)
        btn_start.clicked.connect(self.start_generate)
        btn_start.setStyleSheet("""
            QPushButton {
                background: #4f8cff;
                color: white;
                border-radius: 10px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background: #6aa1ff;
            }
        """)

        btn_reset = QPushButton("恢复默认参数")
        btn_reset.setMinimumHeight(50)
        btn_reset.clicked.connect(self.reset_params)
        btn_reset.setStyleSheet("""
            QPushButton {
                background: #666;
                color: white;
                border-radius: 10px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background: #888;
            }
        """)

        btn_layout.addWidget(btn_start)
        btn_layout.addWidget(btn_reset)

        root_layout.addLayout(btn_layout)
        root_layout.addStretch()

        # ============================================
        # 加载版本列表
        # ============================================
        self.load_versions()

    # ============================================
    # 选择输出目录
    # ============================================
    def select_out_dir(self):
        """选择输出目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if directory:
            self.out_dir_edit.setText(directory)

    # ============================================
    # 恢复默认参数
    # ============================================
    def reset_params(self):
        """恢复默认参数"""
        self.porosity_input.setValue(0.2)
        self.tau_input.setValue(7)
        self.surface_input.setValue(1150)
        QMessageBox.information(self, "已恢复", "所有参数已恢复为默认值！")

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

    # ============================================
    # 开始生成（：校验 + 提交 + 弹窗 + 跳转）
    # ============================================
    def start_generate(self):
        # ============================================
        # 1. 校验输出目录（必须先做！）
        # ============================================
        # if not self.out_dir_edit.text().strip():
        #     QMessageBox.warning(
        #         self,
        #         "错误",
        #         "请先选择输出目录"
        #     )
        #     return

        # ============================================
        # 2. 提交任务到后台
        # ============================================
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

        # ============================================
        # 3. 弹窗提示 + 跳转到任务中心
        # ============================================
        msg = QMessageBox(self)
        msg.setWindowTitle("任务已提交")
        msg.setText(" Stage4 生成任务已提交！")
        msg.setInformativeText(
            "任务正在后台运行，请前往「历史任务中心」查看进度和结果。"
        )
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Ok)

        go_btn = msg.addButton("前往任务中心", QMessageBox.ActionRole)
        ret = msg.exec_()

        if ret == QMessageBox.Ok or msg.clickedButton() == go_btn:
            self.main_window.menu.setCurrentRow(5)  #  跳转到历史任务中心（索引 5）

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