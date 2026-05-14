# -*- coding: utf-8 -*-
import requests
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QMessageBox,
    QComboBox,
    QDoubleSpinBox, QFileDialog, QGroupBox, QLineEdit, QSpinBox,
)

from api_client import (
    create_task,
    query_task,
     API_BASE,
)


class Stage4Page(QWidget):

    def __init__(self, main_window):  # 接收 MainWindow 引用
        super().__init__()
        self.main_window = main_window  # 保存引用
        self.task_id = None

        #  保存用户选择的模型路径
        self.selected_vae_path = None
        self.selected_ldm_path = None

        # ⭐ 默认参数（用于恢复默认）
        self.default_device = "cuda"
        self.default_num_samples = 32
        self.default_porosity = 0.2
        self.default_tau = 7
        self.default_surface = 1150

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
        # 模型选择区域（ 重构）
        # ============================================
        model_group = QGroupBox("模型选择")
        model_layout = QHBoxLayout(model_group)
        model_layout.setSpacing(15)

        #  VAE 模型选择
        vae_label = QLabel("VAE 模型:")
        self.vae_combo = QComboBox()
        self.vae_combo.setMinimumWidth(250)
        self.vae_combo.currentIndexChanged.connect(self.on_vae_changed)

        #  LDM 模型选择
        ldm_label = QLabel("LDM 模型:")
        self.ldm_combo = QComboBox()
        self.ldm_combo.setMinimumWidth(250)
        self.ldm_combo.currentIndexChanged.connect(self.on_ldm_changed)

        #  显示选中的路径
        self.path_label = QLabel("未选择模型")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("""
                    QLabel {
                        font-size: 11px;
                        color: #888;
                        padding: 5px;
                        background: #1e1f24;
                        border-radius: 5px;
                    }
                """)

        model_layout.addWidget(vae_label)
        model_layout.addWidget(self.vae_combo)
        model_layout.addWidget(ldm_label)
        model_layout.addWidget(self.ldm_combo)
        model_layout.addWidget(self.path_label)
        model_layout.addStretch()

        root_layout.addWidget(model_group)

        # ============================================
        # ⭐ 设备和采样数选择（新增）
        # ============================================
        device_group = QGroupBox("设备与采样")
        device_layout = QHBoxLayout(device_group)
        device_layout.setSpacing(15)

        # DEVICE 选择
        device_label = QLabel("设备 (DEVICE):")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        self.device_combo.setCurrentText("cuda")  # ⭐ 默认 cuda
        self.device_combo.setMinimumWidth(150)
        self.device_combo.currentIndexChanged.connect(self.on_device_changed)

        # NUM_SAMPLES 选择
        samples_label = QLabel("采样数 (NUM_SAMPLES):")
        self.samples_spin = QSpinBox()  # ⭐ 使用 QSpinBox
        self.samples_spin.setRange(8, 512)  # ⭐ 范围：8 ~ 512
        self.samples_spin.setSingleStep(8)  # ⭐ 每次增减 8
        self.samples_spin.setValue(32)  # ⭐ 默认 32
        self.samples_spin.setMinimumWidth(150)
        self.samples_spin.valueChanged.connect(self.on_samples_changed)

        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo)
        device_layout.addWidget(samples_label)
        device_layout.addWidget(self.samples_spin)
        device_layout.addStretch()

        root_layout.addWidget(device_group)

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
        # 输出目录选择（ 新增）
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
        # 按钮区域（ 修改）
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
        #  加载模型列表
        # ============================================
        self.load_models()

    # ============================================
    # 选择输出目录
    # ============================================
    def select_out_dir(self):
        """选择输出目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if directory:
            self.out_dir_edit.setText(directory)

    # ============================================
    #  恢复默认参数（ 恢复所有参数，包括 device 和 num_samples）
    # ============================================
    def reset_params(self):
        """恢复所有默认参数"""
        #  恢复条件参数
        self.porosity_input.setValue(self.default_porosity)
        self.tau_input.setValue(self.default_tau)
        self.surface_input.setValue(self.default_surface)

        #  恢复设备（默认 cuda）
        self.device_combo.setCurrentText(self.default_device)

        #  恢复采样数（默认 32）
        self.samples_spin.setValue(self.default_num_samples)

        #  恢复模型为最新（已经在 load_models 中默认选中）
        # 如果需要强制刷新，可以重新调用 load_models()
        self.load_models()

        print(" 所有参数已恢复默认")
        self.main_window.show_message("提示", "所有参数已恢复默认！")

    # ============================================
    #  加载模型（适配新接口）
    # ============================================
    def load_models(self):
        """从接口加载模型列表"""
        try:
            response = requests.get(f"{API_BASE}/models/versions")
            data = response.json()

            #  新接口返回格式：{"vae_models": [...], "ldm_models": [...]}
            vae_models = data.get("vae_models", [])
            ldm_models = data.get("ldm_models", [])

            print(f"[OK] 加载到 {len(vae_models)} 个VAE模型, {len(ldm_models)} 个LDM模型")

            self.model_data = {
                "vae_models": vae_models,
                "ldm_models": ldm_models,
            }

            #  填充 VAE 下拉框（显示文件名 + 时间）
            self.vae_combo.clear()
            for model in vae_models:
                display_text = f"{model['file_name']} ({model['create_time']})"
                self.vae_combo.addItem(display_text, model["full_path"])  # 保存完整路径

            #  填充 LDM 下拉框
            self.ldm_combo.clear()
            for model in ldm_models:
                display_text = f"{model['file_name']} ({model['create_time']})"
                self.ldm_combo.addItem(display_text, model["full_path"])  # 保存完整路径

            #  默认选中最新的模型
            if vae_models:
                self.vae_combo.setCurrentIndex(0)
            if ldm_models:
                self.ldm_combo.setCurrentIndex(0)

            self.update_path_label()

        except Exception as e:
            print(f"[错误] 加载模型失败: {e}")
            self.vae_combo.addItem("无可用模型", "")
            self.ldm_combo.addItem("无可用模型", "")

    # ============================================
    #  DEVICE 选择变化（新增）
    # ============================================
    def on_device_changed(self):
        """设备选择变化时的处理（可选）"""
        print(f"[DEVICE] 当前选择: {self.device_combo.currentText()}")

    # ============================================
    #  NUM_SAMPLES 变化（新增）
    # ============================================
    def on_samples_changed(self):
        """采样数变化时的处理（可选）"""
        print(f"[NUM_SAMPLES] 当前值: {self.samples_spin.value()}")

    # ============================================
    #  VAE 选择变化
    # ============================================
    def on_vae_changed(self):
        """VAE模型选择变化时更新路径显示"""
        idx = self.vae_combo.currentIndex()
        if idx >= 0:
            self.selected_vae_path = self.vae_combo.currentData()  # 获取完整路径
        self.update_path_label()

    # ============================================
    #  LDM 选择变化
    # ============================================
    def on_ldm_changed(self):
        """LDM模型选择变化时更新路径显示"""
        idx = self.ldm_combo.currentIndex()
        if idx >= 0:
            self.selected_ldm_path = self.ldm_combo.currentData()  # 获取完整路径
        self.update_path_label()

    # ============================================
    #  更新路径显示
    # ============================================
    def update_path_label(self):
        """在界面上显示当前选中的模型路径"""
        vae_name = self.vae_combo.currentText().split(" (")[0] if self.vae_combo.currentText() else "未选择"
        ldm_name = self.ldm_combo.currentText().split(" (")[0] if self.ldm_combo.currentText() else "未选择"

        self.path_label.setText(
            f"VAE: {vae_name}\n"
            f"LDM: {ldm_name}"
        )

    # ============================================
    #  获取选中的模型路径（ 关键方法）
    # ============================================
    def get_selected_models(self):
        """
         获取用户选择的模型完整路径
        返回: {
            "vae_path": "完整路径/xxx.ckpt",
            "ldm_path": "完整路径/xxx.ckpt",
            "vae_name": "文件名.ckpt",
            "ldm_name": "文件名.ckpt"
            "设备"
            "采样数"
        }
        """
        return {
            "vae_path": self.selected_vae_path,
            "ldm_path": self.selected_ldm_path,
            "vae_name": self.vae_combo.currentText().split(" (")[0] if self.vae_combo.currentText() else None,
            "ldm_name": self.ldm_combo.currentText().split(" (")[0] if self.ldm_combo.currentText() else None,
            "device": self.device_combo.currentText(),
            "num_samples": self.samples_spin.value(),
        }

    # ============================================
    # 开始生成（：校验 + 提交 + 弹窗 + 跳转）
    # ============================================
    def start_generate(self):
        models = self.get_selected_models()

        #  1.检查是否选择了模型
        if not models["vae_path"] or not models["ldm_path"]:
            self.main_window.show_message("错误", "请先选择VAE和LDM模型！")
            return

        # ============================================
        # 2. 提交任务到后台（ 新增模型路径）
        # ============================================
        payload = {
            "porosity": self.porosity_input.value(),
            "tau_z": self.tau_input.value(),
            "surface_area": self.surface_input.value(),
            #  新增：模型路径
            "vae_path": models["vae_path"],
            "ldm_path": models["ldm_path"],
            # 新增：设备和采样数
            "device": models["device"],
            "num_samples": models["num_samples"],
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
        msg.setText("Stage4 生成任务已提交！")
        msg.setInformativeText(
            "任务正在后台运行，请前往「历史任务中心」查看进度和结果。"
        )
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Ok)

        go_btn = msg.addButton("前往任务中心", QMessageBox.ActionRole)
        ret = msg.exec_()

        if ret == QMessageBox.Ok or msg.clickedButton() == go_btn:
            self.main_window.menu.setCurrentRow(5)

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