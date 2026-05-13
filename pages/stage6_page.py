# -*- coding: utf-8 -*-
import os
import requests

from PySide6.QtCore import Qt

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QMessageBox,
    QFileDialog,
    QTabWidget,
    QGroupBox,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QLineEdit,
)

from api_client import (
    create_task,
    query_task,
    API_BASE,
)


class Stage6Page(QWidget):

    def __init__(self, parent_window):  # 接收 MainWindow
        super().__init__()
        self.main_window = parent_window  # 保存引用

        self.task_id = None
        self.fit_task_id = None

        self.upload_task_id = None
        self.input_file = None

        self.init_ui()

    # ========================================================
    # UI
    # ========================================================
    def init_ui(self):

        root_layout = QVBoxLayout(self)

        title = QLabel(
            "Stage6 CBD三相电极结构生成与参数拟合"
        )

        title.setStyleSheet("""
        font-size:24px;
        font-weight:bold;
        color:white;
        padding:10px;
        """)

        root_layout.addWidget(title)

        self.tabs = QTabWidget()

        # ✅ 设置 Tab 标题样式
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #444;
                background: #2b2d31;
                border-radius: 8px;
            }

            QTabBar::tab {
                background: #25262b;
                color: white;
                padding: 12px 24px;        /* ✅ 内边距，让文字更饱满 */
                font-size: 14px;           /* ✅ Tab 标题字体大小（从 10 → 14） */
                font-weight: bold;         /* ✅ 加粗 */
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 5px;         /* ✅ Tab 之间的间距 */
            }

            QTabBar::tab:selected {
                background: #4f8cff;       /* ✅ 选中时蓝色背景 */
                color: white;
            }

            QTabBar::tab:hover {
                background: #3a3b3f;       /* ✅ 悬停时灰色背景 */
            }
        """)

        root_layout.addWidget(self.tabs)

        # ====================================================
        # tab1
        # ====================================================
        self.tab_generate = QWidget()

        self.tabs.addTab(
            self.tab_generate,
            "CBD三相结构生成"
        )

        self.build_generate_tab()

        # ====================================================
        # tab2
        # ====================================================
        self.tab_fit = QWidget()

        self.tabs.addTab(
            self.tab_fit,
            "CBD参数拟合"
        )

        self.build_fit_tab()

    # ========================================================
    # TAB1
    # ========================================================
    def build_generate_tab(self):
        layout = QVBoxLayout(self.tab_generate)  # ✅ 改为垂直布局
        layout.setSpacing(15)

        # 顶部提示
        tip_label = QLabel(
            "💡 选择.npy文件上传后, 进行参数设置, 点击「开始CBD三相生成」后，任务将提交到「任务中心」,可前往进行查看状态\n"
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
        layout.addWidget(tip_label)

        # ====================================================
        # 文件上传
        # ====================================================
        file_group = QGroupBox("二相结构文件")
        file_layout = QVBoxLayout(file_group)

        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("请选择 .npy 文件...")
        self.file_edit.setMinimumWidth(400)

        btn_select_file = QPushButton("选择 .npy 文件")
        btn_select_file.clicked.connect(self.select_npy_file)

        btn_upload_file = QPushButton("上传文件")  # 新增上传按钮
        btn_upload_file.clicked.connect(self.upload_file)  # 连接到 upload_file 函数
        btn_upload_file.setMinimumHeight(40)
        btn_upload_file.setStyleSheet("""
            QPushButton {
                background: #4f8cff;
                color: white;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #6aa1ff;
            }
        """)

        file_layout.addWidget(self.file_edit)
        file_layout.addWidget(btn_select_file)
        file_layout.addWidget(btn_upload_file)  # 添加上传按钮到布局

        layout.addWidget(file_group)

        # ====================================================
        # CBD参数（水平排列 + 强制范围）
        # ====================================================
        cbd_group = QGroupBox("CBD参数")
        cbd_layout = QHBoxLayout(cbd_group)
        cbd_layout.setSpacing(15)

        self.target_cbd = QDoubleSpinBox()
        self.target_cbd.setRange(0, 999999)  # 强制范围
        self.target_cbd.setSingleStep(0.001)  # ✅ 每次增减 0.001
        self.target_cbd.setDecimals(4)
        self.target_cbd.setValue(0.05)
        self.target_cbd.setMinimumWidth(150)

        self.w_um = QDoubleSpinBox()
        self.w_um.setRange(0, 999999)  # 强制范围
        self.w_um.setSingleStep(0.001)  # 每次增减 0.001
        self.w_um.setDecimals(4)
        self.w_um.setValue(0.08)
        self.w_um.setMinimumWidth(150)

        cbd_layout.addWidget(QLabel("Target CBD:"))
        cbd_layout.addWidget(self.target_cbd)
        cbd_layout.addStretch()
        cbd_layout.addWidget(QLabel("W Parameter:"))
        cbd_layout.addWidget(self.w_um)

        layout.addWidget(cbd_group)

        # ====================================================
        # 相标签（水平排列 + 强制范围）
        # ====================================================
        phase_group = QGroupBox("相标签")
        phase_layout = QHBoxLayout(phase_group)
        phase_layout.setSpacing(15)

        self.pore_value = QSpinBox()
        self.pore_value.setRange(0, 999999)  # 强制范围
        self.pore_value.setSingleStep(1)  # 每次增减 1
        self.pore_value.setMinimumWidth(100)

        self.am_value = QSpinBox()
        self.am_value.setRange(0, 999999)  # 强制范围
        self.am_value.setSingleStep(1)
        self.am_value.setValue(1)
        self.am_value.setMinimumWidth(100)

        self.cbd_value = QSpinBox()
        self.cbd_value.setRange(0, 999999)  # 强制范围
        self.cbd_value.setSingleStep(1)
        self.cbd_value.setValue(2)
        self.cbd_value.setMinimumWidth(100)

        phase_layout.addWidget(QLabel("Pore:"))
        phase_layout.addWidget(self.pore_value)
        phase_layout.addWidget(QLabel("AM:"))
        phase_layout.addWidget(self.am_value)
        phase_layout.addWidget(QLabel("CBD:"))
        phase_layout.addWidget(self.cbd_value)
        phase_layout.addStretch()

        layout.addWidget(phase_group)

        # ====================================================
        # Voxel Size（水平排列 + 强制范围）
        # ====================================================
        voxel_group = QGroupBox("Voxel Size")
        voxel_layout = QHBoxLayout(voxel_group)
        voxel_layout.setSpacing(15)

        self.voxel_x = QDoubleSpinBox()
        self.voxel_x.setRange(0, 999999)  # ✅ 强制范围
        self.voxel_x.setSingleStep(0.0001)  # ✅ 每次增减 0.0001
        self.voxel_x.setDecimals(5)
        self.voxel_x.setValue(0.02791)
        self.voxel_x.setMinimumWidth(120)

        self.voxel_y = QDoubleSpinBox()
        self.voxel_y.setRange(0, 999999)
        self.voxel_y.setSingleStep(0.0001)
        self.voxel_y.setDecimals(5)
        self.voxel_y.setValue(0.0315)
        self.voxel_y.setMinimumWidth(120)

        self.voxel_z = QDoubleSpinBox()
        self.voxel_z.setRange(0, 999999)
        self.voxel_z.setSingleStep(0.0001)
        self.voxel_z.setDecimals(5)
        self.voxel_z.setValue(0.02791)
        self.voxel_z.setMinimumWidth(120)

        voxel_layout.addWidget(QLabel("X:"))
        voxel_layout.addWidget(self.voxel_x)
        voxel_layout.addWidget(QLabel("Y:"))
        voxel_layout.addWidget(self.voxel_y)
        voxel_layout.addWidget(QLabel("Z:"))
        voxel_layout.addWidget(self.voxel_z)
        voxel_layout.addStretch()

        layout.addWidget(voxel_group)

        # ====================================================
        # 高级参数（水平排列 + 强制范围）
        # ====================================================
        adv_group = QGroupBox("高级参数")
        adv_layout = QHBoxLayout(adv_group)
        adv_layout.setSpacing(15)

        self.max_growth = QDoubleSpinBox()
        self.max_growth.setRange(0, 999999)  # ✅ 强制范围
        self.max_growth.setSingleStep(0.1)  # ✅ 每次增减 0.1
        self.max_growth.setDecimals(2)
        self.max_growth.setValue(4.0)
        self.max_growth.setMinimumWidth(120)

        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)  # ✅ 强制范围
        self.seed.setSingleStep(1)
        self.seed.setValue(42)
        self.seed.setMinimumWidth(120)

        self.remove_isolated = QCheckBox("Remove Isolated CBD")
        self.remove_isolated.setChecked(True)

        adv_layout.addWidget(QLabel("Max Growth:"))
        adv_layout.addWidget(self.max_growth)
        adv_layout.addWidget(QLabel("Seed:"))
        adv_layout.addWidget(self.seed)
        adv_layout.addStretch()
        adv_layout.addWidget(self.remove_isolated)

        layout.addWidget(adv_group)

        # ====================================================
        # 按钮区域（开始生成 + 恢复默认）
        # ====================================================
        btn_layout = QHBoxLayout()

        btn_generate = QPushButton("开始CBD三相生成")
        btn_generate.setMinimumHeight(50)
        btn_generate.clicked.connect(self.start_generate)

        btn_reset = QPushButton("恢复默认参数")
        btn_reset.setMinimumHeight(50)
        btn_reset.clicked.connect(self.reset_generate_params)
        btn_reset.setStyleSheet("""
            QPushButton {
                background: #666;
                color: white;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #888;
            }
        """)

        btn_layout.addWidget(btn_generate)
        btn_layout.addWidget(btn_reset)

        layout.addLayout(btn_layout)
        layout.addStretch()  # 底部留白

    # ========================================================
    # generate
    # ========================================================
    def start_generate(self):
        # ============================================
        # 1. 先验证是否上传文件（最重要！）
        # ============================================
        if not self.input_file:
            QMessageBox.warning(
                self,
                "错误",
                "请先上传文件"
            )
            return

        # ============================================
        # 2. 验证通过后，提交任务到后台
        # ============================================
        payload = {
            "task_id": self.upload_task_id,
            "input_volume_path": self.input_file,
            "target_cbd_vol_frac": self.target_cbd.value(),
            "w_um": self.w_um.value(),
            "pore_value": self.pore_value.value(),
            "am_value": self.am_value.value(),
            "cbd_value": self.cbd_value.value(),
            "voxel_size_x": self.voxel_x.value(),
            "voxel_size_y": self.voxel_y.value(),
            "voxel_size_z": self.voxel_z.value(),
            "max_growth_distance_factor": self.max_growth.value(),
            "remove_isolated_cbd": self.remove_isolated.isChecked(),
            "seed": self.seed.value(),
        }

        result = create_task(
            "/stage6/cbd-generate",
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
        # 3. 任务提交后, 清空文件选择框
        # ============================================
        self.file_edit.clear()

        # ============================================
        # 4. 弹窗提示 + 跳转到任务中心
        # ============================================
        msg = QMessageBox(self)
        msg.setWindowTitle("任务已提交")
        msg.setText("✅ CBD三相结构生成任务已提交！")
        msg.setInformativeText(
            "任务正在后台运行，请前往「历史任务中心」查看进度和结果。"
        )
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Ok)

        # 5. 添加「去任务中心」按钮
        go_btn = msg.addButton("前往任务中心", QMessageBox.ActionRole)
        ret = msg.exec_()

        # 5. 跳转到任务中心（无论点 OK 还是「前往任务中心」按钮）
        if ret == QMessageBox.Ok or msg.clickedButton() == go_btn:
            self.main_window.menu.setCurrentRow(5)  # 「历史任务中心」是第 6 个（索引 5）

    def reset_generate_params(self):
        """恢复所有参数为默认值"""

        # ✅ 清空文件
        self.file_edit.clear()

        # ✅ 恢复 CBD 参数
        self.target_cbd.setValue(0.05)
        self.w_um.setValue(0.08)

        # ✅ 恢复相标签
        self.pore_value.setValue(0)  # ✅ 默认值（根据你的实际需求）
        self.am_value.setValue(1)
        self.cbd_value.setValue(2)

        # ✅ 恢复 Voxel Size
        self.voxel_x.setValue(0.02791)
        self.voxel_y.setValue(0.0315)
        self.voxel_z.setValue(0.02791)

        # ✅ 恢复高级参数
        self.max_growth.setValue(4.0)
        self.seed.setValue(42)
        self.remove_isolated.setChecked(True)

        # ✅ 提示
        QMessageBox.information(self, "已恢复", "所有参数已恢复为默认值！")

    # ========================================================
    # TAB2
    # ========================================================
    def build_fit_tab(self):
        layout = QVBoxLayout(self.tab_fit)
        layout.setSpacing(15)

        # ====================================================
        # 顶部提示
        # ====================================================
        tip_label = QLabel(
            "💡 选择真实三相结构目录, 选择输出目录，设置参数后点击「开始CBD参数拟合」任务将提交到「任务中心」,可前往进行查看状态\n"
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
        layout.addWidget(tip_label)

        # ====================================================
        # 路径选择（水平排列）
        # ====================================================
        path_group = QGroupBox("输入路径")
        path_layout = QHBoxLayout(path_group)
        path_layout.setSpacing(15)

        self.real_dir_edit = QLineEdit()
        self.real_dir_edit.setPlaceholderText("选择真实三相结构目录...")
        self.real_dir_edit.setMinimumWidth(300)

        btn_real = QPushButton("选择真实三相结构目录")
        btn_real.clicked.connect(self.select_real_dir)

        self.out_dir_edit = QLineEdit()
        self.out_dir_edit.setPlaceholderText("选择输出目录...")
        self.out_dir_edit.setMinimumWidth(300)

        btn_out = QPushButton("选择输出目录")
        btn_out.clicked.connect(self.select_out_dir)

        path_layout.addWidget(QLabel("真实三相结构目录:"))
        path_layout.addWidget(self.real_dir_edit)
        path_layout.addWidget(btn_real)
        path_layout.addSpacing(20)
        path_layout.addWidget(QLabel("输出目录:"))
        path_layout.addWidget(self.out_dir_edit)
        path_layout.addWidget(btn_out)

        layout.addWidget(path_group)

        # ====================================================
        # 相标签（水平排列）
        # ====================================================
        phase_group = QGroupBox("相标签定义")
        phase_layout = QHBoxLayout(phase_group)
        phase_layout.setSpacing(15)

        self.fit_pore = QSpinBox()
        self.fit_pore.setRange(0, 999999)
        self.fit_pore.setSingleStep(1)
        self.fit_pore.setValue(0)
        self.fit_pore.setMinimumWidth(100)

        self.fit_am = QSpinBox()
        self.fit_am.setRange(0, 999999)
        self.fit_am.setSingleStep(1)
        self.fit_am.setValue(1)
        self.fit_am.setMinimumWidth(100)

        self.fit_cbd = QSpinBox()
        self.fit_cbd.setRange(0, 999999)
        self.fit_cbd.setSingleStep(1)
        self.fit_cbd.setValue(2)
        self.fit_cbd.setMinimumWidth(100)

        phase_layout.addWidget(QLabel("Pore:"))
        phase_layout.addWidget(self.fit_pore)
        phase_layout.addWidget(QLabel("AM:"))
        phase_layout.addWidget(self.fit_am)
        phase_layout.addWidget(QLabel("CBD:"))
        phase_layout.addWidget(self.fit_cbd)
        phase_layout.addStretch()

        layout.addWidget(phase_group)

        # ====================================================
        # W扫描参数（水平排列）
        # ====================================================
        w_group = QGroupBox("W扫描参数")
        w_layout = QHBoxLayout(w_group)
        w_layout.setSpacing(15)

        self.w_min = QDoubleSpinBox()
        self.w_min.setRange(0, 999999)
        self.w_min.setSingleStep(0.01)
        self.w_min.setDecimals(4)
        self.w_min.setValue(0.02)
        self.w_min.setMinimumWidth(120)

        self.w_max = QDoubleSpinBox()
        self.w_max.setRange(0, 999999)
        self.w_max.setSingleStep(0.01)
        self.w_max.setDecimals(4)
        self.w_max.setValue(0.30)
        self.w_max.setMinimumWidth(120)

        self.num_w = QSpinBox()
        self.num_w.setRange(0, 999999)
        self.num_w.setSingleStep(1)
        self.num_w.setValue(20)
        self.num_w.setMinimumWidth(100)

        w_layout.addWidget(QLabel("W Min:"))
        w_layout.addWidget(self.w_min)
        w_layout.addWidget(QLabel("W Max:"))
        w_layout.addWidget(self.w_max)
        w_layout.addWidget(QLabel("Num W:"))
        w_layout.addWidget(self.num_w)
        w_layout.addStretch()

        layout.addWidget(w_group)

        # ====================================================
        # Voxel Size（水平排列）
        # ====================================================
        voxel_group = QGroupBox("Voxel Size")
        voxel_layout = QHBoxLayout(voxel_group)
        voxel_layout.setSpacing(15)

        self.fit_voxel_x = QDoubleSpinBox()
        self.fit_voxel_x.setRange(0, 999999)
        self.fit_voxel_x.setSingleStep(0.0001)
        self.fit_voxel_x.setDecimals(5)
        self.fit_voxel_x.setValue(0.02791)
        self.fit_voxel_x.setMinimumWidth(120)

        self.fit_voxel_y = QDoubleSpinBox()
        self.fit_voxel_y.setRange(0, 999999)
        self.fit_voxel_y.setSingleStep(0.0001)
        self.fit_voxel_y.setDecimals(5)
        self.fit_voxel_y.setValue(0.0315)
        self.fit_voxel_y.setMinimumWidth(120)

        self.fit_voxel_z = QDoubleSpinBox()
        self.fit_voxel_z.setRange(0, 999999)
        self.fit_voxel_z.setSingleStep(0.0001)
        self.fit_voxel_z.setDecimals(5)
        self.fit_voxel_z.setValue(0.02791)
        self.fit_voxel_z.setMinimumWidth(120)

        voxel_layout.addWidget(QLabel("X:"))
        voxel_layout.addWidget(self.fit_voxel_x)
        voxel_layout.addWidget(QLabel("Y:"))
        voxel_layout.addWidget(self.fit_voxel_y)
        voxel_layout.addWidget(QLabel("Z:"))
        voxel_layout.addWidget(self.fit_voxel_z)
        voxel_layout.addStretch()

        layout.addWidget(voxel_group)

        # ====================================================
        # 高级参数（水平排列）
        # ====================================================
        adv_group = QGroupBox("高级参数")
        adv_layout = QHBoxLayout(adv_group)
        adv_layout.setSpacing(15)

        self.fit_growth = QDoubleSpinBox()
        self.fit_growth.setRange(0, 999999)
        self.fit_growth.setSingleStep(0.1)
        self.fit_growth.setDecimals(2)
        self.fit_growth.setValue(4.0)
        self.fit_growth.setMinimumWidth(120)

        self.fit_seed = QSpinBox()
        self.fit_seed.setRange(0, 999999)
        self.fit_seed.setSingleStep(1)
        self.fit_seed.setValue(42)
        self.fit_seed.setMinimumWidth(120)

        self.fit_remove = QCheckBox("Remove Isolated CBD")
        self.fit_remove.setChecked(True)

        adv_layout.addWidget(QLabel("Max Growth:"))
        adv_layout.addWidget(self.fit_growth)
        adv_layout.addWidget(QLabel("Seed:"))
        adv_layout.addWidget(self.fit_seed)
        adv_layout.addStretch()
        adv_layout.addWidget(self.fit_remove)

        layout.addWidget(adv_group)

        # ====================================================
        # 按钮区域（用你的真实 start_fit）
        # ====================================================
        btn_layout = QHBoxLayout()

        btn_fit = QPushButton("开始CBD参数拟合")
        btn_fit.setMinimumHeight(50)
        btn_fit.clicked.connect(self.start_fit)

        btn_reset = QPushButton("恢复默认参数")
        btn_reset.setMinimumHeight(50)
        btn_reset.clicked.connect(self.reset_fit_params)
        btn_reset.setStyleSheet("""
            QPushButton {
                background: #666;
                color: white;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #888;
            }
        """)

        btn_layout.addWidget(btn_fit)
        btn_layout.addWidget(btn_reset)

        layout.addLayout(btn_layout)
        layout.addStretch()

    # ========================================================
    # select file
    # ========================================================
    def select_npy_file(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择npy文件",
            "",
            "NumPy File (*.npy)"
        )

        if path:
            self.file_edit.setText(path)

    # ========================================================
    # upload
    # ========================================================
    def upload_file(self):

        path = self.file_edit.text()

        if not path:
            QMessageBox.warning(
                self,
                "错误",
                "请先选择文件"
            )

            return

        try:

            with open(path, "rb") as f:

                files = {
                    "file": (
                        os.path.basename(path),
                        f,
                        "application/octet-stream"
                    )
                }

                response = requests.post(
                    f"{API_BASE}/upload/b2ps",
                    files=files,
                    timeout=300,
                )

            result = response.json()

            if not result.get("success"):
                QMessageBox.warning(
                    self,
                    "错误",
                    str(result)
                )

                return

            self.upload_task_id = result["task_id"]

            self.input_file = result["input_file"]

            QMessageBox.information(
                self,
                "成功",
                "文件上传成功"
            )

        except Exception as e:

            QMessageBox.warning(
                self,
                "错误",
                str(e)
            )

    # ========================================================
    # fit
    # ========================================================
    def start_fit(self):
        # ============================================
        # 1. 校验输入输出目录（必须先做！）
        # ============================================
        if not self.real_dir_edit.text().strip():
            QMessageBox.warning(
                self,
                "错误",
                "请先选择真实三相结构目录"
            )
            return

        if not self.out_dir_edit.text().strip():
            QMessageBox.warning(
                self,
                "错误",
                "请先选择输出目录"
            )
            return

        # ============================================
        # 2. 提交任务到后台
        # ============================================
        payload = {
            "real_3phase_slice_dir": self.real_dir_edit.text(),
            "out_dir": self.out_dir_edit.text(),
            "pore_value": self.fit_pore.value(),
            "am_value": self.fit_am.value(),
            "cbd_value": self.fit_cbd.value(),
            "w_min": self.w_min.value(),
            "w_max": self.w_max.value(),
            "num_w": self.num_w.value(),
            "voxel_size_x": self.fit_voxel_x.value(),
            "voxel_size_y": self.fit_voxel_y.value(),
            "voxel_size_z": self.fit_voxel_z.value(),
            "max_growth_distance_factor": self.fit_growth.value(),
            "remove_isolated_cbd": self.fit_remove.isChecked(),
            "seed": self.fit_seed.value(),
        }

        result = create_task(
            "/stage6/fit-cbd-spreading-parameter",
            payload,
        )

        if "task_id" not in result:
            QMessageBox.warning(
                self,
                "错误",
                str(result)
            )
            return

        self.fit_task_id = result["task_id"]

        # ============================================
        # 3. 任务提交后, 清空文件选择框
        # ============================================
        self.file_edit.clear()

        # ============================================
        # 4. 弹窗提示 + 跳转到任务中心
        # ============================================
        msg = QMessageBox(self)
        msg.setWindowTitle("任务已提交")
        msg.setText("✅ CBD参数拟合任务已提交！")
        msg.setInformativeText(
            "任务正在后台运行，请前往「历史任务中心」查看进度和结果。"
        )
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Ok)

        # 5. 添加「去任务中心」按钮
        go_btn = msg.addButton("前往任务中心", QMessageBox.ActionRole)
        ret = msg.exec_()

        # 6. 跳转到任务中心（无论点 OK 还是「前往任务中心」按钮）
        if ret == QMessageBox.Ok or msg.clickedButton() == go_btn:
            self.main_window.menu.setCurrentRow(5)  # 「历史任务中心」是第 6 个（索引 5）

    # ========================================================
    # refresh generate
    # ========================================================
    def refresh_generate_task(self):

        if not self.task_id:
            return

        task = query_task(self.task_id)

        self.generate_progress.setValue(
            int(task.get("progress", 0))
        )

        logs = task.get("logs", [])

        self.generate_log.setPlainText(
            "\n".join(logs[-200:])
        )

        status = task.get("status", "")

        if status == "finished":

            self.generate_timer.stop()

            QMessageBox.information(
                self,
                "完成",
                "CBD三相结构生成完成"
            )

        elif status == "failed":

            self.generate_timer.stop()

            QMessageBox.warning(
                self,
                "失败",
                task.get("error", "")
            )

    # ========================================================
    # refresh fit
    # ========================================================
    def refresh_fit_task(self):

        if not self.fit_task_id:
            return

        task = query_task(self.fit_task_id)

        self.fit_progress.setValue(
            int(task.get("progress", 0))
        )

        logs = task.get("logs", [])

        self.fit_log.setPlainText(
            "\n".join(logs[-200:])
        )

        status = task.get("status", "")

        if status == "finished":

            self.fit_timer.stop()

            QMessageBox.information(
                self,
                "完成",
                "CBD参数拟合完成"
            )

        elif status == "failed":

            self.fit_timer.stop()

            QMessageBox.warning(
                self,
                "失败",
                task.get("error", "")
            )

    # ========================================================
    # dir
    # ========================================================
    def select_real_dir(self):

        path = QFileDialog.getExistingDirectory(
            self,
            "选择真实三相结构目录"
        )

        if path:
            self.real_dir_edit.setText(path)

    def select_out_dir(self):

        path = QFileDialog.getExistingDirectory(
            self,
            "选择输出目录"
        )

        if path:
            self.out_dir_edit.setText(path)

    def reset_fit_params(self):
        """恢复默认参数"""
        self.fit_pore.setValue(0)
        self.fit_am.setValue(1)
        self.fit_cbd.setValue(2)
        self.w_min.setValue(0.02)
        self.w_max.setValue(0.30)
        self.num_w.setValue(20)
        self.fit_voxel_x.setValue(0.02791)
        self.fit_voxel_y.setValue(0.0315)
        self.fit_voxel_z.setValue(0.02791)
        self.fit_growth.setValue(4.0)
        self.fit_seed.setValue(42)
        self.fit_remove.setChecked(True)
        QMessageBox.information(self, "已恢复", "所有参数已恢复为默认值！")
