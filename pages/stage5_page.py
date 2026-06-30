# -*- coding: utf-8 -*-
import os

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


class Stage5Page(QWidget):

    def __init__(self, parent_window):  # 接收 MainWindow
        super().__init__()
        self.main_window = parent_window  # 保存引用

        self.task_id = None
        self.fit_task_id = None

        self.init_ui()

    # ========================================================
    # UI
    # ========================================================
    def init_ui(self):

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
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
        layout = QVBoxLayout(self.tab_generate)
        layout.setSpacing(4)
        layout.setContentsMargins(2, 2, 2, 2)

        # 顶部说明
        tip = QLabel("选择 .npy 文件上传后进行参数设置，点击按钮提交任务到「任务中心」。")
        tip.setWordWrap(True)
        tip.setAlignment(Qt.AlignCenter)
        layout.addWidget(tip)

        # 文件 + 输出
        io_group = QGroupBox("输入 / 输出")
        io_layout = QVBoxLayout(io_group)
        io_layout.setSpacing(2); io_layout.setContentsMargins(4, 2, 4, 2)

        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText(".npy 文件...")
        btn_select_file = QPushButton("选择 .npy")
        btn_select_file.clicked.connect(self.select_npy_file)
        io_layout.addWidget(self.file_edit)
        io_layout.addWidget(btn_select_file)

        out_row = QHBoxLayout(); out_row.setSpacing(2)
        self.gen_out_dir_edit = QLineEdit()
        self.gen_out_dir_edit.setPlaceholderText("输出目录...")
        btn_out = QPushButton("选择")
        btn_out.clicked.connect(self.select_gen_out_dir)
        out_row.addWidget(QLabel("输出:"))
        out_row.addWidget(self.gen_out_dir_edit, 1)
        out_row.addWidget(btn_out)
        io_layout.addLayout(out_row)
        layout.addWidget(io_group)

        # CBD + Voxel + 高级
        param_group = QGroupBox("生成参数")
        param_layout = QVBoxLayout(param_group)
        param_layout.setSpacing(2); param_layout.setContentsMargins(4, 2, 4, 2)

        r1 = QHBoxLayout(); r1.setSpacing(3)
        self.target_cbd = QDoubleSpinBox(); self.target_cbd.setDecimals(4); self.target_cbd.setValue(0.05); self.target_cbd.setRange(0, 999999)
        self.w_um = QDoubleSpinBox(); self.w_um.setDecimals(4); self.w_um.setValue(0.08); self.w_um.setRange(0, 999999)
        r1.addWidget(QLabel("CBD vol:")); r1.addWidget(self.target_cbd)
        r1.addWidget(QLabel("W(μm):")); r1.addWidget(self.w_um)
        param_layout.addLayout(r1)

        r2 = QHBoxLayout(); r2.setSpacing(2)
        self.voxel_x = QDoubleSpinBox(); self.voxel_x.setDecimals(5); self.voxel_x.setValue(0.02791); self.voxel_x.setRange(0, 999999)
        self.voxel_y = QDoubleSpinBox(); self.voxel_y.setDecimals(5); self.voxel_y.setValue(0.0315); self.voxel_y.setRange(0, 999999)
        self.voxel_z = QDoubleSpinBox(); self.voxel_z.setDecimals(5); self.voxel_z.setValue(0.02791); self.voxel_z.setRange(0, 999999)
        r2.addWidget(QLabel("Voxel:")); r2.addWidget(QLabel("X")); r2.addWidget(self.voxel_x)
        r2.addWidget(QLabel("Y")); r2.addWidget(self.voxel_y)
        r2.addWidget(QLabel("Z")); r2.addWidget(self.voxel_z)
        param_layout.addLayout(r2)

        r3 = QHBoxLayout(); r3.setSpacing(3)
        self.max_growth = QDoubleSpinBox(); self.max_growth.setRange(0, 999999); self.max_growth.setDecimals(2); self.max_growth.setValue(4.0)
        self.remove_isolated = QCheckBox("Remove Isolated CBD"); self.remove_isolated.setChecked(True)
        r3.addWidget(QLabel("Growth:")); r3.addWidget(self.max_growth)
        r3.addWidget(self.remove_isolated)
        param_layout.addLayout(r3)
        layout.addWidget(param_group)

        # 按钮
        btn_layout = QHBoxLayout(); btn_layout.setSpacing(6)
        btn_generate = QPushButton("开始CBD三相生成")
        btn_generate.setMinimumHeight(45)
        btn_generate.clicked.connect(self.start_generate)
        btn_generate.setStyleSheet("QPushButton { font-weight: bold; }")
        btn_reset = QPushButton("恢复默认参数")
        btn_reset.setMinimumHeight(45)
        btn_reset.clicked.connect(self.reset_generate_params)
        btn_reset.setStyleSheet("QPushButton { font-weight: bold; }")
        btn_layout.addWidget(btn_generate)
        btn_layout.addWidget(btn_reset)
        layout.addLayout(btn_layout)
        layout.addStretch()

    # ========================================================
    # generate
    # ========================================================
    def start_generate(self):
        # ============================================
        # 1. 先验证是否选择了文件
        # ============================================
        file_path = self.file_edit.text().strip()
        if not file_path:
            QMessageBox.warning(
                self,
                "错误",
                "请先选择 .npy 文件"
            )
            return

        # ============================================
        # 2. 验证输出目录
        # ============================================
        out_path = self.gen_out_dir_edit.text().strip()
        if not out_path:
            QMessageBox.warning(
                self,
                "错误",
                "请先选择输出目录"
            )
            return

        # ============================================
        # 3. 验证通过后，提交任务到后台
        # ============================================
        payload = {
            "input_volume_path": file_path,
            "out_dir": out_path,
            "target_cbd_vol_frac": self.target_cbd.value(),
            "w_um": self.w_um.value(),
            "pore_value": 0,
            "am_value": 1,
            "cbd_value": 2,
            "voxel_size_x": self.voxel_x.value(),
            "voxel_size_y": self.voxel_y.value(),
            "voxel_size_z": self.voxel_z.value(),
            "max_growth_distance_factor": self.max_growth.value(),
            "remove_isolated_cbd": self.remove_isolated.isChecked(),
            "seed": 42,
        }

        result = create_task(
            "/stage5/cbd-generate",
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
        msg.setText(" CBD三相结构生成任务已提交！")
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
            self.main_window.history_page.refresh_task_list()

    def reset_generate_params(self):
        """恢复所有参数为默认值"""

        self.file_edit.clear()
        self.gen_out_dir_edit.clear()
        self.target_cbd.setValue(0.05)
        self.w_um.setValue(0.08)
        self.voxel_x.setValue(0.02791)
        self.voxel_y.setValue(0.0315)
        self.voxel_z.setValue(0.02791)
        self.max_growth.setValue(4.0)
        self.remove_isolated.setChecked(True)
        QMessageBox.information(self, "已恢复", "所有参数已恢复为默认值！")

    # ========================================================
    # TAB2
    # ========================================================
    def build_fit_tab(self):
        layout = QVBoxLayout(self.tab_fit)
        layout.setSpacing(4)
        layout.setContentsMargins(2, 2, 2, 2)

        # 顶部说明
        tip = QLabel("选择真实三相结构目录，设置参数后点击按钮提交拟合任务到「任务中心」。")
        tip.setWordWrap(True)
        tip.setAlignment(Qt.AlignCenter)
        layout.addWidget(tip)

        # 输入输出
        io_group = QGroupBox("输入 / 输出")
        io_layout = QVBoxLayout(io_group)
        io_layout.setSpacing(2); io_layout.setContentsMargins(4, 2, 4, 2)

        r = QHBoxLayout(); r.setSpacing(2)
        self.real_dir_edit = QLineEdit()
        self.real_dir_edit.setPlaceholderText("真实三相结构目录...")
        btn_real = QPushButton("选择")
        btn_real.clicked.connect(self.select_real_dir)
        r.addWidget(QLabel("输入:")); r.addWidget(self.real_dir_edit, 1); r.addWidget(btn_real)
        io_layout.addLayout(r)

        r = QHBoxLayout(); r.setSpacing(2)
        self.out_dir_edit = QLineEdit()
        self.out_dir_edit.setPlaceholderText("输出目录...")
        btn_out = QPushButton("选择")
        btn_out.clicked.connect(self.select_out_dir)
        r.addWidget(QLabel("输出:")); r.addWidget(self.out_dir_edit, 1); r.addWidget(btn_out)
        io_layout.addLayout(r)
        layout.addWidget(io_group)

        # W + Voxel + 高级
        param_group = QGroupBox("拟合参数")
        param_layout = QVBoxLayout(param_group)
        param_layout.setSpacing(2); param_layout.setContentsMargins(4, 2, 4, 2)

        r1 = QHBoxLayout(); r1.setSpacing(3)
        self.w_min = QDoubleSpinBox(); self.w_min.setDecimals(4); self.w_min.setValue(0.02); self.w_min.setRange(0, 999999)
        self.w_max = QDoubleSpinBox(); self.w_max.setDecimals(4); self.w_max.setValue(0.30); self.w_max.setRange(0, 999999)
        self.num_w = QSpinBox(); self.num_w.setRange(0, 999999); self.num_w.setValue(20)
        r1.addWidget(QLabel("w_min:")); r1.addWidget(self.w_min)
        r1.addWidget(QLabel("w_max:")); r1.addWidget(self.w_max)
        r1.addWidget(QLabel("num_w:")); r1.addWidget(self.num_w)
        param_layout.addLayout(r1)

        r2 = QHBoxLayout(); r2.setSpacing(2)
        self.fit_voxel_x = QDoubleSpinBox(); self.fit_voxel_x.setDecimals(5); self.fit_voxel_x.setValue(0.02791); self.fit_voxel_x.setRange(0, 999999)
        self.fit_voxel_y = QDoubleSpinBox(); self.fit_voxel_y.setDecimals(5); self.fit_voxel_y.setValue(0.0315); self.fit_voxel_y.setRange(0, 999999)
        self.fit_voxel_z = QDoubleSpinBox(); self.fit_voxel_z.setDecimals(5); self.fit_voxel_z.setValue(0.02791); self.fit_voxel_z.setRange(0, 999999)
        r2.addWidget(QLabel("Voxel:")); r2.addWidget(QLabel("X")); r2.addWidget(self.fit_voxel_x)
        r2.addWidget(QLabel("Y")); r2.addWidget(self.fit_voxel_y)
        r2.addWidget(QLabel("Z")); r2.addWidget(self.fit_voxel_z)
        param_layout.addLayout(r2)

        r3 = QHBoxLayout(); r3.setSpacing(3)
        self.fit_growth = QDoubleSpinBox(); self.fit_growth.setRange(0, 999999); self.fit_growth.setDecimals(2); self.fit_growth.setValue(4.0)
        self.fit_remove = QCheckBox("Remove Isolated CBD"); self.fit_remove.setChecked(True)
        r3.addWidget(QLabel("Growth:")); r3.addWidget(self.fit_growth)
        r3.addWidget(self.fit_remove)
        param_layout.addLayout(r3)
        layout.addWidget(param_group)

        # 按钮
        btn_layout = QHBoxLayout(); btn_layout.setSpacing(6)
        btn_fit = QPushButton("开始CBD参数拟合")
        btn_fit.setMinimumHeight(45)
        btn_fit.clicked.connect(self.start_fit)
        btn_fit.setStyleSheet("QPushButton { font-weight: bold; }")
        btn_reset = QPushButton("恢复默认参数")
        btn_reset.setMinimumHeight(45)
        btn_reset.clicked.connect(self.reset_fit_params)
        btn_reset.setStyleSheet("QPushButton { font-weight: bold; }")
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
            # 文件类型校验：必须是 .npy 后缀
            if not path.lower().endswith(".npy"):
                QMessageBox.warning(
                    self,
                    "文件类型错误",
                    f"请选择 .npy 格式的文件！\n当前文件: {os.path.basename(path)}"
                )
                return

            # 校验文件是否可读
            if not os.path.isfile(path):
                QMessageBox.warning(
                    self,
                    "文件错误",
                    "所选文件不存在或无法访问！"
                )
                return

            self.file_edit.setText(path)

    # ========================================================
    # select gen out dir
    # ========================================================
    def select_gen_out_dir(self):

        path = QFileDialog.getExistingDirectory(
            self,
            "选择输出目录"
        )

        if path:
            self.gen_out_dir_edit.setText(path)

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
            "pore_value": 0,
            "am_value": 1,
            "cbd_value": 2,
            "w_min": self.w_min.value(),
            "w_max": self.w_max.value(),
            "num_w": self.num_w.value(),
            "voxel_size_x": self.fit_voxel_x.value(),
            "voxel_size_y": self.fit_voxel_y.value(),
            "voxel_size_z": self.fit_voxel_z.value(),
            "max_growth_distance_factor": self.fit_growth.value(),
            "remove_isolated_cbd": self.fit_remove.isChecked(),
            "seed": 42,
        }

        result = create_task(
            "/stage5/fit-cbd-spreading-parameter",
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
        msg.setText(" CBD参数拟合任务已提交！")
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
            self.main_window.history_page.refresh_task_list()

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
        self.w_min.setValue(0.02)
        self.w_max.setValue(0.30)
        self.num_w.setValue(20)
        self.fit_voxel_x.setValue(0.02791)
        self.fit_voxel_y.setValue(0.0315)
        self.fit_voxel_z.setValue(0.02791)
        self.fit_growth.setValue(4.0)
        self.fit_remove.setChecked(True)
        QMessageBox.information(self, "已恢复", "所有参数已恢复为默认值！")
