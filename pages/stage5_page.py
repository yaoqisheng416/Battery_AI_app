# -*- coding: utf-8 -*-
import os
import sys

import requests
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QMessageBox,
    QComboBox,
    QDoubleSpinBox, QFileDialog, QGroupBox, QLineEdit, QSpinBox, QCheckBox,
    QTabWidget,
)

from api_client import (
    create_task,
     API_BASE,
)


def resource_path(relative_path):

    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class Stage5Page(QWidget):

    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window

        self.selected_vae_path = None
        self.selected_ldm_path = None

        self.default_summary_json = resource_path(
            "backend/electrode_twin/latent_dataset/dataset_summary.json"
        )

        self.init_ui()

        self.load_models()

    # =========================================================
    # UI
    # =========================================================
    def init_ui(self):

        root_layout = QVBoxLayout(self)

        # =====================================================
        # title
        # =====================================================
        title = QLabel(
            "Stage5 大体积结构生成"
        )

        title.setStyleSheet("""
        font-size:24px;
        font-weight:bold;
        color:white;
        padding:10px;
        """)

        root_layout.addWidget(title)

        # =====================================================
        # tabs
        # =====================================================
        self.tabs = QTabWidget()

        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #444;
                background: #2b2d31;
                border-radius: 8px;
            }

            QTabBar::tab {
                background: #25262b;
                color: white;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 5px;
            }

            QTabBar::tab:selected {
                background: #4f8cff;
                color: white;
            }

            QTabBar::tab:hover {
                background: #3a3b3f;
            }
        """)

        root_layout.addWidget(self.tabs)

        # =====================================================
        # tab1
        # =====================================================
        self.tab_local_condition = QWidget()

        self.tabs.addTab(
            self.tab_local_condition,
            "步骤1：构建 Local Conditions"
        )

        self.build_local_condition_tab()

        # =====================================================
        # tab2
        # =====================================================
        self.tab_large_volume = QWidget()

        self.tabs.addTab(
            self.tab_large_volume,
            "步骤2：生成224³大体积"
        )

        self.build_large_volume_tab()

    # =========================================================
    # helper
    # =========================================================
    def create_form_row(
            self,
            label_text,
            widget,
            layout,
            label_width=220,
    ):
        row = QHBoxLayout()

        label = QLabel(label_text)
        label.setFixedWidth(label_width)

        row.addWidget(label)
        row.addWidget(widget)

        row.addStretch()

        layout.addLayout(row)

    # =========================================================
    # tab1
    # =========================================================
    def build_local_condition_tab(self):

        layout = QVBoxLayout(self.tab_local_condition)

        # =====================================================
        # tip
        # =====================================================
        tip = QLabel(
            "从真实大体积中裁剪224³区域，并生成8个local conditions"
        )

        tip.setWordWrap(True)

        tip.setStyleSheet("""
            QLabel{
                background:#25262b;
                border:1px solid #4f8cff;
                border-radius:8px;
                padding:12px;
                color:#4f8cff;
                font-size:12px;
            }
        """)

        layout.addWidget(tip)

        # =====================================================
        # file
        # =====================================================
        file_group = QGroupBox("文件路径")

        file_layout = QVBoxLayout(file_group)

        self.real_volume_edit = QLineEdit()
        self.real_volume_edit.setPlaceholderText(
            "真实体积npy路径"
        )

        self.out_dir1_edit = QLineEdit()
        self.out_dir1_edit.setPlaceholderText(
            "输出目录"
        )

        btn_real = QPushButton("选择真实体积")
        btn_real.clicked.connect(
            self.select_real_volume
        )

        btn_out1 = QPushButton("选择输出目录")
        btn_out1.clicked.connect(
            self.select_out_dir1
        )

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("真实体积"))
        row1.addWidget(self.real_volume_edit)
        row1.addWidget(btn_real)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("输出目录"))
        row2.addWidget(self.out_dir1_edit)
        row2.addWidget(btn_out1)

        file_layout.addLayout(row1)
        file_layout.addLayout(row2)

        layout.addWidget(file_group)

        # =====================================================
        # params
        # =====================================================
        param_group = QGroupBox("参数设置")

        param_layout = QVBoxLayout(param_group)

        # crop
        self.crop_y_spin = QSpinBox()
        self.crop_y_spin.setRange(0, 99999)
        self.crop_y_spin.setValue(0)

        self.crop_z_spin = QSpinBox()
        self.crop_z_spin.setRange(0, 99999)
        self.crop_z_spin.setValue(100)

        self.crop_x_spin = QSpinBox()
        self.crop_x_spin.setRange(0, 99999)
        self.crop_x_spin.setValue(0)

        self.large_vol_spin = QSpinBox()
        self.large_vol_spin.setRange(32, 2048)
        self.large_vol_spin.setValue(224)

        self.patch_size1_spin = QSpinBox()
        self.patch_size1_spin.setRange(32, 512)
        self.patch_size1_spin.setValue(128)

        self.overlap1_spin = QSpinBox()
        self.overlap1_spin.setRange(0, 256)
        self.overlap1_spin.setValue(32)

        self.pore1_spin = QSpinBox()
        self.pore1_spin.setValue(0)

        self.solid1_spin = QSpinBox()
        self.solid1_spin.setValue(1)

        self.voxel_y1_spin = QDoubleSpinBox()
        self.voxel_y1_spin.setDecimals(6)
        self.voxel_y1_spin.setValue(0.0315)

        self.voxel_z1_spin = QDoubleSpinBox()
        self.voxel_z1_spin.setDecimals(6)
        self.voxel_z1_spin.setValue(0.02791)

        self.voxel_x1_spin = QDoubleSpinBox()
        self.voxel_x1_spin.setDecimals(6)
        self.voxel_x1_spin.setValue(0.02791)

        self.remove_small_checkbox = QCheckBox()
        self.remove_small_checkbox.setChecked(True)

        self.min_pore_spin = QSpinBox()
        self.min_pore_spin.setRange(1, 100000)
        self.min_pore_spin.setValue(10)

        self.tau_nonperc_spin = QDoubleSpinBox()
        self.tau_nonperc_spin.setDecimals(2)
        self.tau_nonperc_spin.setMaximum(1e9)
        self.tau_nonperc_spin.setValue(1e6)

        self.suppress_checkbox = QCheckBox()
        self.suppress_checkbox.setChecked(True)

        self.create_form_row(
            "crop_start_y",
            self.crop_y_spin,
            param_layout
        )

        self.create_form_row(
            "crop_start_z",
            self.crop_z_spin,
            param_layout
        )

        self.create_form_row(
            "crop_start_x",
            self.crop_x_spin,
            param_layout
        )

        self.create_form_row(
            "large_vol_size",
            self.large_vol_spin,
            param_layout
        )

        self.create_form_row(
            "patch_size",
            self.patch_size1_spin,
            param_layout
        )

        self.create_form_row(
            "overlap",
            self.overlap1_spin,
            param_layout
        )

        self.create_form_row(
            "pore_value",
            self.pore1_spin,
            param_layout
        )

        self.create_form_row(
            "solid_value",
            self.solid1_spin,
            param_layout
        )

        self.create_form_row(
            "voxel_size_y",
            self.voxel_y1_spin,
            param_layout
        )

        self.create_form_row(
            "voxel_size_z",
            self.voxel_z1_spin,
            param_layout
        )

        self.create_form_row(
            "voxel_size_x",
            self.voxel_x1_spin,
            param_layout
        )

        self.create_form_row(
            "remove_small_pore_components",
            self.remove_small_checkbox,
            param_layout
        )

        self.create_form_row(
            "min_pore_component_size",
            self.min_pore_spin,
            param_layout
        )

        self.create_form_row(
            "tau_nonperc_value",
            self.tau_nonperc_spin,
            param_layout
        )

        self.create_form_row(
            "suppress_taufactor_output",
            self.suppress_checkbox,
            param_layout
        )

        layout.addWidget(param_group)

        # =====================================================
        # btn
        # =====================================================

        btn_layout = QHBoxLayout()

        btn = QPushButton(
            "开始执行Stage5-1：构建空间异质局部条件场"
        )

        btn.setMinimumHeight(45)

        btn.setStyleSheet("""
            QPushButton{
                background:#4f8cff;
                color:white;
                border-radius:8px;
                font-size:15px;
                font-weight:bold;
            }
            QPushButton:hover{
                background:#6aa1ff;
            }
        """)

        btn.clicked.connect(
            self.start_local_condition_task
        )

        # ============================================
        # reset
        # ============================================

        btn_reset = QPushButton("恢复默认参数")

        btn_reset.setMinimumHeight(45)

        btn_reset.setStyleSheet("""
            QPushButton{
                background:#666;
                color:white;
                border-radius:8px;
                font-size:15px;
                font-weight:bold;
            }

            QPushButton:hover{
                background:#888;
            }
        """)

        btn_reset.clicked.connect(
            self.reset_local_condition_params
        )

        btn_layout.addWidget(btn)
        btn_layout.addWidget(btn_reset)

        layout.addLayout(btn_layout)

    # =========================================================
    # tab2
    # =========================================================
    def build_large_volume_tab(self):

        layout = QVBoxLayout(self.tab_large_volume)

        # =====================================================
        # tip
        # =====================================================
        tip = QLabel(
            "使用步骤1生成的local_conditions_json，生成224³大体积"
        )

        tip.setWordWrap(True)

        tip.setStyleSheet("""
            QLabel{
                background:#25262b;
                border:1px solid #4f8cff;
                border-radius:8px;
                padding:12px;
                color:#4f8cff;
                font-size:12px;
            }
        """)

        layout.addWidget(tip)

        # =====================================================
        # model group
        # =====================================================
        model_group = QGroupBox("模型选择")

        model_layout = QVBoxLayout(model_group)

        self.vae_combo = QComboBox()
        self.ldm_combo = QComboBox()

        self.model_path_label = QLabel("未选择模型")

        self.model_path_label.setStyleSheet("""
            QLabel{
                background:#1e1f24;
                border-radius:5px;
                padding:6px;
                color:#aaa;
            }
        """)

        self.vae_combo.currentIndexChanged.connect(
            self.on_vae_changed
        )

        self.ldm_combo.currentIndexChanged.connect(
            self.on_ldm_changed
        )

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("VAE模型"))
        row1.addWidget(self.vae_combo)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("LDM模型"))
        row2.addWidget(self.ldm_combo)

        model_layout.addLayout(row1)
        model_layout.addLayout(row2)
        model_layout.addWidget(self.model_path_label)

        layout.addWidget(model_group)

        # =====================================================
        # file
        # =====================================================
        file_group = QGroupBox("输入输出")

        file_layout = QVBoxLayout(file_group)

        self.local_json_edit = QLineEdit()
        self.summary_json_edit = QLineEdit()

        default_summary_json = resource_path(
            "backend/electrode_twin/latent_dataset/dataset_summary.json"
        )

        self.summary_json_edit.setText(
            default_summary_json
        )
        self.out_dir2_edit = QLineEdit()

        btn_local = QPushButton("选择local_conditions_json")
        btn_summary = QPushButton("选择summary_json")
        btn_out2 = QPushButton("选择输出目录")

        btn_local.clicked.connect(
            self.select_local_json
        )

        btn_summary.clicked.connect(
            self.select_summary_json
        )

        btn_out2.clicked.connect(
            self.select_out_dir2
        )

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("local json"))
        row1.addWidget(self.local_json_edit)
        row1.addWidget(btn_local)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("summary json"))
        row2.addWidget(self.summary_json_edit)
        row2.addWidget(btn_summary)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("输出目录"))
        row3.addWidget(self.out_dir2_edit)
        row3.addWidget(btn_out2)

        file_layout.addLayout(row1)
        file_layout.addLayout(row2)
        file_layout.addLayout(row3)

        layout.addWidget(file_group)

        # =====================================================
        # params
        # =====================================================
        param_group = QGroupBox("生成参数")

        param_layout = QVBoxLayout(param_group)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        self.device_combo.setCurrentText("cuda")

        self.patch_size2_spin = QSpinBox()
        self.patch_size2_spin.setRange(32, 512)
        self.patch_size2_spin.setValue(128)

        self.overlap2_spin = QSpinBox()
        self.overlap2_spin.setRange(0, 256)
        self.overlap2_spin.setValue(32)

        self.num_samples_spin = QSpinBox()
        self.num_samples_spin.setRange(1, 2048)
        self.num_samples_spin.setValue(64)

        self.adaptive_checkbox = QCheckBox()
        self.adaptive_checkbox.setChecked(True)

        self.threshold_iter_spin = QSpinBox()
        self.threshold_iter_spin.setRange(1, 1000)
        self.threshold_iter_spin.setValue(25)

        self.threshold_tol_spin = QDoubleSpinBox()
        self.threshold_tol_spin.setDecimals(8)
        self.threshold_tol_spin.setValue(1e-4)

        self.topology_spin = QDoubleSpinBox()
        self.topology_spin.setValue(1.0)

        self.topk_spin = QSpinBox()
        self.topk_spin.setValue(3)

        self.warn_checkbox = QCheckBox()
        self.warn_checkbox.setChecked(True)

        self.clip_checkbox = QCheckBox()
        self.clip_checkbox.setChecked(False)

        self.create_form_row(
            "device",
            self.device_combo,
            param_layout
        )

        self.create_form_row(
            "patch_size",
            self.patch_size2_spin,
            param_layout
        )

        self.create_form_row(
            "overlap",
            self.overlap2_spin,
            param_layout
        )

        self.create_form_row(
            "num_samples_per_patch",
            self.num_samples_spin,
            param_layout
        )

        self.create_form_row(
            "use_adaptive_threshold",
            self.adaptive_checkbox,
            param_layout
        )

        self.create_form_row(
            "adaptive_threshold_max_iters",
            self.threshold_iter_spin,
            param_layout
        )

        self.create_form_row(
            "adaptive_threshold_tol",
            self.threshold_tol_spin,
            param_layout
        )

        self.create_form_row(
            "topology_penalty_weight",
            self.topology_spin,
            param_layout
        )

        self.create_form_row(
            "exact_eval_topk_per_candidate",
            self.topk_spin,
            param_layout
        )

        self.create_form_row(
            "warn_if_target_ood",
            self.warn_checkbox,
            param_layout
        )

        self.create_form_row(
            "clip_condition_to_train_range",
            self.clip_checkbox,
            param_layout
        )

        layout.addWidget(param_group)

        btn_layout = QHBoxLayout()

        btn = QPushButton(
            "开始执行Stage5-2：生成224³大体积"
        )

        btn.setMinimumHeight(45)

        btn.setStyleSheet("""
            QPushButton{
                background:#4f8cff;
                color:white;
                border-radius:8px;
                font-size:15px;
                font-weight:bold;
            }

            QPushButton:hover{
                background:#6aa1ff;
            }
        """)

        btn.clicked.connect(
            self.start_large_volume_task
        )

        # ============================================
        # reset
        # ============================================

        btn_reset = QPushButton("恢复默认参数")

        btn_reset.setMinimumHeight(45)

        btn_reset.setStyleSheet("""
            QPushButton{
                background:#666;
                color:white;
                border-radius:8px;
                font-size:15px;
                font-weight:bold;
            }

            QPushButton:hover{
                background:#888;
            }
        """)

        btn_reset.clicked.connect(
            self.reset_large_volume_params
        )

        btn_layout.addWidget(btn)
        btn_layout.addWidget(btn_reset)

        layout.addLayout(btn_layout)

    # =========================================================
    # model api
    # =========================================================
    def load_models(self):

        try:

            response = requests.get(
                f"{API_BASE}/models/versions"
            )

            data = response.json()

            vae_models = data.get(
                "vae_models",
                []
            )

            ldm_models = data.get(
                "ldm_models",
                []
            )

            self.vae_combo.clear()
            self.ldm_combo.clear()

            for model in vae_models:

                text = (
                    f"{model['file_name']} "
                    f"({model['create_time']})"
                )

                self.vae_combo.addItem(
                    text,
                    model["full_path"]
                )

            for model in ldm_models:

                text = (
                    f"{model['file_name']} "
                    f"({model['create_time']})"
                )

                self.ldm_combo.addItem(
                    text,
                    model["full_path"]
                )

            if vae_models:
                self.vae_combo.setCurrentIndex(0)

            if ldm_models:
                self.ldm_combo.setCurrentIndex(0)

            self.update_model_label()

        except Exception as e:

            QMessageBox.warning(
                self,
                "错误",
                str(e)
            )

    # =========================================================
    # model changed
    # =========================================================
    def on_vae_changed(self):

        self.selected_vae_path = \
            self.vae_combo.currentData()

        self.update_model_label()

    def on_ldm_changed(self):

        self.selected_ldm_path = \
            self.ldm_combo.currentData()

        self.update_model_label()

    def update_model_label(self):

        vae_name = self.vae_combo.currentText()
        ldm_name = self.ldm_combo.currentText()

        self.model_path_label.setText(
            f"VAE: {vae_name}\n"
            f"LDM: {ldm_name}"
        )

    # =========================================================
    # select file
    # =========================================================
    def select_real_volume(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择真实体积",
            "",
            "NumPy (*.npy)"
        )

        if path:
            self.real_volume_edit.setText(path)

    def select_out_dir1(self):

        path = QFileDialog.getExistingDirectory(
            self,
            "选择输出目录"
        )

        if path:
            self.out_dir1_edit.setText(path)

    def select_local_json(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择local json",
            "",
            "JSON (*.json)"
        )

        if path:
            self.local_json_edit.setText(path)

    def select_summary_json(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 summary json",
            "",
            "JSON Files (*.json)"
        )

        if file_path:
            self.summary_json_edit.setText(file_path)

    def select_out_dir2(self):

        path = QFileDialog.getExistingDirectory(
            self,
            "选择输出目录"
        )

        if path:
            self.out_dir2_edit.setText(path)

    # =========================================================
    # start task1
    # =========================================================
    def start_local_condition_task(self):
        # ============================================
        # check
        # ============================================

        if not self.real_volume_edit.text().strip():
            QMessageBox.warning(
                self,
                "错误",
                "请选择真实体积文件"
            )

            return

        if not self.out_dir1_edit.text().strip():
            QMessageBox.warning(
                self,
                "错误",
                "请选择输出目录"
            )

            return

        payload = {

            "real_volume_path":
                self.real_volume_edit.text(),

            "out_dir":
                self.out_dir1_edit.text(),

            "crop_start_y":
                self.crop_y_spin.value(),

            "crop_start_z":
                self.crop_z_spin.value(),

            "crop_start_x":
                self.crop_x_spin.value(),

            "large_vol_size":
                self.large_vol_spin.value(),

            "patch_size":
                self.patch_size1_spin.value(),

            "overlap":
                self.overlap1_spin.value(),

            "pore_value":
                self.pore1_spin.value(),

            "solid_value":
                self.solid1_spin.value(),

            "voxel_size_y":
                self.voxel_y1_spin.value(),

            "voxel_size_z":
                self.voxel_z1_spin.value(),

            "voxel_size_x":
                self.voxel_x1_spin.value(),

            "remove_small_pore_components_flag":
                self.remove_small_checkbox.isChecked(),

            "min_pore_component_size":
                self.min_pore_spin.value(),

            "tau_nonperc_value":
                self.tau_nonperc_spin.value(),

            "suppress_taufactor_output":
                self.suppress_checkbox.isChecked(),
        }

        result = create_task(
            "/stage5/local-conditions-generate",
            payload
        )

        # 提交成功弹窗
        msg = QMessageBox(self)

        msg.setWindowTitle("任务已提交")

        msg.setText(
            f"Stage5-1 任务已提交\n{result}！"
        )

        msg.setInformativeText(
            "任务正在后台运行，请前往「历史任务中心」查看进度和结果。"
        )

        msg.setStandardButtons(
            QMessageBox.Ok | QMessageBox.Cancel
        )

        msg.setDefaultButton(
            QMessageBox.Ok
        )
        # 添加按钮
        go_btn = msg.addButton(
            "前往任务中心",
            QMessageBox.ActionRole
        )
        ret = msg.exec_()
        # 跳转任务中心
        if (
                ret == QMessageBox.Ok
                or msg.clickedButton() == go_btn
        ):
            self.main_window.menu.setCurrentRow(6)

    # =========================================================
    # start task2
    # =========================================================
    def start_large_volume_task(self):

        # ============================================
        # check
        # ============================================

        if not self.local_json_edit.text().strip():
            QMessageBox.warning(
                self,
                "错误",
                "请选择local_conditions_json"
            )

            return

        if not self.summary_json_edit.text().strip():
            QMessageBox.warning(
                self,
                "错误",
                "请选择summary_json"
            )

            return

        if not self.out_dir2_edit.text().strip():
            QMessageBox.warning(
                self,
                "错误",
                "请选择输出目录"
            )

            return

        payload = {

            "local_conditions_json":
                self.local_json_edit.text(),

            "summary_json_path":
                self.summary_json_edit.text(),

            "ldm_ckpt_path":
                self.selected_ldm_path,

            "vae_ckpt_path":
                self.selected_vae_path,

            "out_dir":
                self.out_dir2_edit.text(),

            "device":
                self.device_combo.currentText(),

            "patch_size":
                self.patch_size2_spin.value(),

            "overlap":
                self.overlap2_spin.value(),

            "num_samples_per_patch":
                self.num_samples_spin.value(),

            "use_adaptive_threshold_for_porosity":
                self.adaptive_checkbox.isChecked(),

            "adaptive_threshold_max_iters":
                self.threshold_iter_spin.value(),

            "adaptive_threshold_tol":
                self.threshold_tol_spin.value(),

            "topology_penalty_weight":
                self.topology_spin.value(),

            "exact_eval_topk_per_candidate":
                self.topk_spin.value(),

            "warn_if_target_ood":
                self.warn_checkbox.isChecked(),

            "clip_normalized_condition_to_train_range":
                self.clip_checkbox.isChecked(),
        }

        result = create_task(
            "/stage5/large-volume-generate",
            payload
        )

        # 提交成功弹窗
        msg = QMessageBox(self)

        msg.setWindowTitle("任务已提交")

        msg.setText(
            f"Stage5-1 任务已提交\n{result}！"
        )

        msg.setInformativeText(
            "任务正在后台运行，请前往「历史任务中心」查看进度和结果。"
        )

        msg.setStandardButtons(
            QMessageBox.Ok | QMessageBox.Cancel
        )

        msg.setDefaultButton(
            QMessageBox.Ok
        )
        # 添加按钮
        go_btn = msg.addButton(
            "前往任务中心",
            QMessageBox.ActionRole
        )
        ret = msg.exec_()
        # 跳转任务中心
        if (
                ret == QMessageBox.Ok
                or msg.clickedButton() == go_btn
        ):
            self.main_window.menu.setCurrentRow(6)

    def reset_local_condition_params(self):

        # file
        self.real_volume_edit.clear()
        self.out_dir1_edit.clear()

        # crop
        self.crop_y_spin.setValue(0)
        self.crop_z_spin.setValue(100)
        self.crop_x_spin.setValue(0)

        # volume
        self.large_vol_spin.setValue(224)

        # patch
        self.patch_size1_spin.setValue(128)
        self.overlap1_spin.setValue(32)

        # phase
        self.pore1_spin.setValue(0)
        self.solid1_spin.setValue(1)

        # voxel
        self.voxel_y1_spin.setValue(0.0315)
        self.voxel_z1_spin.setValue(0.02791)
        self.voxel_x1_spin.setValue(0.02791)

        # clean
        self.remove_small_checkbox.setChecked(True)

        self.min_pore_spin.setValue(10)

        # tau
        self.tau_nonperc_spin.setValue(1e6)

        self.suppress_checkbox.setChecked(True)

        QMessageBox.information(
            self,
            "提示",
            "Stage5-1 参数已恢复默认"
        )

    def reset_large_volume_params(self):

        # file
        self.local_json_edit.clear()

        self.summary_json_edit.setText(
            self.default_summary_json
        )

        self.out_dir2_edit.clear()

        # model
        if self.vae_combo.count() > 0:
            self.vae_combo.setCurrentIndex(0)

        if self.ldm_combo.count() > 0:
            self.ldm_combo.setCurrentIndex(0)

        # device
        self.device_combo.setCurrentText("cuda")

        # patch
        self.patch_size2_spin.setValue(128)
        self.overlap2_spin.setValue(32)

        # sample
        self.num_samples_spin.setValue(64)

        # threshold
        self.adaptive_checkbox.setChecked(True)

        self.threshold_iter_spin.setValue(25)

        self.threshold_tol_spin.setValue(1e-4)

        # topology
        self.topology_spin.setValue(1.0)

        self.topk_spin.setValue(3)

        # flags
        self.warn_checkbox.setChecked(True)

        self.clip_checkbox.setChecked(False)

        QMessageBox.information(
            self,
            "提示",
            "Stage5-2 参数已恢复默认"
        )
