# -*- coding: utf-8 -*-
import json
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
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QLineEdit,
    QSpinBox,
    QCheckBox,
    QTextEdit,
    QScrollArea,
    QGridLayout,
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

        self.manual_patch_widgets = []

        self.default_summary_json = resource_path(
            "backend/electrode_twin/latent_dataset/dataset_summary.json"
        )

        self.default_metrics_csv = resource_path(
            "backend/electrode_twin/latent_dataset/train_metrics_table.csv"
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
            "Stage5 生成特定体积"
        )

        title.setStyleSheet("""
        font-size:24px;
        font-weight:bold;
        color:white;
        padding:10px;
        """)

        root_layout.addWidget(title)

        # =====================================================
        # scroll
        # =====================================================
        scroll = QScrollArea()

        scroll.setWidgetResizable(True)

        root_layout.addWidget(scroll)

        container = QWidget()

        scroll.setWidget(container)

        layout = QVBoxLayout(container)

        # =====================================================
        # tip
        # =====================================================
        tip = QLabel(
            "根据用户指定 porosity / tau_z 条件，自动生成 224³ AM-pore twin volume"
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
        # model
        # =====================================================
        self.build_model_group(layout)

        # =====================================================
        # path
        # =====================================================
        self.build_path_group(layout)

        # =====================================================
        # condition
        # =====================================================
        self.build_condition_group(layout)

        # =====================================================
        # generation
        # =====================================================
        self.build_generation_group(layout)

        # =====================================================
        # threshold
        # =====================================================
        self.build_threshold_group(layout)

        # =====================================================
        # score
        # =====================================================
        self.build_score_group(layout)

        # =====================================================
        # slice
        # =====================================================
        self.build_slice_group(layout)

        # =====================================================
        # btn
        # =====================================================
        btn_layout = QHBoxLayout()

        btn_run = QPushButton(
            "开始执行 Stage5 生成特定体积"
        )

        btn_run.setMinimumHeight(50)

        btn_run.setStyleSheet("""
            QPushButton{
                background:#4f8cff;
                color:white;
                border-radius:8px;
                font-size:16px;
                font-weight:bold;
            }

            QPushButton:hover{
                background:#6aa1ff;
            }
        """)

        btn_run.clicked.connect(
            self.start_task
        )

        btn_reset = QPushButton(
            "恢复默认参数"
        )

        btn_reset.setMinimumHeight(50)

        btn_reset.setStyleSheet("""
            QPushButton{
                background:#666;
                color:white;
                border-radius:8px;
                font-size:16px;
                font-weight:bold;
            }

            QPushButton:hover{
                background:#888;
            }
        """)

        btn_reset.clicked.connect(
            self.reset_params
        )

        btn_layout.addWidget(btn_run)
        btn_layout.addWidget(btn_reset)

        layout.addLayout(btn_layout)

        layout.addStretch()

    # =========================================================
    # helper
    # =========================================================
    def create_form_row(
            self,
            label_text,
            widget,
            layout,
            label_width=260,
    ):

        row = QHBoxLayout()

        label = QLabel(label_text)

        label.setFixedWidth(label_width)

        row.addWidget(label)
        row.addWidget(widget)

        row.addStretch()

        layout.addLayout(row)

    # =========================================================
    # model
    # =========================================================
    def build_model_group(self, layout):

        group = QGroupBox("模型选择")

        g_layout = QVBoxLayout(group)

        self.vae_combo = QComboBox()
        self.ldm_combo = QComboBox()

        self.model_label = QLabel("未选择模型")

        self.model_label.setStyleSheet("""
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
        row1.addWidget(QLabel("VAE"))
        row1.addWidget(self.vae_combo)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("LDM"))
        row2.addWidget(self.ldm_combo)

        g_layout.addLayout(row1)
        g_layout.addLayout(row2)
        g_layout.addWidget(self.model_label)

        layout.addWidget(group)

    # =========================================================
    # path
    # =========================================================
    def build_path_group(self, layout):

        group = QGroupBox("路径参数")

        g_layout = QVBoxLayout(group)

        self.summary_json_edit = QLineEdit()
        self.summary_json_edit.setText(
            self.default_summary_json
        )

        self.metrics_csv_edit = QLineEdit()
        self.metrics_csv_edit.setText(
            self.default_metrics_csv
        )

        self.out_dir_edit = QLineEdit()

        btn_summary = QPushButton("选择")
        btn_metrics = QPushButton("选择")
        btn_out = QPushButton("选择")

        btn_summary.clicked.connect(
            self.select_summary_json
        )

        btn_metrics.clicked.connect(
            self.select_metrics_csv
        )

        btn_out.clicked.connect(
            self.select_out_dir
        )

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("summary_json"))
        row1.addWidget(self.summary_json_edit)
        row1.addWidget(btn_summary)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("train_metrics_table"))
        row2.addWidget(self.metrics_csv_edit)
        row2.addWidget(btn_metrics)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("输出目录"))
        row3.addWidget(self.out_dir_edit)
        row3.addWidget(btn_out)

        g_layout.addLayout(row1)
        g_layout.addLayout(row2)
        g_layout.addLayout(row3)

        layout.addWidget(group)

    # =========================================================
    # condition
    # =========================================================
    def build_condition_group(self, layout):

        group = QGroupBox("条件输入")

        g_layout = QVBoxLayout(group)

        # =====================================================
        # grid
        # =====================================================
        self.grid_y_spin = QSpinBox()
        self.grid_z_spin = QSpinBox()
        self.grid_x_spin = QSpinBox()

        self.grid_y_spin.setValue(2)
        self.grid_z_spin.setValue(2)
        self.grid_x_spin.setValue(2)

        grid_row = QHBoxLayout()

        grid_row.addWidget(QLabel("GRID_SHAPE"))

        grid_row.addWidget(QLabel("Y"))
        grid_row.addWidget(self.grid_y_spin)

        grid_row.addWidget(QLabel("Z"))
        grid_row.addWidget(self.grid_z_spin)

        grid_row.addWidget(QLabel("X"))
        grid_row.addWidget(self.grid_x_spin)

        btn_refresh = QPushButton("刷新 Manual Patch 表格")

        btn_refresh.clicked.connect(
            self.build_manual_patch_editor
        )

        grid_row.addWidget(btn_refresh)

        g_layout.addLayout(grid_row)

        # =====================================================
        # mode
        # =====================================================
        self.condition_mode_combo = QComboBox()

        self.condition_mode_combo.addItems([
            "uniform_porosity",
            "manual_user",
        ])

        self.condition_mode_combo.currentTextChanged.connect(
            self.on_condition_mode_changed
        )

        self.create_form_row(
            "CONDITION_INPUT_MODE",
            self.condition_mode_combo,
            g_layout
        )

        # =====================================================
        # uniform
        # =====================================================
        self.uniform_group = QGroupBox(
            "uniform_porosity 模式"
        )

        uniform_layout = QVBoxLayout(
            self.uniform_group
        )

        self.target_porosity_spin = QDoubleSpinBox()
        self.target_porosity_spin.setDecimals(4)
        self.target_porosity_spin.setRange(0, 1)
        self.target_porosity_spin.setValue(0.30)

        self.target_tau_spin = QDoubleSpinBox()
        self.target_tau_spin.setDecimals(4)
        self.target_tau_spin.setRange(0, 9999)
        self.target_tau_spin.setValue(3.30)

        self.create_form_row(
            "TARGET_PATCH_POROSITY",
            self.target_porosity_spin,
            uniform_layout
        )

        self.create_form_row(
            "TARGET_PATCH_TAU_Z",
            self.target_tau_spin,
            uniform_layout
        )

        g_layout.addWidget(
            self.uniform_group
        )

        # =====================================================
        # manual
        # =====================================================
        self.manual_group = QGroupBox(
            "manual_user 模式"
        )

        manual_layout = QVBoxLayout(
            self.manual_group
        )

        tip = QLabel(
            "请为每个 patch 填写 porosity 与 tau_z"
        )

        tip.setStyleSheet("""
            color:#4f8cff;
            padding:6px;
        """)

        manual_layout.addWidget(tip)

        self.manual_grid_layout = QGridLayout()

        manual_layout.addLayout(
            self.manual_grid_layout
        )

        g_layout.addWidget(
            self.manual_group
        )

        layout.addWidget(group)

        self.build_manual_patch_editor()

        self.on_condition_mode_changed()

    # =========================================================
    # generation
    # =========================================================
    def build_generation_group(self, layout):

        group = QGroupBox("生成参数")

        g_layout = QVBoxLayout(group)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])

        self.patch_size_spin = QSpinBox()
        self.patch_size_spin.setValue(128)

        self.overlap_spin = QSpinBox()
        self.overlap_spin.setValue(32)

        self.num_samples_spin = QSpinBox()
        self.num_samples_spin.setValue(32)

        self.auto_surface_combo = QComboBox()
        self.auto_surface_combo.addItems([
            "nearest_training_porosity_tau"
        ])

        self.auto_deff_combo = QComboBox()
        self.auto_deff_combo.addItems([
            "porosity_over_tau"
        ])

        self.pore_spin = QSpinBox()
        self.pore_spin.setValue(0)

        self.solid_spin = QSpinBox()
        self.solid_spin.setValue(1)

        self.voxel_y_spin = QDoubleSpinBox()
        self.voxel_y_spin.setDecimals(6)
        self.voxel_y_spin.setValue(0.0315)

        self.voxel_z_spin = QDoubleSpinBox()
        self.voxel_z_spin.setDecimals(6)
        self.voxel_z_spin.setValue(0.02791)

        self.voxel_x_spin = QDoubleSpinBox()
        self.voxel_x_spin.setDecimals(6)
        self.voxel_x_spin.setValue(0.02791)

        self.remove_small_checkbox = QCheckBox()
        self.remove_small_checkbox.setChecked(True)

        self.min_pore_spin = QSpinBox()
        self.min_pore_spin.setValue(10)

        self.tau_nonperc_spin = QDoubleSpinBox()
        self.tau_nonperc_spin.setMaximum(1e9)
        self.tau_nonperc_spin.setValue(1e6)

        self.suppress_checkbox = QCheckBox()
        self.suppress_checkbox.setChecked(True)

        self.create_form_row(
            "DEVICE",
            self.device_combo,
            g_layout
        )

        self.create_form_row(
            "PATCH_SIZE",
            self.patch_size_spin,
            g_layout
        )

        self.create_form_row(
            "OVERLAP",
            self.overlap_spin,
            g_layout
        )

        self.create_form_row(
            "NUM_SAMPLES_PER_PATCH",
            self.num_samples_spin,
            g_layout
        )

        self.create_form_row(
            "AUTO_SURFACE_MODE",
            self.auto_surface_combo,
            g_layout
        )

        self.create_form_row(
            "AUTO_DEFF_MODE",
            self.auto_deff_combo,
            g_layout
        )

        self.create_form_row(
            "PORE_VALUE",
            self.pore_spin,
            g_layout
        )

        self.create_form_row(
            "SOLID_VALUE",
            self.solid_spin,
            g_layout
        )

        self.create_form_row(
            "VOXEL_SIZE_Y",
            self.voxel_y_spin,
            g_layout
        )

        self.create_form_row(
            "VOXEL_SIZE_Z",
            self.voxel_z_spin,
            g_layout
        )

        self.create_form_row(
            "VOXEL_SIZE_X",
            self.voxel_x_spin,
            g_layout
        )

        self.create_form_row(
            "REMOVE_SMALL_PORE_COMPONENTS",
            self.remove_small_checkbox,
            g_layout
        )

        self.create_form_row(
            "MIN_PORE_COMPONENT_SIZE",
            self.min_pore_spin,
            g_layout
        )

        self.create_form_row(
            "TAU_NONPERC_VALUE",
            self.tau_nonperc_spin,
            g_layout
        )

        self.create_form_row(
            "SUPPRESS_TAUFACTOR_OUTPUT",
            self.suppress_checkbox,
            g_layout
        )

        layout.addWidget(group)

    # =========================================================
    # threshold
    # =========================================================
    def build_threshold_group(self, layout):

        group = QGroupBox("Threshold / Postprocess")

        g_layout = QVBoxLayout(group)

        self.adaptive_checkbox = QCheckBox()
        self.adaptive_checkbox.setChecked(True)

        self.adaptive_iter_spin = QSpinBox()
        self.adaptive_iter_spin.setValue(25)

        self.adaptive_tol_spin = QDoubleSpinBox()
        self.adaptive_tol_spin.setDecimals(8)
        self.adaptive_tol_spin.setValue(1e-4)

        self.threshold_offsets_edit = QTextEdit()
        self.threshold_offsets_edit.setFixedHeight(80)

        self.threshold_offsets_edit.setPlainText(
            json.dumps(
                [-0.04, -0.03, -0.02, -0.01,
                 0.0,
                 0.01, 0.02, 0.03, 0.04],
                indent=2
            )
        )

        self.postprocess_edit = QTextEdit()

        self.postprocess_edit.setFixedHeight(180)

        self.postprocess_edit.setPlainText(
            json.dumps([
                {
                    "name": "raw",
                    "mode": "none",
                },
                {
                    "name": "erode1",
                    "mode": "erode",
                    "iters": 1,
                },
                {
                    "name": "open1",
                    "mode": "open",
                    "iters": 1,
                },
            ], indent=2)
        )

        self.create_form_row(
            "USE_ADAPTIVE_THRESHOLD_FOR_POROSITY",
            self.adaptive_checkbox,
            g_layout
        )

        self.create_form_row(
            "ADAPTIVE_THRESHOLD_MAX_ITERS",
            self.adaptive_iter_spin,
            g_layout
        )

        self.create_form_row(
            "ADAPTIVE_THRESHOLD_TOL",
            self.adaptive_tol_spin,
            g_layout
        )

        g_layout.addWidget(
            QLabel("THRESHOLD_OFFSETS")
        )

        g_layout.addWidget(
            self.threshold_offsets_edit
        )

        g_layout.addWidget(
            QLabel("POSTPROCESS_CONFIGS")
        )

        g_layout.addWidget(
            self.postprocess_edit
        )

        layout.addWidget(group)

    # =========================================================
    # score
    # =========================================================
    def build_score_group(self, layout):

        group = QGroupBox("Scoring / OOD")

        g_layout = QVBoxLayout(group)

        self.cheap_weights_edit = QTextEdit()
        self.cheap_weights_edit.setFixedHeight(80)

        self.cheap_weights_edit.setPlainText(
            json.dumps({
                "porosity": 4.0,
            }, indent=2)
        )

        self.final_weights_edit = QTextEdit()
        self.final_weights_edit.setFixedHeight(120)

        self.final_weights_edit.setPlainText(
            json.dumps({
                "porosity": 4.0,
                "tau_z": 5.0,
                "deff_z": 0.5,
            }, indent=2)
        )

        self.use_std_checkbox = QCheckBox()
        self.use_std_checkbox.setChecked(True)

        self.topology_spin = QDoubleSpinBox()
        self.topology_spin.setValue(1.0)

        self.min_solid_spin = QSpinBox()
        self.min_solid_spin.setValue(10)

        self.topk_spin = QSpinBox()
        self.topk_spin.setValue(3)

        self.warn_checkbox = QCheckBox()
        self.warn_checkbox.setChecked(True)

        self.clip_checkbox = QCheckBox()
        self.clip_checkbox.setChecked(False)

        g_layout.addWidget(
            QLabel("CHEAP_ERROR_WEIGHTS")
        )

        g_layout.addWidget(
            self.cheap_weights_edit
        )

        g_layout.addWidget(
            QLabel("FINAL_ERROR_WEIGHTS")
        )

        g_layout.addWidget(
            self.final_weights_edit
        )

        self.create_form_row(
            "USE_STD_NORMALIZED_ERROR",
            self.use_std_checkbox,
            g_layout
        )

        self.create_form_row(
            "TOPOLOGY_PENALTY_WEIGHT",
            self.topology_spin,
            g_layout
        )

        self.create_form_row(
            "MIN_SOLID_COMPONENT_COUNT_SOFT",
            self.min_solid_spin,
            g_layout
        )

        self.create_form_row(
            "EXACT_EVAL_TOPK_PER_CANDIDATE",
            self.topk_spin,
            g_layout
        )

        self.create_form_row(
            "WARN_IF_TARGET_OOD",
            self.warn_checkbox,
            g_layout
        )

        self.create_form_row(
            "CLIP_NORMALIZED_CONDITION_TO_TRAIN_RANGE",
            self.clip_checkbox,
            g_layout
        )

        layout.addWidget(group)

    # =========================================================
    # slice
    # =========================================================
    def build_slice_group(self, layout):

        group = QGroupBox("Slice Visualization")

        g_layout = QVBoxLayout(group)

        self.save_slice_checkbox = QCheckBox()
        self.save_slice_checkbox.setChecked(True)

        self.slice_style_combo = QComboBox()

        self.slice_style_combo.addItems([
            "black_yellow",
            "white_blue",
        ])

        self.slice_axis_checkbox = QCheckBox()
        self.slice_axis_checkbox.setChecked(False)

        self.slice_dpi_spin = QSpinBox()
        self.slice_dpi_spin.setValue(200)

        self.create_form_row(
            "SAVE_ALL_Y_ZX_SLICE_PNG",
            self.save_slice_checkbox,
            g_layout
        )

        self.create_form_row(
            "SLICE_COLOR_STYLE",
            self.slice_style_combo,
            g_layout
        )

        self.create_form_row(
            "SLICE_SHOW_AXIS",
            self.slice_axis_checkbox,
            g_layout
        )

        self.create_form_row(
            "SLICE_DPI",
            self.slice_dpi_spin,
            g_layout
        )

        layout.addWidget(group)

    # =========================================================
    # manual patch
    # =========================================================
    def build_manual_patch_editor(self):

        while self.manual_grid_layout.count():

            item = self.manual_grid_layout.takeAt(0)

            widget = item.widget()

            if widget:
                widget.deleteLater()

        self.manual_patch_widgets = []

        gy = self.grid_y_spin.value()
        gz = self.grid_z_spin.value()
        gx = self.grid_x_spin.value()

        row = 0

        for iy in range(gy):
            for iz in range(gz):
                for ix in range(gx):

                    label = QLabel(
                        f"Patch [{iy}, {iz}, {ix}]"
                    )

                    porosity_spin = QDoubleSpinBox()
                    porosity_spin.setDecimals(4)
                    porosity_spin.setRange(0, 1)
                    porosity_spin.setValue(0.30)

                    tau_spin = QDoubleSpinBox()
                    tau_spin.setDecimals(4)
                    tau_spin.setRange(0, 9999)
                    tau_spin.setValue(3.30)

                    self.manual_grid_layout.addWidget(
                        label,
                        row,
                        0
                    )

                    self.manual_grid_layout.addWidget(
                        QLabel("porosity"),
                        row,
                        1
                    )

                    self.manual_grid_layout.addWidget(
                        porosity_spin,
                        row,
                        2
                    )

                    self.manual_grid_layout.addWidget(
                        QLabel("tau_z"),
                        row,
                        3
                    )

                    self.manual_grid_layout.addWidget(
                        tau_spin,
                        row,
                        4
                    )

                    self.manual_patch_widgets.append({
                        "grid_index": [iy, iz, ix],
                        "porosity": porosity_spin,
                        "tau_z": tau_spin,
                    })

                    row += 1

    # =========================================================
    # condition mode
    # =========================================================
    def on_condition_mode_changed(self):

        mode = self.condition_mode_combo.currentText()

        if mode == "uniform_porosity":

            self.uniform_group.setVisible(True)
            self.manual_group.setVisible(False)

        else:

            self.uniform_group.setVisible(False)
            self.manual_group.setVisible(True)

    # =========================================================
    # model
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

        self.model_label.setText(
            f"VAE: {vae_name}\n"
            f"LDM: {ldm_name}"
        )

    # =========================================================
    # select
    # =========================================================
    def select_summary_json(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 summary json",
            "",
            "JSON (*.json)"
        )

        if path:
            self.summary_json_edit.setText(path)

    def select_metrics_csv(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 metrics csv",
            "",
            "CSV (*.csv)"
        )

        if path:
            self.metrics_csv_edit.setText(path)

    def select_out_dir(self):

        path = QFileDialog.getExistingDirectory(
            self,
            "选择输出目录"
        )

        if path:
            self.out_dir_edit.setText(path)

    # =========================================================
    # payload
    # =========================================================
    def build_manual_conditions(self):

        result = []

        for item in self.manual_patch_widgets:

            result.append({
                "grid_index":
                    item["grid_index"],

                "porosity":
                    item["porosity"].value(),

                "tau_z":
                    item["tau_z"].value(),
            })

        return result

    # =========================================================
    # task
    # =========================================================
    def start_task(self):

        if not self.out_dir_edit.text().strip():

            QMessageBox.warning(
                self,
                "错误",
                "请选择输出目录"
            )

            return

        try:

            threshold_offsets = json.loads(
                self.threshold_offsets_edit.toPlainText()
            )

            postprocess_configs = json.loads(
                self.postprocess_edit.toPlainText()
            )

            cheap_weights = json.loads(
                self.cheap_weights_edit.toPlainText()
            )

            final_weights = json.loads(
                self.final_weights_edit.toPlainText()
            )

        except Exception as e:

            QMessageBox.warning(
                self,
                "JSON错误",
                str(e)
            )

            return

        payload = {

            # =================================================
            # path
            # =================================================
            "summary_json_path":
                self.summary_json_edit.text(),

            "train_metrics_table_path":
                self.metrics_csv_edit.text(),

            "ldm_ckpt_path":
                self.selected_ldm_path,

            "vae_ckpt_path":
                self.selected_vae_path,

            "out_dir":
                self.out_dir_edit.text(),

            # =================================================
            # device
            # =================================================
            "device":
                self.device_combo.currentText(),

            # =================================================
            # patch
            # =================================================
            "patch_size":
                self.patch_size_spin.value(),

            "overlap":
                self.overlap_spin.value(),

            "grid_shape": [
                self.grid_y_spin.value(),
                self.grid_z_spin.value(),
                self.grid_x_spin.value(),
            ],

            # =================================================
            # condition
            # =================================================
            "condition_input_mode":
                self.condition_mode_combo.currentText(),

            "target_patch_porosity":
                self.target_porosity_spin.value(),

            "target_patch_tau_z":
                self.target_tau_spin.value(),

            "manual_patch_conditions":
                self.build_manual_conditions(),

            # =================================================
            # auto
            # =================================================
            "auto_surface_mode":
                self.auto_surface_combo.currentText(),

            "auto_deff_mode":
                self.auto_deff_combo.currentText(),

            # =================================================
            # generation
            # =================================================
            "num_samples_per_patch":
                self.num_samples_spin.value(),

            "pore_value":
                self.pore_spin.value(),

            "solid_value":
                self.solid_spin.value(),

            # =================================================
            # voxel
            # =================================================
            "voxel_size_y":
                self.voxel_y_spin.value(),

            "voxel_size_z":
                self.voxel_z_spin.value(),

            "voxel_size_x":
                self.voxel_x_spin.value(),

            # =================================================
            # clean
            # =================================================
            "remove_small_pore_components":
                self.remove_small_checkbox.isChecked(),

            "min_pore_component_size":
                self.min_pore_spin.value(),

            # =================================================
            # threshold
            # =================================================
            "use_adaptive_threshold_for_porosity":
                self.adaptive_checkbox.isChecked(),

            "adaptive_threshold_max_iters":
                self.adaptive_iter_spin.value(),

            "adaptive_threshold_tol":
                self.adaptive_tol_spin.value(),

            "threshold_offsets":
                threshold_offsets,

            # =================================================
            # score
            # =================================================
            "cheap_error_weights":
                cheap_weights,

            "final_error_weights":
                final_weights,

            "use_std_normalized_error":
                self.use_std_checkbox.isChecked(),

            "topology_penalty_weight":
                self.topology_spin.value(),

            "min_solid_component_count_soft":
                self.min_solid_spin.value(),

            "exact_eval_topk_per_candidate":
                self.topk_spin.value(),

            # =================================================
            # ood
            # =================================================
            "warn_if_target_ood":
                self.warn_checkbox.isChecked(),

            "clip_normalized_condition_to_train_range":
                self.clip_checkbox.isChecked(),

            # =================================================
            # tau
            # =================================================
            "tau_nonperc_value":
                self.tau_nonperc_spin.value(),

            "suppress_taufactor_output":
                self.suppress_checkbox.isChecked(),

            # =================================================
            # slice
            # =================================================
            "save_all_y_zx_slice_png":
                self.save_slice_checkbox.isChecked(),

            "slice_color_style":
                self.slice_style_combo.currentText(),

            "slice_show_axis":
                self.slice_axis_checkbox.isChecked(),

            "slice_dpi":
                self.slice_dpi_spin.value(),
        }

        result = create_task(
            "/stage5/generate-specific-volume",
            payload
        )

        msg = QMessageBox(self)

        msg.setWindowTitle("任务已提交")

        msg.setText(
            f"Stage5 任务已提交\n{result}"
        )

        msg.setInformativeText(
            "任务正在后台运行，请前往历史任务中心查看"
        )

        msg.setStandardButtons(
            QMessageBox.Ok | QMessageBox.Cancel
        )

        go_btn = msg.addButton(
            "前往任务中心",
            QMessageBox.ActionRole
        )

        ret = msg.exec()

        if (
                ret == QMessageBox.Ok
                or msg.clickedButton() == go_btn
        ):
            self.main_window.menu.setCurrentRow(6)

    # =========================================================
    # reset
    # =========================================================
    def reset_params(self):

        self.device_combo.setCurrentText("cuda")

        self.patch_size_spin.setValue(128)

        self.overlap_spin.setValue(32)

        self.grid_y_spin.setValue(2)
        self.grid_z_spin.setValue(2)
        self.grid_x_spin.setValue(2)

        self.condition_mode_combo.setCurrentText(
            "uniform_porosity"
        )

        self.target_porosity_spin.setValue(0.30)

        self.target_tau_spin.setValue(3.30)

        self.num_samples_spin.setValue(32)

        self.remove_small_checkbox.setChecked(True)

        self.min_pore_spin.setValue(10)

        self.adaptive_checkbox.setChecked(True)

        self.adaptive_iter_spin.setValue(25)

        self.adaptive_tol_spin.setValue(1e-4)

        self.topology_spin.setValue(1.0)

        self.min_solid_spin.setValue(10)

        self.topk_spin.setValue(3)

        self.warn_checkbox.setChecked(True)

        self.clip_checkbox.setChecked(False)

        self.save_slice_checkbox.setChecked(True)

        self.slice_style_combo.setCurrentText(
            "black_yellow"
        )

        self.slice_axis_checkbox.setChecked(False)

        self.slice_dpi_spin.setValue(200)

        self.build_manual_patch_editor()

        QMessageBox.information(
            self,
            "提示",
            "参数已恢复默认"
        )