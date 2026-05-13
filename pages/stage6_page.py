# -*- coding: utf-8 -*-
import os
import requests

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
    QFileDialog,
    QTabWidget,
    QGroupBox,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QLineEdit,
    QSplitter,
)

from api_client import (
    create_task,
    query_task,
    API_BASE,
)


class Stage6Page(QWidget):

    def __init__(self):
        super().__init__()

        self.task_id = None
        self.fit_task_id = None

        self.upload_task_id = None
        self.input_file = None

        self.init_ui()

        # ====================================================
        # timer
        # ====================================================
        self.generate_timer = QTimer()

        self.generate_timer.timeout.connect(
            self.refresh_generate_task
        )

        self.fit_timer = QTimer()

        self.fit_timer.timeout.connect(
            self.refresh_fit_task
        )

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

        layout = QHBoxLayout(self.tab_generate)

        splitter = QSplitter(Qt.Horizontal)

        layout.addWidget(splitter)

        # ====================================================
        # LEFT
        # ====================================================
        left = QWidget()

        left_layout = QVBoxLayout(left)

        # ====================================================
        # 文件上传
        # ====================================================
        file_group = QGroupBox("二相结构文件")

        file_layout = QVBoxLayout(file_group)

        self.file_edit = QLineEdit()

        btn_select_file = QPushButton(
            "选择 .npy 文件"
        )

        btn_select_file.clicked.connect(
            self.select_npy_file
        )

        btn_upload = QPushButton(
            "上传文件"
        )

        btn_upload.clicked.connect(
            self.upload_file
        )

        file_layout.addWidget(self.file_edit)

        file_layout.addWidget(btn_select_file)

        file_layout.addWidget(btn_upload)

        left_layout.addWidget(file_group)

        # ====================================================
        # CBD参数
        # ====================================================
        cbd_group = QGroupBox("CBD参数")

        cbd_layout = QVBoxLayout(cbd_group)

        self.target_cbd = QDoubleSpinBox()

        self.target_cbd.setValue(0.05)

        self.target_cbd.setDecimals(4)

        self.w_um = QDoubleSpinBox()

        self.w_um.setValue(0.08)

        self.w_um.setDecimals(4)

        cbd_layout.addWidget(
            QLabel("Target CBD Volume Fraction")
        )

        cbd_layout.addWidget(self.target_cbd)

        cbd_layout.addWidget(
            QLabel("W Parameter")
        )

        cbd_layout.addWidget(self.w_um)

        left_layout.addWidget(cbd_group)

        # ====================================================
        # 相标签
        # ====================================================
        phase_group = QGroupBox("相标签")

        phase_layout = QVBoxLayout(phase_group)

        self.pore_value = QSpinBox()

        self.am_value = QSpinBox()

        self.am_value.setValue(1)

        self.cbd_value = QSpinBox()

        self.cbd_value.setValue(2)

        phase_layout.addWidget(QLabel("Pore Value"))

        phase_layout.addWidget(self.pore_value)

        phase_layout.addWidget(QLabel("AM Value"))

        phase_layout.addWidget(self.am_value)

        phase_layout.addWidget(QLabel("CBD Value"))

        phase_layout.addWidget(self.cbd_value)

        left_layout.addWidget(phase_group)

        # ====================================================
        # voxel
        # ====================================================
        voxel_group = QGroupBox("Voxel Size")

        voxel_layout = QVBoxLayout(voxel_group)

        self.voxel_x = QDoubleSpinBox()

        self.voxel_y = QDoubleSpinBox()

        self.voxel_z = QDoubleSpinBox()

        self.voxel_x.setValue(0.02791)

        self.voxel_y.setValue(0.0315)

        self.voxel_z.setValue(0.02791)

        voxel_layout.addWidget(QLabel("Voxel Size X"))

        voxel_layout.addWidget(self.voxel_x)

        voxel_layout.addWidget(QLabel("Voxel Size Y"))

        voxel_layout.addWidget(self.voxel_y)

        voxel_layout.addWidget(QLabel("Voxel Size Z"))

        voxel_layout.addWidget(self.voxel_z)

        left_layout.addWidget(voxel_group)

        # ====================================================
        # advanced
        # ====================================================
        adv_group = QGroupBox("高级参数")

        adv_layout = QVBoxLayout(adv_group)

        self.max_growth = QDoubleSpinBox()

        self.max_growth.setValue(4.0)

        self.seed = QSpinBox()

        self.seed.setValue(42)

        self.remove_isolated = QCheckBox(
            "Remove Isolated CBD"
        )

        self.remove_isolated.setChecked(True)

        adv_layout.addWidget(
            QLabel("Max Growth Distance")
        )

        adv_layout.addWidget(self.max_growth)

        adv_layout.addWidget(QLabel("Random Seed"))

        adv_layout.addWidget(self.seed)

        adv_layout.addWidget(self.remove_isolated)

        left_layout.addWidget(adv_group)

        # ====================================================
        # submit
        # ====================================================
        btn_generate = QPushButton(
            "开始CBD三相生成"
        )

        btn_generate.setMinimumHeight(50)

        btn_generate.clicked.connect(
            self.start_generate
        )

        left_layout.addWidget(btn_generate)

        left_layout.addStretch()

        # ====================================================
        # RIGHT
        # ====================================================
        right = QWidget()

        right_layout = QVBoxLayout(right)

        self.generate_progress = QProgressBar()

        self.generate_log = QTextEdit()

        self.generate_log.setReadOnly(True)

        right_layout.addWidget(
            QLabel("生成进度")
        )

        right_layout.addWidget(
            self.generate_progress
        )

        right_layout.addWidget(
            QLabel("实时日志")
        )

        right_layout.addWidget(
            self.generate_log
        )

        splitter.addWidget(left)

        splitter.addWidget(right)

        splitter.setSizes([500, 700])

    # ========================================================
    # TAB2
    # ========================================================
    def build_fit_tab(self):

        layout = QHBoxLayout(self.tab_fit)

        splitter = QSplitter(Qt.Horizontal)

        layout.addWidget(splitter)

        # ====================================================
        # LEFT
        # ====================================================
        left = QWidget()

        left_layout = QVBoxLayout(left)

        # ====================================================
        # path
        # ====================================================
        path_group = QGroupBox("输入路径")

        path_layout = QVBoxLayout(path_group)

        self.real_dir_edit = QLineEdit()

        self.out_dir_edit = QLineEdit()

        btn_real = QPushButton(
            "选择真实三相结构目录"
        )

        btn_real.clicked.connect(
            self.select_real_dir
        )

        btn_out = QPushButton(
            "选择输出目录"
        )

        btn_out.clicked.connect(
            self.select_out_dir
        )

        path_layout.addWidget(
            QLabel("真实三相结构目录")
        )

        path_layout.addWidget(
            self.real_dir_edit
        )

        path_layout.addWidget(btn_real)

        path_layout.addSpacing(10)

        path_layout.addWidget(
            QLabel("输出目录")
        )

        path_layout.addWidget(
            self.out_dir_edit
        )

        path_layout.addWidget(btn_out)

        left_layout.addWidget(path_group)

        # ====================================================
        # phase
        # ====================================================
        phase_group = QGroupBox("相标签定义")

        phase_layout = QVBoxLayout(phase_group)

        self.fit_pore = QSpinBox()

        self.fit_am = QSpinBox()

        self.fit_am.setValue(1)

        self.fit_cbd = QSpinBox()

        self.fit_cbd.setValue(2)

        phase_layout.addWidget(QLabel("Pore Value"))

        phase_layout.addWidget(self.fit_pore)

        phase_layout.addWidget(QLabel("AM Value"))

        phase_layout.addWidget(self.fit_am)

        phase_layout.addWidget(QLabel("CBD Value"))

        phase_layout.addWidget(self.fit_cbd)

        left_layout.addWidget(phase_group)

        # ====================================================
        # W扫描
        # ====================================================
        w_group = QGroupBox("W扫描参数")

        w_layout = QVBoxLayout(w_group)

        self.w_min = QDoubleSpinBox()

        self.w_min.setValue(0.02)

        self.w_max = QDoubleSpinBox()

        self.w_max.setValue(0.30)

        self.num_w = QSpinBox()

        self.num_w.setValue(20)

        w_layout.addWidget(QLabel("W Min"))

        w_layout.addWidget(self.w_min)

        w_layout.addWidget(QLabel("W Max"))

        w_layout.addWidget(self.w_max)

        w_layout.addWidget(QLabel("Num W"))

        w_layout.addWidget(self.num_w)

        left_layout.addWidget(w_group)

        # ====================================================
        # voxel
        # ====================================================
        voxel_group = QGroupBox("Voxel Size")

        voxel_layout = QVBoxLayout(voxel_group)

        self.fit_voxel_x = QDoubleSpinBox()

        self.fit_voxel_y = QDoubleSpinBox()

        self.fit_voxel_z = QDoubleSpinBox()

        self.fit_voxel_x.setValue(0.02791)

        self.fit_voxel_y.setValue(0.0315)

        self.fit_voxel_z.setValue(0.02791)

        voxel_layout.addWidget(QLabel("Voxel Size X"))

        voxel_layout.addWidget(self.fit_voxel_x)

        voxel_layout.addWidget(QLabel("Voxel Size Y"))

        voxel_layout.addWidget(self.fit_voxel_y)

        voxel_layout.addWidget(QLabel("Voxel Size Z"))

        voxel_layout.addWidget(self.fit_voxel_z)

        left_layout.addWidget(voxel_group)

        # ====================================================
        # advanced
        # ====================================================
        adv_group = QGroupBox("高级参数")

        adv_layout = QVBoxLayout(adv_group)

        self.fit_growth = QDoubleSpinBox()

        self.fit_growth.setValue(4.0)

        self.fit_seed = QSpinBox()

        self.fit_seed.setValue(42)

        self.fit_remove = QCheckBox(
            "Remove Isolated CBD"
        )

        self.fit_remove.setChecked(True)

        adv_layout.addWidget(
            QLabel("Max Growth Distance")
        )

        adv_layout.addWidget(self.fit_growth)

        adv_layout.addWidget(QLabel("Random Seed"))

        adv_layout.addWidget(self.fit_seed)

        adv_layout.addWidget(self.fit_remove)

        left_layout.addWidget(adv_group)

        # ====================================================
        # submit
        # ====================================================
        btn_fit = QPushButton(
            "开始CBD参数拟合"
        )

        btn_fit.setMinimumHeight(50)

        btn_fit.clicked.connect(
            self.start_fit
        )

        left_layout.addWidget(btn_fit)

        left_layout.addStretch()

        # ====================================================
        # RIGHT
        # ====================================================
        right = QWidget()

        right_layout = QVBoxLayout(right)

        self.fit_progress = QProgressBar()

        self.fit_log = QTextEdit()

        self.fit_log.setReadOnly(True)

        right_layout.addWidget(
            QLabel("拟合进度")
        )

        right_layout.addWidget(
            self.fit_progress
        )

        right_layout.addWidget(
            QLabel("实时日志")
        )

        right_layout.addWidget(
            self.fit_log
        )

        splitter.addWidget(left)

        splitter.addWidget(right)

        splitter.setSizes([500, 700])

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
    # generate
    # ========================================================
    def start_generate(self):

        if not self.input_file:

            QMessageBox.warning(
                self,
                "错误",
                "请先上传文件"
            )

            return

        payload = {

            "task_id":
                self.upload_task_id,

            "input_volume_path":
                self.input_file,

            "target_cbd_vol_frac":
                self.target_cbd.value(),

            "w_um":
                self.w_um.value(),

            "pore_value":
                self.pore_value.value(),

            "am_value":
                self.am_value.value(),

            "cbd_value":
                self.cbd_value.value(),

            "voxel_size_x":
                self.voxel_x.value(),

            "voxel_size_y":
                self.voxel_y.value(),

            "voxel_size_z":
                self.voxel_z.value(),

            "max_growth_distance_factor":
                self.max_growth.value(),

            "remove_isolated_cbd":
                self.remove_isolated.isChecked(),

            "seed":
                self.seed.value(),
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

        self.generate_timer.start(2000)

    # ========================================================
    # fit
    # ========================================================
    def start_fit(self):

        payload = {

            "real_3phase_slice_dir":
                self.real_dir_edit.text(),

            "out_dir":
                self.out_dir_edit.text(),

            "pore_value":
                self.fit_pore.value(),

            "am_value":
                self.fit_am.value(),

            "cbd_value":
                self.fit_cbd.value(),

            "w_min":
                self.w_min.value(),

            "w_max":
                self.w_max.value(),

            "num_w":
                self.num_w.value(),

            "voxel_size_x":
                self.fit_voxel_x.value(),

            "voxel_size_y":
                self.fit_voxel_y.value(),

            "voxel_size_z":
                self.fit_voxel_z.value(),

            "max_growth_distance_factor":
                self.fit_growth.value(),

            "remove_isolated_cbd":
                self.fit_remove.isChecked(),

            "seed":
                self.fit_seed.value(),
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

        self.fit_timer.start(2000)

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
