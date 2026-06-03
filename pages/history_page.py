# -*- coding: utf-8 -*-
import os
import requests

from PySide6.QtCore import Qt, QTimer

from PySide6.QtGui import QPixmap

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QTextEdit,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QSplitter,
    QComboBox,
)

from api_client import (
    API_BASE,
    query_task,
    query_all_tasks,
)


class HistoryPage(QWidget):

    def __init__(self):
        super().__init__()
        self.current_task_id = None
        self.current_files = []
        self.last_log_count = 0  # ← 记录上次日志条数
        self.init_ui()
        self.load_tasks()
        # ========== 日志定时刷新 ==========
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.refresh_logs)

    # ========== 新增：公开刷新方法 ==========
    def refresh_task_list(self):
        self.load_tasks()

    # =====================================================
    # UI
    # =====================================================
    def init_ui(self):

        root_layout = QHBoxLayout(self)

        splitter = QSplitter(Qt.Horizontal)

        root_layout.addWidget(splitter)

        # =================================================
        # LEFT
        # =================================================
        left = QWidget()

        left_layout = QVBoxLayout(left)

        title = QLabel("历史任务中心")

        title.setStyleSheet("""
        font-size:24px;
        font-weight:bold;
        padding:10px;
        """)

        left_layout.addWidget(title)

        self.task_list = QListWidget()

        self.task_list.itemClicked.connect(
            self.on_task_clicked
        )

        left_layout.addWidget(self.task_list)

        btn_refresh = QPushButton(
            "刷新任务列表"
        )

        btn_refresh.clicked.connect(
            self.load_tasks
        )

        left_layout.addWidget(btn_refresh)

        # =================================================
        # RIGHT
        # =================================================
        right = QWidget()

        right_layout = QVBoxLayout(right)

        # =================================================
        # info
        # =================================================
        self.info_label = QLabel(
            "请选择任务"
        )

        self.info_label.setStyleSheet("""
        font-size:18px;
        padding:10px;
        """)

        right_layout.addWidget(
            self.info_label
        )

        # =================================================
        # progress
        # =================================================
        self.progress = QProgressBar()

        right_layout.addWidget(
            self.progress
        )

        # =================================================
        # logs
        # =================================================
        log_title = QLabel("实时日志")

        right_layout.addWidget(log_title)

        self.log_text = QTextEdit()

        self.log_text.setReadOnly(True)

        right_layout.addWidget(
            self.log_text,
            3
        )

        # =================================================
        # files
        # =================================================
        file_title = QLabel("结果文件")

        right_layout.addWidget(file_title)

        self.file_combo = QComboBox()

        self.file_combo.currentIndexChanged.connect(
            self.preview_file
        )

        right_layout.addWidget(
            self.file_combo
        )

        # =================================================
        # image preview
        # =================================================
        self.image_label = QLabel()

        self.image_label.setAlignment(
            Qt.AlignCenter
        )

        self.image_label.setMinimumHeight(400)

        self.image_label.setStyleSheet("""
        border:1px solid #444;
        border-radius:10px;
        """)

        right_layout.addWidget(
            self.image_label,
            4
        )

        # =================================================
        # buttons
        # =================================================
        btn_layout = QHBoxLayout()

        self.btn_download_file = QPushButton(
            "下载文件"
        )

        self.btn_download_dir = QPushButton(
            "下载目录ZIP"
        )

        self.btn_view_json = QPushButton(
            "查看JSON"
        )

        self.btn_download_file.clicked.connect(
            self.download_file
        )

        self.btn_download_dir.clicked.connect(
            self.download_dir
        )

        self.btn_view_json.clicked.connect(
            self.view_json
        )

        btn_layout.addWidget(
            self.btn_download_file
        )

        btn_layout.addWidget(
            self.btn_download_dir
        )

        btn_layout.addWidget(
            self.btn_view_json
        )

        right_layout.addLayout(btn_layout)

        splitter.addWidget(left)

        splitter.addWidget(right)

        splitter.setSizes([350, 1200])

    # =====================================================
    # load tasks
    # =====================================================
    def load_tasks(self):
        self.task_list.clear()

        try:
            tasks = query_all_tasks()
        except Exception as e:
            QMessageBox.warning(self, "错误", str(e))
            return

        for item in reversed(tasks):
            task_id = item.get("task_id", "")
            title = item.get("title", "未知任务")
            create_time = item.get("create_time", "")
            status = item.get("status", "unknown")

            text = f"{title}\n{create_time}\n{status}"

            list_item = QListWidgetItem(text)
            list_item.setData(Qt.UserRole, task_id)
            self.task_list.addItem(list_item)

    def on_task_clicked(self, item):
        task_id = item.data(Qt.UserRole)
        self.current_task_id = task_id

        # 停止旧定时器，启动新的
        self.log_timer.stop()
        self.log_timer.start(20000)  # 20秒刷新一次

        task = query_task(task_id)
        status = task.get("status", "")
        progress = task.get("progress", 0)
        logs = task.get("logs", [])

        self.info_label.setText(f"任务ID: {task_id} | 状态: {status}")
        self.progress.setValue(int(progress))
        self.log_text.setPlainText("\n".join(logs[-200:]))

        self.load_files(task_id)

        # ========== 定时刷新：只追加新行 ==========

    def refresh_logs(self):
        if not self.current_task_id:
            return

        task = query_task(self.current_task_id)
        status = task.get("status", "")
        progress = task.get("progress", 0)
        logs = task.get("logs", [])

        self.progress.setValue(int(progress))

        # 只追加新增的日志，不覆盖
        new_logs = logs[self.last_log_count:]
        if new_logs:
            for line in new_logs:
                self.log_text.append(line)
            self.last_log_count = len(logs)  # ← 更新条数

        if status in ("finished", "failed"):
            self.log_timer.stop()

    # =====================================================
    # load files
    # =====================================================
    def load_files(self, task_id):

        self.file_combo.clear()

        try:

            res = requests.get(
                f"{API_BASE}/task/results/list",
                params={"task_id": task_id},
                timeout=10,
            ).json()

        except Exception:

            return

        files = res.get("files", [])

        self.current_files = files

        for f in files:

            text = (
                f"{f['name']} | "
                f"{f['path']}"
            )

            self.file_combo.addItem(text)

    # =====================================================
    # preview file
    # =====================================================
    def preview_file(self):

        idx = self.file_combo.currentIndex()

        if idx < 0:
            return

        file_info = self.current_files[idx]

        path = file_info["path"]

        ext = os.path.splitext(path)[-1].lower()

        file_url = (
            f"{API_BASE}/download/file"
            f"?path={path}"
        )

        # =================================================
        # image
        # =================================================
        if ext in [".png", ".jpg", ".jpeg"]:

            try:

                response = requests.get(
                    file_url
                )

                temp_path = "temp_preview.png"

                with open(
                        temp_path,
                        "wb"
                ) as f:

                    f.write(response.content)

                pixmap = QPixmap(temp_path)

                self.image_label.setPixmap(
                    pixmap.scaled(
                        self.image_label.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )

            except Exception as e:

                print(e)

        else:

            self.image_label.clear()

    # =====================================================
    # download file
    # =====================================================
    def download_file(self):

        idx = self.file_combo.currentIndex()

        if idx < 0:
            return

        file_info = self.current_files[idx]

        path = file_info["path"]

        url = (
            f"{API_BASE}/download/file"
            f"?path={path}"
        )

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存文件",
            os.path.basename(path)
        )

        if not save_path:
            return

        try:

            r = requests.get(url)

            with open(save_path, "wb") as f:

                f.write(r.content)

            QMessageBox.information(
                self,
                "成功",
                "文件下载完成"
            )

        except Exception as e:

            QMessageBox.warning(
                self,
                "错误",
                str(e)
            )

    # =====================================================
    # download dir
    # =====================================================
    def download_dir(self):

        idx = self.file_combo.currentIndex()

        if idx < 0:
            return

        file_info = self.current_files[idx]

        path = file_info["path"]

        dir_path = os.path.dirname(path)

        url = (
            f"{API_BASE}/download/dir"
            f"?path={dir_path}"
        )

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存ZIP",
            "result.zip"
        )

        if not save_path:
            return

        try:

            r = requests.get(url)

            with open(save_path, "wb") as f:

                f.write(r.content)

            QMessageBox.information(
                self,
                "成功",
                "ZIP下载完成"
            )

        except Exception as e:

            QMessageBox.warning(
                self,
                "错误",
                str(e)
            )

    # =====================================================
    # json
    # =====================================================
    def view_json(self):

        idx = self.file_combo.currentIndex()

        if idx < 0:
            return

        file_info = self.current_files[idx]

        path = file_info["path"]

        ext = os.path.splitext(path)[-1]

        if ext != ".json":

            QMessageBox.information(
                self,
                "提示",
                "当前不是JSON文件"
            )

            return

        try:

            url = (
                f"{API_BASE}/download/file"
                f"?path={path}"
            )

            data = requests.get(url).json()

            self.log_text.setPlainText(
                str(data)
            )

        except Exception as e:

            QMessageBox.warning(
                self,
                "错误",
                str(e)
            )
