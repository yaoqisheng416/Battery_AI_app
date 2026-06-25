# -*- coding: utf-8 -*-

from PySide6.QtCore import Qt, QTimer

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QTextEdit,
    QPushButton,
    QMessageBox,
    QSplitter,
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
        # logs
        # =================================================
        log_title = QLabel("实时日志")

        right_layout.addWidget(log_title)

        self.log_text = QTextEdit()

        self.log_text.setReadOnly(True)

        right_layout.addWidget(
            self.log_text,
            1
        )

        # =================================================
        # buttons
        # =================================================
        btn_layout = QHBoxLayout()

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
        self.log_timer.start(5000)  # 5秒刷新一次

        task = query_task(task_id)
        status = task.get("status", "")
        logs = task.get("logs", [])

        self.info_label.setText(f"任务ID: {task_id} | 状态: {status}")
        self.log_text.setPlainText("\n".join(logs[-200:]))

        # ========== 定时刷新：只追加新行 ==========

    def refresh_logs(self):
        if not self.current_task_id:
            return

        task = query_task(self.current_task_id)
        status = task.get("status", "")
        logs = task.get("logs", [])


        # 只追加新增的日志，不覆盖
        new_logs = logs[self.last_log_count:]
        if new_logs:
            for line in new_logs:
                self.log_text.append(line)
            self.last_log_count = len(logs)  # ← 更新条数
