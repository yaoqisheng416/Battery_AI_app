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

        root_layout = QVBoxLayout(self)
        root_layout.setSpacing(6)

        title = QLabel("历史任务中心")
        title.setStyleSheet("font-weight: bold; padding: 6px;")
        root_layout.addWidget(title)

        self.task_list = QListWidget()
        self.task_list.setMaximumHeight(150)
        self.task_list.itemClicked.connect(self.on_task_clicked)
        root_layout.addWidget(self.task_list)

        btn_refresh = QPushButton("刷新任务列表")
        btn_refresh.clicked.connect(self.load_tasks)
        root_layout.addWidget(btn_refresh)

        # 任务信息
        self.info_label = QLabel("请选择任务")
        self.info_label.setStyleSheet("padding: 4px;")
        self.info_label.setWordWrap(True)
        root_layout.addWidget(self.info_label)

        # 日志
        root_layout.addWidget(QLabel("实时日志"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        root_layout.addWidget(self.log_text, 1)

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
