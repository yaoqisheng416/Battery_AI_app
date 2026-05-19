# -*- coding: utf-8 -*-
import multiprocessing
import socket
import subprocess
import sys
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget, QPushButton, QScrollArea,
    # 新增
    QMessageBox,
    QProgressBar,
)

from pages.history_page import HistoryPage
from pages.stage1_page import Stage1Page
from pages.stage2_page import Stage2Page
from pages.stage3_page import Stage3Page
from pages.stage4_page import Stage4Page
from pages.stage5_page import Stage5Page
from pages.stage6_page import Stage6Page
import sys
import os

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")

if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("极片三维数字孪生")
        self.resize(1600, 950)

        # 全局字体
        font = QFont()
        font.setFamily("Microsoft YaHei")
        font.setPointSize(12)
        QApplication.instance().setFont(font)

        self.current_theme = "light"

        # 中央布局
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(20, 20, 20, 20)
        root_layout.setSpacing(15)

        # ========== 顶部标题区域 ==========
        title_frame = QFrame()
        title_frame.setObjectName("titleFrame")
        # ❌ 删除 setFixedHeight(280)

        title_layout = QVBoxLayout(title_frame)
        title_layout.setContentsMargins(20, 15, 20, 15)
        title_layout.setSpacing(10)

        title = QLabel("🧪 极片三维数字孪生模型训练与推理应用")
        title.setObjectName("mainTitle")

        #  换成 QLabel
        desc = QLabel(
            "该平台用于：\n\n"
            "• Stage1~3：模型训练\n"
            "• Stage4~6：模型推理与三维结构生成\n"
            "• 实时日志展示\n"
            "• 实时任务状态查询\n"
            "• 模型版本管理\n"
            "• 图片结果展示\n"
            "• 文件下载"
        )
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignTop)
        desc.setObjectName("descLabel")

        desc.setStyleSheet("""
            #descLabel {
                font-size: 18px;
                font-weight: bold;
                line-height: 1.8;
                color: #e0e0e0;
                background-color: #2b2d31;
                border: 2px solid #444;
                border-radius: 10px;
                padding: 15px;
            }
        """)

        title_layout.addWidget(title)
        title_layout.addWidget(desc)

        root_layout.addWidget(title_frame)  # 自适应高度

        # ========== 主题切换按钮 ==========
        theme_bar = QHBoxLayout()
        theme_bar.addStretch()

        self.theme_btn_dark = QPushButton("🌙 深色模式")
        self.theme_btn_dark.setFixedSize(120, 35)
        self.theme_btn_dark.clicked.connect(lambda: self.switch_theme("dark"))

        self.theme_btn_light = QPushButton("☀️ 浅色模式")
        self.theme_btn_light.setFixedSize(120, 35)
        self.theme_btn_light.clicked.connect(lambda: self.switch_theme("light"))

        theme_bar.addWidget(self.theme_btn_dark)
        theme_bar.addWidget(self.theme_btn_light)

        root_layout.addLayout(theme_bar)

        # ========== 主体区域（QScrollArea 包裹）==========
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 水平滚动条：已隐藏 
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 垂直滚动条：永久隐藏

        body_widget = QWidget()
        body_layout = QHBoxLayout(body_widget)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(15)

        # 左侧导航
        self.menu = QListWidget()
        self.menu.setObjectName("leftMenu")
        self.menu.setFixedWidth(400)

        menus = [
            "Stage1 模型训练",
            "Stage2 模型训练",
            "Stage3 模型训练",
            "Stage4 条件可控的两相结构生成",
            "Stage5 任意大体积组装生成与验证",
            "Stage6 CBD三相电极结构生成与参数拟合",
            "任务中心",
        ]

        for m in menus:
            item = QListWidgetItem(m)
            self.menu.addItem(item)

        # 右侧页面
        self.stack = QStackedWidget()
        self.stage1_page = Stage1Page()
        self.stage2_page = Stage2Page()
        self.stage3_page = Stage3Page()
        self.stage4_page = Stage4Page(self)
        self.stage5_page = Stage5Page(self)
        self.stage6_page = Stage6Page(self)
        self.history_page = HistoryPage()

        self.stack.addWidget(self.stage1_page)
        self.stack.addWidget(self.stage2_page)
        self.stack.addWidget(self.stage3_page)
        self.stack.addWidget(self.stage4_page)
        self.stack.addWidget(self.stage5_page)
        self.stack.addWidget(self.stage6_page)
        self.stack.addWidget(self.history_page)

        body_layout.addWidget(self.menu)
        body_layout.addWidget(self.stack, 1)  # stack 占剩余空间

        scroll.setWidget(body_widget)
        root_layout.addWidget(scroll)

        self.menu.currentRowChanged.connect(self.stack.setCurrentIndex)

        self.load_qss()

    def switch_theme(self, theme):
        self.current_theme = theme
        self.load_qss()

    def load_qss(self):
        if self.current_theme == "dark":
            self.setStyleSheet("""
                QMainWindow {
                    background: #1e1f22;
                }

                QLabel {
                    color: white;
                }

                QTextEdit {
                    background: #2b2d31;
                    color: #e0e0e0;
                    border-radius: 10px;
                    padding: 15px;
                    border: 2px solid #444;
                }

                QLineEdit {
                    background: #2b2d31;
                    color: white;
                    border-radius: 8px;
                    padding: 8px;
                    border: 1px solid #444;
                }

                QPushButton {
                    background: #4f8cff;
                    color: white;
                    border-radius: 10px;
                    padding: 10px;
                    font-weight: bold;
                    border: none;
                }

                QPushButton:hover {
                    background: #6aa1ff;
                }

                /* ========== ? QSpinBox / QDoubleSpinBox（暗色主题）========== */
                QSpinBox, QDoubleSpinBox {
                    background: #2b2d31;
                    color: white;
                    border-radius: 8px;
                    padding: 5px;
                    border: 1px solid #444;
                }
        
                /* ? 上调按钮（右上角） */
                QSpinBox::up-button, QDoubleSpinBox::up-button {
                    background: #3a3b3f;
                    width: 24px;
                    subcontrol-position: top right;
                    border-top-right-radius: 8px;
                }
        
                QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
                    background: #4f8cff;
                }
        
                QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {
                    background: #3a7ae6;
                }
        
                /* ? 下调按钮（右下角） */
                QSpinBox::down-button, QDoubleSpinBox::down-button {
                    background: #3a3b3f;
                    width: 24px;
                    subcontrol-position: bottom right;
                    border-bottom-right-radius: 8px;
                }
        
                QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                    background: #4f8cff;
                }
        
                QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
                    background: #3a7ae6;
                }
        
                /* ? 关键：三角形箭头（用 border 画） */
                QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                    width: 0px;
                    height: 0px;
                    border-left: 6px solid transparent;
                    border-right: 6px solid transparent;
                    border-bottom: 8px solid white;  /* ▲ 上箭头（三角形朝上） */
                }
        
                QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                    width: 0px;
                    height: 0px;
                    border-left: 6px solid transparent;
                    border-right: 6px solid transparent;
                    border-top: 8px solid white;  /* ▼ 下箭头（三角形朝下） */
                }


                /* ========== 其他组件（保持不变）========== */
                QListWidget {
                    background: #25262b;
                    color: white;
                    border: none;
                    font-size: 16px;
                    outline: none;
                }

                QListWidget::item {
                    height: 50px;
                    padding-left: 10px;
                    border-radius: 8px;
                    margin: 3px 5px;
                }

                QListWidget::item:selected {
                    background: #4f8cff;
                    border-radius: 8px;
                }

                QListWidget::item:hover {
                    background: #3a3b3f;
                }

                #mainTitle {
                    font-size: 28px;
                    font-weight: bold;
                    color: #ffffff;
                }

                #titleFrame {
                    background: #25262b;
                    border-radius: 16px;
                    padding: 15px;
                }

                QFrame {
                    background: #25262b;
                    border-radius: 16px;
                }

                QProgressBar {
                    height: 20px;
                    border-radius: 10px;
                    background: #333;
                    text-align: center;
                    color: white;
                }

                QProgressBar::chunk {
                    background: #4f8cff;
                    border-radius: 10px;
                }

                QScrollArea {
                    border: none;
                    background: #1e1f22;
                }

                QScrollBar:vertical {
                    background: #25262b;
                    width: 12px;
                    border-radius: 6px;
                }

                QScrollBar::handle:vertical {
                    background: #4f8cff;
                    border-radius: 6px;
                    min-height: 30px;
                }

                QScrollBar::handle:vertical:hover {
                    background: #6aa1ff;
                }

                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0px;
                }
            """)
        else:  # light theme
            self.setStyleSheet("""
                QMainWindow {
                    background: #f0f2f5;
                }

                QLabel {
                    color: #333;
                }

                QTextEdit {
                    background: #ffffff;
                    color: #333;
                    border-radius: 10px;
                    padding: 15px;
                    border: 2px solid #ddd;
                }

                QLineEdit {
                    background: #ffffff;
                    color: #333;
                    border-radius: 8px;
                    padding: 8px;
                    border: 1px solid #ccc;
                }

                QPushButton {
                    background: #4f8cff;
                    color: white;
                    border-radius: 10px;
                    padding: 10px;
                    font-weight: bold;
                    border: none;
                }

                QPushButton:hover {
                    background: #6aa1ff;
                }

                /* ========== ? QSpinBox / QDoubleSpinBox（亮色主题）========== */
                QSpinBox, QDoubleSpinBox {
                    background: #ffffff;
                    color: #333;
                    border-radius: 8px;
                    padding: 5px;
                    border: 1px solid #ccc;
                }
    
                /* ? 上调按钮（右上角） */
                QSpinBox::up-button, QDoubleSpinBox::up-button {
                    background: #e0e0e0;
                    width: 24px;
                    subcontrol-position: top right;
                    border-top-right-radius: 8px;
                }
    
                QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
                    background: #4f8cff;
                }
    
                QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {
                    background: #3a7ae6;
                }
    
                /* ? 下调按钮（右下角） */
                QSpinBox::down-button, QDoubleSpinBox::down-button {
                    background: #e0e0e0;
                    width: 24px;
                    subcontrol-position: bottom right;
                    border-bottom-right-radius: 8px;
                }
    
                QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                    background: #4f8cff;
                }
    
                QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
                    background: #3a7ae6;
                }
    
                /* ? 关键：三角形箭头（用 border 画） */
                QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                    width: 0px;
                    height: 0px;
                    border-left: 6px solid transparent;
                    border-right: 6px solid transparent;
                    border-bottom: 8px solid #333;  /* ▲ 上箭头（三角形朝上） */
                }
    
                QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                    width: 0px;
                    height: 0px;
                    border-left: 6px solid transparent;
                    border-right: 6px solid transparent;
                    border-top: 8px solid #333;  /* ▼ 下箭头（三角形朝下） */
                }


                /* ========== 其他组件（保持不变）========== */
                QListWidget {
                    background: #ffffff;
                    color: #333;
                    border: 1px solid #ddd;
                    font-size: 16px;
                    outline: none;
                }

                QListWidget::item {
                    height: 50px;
                    padding-left: 10px;
                    border-radius: 8px;
                    margin: 3px 5px;
                }

                QListWidget::item:selected {
                    background: #4f8cff;
                    color: white;
                    border-radius: 8px;
                }

                QListWidget::item:hover {
                    background: #e8f0fe;
                }

                #mainTitle {
                    font-size: 28px;
                    font-weight: bold;
                    color: #1a1a2e;
                }

                #titleFrame {
                    background: #ffffff;
                    border-radius: 16px;
                    padding: 15px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
                }

                QFrame {
                    background: #ffffff;
                    border-radius: 16px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
                }

                QProgressBar {
                    height: 20px;
                    border-radius: 10px;
                    background: #e0e0e0;
                    text-align: center;
                    color: #333;
                }

                QProgressBar::chunk {
                    background: #4f8cff;
                    border-radius: 10px;
                }

                QScrollArea {
                    border: none;
                    background: #f0f2f5;
                }

                QScrollBar:vertical {
                    background: #e0e0e0;
                    width: 12px;
                    border-radius: 6px;
                }

                QScrollBar::handle:vertical {
                    background: #4f8cff;
                    border-radius: 6px;
                    min-height: 30px;
                }

                QScrollBar::handle:vertical:hover {
                    background: #6aa1ff;
                }

                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0px;
                }
            """)

        #  更新主题按钮样式（保持不变）
        if self.current_theme == "dark":
            self.theme_btn_dark.setStyleSheet("""
                QPushButton {
                    background: #4f8cff;
                    color: white;
                    border-radius: 10px;
                    padding: 8px;
                    font-weight: bold;
                    border: 2px solid #4f8cff;
                }
            """)
            self.theme_btn_light.setStyleSheet("""
                QPushButton {
                    background: #25262b;
                    color: #aaa;
                    border-radius: 10px;
                    padding: 8px;
                    border: 2px solid #444;
                }
            """)
        else:
            self.theme_btn_dark.setStyleSheet("""
                QPushButton {
                    background: #e0e0e0;
                    color: #666;
                    border-radius: 10px;
                    padding: 8px;
                    border: 2px solid #ccc;
                }
            """)
            self.theme_btn_light.setStyleSheet("""
                QPushButton {
                    background: #4f8cff;
                    color: white;
                    border-radius: 10px;
                    padding: 8px;
                    font-weight: bold;
                    border: 2px solid #4f8cff;
                }
            """)


# ============================================
# 单实例锁
# 防止重复启动
# ============================================

LOCK_PORT = 54321


def already_running():

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        s.bind(("127.0.0.1", LOCK_PORT))
        return False

    except socket.error:
        return True


# ============================================
# 启动 Splash
# ============================================

class SplashScreen(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Battery AI")

        self.setFixedSize(420, 220)

        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.CustomizeWindowHint
        )

        layout = QVBoxLayout()

        self.label = QLabel("正在启动 三维数字孪生应用...")

        self.label.setAlignment(Qt.AlignCenter)

        self.label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
        """)

        self.progress = QProgressBar()

        self.progress.setRange(0, 100)

        self.progress.setValue(0)

        layout.addStretch()

        layout.addWidget(self.label)

        layout.addWidget(self.progress)

        layout.addStretch()

        self.setLayout(layout)

        self.timer = QTimer()

        self.timer.timeout.connect(self.fake_progress)

        self.current = 0

        self.timer.start(120)

    def fake_progress(self):

        if self.current < 90:

            self.current += 1

            self.progress.setValue(self.current)

    def finish(self):

        self.timer.stop()

        self.progress.setValue(100)

        self.label.setText("启动完成")


def run_api_server():

    from backend.api_server import start_server

    start_server()


def wait_for_api(timeout=120):

    import requests

    start_time = time.time()

    while time.time() - start_time < timeout:

        try:

            r = requests.get(
                "http://127.0.0.1:8001/health",
                timeout=2
            )

            if r.status_code == 200:

                data = r.json()

                if data.get("ready") is True:

                    print("[OK] API 已完全启动")

                    return True

                else:

                    print("API 正在初始化模型...")

        except Exception as e:

            print("等待 API:", e)

        time.sleep(1)

    return False


if __name__ == "__main__":

    multiprocessing.freeze_support()

    if already_running():

        app = QApplication(sys.argv)

        QMessageBox.warning(
            None,
            "提示",
            "BatteryAI 已经在运行中"
        )

        sys.exit(0)

    print("=" * 50)
    print("开始启动应用...")
    print("=" * 50)

    app = QApplication(sys.argv)

    splash = SplashScreen()

    splash.show()

    app.processEvents()

    # =====================================
    # 启动 API
    # =====================================
    splash.label.setText("正在启动后端 API...")

    api_process = multiprocessing.Process(
        target=run_api_server,
        daemon=True
    )

    api_process.start()

    # =====================================
    # 等待 API 完全初始化
    # =====================================
    splash.label.setText("正在加载 AI 模型（请稍等）...")

    print("[2] 等待 API 完全初始化...")

    if not wait_for_api(timeout=300):  # 建议加大

        QMessageBox.critical(
            None,
            "错误",
            "后端 API 初始化失败"
        )

        api_process.terminate()
        sys.exit(1)

    # API完全ready后才到这里
    splash.label.setText("启动界面...")

    splash.repaint()
    app.processEvents()

    window = MainWindow()

    window.show()

    splash.close()

    # =====================================
    # 清理
    # =====================================

    def cleanup():

        if api_process.is_alive():

            print("关闭 API...")

            api_process.terminate()

            api_process.join(timeout=3)

    app.aboutToQuit.connect(cleanup)

    print("=" * 50)
    print("GUI 已启动")
    print("=" * 50)

    sys.exit(app.exec())
