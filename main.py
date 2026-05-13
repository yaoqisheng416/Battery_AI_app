# -*- coding: utf-8 -*-
import sys

from PySide6.QtCore import Qt
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
)

from pages.history_page import HistoryPage
from pages.stage1_page import Stage1Page
from pages.stage2_page import Stage2Page
from pages.stage3_page import Stage3Page
from pages.stage4_page import Stage4Page
from pages.stage6_page import Stage6Page


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

        # ✅ 换成 QLabel
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

        root_layout.addWidget(title_frame)  # ✅ 自适应高度

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
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 水平滚动条：已隐藏 ✅
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # ✅ 垂直滚动条：永久隐藏

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
        self.stage4_page = Stage4Page()
        self.stage6_page = Stage6Page()
        self.history_page = HistoryPage()

        self.stack.addWidget(self.stage1_page)
        self.stack.addWidget(self.stage2_page)
        self.stack.addWidget(self.stage3_page)
        self.stack.addWidget(self.stage4_page)
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

        # ✅ 更新主题按钮样式（保持不变）
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


if __name__ == "__main__":
    app = QApplication(sys.argv)

    win = MainWindow()

    win.show()

    sys.exit(app.exec())
