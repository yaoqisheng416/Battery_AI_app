# -*- coding: utf-8 -*-
import sys

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
    QWidget,
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

        self.setWindowTitle(
            "极片三维数字孪生软件"
        )

        self.resize(1600, 950)

        # 全局字体
        font = QFont()

        font.setFamily("Microsoft YaHei")

        font.setPointSize(10)

        QApplication.instance().setFont(font)

        # 中央布局
        central = QWidget()

        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)

        # 顶部标题
        title_frame = QFrame()

        title_frame.setObjectName("titleFrame")

        title_layout = QVBoxLayout(title_frame)

        title = QLabel(
            "🧪 极片三维数字孪生模型训练与推理应用"
        )

        title.setObjectName("mainTitle")

        desc = QTextEdit()

        desc.setReadOnly(True)

        desc.setMaximumHeight(360)

        # ✅ 设置样式
        desc.setStyleSheet("""
            QTextEdit {
                font-size: 18px;              /* 字体大小 */
                font-weight: bold;            /* 加粗 */
                line-height: 1.8;             /* 行高（1.8倍） */
                color: #333333;               /* 文字颜色 */
                background-color: #f9f9f9;    /* 背景色 */
                border: 2px solid #ddd;       /* 边框 */
                border-radius: 8px;           /* 圆角 */
                padding: 15px;                /* 内边距 */
            }
        """)

        desc.setText(
            "该平台用于：\n\n"
            "• Stage1~3：模型训练\n"
            "• Stage4~6：模型推理与三维结构生成\n"
            "• 实时日志展示\n"
            "• 实时任务状态查询\n"
            "• 模型版本管理\n"
            "• 图片结果展示\n"
            "• 文件下载"
        )

        title_layout.addWidget(title)

        title_layout.addWidget(desc)

        root_layout.addWidget(title_frame)

        # 主体区域
        body_layout = QHBoxLayout()

        # 左侧导航
        self.menu = QListWidget()

        self.menu.setObjectName("leftMenu")

        self.menu.setFixedWidth(460)

        menus = [
            "Stage1 模型训练",
            "Stage2 模型训练",
            "Stage3 模型训练",
            "Stage4 条件可控的两相结构生成",
            # "Stage5 任意大体积组装生成与验证",
            "Stage6 CBD三相电极结构生成与参数拟合",
            "历史任务中心",
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

        body_layout.addWidget(self.stack)

        root_layout.addLayout(body_layout)

        self.menu.currentRowChanged.connect(
            self.stack.setCurrentIndex
        )

        self.load_qss()

    def load_qss(self):
        self.setStyleSheet("""

        QMainWindow {
            background: #1e1f22;
        }

        QLabel {
            color: white;
        }

        QTextEdit {
            background: #2b2d31;
            color: white;
            border-radius: 10px;
            padding: 8px;
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
        }

        QPushButton:hover {
            background: #6aa1ff;
        }

        QSpinBox, QDoubleSpinBox {
            background: #2b2d31;
            color: white;
            border-radius: 8px;
            padding: 5px;
        }

        QListWidget {
            background: #25262b;
            color: white;
            border: none;
            font-size: 16px;
        }

        QListWidget::item {
            height: 50px;
            padding-left: 10px;
        }

        QListWidget::item:selected {
            background: #4f8cff;
            border-radius: 8px;
        }

        #mainTitle {
            font-size: 28px;
            font-weight: bold;
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
        }

        QProgressBar::chunk {
            background: #4f8cff;
            border-radius: 10px;
        }
        """)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    win = MainWindow()

    win.show()

    sys.exit(app.exec())
