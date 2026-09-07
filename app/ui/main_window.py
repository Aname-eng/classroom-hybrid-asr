# coding: utf-8
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, Optional, List, Callable

from PySide6.QtCore import Qt, QObject, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QColor, QIcon, QAction, QKeyEvent, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QScrollArea, QFrame, QProgressBar,
    QStatusBar, QMessageBox, QSplitter, QTextEdit, QPlainTextEdit,
    QLineEdit, QDialog, QSizePolicy
)
import sounddevice as sd
import numpy as np

from app.config import (
    AppConfig, SESSIONS_DIR, COURSES_DIR,
    CAPSWRITER_SERVER_EXE, QWEN_MODEL_DIR
)
from app.courses.course_manager import CourseManager, CourseInfo
from app.pipeline.session_manager import SessionManager


class UiBridge(QObject):
    """
    后台管道与 Qt UI 主线程之间的异步信号桥梁
    """
    partial_received = Signal(int, str, float)             # segment_id, text, timestamp
    final_received = Signal(int, str, str, float, bool, str)  # segment_id, text, model, timestamp, success, fallback_reason
    queue_changed = Signal(int)                            # backlog_count
    status_changed = Signal(str, str)                      # key, message
    audio_level_updated = Signal(float)                    # rms level (0.0 ~ 1.0)
    session_finished = Signal(str, str)                    # session_id, session_path


class InlineTextEdit(QPlainTextEdit):
    """
    单句卡片内嵌编辑框，支持快捷键保存与取消
    """
    save_requested = Signal()
    cancel_requested = Signal()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and (
            event.modifiers() & Qt.ControlModifier or event.modifiers() & Qt.ShiftModifier
        ):
            self.save_requested.emit()
            event.accept()
        elif event.key() == Qt.Key_Escape:
            self.cancel_requested.emit()
            event.accept()
        else:
            super().keyPressEvent(event)


class SegmentCard(QFrame):
    """
    单句字幕展示卡片，支持随听随改交互
    """
    def __init__(
        self,
        segment_id: int,
        timestamp_sec: float,
        on_text_edited: Optional[Callable[[int, str], None]] = None,
        parent=None
    ):
        super().__init__(parent)
        self.segment_id = segment_id
        self.timestamp_sec = timestamp_sec
        self.is_final = False
        self.is_manual_edited = False
        self.is_editing = False
        self.on_text_edited = on_text_edited

        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setObjectName("segmentCard")
        
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(12, 8, 12, 10)
        self.main_layout.setSpacing(6)

        # -------------------------------------------------------------
        # Header Row
        # -------------------------------------------------------------
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        mins = int(timestamp_sec) // 60
        secs = int(timestamp_sec) % 60
        time_str = f"{mins:02d}:{secs:02d}"

        self.lbl_tag = QLabel(f"#{segment_id:02d}  [{time_str}]")
        self.lbl_tag.setStyleSheet("color: #718096; font-size: 12px; font-weight: 600;")
        header_layout.addWidget(self.lbl_tag)

        header_layout.addStretch()

        # 编辑按钮
        self.btn_edit = QPushButton("✏️ 改错")
        self.btn_edit.setFixedHeight(22)
        self.btn_edit.setCursor(Qt.PointingHandCursor)
        self.btn_edit.setToolTip("点击或双击文本可手动修改此句（随听随改）")
        self.btn_edit.setStyleSheet("""
            QPushButton {
                background-color: #FFFFFF;
                color: #4A5568;
                border: 1px solid #CBD5E0;
                border-radius: 4px;
                padding: 1px 8px;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #EDF2F7;
                color: #2B6CB0;
                border-color: #3182CE;
            }
        """)
        self.btn_edit.clicked.connect(self.enter_edit_mode)
        header_layout.addWidget(self.btn_edit)

        # 状态徽章
        self.lbl_badge = QLabel("⚡ 实时流式")
        self.lbl_badge.setStyleSheet(
            "background-color: #FEFCBF; color: #B7791F; "
            "border-radius: 4px; padding: 2px 6px; font-size: 11px; font-weight: 600;"
        )
        header_layout.addWidget(self.lbl_badge)
        self.main_layout.addLayout(header_layout)

        # -------------------------------------------------------------
        # Content Text Label (Normal View)
        # -------------------------------------------------------------
        self.lbl_text = QLabel("...")
        self.lbl_text.setWordWrap(True)
        self.lbl_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_text.setToolTip("双击可直接编辑修改此句")
        self.lbl_text.setStyleSheet("font-size: 15px; color: #2D3748; line-height: 1.5; padding: 2px 0;")
        self.main_layout.addWidget(self.lbl_text)

        # -------------------------------------------------------------
        # In-Place Edit Container (Hidden by default)
        # -------------------------------------------------------------
        self.edit_container = QWidget()
        edit_layout = QVBoxLayout(self.edit_container)
        edit_layout.setContentsMargins(0, 4, 0, 0)
        edit_layout.setSpacing(6)

        self.text_editor = InlineTextEdit()
        self.text_editor.setFixedHeight(65)
        self.text_editor.setStyleSheet("""
            QPlainTextEdit {
                border: 1.5px solid #3182CE;
                border-radius: 6px;
                background-color: #FFFFFF;
                font-size: 14px;
                color: #1A202C;
                padding: 4px;
            }
        """)
        self.text_editor.save_requested.connect(self.save_edit)
        self.text_editor.cancel_requested.connect(self.cancel_edit)
        edit_layout.addWidget(self.text_editor)

        # Save / Cancel Buttons Row
        edit_btn_layout = QHBoxLayout()
        edit_btn_layout.setContentsMargins(0, 0, 0, 0)
        edit_btn_layout.setSpacing(8)

        lbl_edit_hint = QLabel("💡 提示: Ctrl+Enter 快速保存，Esc 放弃")
        lbl_edit_hint.setStyleSheet("font-size: 11px; color: #718096;")
        edit_btn_layout.addWidget(lbl_edit_hint)

        edit_btn_layout.addStretch()

        self.btn_cancel_edit = QPushButton("取消")
        self.btn_cancel_edit.setFixedHeight(26)
        self.btn_cancel_edit.setCursor(Qt.PointingHandCursor)
        self.btn_cancel_edit.setStyleSheet("""
            QPushButton {
                background-color: #EDF2F7;
                color: #4A5568;
                border: 1px solid #CBD5E0;
                border-radius: 4px;
                padding: 2px 10px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #E2E8F0;
            }
        """)
        self.btn_cancel_edit.clicked.connect(self.cancel_edit)
        edit_btn_layout.addWidget(self.btn_cancel_edit)

        self.btn_save_edit = QPushButton("💾 保存修改")
        self.btn_save_edit.setFixedHeight(26)
        self.btn_save_edit.setCursor(Qt.PointingHandCursor)
        self.btn_save_edit.setStyleSheet("""
            QPushButton {
                background-color: #319795;
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 4px;
                padding: 2px 12px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #285E61;
            }
        """)
        self.btn_save_edit.clicked.connect(self.save_edit)
        edit_btn_layout.addWidget(self.btn_save_edit)

        edit_layout.addLayout(edit_btn_layout)
        self.edit_container.setVisible(False)
        self.main_layout.addWidget(self.edit_container)

        self.setStyleSheet("""
            QFrame#segmentCard {
                background-color: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                margin-bottom: 6px;
            }
        """)

    def mouseDoubleClickEvent(self, event):
        if not self.is_editing:
            self.enter_edit_mode()
        super().mouseDoubleClickEvent(event)

    def enter_edit_mode(self):
        current_text = self.lbl_text.text()
        if current_text == "..." or current_text == "(无语音内容)":
            current_text = ""
        self.text_editor.setPlainText(current_text)
        
        # 移动光标到末尾
        cursor = self.text_editor.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.text_editor.setTextCursor(cursor)

        self.lbl_text.setVisible(False)
        self.btn_edit.setVisible(False)
        self.edit_container.setVisible(True)
        self.is_editing = True
        self.text_editor.setFocus()

    def cancel_edit(self):
        self.edit_container.setVisible(False)
        self.lbl_text.setVisible(True)
        self.btn_edit.setVisible(True)
        self.is_editing = False

    def save_edit(self):
        new_text = self.text_editor.toPlainText().strip()
        if not new_text:
            new_text = self.lbl_text.text()
        
        self.lbl_text.setText(new_text)
        self.is_manual_edited = True
        self.is_final = True
        self.cancel_edit()

        # 更新徽章与卡片样式为人工修正
        self.lbl_badge.setText("✨ 人工已修正")
        self.lbl_badge.setStyleSheet(
            "background-color: #EBF8FF; color: #2B6CB0; "
            "border-radius: 4px; padding: 2px 6px; font-size: 11px; font-weight: 600;"
        )
        self.setStyleSheet("""
            QFrame#segmentCard {
                background-color: #F7FAFC;
                border: 1px solid #CBD5E0;
                border-left: 4px solid #3182CE;
                border-radius: 8px;
                margin-bottom: 6px;
            }
        """)

        if self.on_text_edited:
            self.on_text_edited(self.segment_id, new_text)

    def update_partial(self, text: str):
        if not self.is_final and not self.is_manual_edited and not self.is_editing:
            self.lbl_text.setText(text if text.strip() else "...")
            self.lbl_badge.setText("⚡ 实时流式")
            self.lbl_badge.setStyleSheet(
                "background-color: #FEFCBF; color: #B7791F; "
                "border-radius: 4px; padding: 2px 6px; font-size: 11px; font-weight: 600;"
            )

    def update_final(self, text: str, model: str, success: bool = True, fallback_reason: Optional[str] = None):
        self.is_final = True
        
        # 若用户此前已手动改错，绝不覆盖用户的人工输入
        if self.is_manual_edited:
            return

        self.lbl_text.setText(text if text.strip() else "(无语音内容)")
        
        if success and "Qwen" in model:
            badge_text = "✨ Qwen3-ASR 权威纠错"
            badge_style = "background-color: #C6F6D5; color: #22543D;"
            border_left = "4px solid #319795"
            self.setToolTip("")
        elif not success:
            badge_text = "⚠️ 实时回退 (Paraformer)"
            badge_style = "background-color: #FEEBC8; color: #7B341E;"
            border_left = "4px solid #DD6B20"
            if fallback_reason:
                self.setToolTip(f"回退原因: {fallback_reason}")
        else:
            badge_text = "✅ 实时已定稿"
            badge_style = "background-color: #E2E8F0; color: #4A5568;"
            border_left = "4px solid #A0AEC0"
            self.setToolTip("")

        self.lbl_badge.setText(badge_text)
        self.lbl_badge.setStyleSheet(
            f"{badge_style} border-radius: 4px; padding: 2px 6px; font-size: 11px; font-weight: 600;"
        )
        self.setStyleSheet(f"""
            QFrame#segmentCard {{
                background-color: #F7FAFC;
                border: 1px solid #CBD5E0;
                border-left: {border_left};
                border-radius: 8px;
                margin-bottom: 6px;
            }}
        """)


class CourseCreateDialog(QDialog):
    """
    新建自定义课程与专属词库对话框
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("➕ 新建课程分类与专属词库")
        self.resize(480, 400)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(18, 18, 18, 18)
        
        lbl_name = QLabel("课程名称 (必填):")
        lbl_name.setStyleSheet("font-weight: 600; color: #2D3748; font-size: 13px;")
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("例如：宏观经济学、深度学习与大模型、公司金融...")
        self.edit_name.setStyleSheet("padding: 6px; border: 1px solid #CBD5E0; border-radius: 6px; font-size: 13px;")
        layout.addWidget(lbl_name)
        layout.addWidget(self.edit_name)
        
        lbl_desc = QLabel("课程简述 (可选):")
        lbl_desc.setStyleSheet("font-weight: 600; color: #2D3748; font-size: 13px;")
        self.edit_desc = QLineEdit()
        self.edit_desc.setPlaceholderText("简短描述课程内容或讲师信息")
        self.edit_desc.setStyleSheet("padding: 6px; border: 1px solid #CBD5E0; border-radius: 6px; font-size: 13px;")
        layout.addWidget(lbl_desc)
        layout.addWidget(self.edit_desc)
        
        lbl_hw = QLabel("专业术语与热词库 (可选，一行一个):")
        lbl_hw.setStyleSheet("font-weight: 600; color: #2D3748; font-size: 13px;")
        self.edit_hotwords = QPlainTextEdit()
        self.edit_hotwords.setPlaceholderText("输入容易识别错误的专有名词、英文缩写或公式术语（每行一个）：\nTransformer\n自注意力机制\nBackpropagation\n动态面板模型")
        self.edit_hotwords.setStyleSheet("padding: 6px; border: 1px solid #CBD5E0; border-radius: 6px; font-size: 13px;")
        layout.addWidget(lbl_hw)
        layout.addWidget(self.edit_hotwords)
        
        btn_box = QHBoxLayout()
        btn_box.setSpacing(10)
        btn_box.addStretch()
        
        btn_cancel = QPushButton("取消")
        btn_cancel.setFixedHeight(32)
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)
        
        btn_ok = QPushButton("确定创建")
        btn_ok.setFixedHeight(32)
        btn_ok.setStyleSheet("""
            QPushButton {
                background-color: #319795;
                color: white;
                font-weight: bold;
                padding: 0 16px;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #285E61;
            }
        """)
        btn_ok.clicked.connect(self._on_confirm)
        btn_box.addWidget(btn_ok)
        
        layout.addLayout(btn_box)

    def _on_confirm(self):
        name = self.edit_name.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请输入课程名称！")
            return
        self.accept()

    def get_data(self):
        name = self.edit_name.text().strip()
        desc = self.edit_desc.text().strip()
        raw_hw = self.edit_hotwords.toPlainText()
        hotwords = [line.strip() for line in raw_hw.splitlines() if line.strip() and not line.startswith("#")]
        return name, desc, hotwords


class CourseHotwordsDialog(QDialog):
    """
    编辑当前课程词库对话框
    """
    def __init__(self, course: CourseInfo, parent=None):
        super().__init__(parent)
        self.course = course
        self.setWindowTitle(f"⚙️ 词库管理 - {course.name}")
        self.resize(500, 440)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(18, 18, 18, 18)
        
        lbl_title = QLabel(f"当前课程: {course.name}")
        lbl_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #2D3748;")
        layout.addWidget(lbl_title)
        
        lbl_hint = QLabel("每行输入一个专业名词或术语。Qwen3-ASR 权威纠错时会优先基于此热词库提升识别准确率：")
        lbl_hint.setStyleSheet("color: #718096; font-size: 12px;")
        lbl_hint.setWordWrap(True)
        layout.addWidget(lbl_hint)
        
        self.edit_hotwords = QPlainTextEdit()
        self.edit_hotwords.setPlainText("\n".join(course.hotwords))
        self.edit_hotwords.setStyleSheet("padding: 8px; border: 1px solid #CBD5E0; border-radius: 6px; font-size: 13px;")
        layout.addWidget(self.edit_hotwords)
        
        btn_box = QHBoxLayout()
        btn_box.setSpacing(10)
        btn_box.addStretch()
        
        btn_cancel = QPushButton("取消")
        btn_cancel.setFixedHeight(32)
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)
        
        btn_save = QPushButton("💾 保存词库")
        btn_save.setFixedHeight(32)
        btn_save.setStyleSheet("""
            QPushButton {
                background-color: #319795;
                color: white;
                font-weight: bold;
                padding: 0 16px;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #285E61;
            }
        """)
        btn_save.clicked.connect(self.accept)
        btn_box.addWidget(btn_save)
        
        layout.addLayout(btn_box)

    def get_hotwords(self) -> List[str]:
        raw_hw = self.edit_hotwords.toPlainText()
        return [line.strip() for line in raw_hw.splitlines() if line.strip() and not line.startswith("#")]


class MainWindow(QMainWindow):
    """
    课堂实时转写系统主界面
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("课堂实时转写 (Hybrid 2-Pass Classroom ASR)")
        self.resize(960, 720)
        self.setMinimumSize(800, 560)

        self.course_manager = CourseManager()
        self.bridge = UiBridge()
        self.session_manager: Optional[SessionManager] = None
        self.active_session_id: Optional[str] = None
        self.active_session_path: Optional[str] = None

        self.segment_cards: Dict[int, SegmentCard] = {}
        self.current_active_segment_id = 0

        self._setup_ui()
        self._bind_signals()
        self._init_session_manager()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 12)
        main_layout.setSpacing(12)

        # -------------------------------------------------------------
        # 1. Top Control Bar (Course, Mic, Badges)
        # -------------------------------------------------------------
        top_card = QFrame()
        top_card.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 10px;
                padding: 4px;
            }
        """)
        top_layout = QVBoxLayout(top_card)
        top_layout.setContentsMargins(12, 10, 12, 10)
        top_layout.setSpacing(10)

        # Row 1: Course Selector + Action Buttons + Mic
        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(8)

        # Course Dropdown
        lbl_course = QLabel("课程分类:")
        lbl_course.setStyleSheet("font-weight: 600; color: #2D3748;")
        self.combo_course = QComboBox()
        self.combo_course.setMinimumWidth(160)
        self._populate_courses()
        self.combo_course.currentIndexChanged.connect(self._on_course_changed)
        row1_layout.addWidget(lbl_course)
        row1_layout.addWidget(self.combo_course)

        # ➕ 新建课程按钮
        self.btn_new_course = QPushButton("➕ 新建")
        self.btn_new_course.setFixedHeight(28)
        self.btn_new_course.setCursor(Qt.PointingHandCursor)
        self.btn_new_course.setToolTip("创建新的课程分类与专属专业词库")
        self.btn_new_course.setStyleSheet("""
            QPushButton {
                background-color: #EDF2F7;
                color: #2D3748;
                border: 1px solid #CBD5E0;
                border-radius: 4px;
                padding: 0 10px;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #E2E8F0;
                color: #319795;
                border-color: #319795;
            }
        """)
        self.btn_new_course.clicked.connect(self._on_new_course)
        row1_layout.addWidget(self.btn_new_course)

        # ⚙️ 词库管理按钮
        self.btn_edit_hotwords = QPushButton("⚙️ 词库")
        self.btn_edit_hotwords.setFixedHeight(28)
        self.btn_edit_hotwords.setCursor(Qt.PointingHandCursor)
        self.btn_edit_hotwords.setToolTip("查看和编辑当前选中课程的专业热词库")
        self.btn_edit_hotwords.setStyleSheet("""
            QPushButton {
                background-color: #EDF2F7;
                color: #2D3748;
                border: 1px solid #CBD5E0;
                border-radius: 4px;
                padding: 0 10px;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #E2E8F0;
                color: #319795;
                border-color: #319795;
            }
        """)
        self.btn_edit_hotwords.clicked.connect(self._on_edit_hotwords)
        row1_layout.addWidget(self.btn_edit_hotwords)

        row1_layout.addSpacing(12)

        # Mic Dropdown
        lbl_mic = QLabel("麦克风设备:")
        lbl_mic.setStyleSheet("font-weight: 600; color: #2D3748;")
        self.combo_mic = QComboBox()
        self.combo_mic.setMinimumWidth(200)
        self._populate_audio_devices()
        row1_layout.addWidget(lbl_mic)
        row1_layout.addWidget(self.combo_mic)

        row1_layout.addStretch()

        # Engine Status Badges
        self.badge_streaming = QLabel("● 流式: Paraformer (CPU)")
        self.badge_streaming.setStyleSheet(
            "background-color: #EDF2F7; color: #2B6CB0; padding: 4px 8px; border-radius: 6px; font-size: 11px; font-weight: 600;"
        )
        row1_layout.addWidget(self.badge_streaming)

        self.badge_qwen = QLabel("● 纠错: Qwen3-ASR (Intel Arc)")
        self.badge_qwen.setStyleSheet(
            "background-color: #EDF2F7; color: #276749; padding: 4px 8px; border-radius: 6px; font-size: 11px; font-weight: 600;"
        )
        row1_layout.addWidget(self.badge_qwen)

        top_layout.addLayout(row1_layout)

        # Row 2: Action Buttons & Course Info
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(10)

        self.btn_start = QPushButton("▶ 开始课堂录音")
        self.btn_start.setFixedHeight(36)
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #319795;
                color: white;
                font-weight: bold;
                font-size: 13px;
                border: none;
                border-radius: 6px;
                padding: 0 16px;
            }
            QPushButton:hover {
                background-color: #285E61;
            }
            QPushButton:disabled {
                background-color: #CBD5E0;
            }
        """)
        self.btn_start.clicked.connect(self.start_session)
        row2_layout.addWidget(self.btn_start)

        self.btn_pause = QPushButton("⏸ 暂停录音")
        self.btn_pause.setFixedHeight(36)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setCursor(Qt.PointingHandCursor)
        self.btn_pause.setStyleSheet("""
            QPushButton {
                background-color: #D69E2E;
                color: white;
                font-weight: bold;
                font-size: 13px;
                border: none;
                border-radius: 6px;
                padding: 0 16px;
            }
            QPushButton:hover {
                background-color: #B7791F;
            }
            QPushButton:disabled {
                background-color: #CBD5E0;
            }
        """)
        self.btn_pause.clicked.connect(self.toggle_pause_session)
        row2_layout.addWidget(self.btn_pause)

        self.btn_end = QPushButton("⏹ 结束课堂")
        self.btn_end.setFixedHeight(36)
        self.btn_end.setEnabled(False)
        self.btn_end.setCursor(Qt.PointingHandCursor)
        self.btn_end.setStyleSheet("""
            QPushButton {
                background-color: #E53E3E;
                color: white;
                font-weight: bold;
                font-size: 13px;
                border: none;
                border-radius: 6px;
                padding: 0 16px;
            }
            QPushButton:hover {
                background-color: #9B2C2C;
            }
            QPushButton:disabled {
                background-color: #CBD5E0;
            }
        """)
        self.btn_end.clicked.connect(self.end_session)
        row2_layout.addWidget(self.btn_end)

        row2_layout.addSpacing(10)

        self.lbl_course_info = QLabel("已加载词库: 0 个专业术语")
        self.lbl_course_info.setStyleSheet("color: #718096; font-size: 12px;")
        row2_layout.addWidget(self.lbl_course_info)

        row2_layout.addStretch()

        self.btn_copy = QPushButton("📋 复制全文")
        self.btn_copy.setFixedHeight(32)
        self.btn_copy.setCursor(Qt.PointingHandCursor)
        self.btn_copy.clicked.connect(self.copy_transcript)
        row2_layout.addWidget(self.btn_copy)

        self.btn_open_dir = QPushButton("📂 打开记录目录")
        self.btn_open_dir.setFixedHeight(32)
        self.btn_open_dir.setCursor(Qt.PointingHandCursor)
        self.btn_open_dir.clicked.connect(self.open_session_dir)
        row2_layout.addWidget(self.btn_open_dir)

        top_layout.addLayout(row2_layout)
        main_layout.addWidget(top_card)

        # -------------------------------------------------------------
        # 2. Live Subtitle Scroll Area
        # -------------------------------------------------------------
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #E2E8F0;
                border-radius: 10px;
                background-color: #F7FAFC;
            }
        """)

        self.subtitle_container = QWidget()
        self.subtitle_layout = QVBoxLayout(self.subtitle_container)
        self.subtitle_layout.setContentsMargins(16, 16, 16, 16)
        self.subtitle_layout.setSpacing(8)
        self.subtitle_layout.addStretch()

        self.scroll_area.setWidget(self.subtitle_container)
        main_layout.addWidget(self.scroll_area, stretch=1)

        # -------------------------------------------------------------
        # 3. Status Bar
        # -------------------------------------------------------------
        status_bar = self.statusBar()
        status_bar.setStyleSheet("background-color: #EDF2F7; color: #4A5568; font-size: 12px;")

        self.lbl_status = QLabel("就绪")
        status_bar.addWidget(self.lbl_status, stretch=1)

        self.lbl_queue = QLabel("离线纠错队列: 0 句")
        self.lbl_queue.setStyleSheet("font-weight: 600; color: #2B6CB0; margin-right: 12px;")
        status_bar.addPermanentWidget(self.lbl_queue)

        self._on_course_changed()

    def _populate_courses(self, select_course_id: Optional[str] = None):
        self.combo_course.blockSignals(True)
        self.combo_course.clear()
        courses = self.course_manager.list_courses()
        selected_index = 0
        for idx, c in enumerate(courses):
            self.combo_course.addItem(f"{c.name}", userData=c.id)
            if select_course_id and c.id == select_course_id:
                selected_index = idx
        self.combo_course.setCurrentIndex(selected_index)
        self.combo_course.blockSignals(False)

    def _populate_audio_devices(self):
        self.combo_mic.clear()
        self.combo_mic.addItem("系统默认麦克风", userData=None)
        try:
            devices = sd.query_devices()
            for idx, dev in enumerate(devices):
                if dev.get("max_input_channels", 0) > 0:
                    name = dev.get("name", f"Device {idx}")
                    self.combo_mic.addItem(f"[{idx}] {name}", userData=idx)
        except Exception as e:
            print(f"[UI] Error querying audio devices: {e}")

    def _bind_signals(self):
        self.bridge.partial_received.connect(self._handle_partial_received)
        self.bridge.final_received.connect(self._handle_final_received)
        self.bridge.queue_changed.connect(self._handle_queue_changed)
        self.bridge.status_changed.connect(self._handle_status_changed)
        self.bridge.session_finished.connect(self._handle_session_finished)

    def _init_session_manager(self):
        def on_partial(segment_id: int, text: str, ts: float):
            self.bridge.partial_received.emit(segment_id, text, ts)

        def on_final(segment_id: int, text: str, model: str, ts: float, success: bool = True, fallback_reason: Optional[str] = None):
            self.bridge.final_received.emit(segment_id, text, model, ts, success, fallback_reason or "")

        def on_status(status_dict: dict):
            if "status" in status_dict:
                self.bridge.status_changed.emit("status", status_dict["status"])

        self.session_manager = SessionManager(
            on_partial_subtitle=on_partial,
            on_final_subtitle=on_final,
            on_status_update=on_status,
            course_manager=self.course_manager
        )
        self.session_manager.qwen_worker.on_queue_change = lambda qsize: self.bridge.queue_changed.emit(qsize)

        # 启动时后台静默拉起/连接 CapsWriter-Offline 离线服务
        import threading
        threading.Thread(
            target=self.session_manager.qwen_adapter.ensure_server_running,
            kwargs={"wait_timeout": 30.0},
            daemon=True,
            name="CapsWriterGuiAutoStartThread"
        ).start()

    def _on_course_changed(self):
        course_id = self.combo_course.currentData()
        if course_id:
            cfg = self.course_manager.get_course(course_id)
            if cfg:
                hotwords = cfg.hotwords
                self.lbl_course_info.setText(f"已加载词库: {len(hotwords)} 个专业术语 ({cfg.name})")

    def _on_new_course(self):
        dlg = CourseCreateDialog(self)
        if dlg.exec() == QDialog.Accepted:
            name, desc, hotwords = dlg.get_data()
            try:
                new_course = self.course_manager.create_course(
                    name=name,
                    hotwords=hotwords,
                    description=desc
                )
                self._populate_courses(select_course_id=new_course.id)
                self._on_course_changed()
                self.lbl_status.setText(f"✅ 已成功创建课程《{new_course.name}》并配置专属词库 ({len(new_course.hotwords)} 词)")
            except Exception as e:
                QMessageBox.critical(self, "创建失败", f"创建课程时发生错误: {e}")

    def _on_edit_hotwords(self):
        course_id = self.combo_course.currentData()
        if not course_id:
            return
        course = self.course_manager.get_course(course_id)
        if not course:
            return
            
        dlg = CourseHotwordsDialog(course, self)
        if dlg.exec() == QDialog.Accepted:
            new_hotwords = dlg.get_hotwords()
            self.course_manager.update_course_hotwords(course_id, new_hotwords)
            self._on_course_changed()
            self.lbl_status.setText(f"✅ 已更新《{course.name}》专业词库: 共 {len(new_hotwords)} 个术语")

    def _handle_segment_edited(self, segment_id: int, new_text: str):
        if self.session_manager and self.session_manager.is_active:
            self.session_manager.update_segment_text(segment_id, new_text)
            self.lbl_status.setText(f"✏️ 已保存第 #{segment_id} 句的人工修改并同步纪要文件")
        elif self.session_manager:
            # 即使在课堂结束之后，也允许直接更新内存和已生成文件
            self.session_manager.update_segment_text(segment_id, new_text)
            self.lbl_status.setText(f"✏️ 已更新第 #{segment_id} 句内容")

    @Slot()
    def toggle_pause_session(self):
        if not self.session_manager or not self.session_manager.is_active:
            return
        if self.session_manager.recorder.is_paused:
            self.session_manager.resume_session()
            self.btn_pause.setText("⏸ 暂停录音")
            self.btn_pause.setStyleSheet("""
                QPushButton {
                    background-color: #D69E2E;
                    color: white;
                    font-weight: bold;
                    font-size: 13px;
                    border: none;
                    border-radius: 6px;
                    padding: 0 16px;
                }
                QPushButton:hover {
                    background-color: #B7791F;
                }
                QPushButton:disabled {
                    background-color: #CBD5E0;
                }
            """)
            self.lbl_status.setText("🔴 正在录音转写中... (支持直接双击句子实时修改)")
        else:
            self.session_manager.pause_session()
            self.btn_pause.setText("▶ 继续录音")
            self.btn_pause.setStyleSheet("""
                QPushButton {
                    background-color: #3182CE;
                    color: white;
                    font-weight: bold;
                    font-size: 13px;
                    border: none;
                    border-radius: 6px;
                    padding: 0 16px;
                }
                QPushButton:hover {
                    background-color: #2B6CB0;
                }
                QPushButton:disabled {
                    background-color: #CBD5E0;
                }
            """)
            self.lbl_status.setText("⏸ 课堂录音已暂停 (点击【▶ 继续录音】继续在当前课堂中记录)")

    @Slot()
    def start_session(self):
        course_id = self.combo_course.currentData() or "political_economy"
        device_idx = self.combo_mic.currentData()

        # Clear existing cards
        self._clear_subtitles()

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("⏸ 暂停录音")
        self.btn_pause.setStyleSheet("""
            QPushButton {
                background-color: #D69E2E;
                color: white;
                font-weight: bold;
                font-size: 13px;
                border: none;
                border-radius: 6px;
                padding: 0 16px;
            }
            QPushButton:hover {
                background-color: #B7791F;
            }
            QPushButton:disabled {
                background-color: #CBD5E0;
            }
        """)
        self.btn_end.setEnabled(True)
        self.combo_course.setEnabled(False)
        self.combo_mic.setEnabled(False)
        self.btn_new_course.setEnabled(False)
        self.btn_edit_hotwords.setEnabled(False)
        self.lbl_status.setText("🔴 正在录音转写中... (支持直接双击句子实时修改)")

        self.active_session_id = self.session_manager.start_session(
            course_id=course_id,
            device_index=device_idx
        )
        self.active_session_path = str(SESSIONS_DIR / self.active_session_id)

    @Slot()
    def end_session(self):
        self.btn_end.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.lbl_status.setText("⏳ 正在等待 Qwen 离线纠错队列收尾完成...")

        # Run end_session in background to keep UI responsive
        import threading
        def _bg_end():
            self.session_manager.end_session()
            self.bridge.session_finished.emit(self.active_session_id, self.active_session_path)

        threading.Thread(target=_bg_end, daemon=True).start()

    @Slot(int, str, float)
    def _handle_partial_received(self, segment_id: int, text: str, ts: float):
        if segment_id not in self.segment_cards:
            card = SegmentCard(segment_id, ts, on_text_edited=self._handle_segment_edited)
            self.segment_cards[segment_id] = card
            # Insert before the last stretch item
            count = self.subtitle_layout.count()
            self.subtitle_layout.insertWidget(count - 1, card)

        self.segment_cards[segment_id].update_partial(text)
        self._scroll_to_bottom()

    @Slot(int, str, str, float, bool, str)
    def _handle_final_received(self, segment_id: int, text: str, model: str, ts: float, success: bool = True, fallback_reason: str = ""):
        if segment_id not in self.segment_cards:
            card = SegmentCard(segment_id, ts, on_text_edited=self._handle_segment_edited)
            self.segment_cards[segment_id] = card
            count = self.subtitle_layout.count()
            self.subtitle_layout.insertWidget(count - 1, card)

        self.segment_cards[segment_id].update_final(text, model, success=success, fallback_reason=fallback_reason)
        self._scroll_to_bottom()

    @Slot(int)
    def _handle_queue_changed(self, qsize: int):
        self.lbl_queue.setText(f"离线纠错队列: {qsize} 句")

    @Slot(str, str)
    def _handle_status_changed(self, key: str, message: str):
        self.lbl_status.setText(message)

    @Slot(str, str)
    def _handle_session_finished(self, session_id: str, session_path: str):
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸ 暂停录音")
        self.btn_end.setEnabled(False)
        self.combo_course.setEnabled(True)
        self.combo_mic.setEnabled(True)
        self.btn_new_course.setEnabled(True)
        self.btn_edit_hotwords.setEnabled(True)
        self.lbl_status.setText(f"✅ 课堂已结束，纪要已保存至: {session_id}")

        QMessageBox.information(
            self,
            "课堂转写已完成",
            f"本次课堂纪要已完整生成并保存！\n\n"
            f"● 最终纪要: transcript_final.md\n"
            f"● 原始音频: audio.wav\n"
            f"● 时间轴数据: events.jsonl\n\n"
            f"存储路径: {session_path}"
        )

    def _scroll_to_bottom(self):
        v_bar = self.scroll_area.verticalScrollBar()
        v_bar.setValue(v_bar.maximum())

    def _clear_subtitles(self):
        self.segment_cards.clear()
        while self.subtitle_layout.count() > 1:
            item = self.subtitle_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    @Slot()
    def copy_transcript(self):
        texts = []
        for seg_id in sorted(self.segment_cards.keys()):
            card = self.segment_cards[seg_id]
            texts.append(card.lbl_text.text())
        full_text = "\n".join(texts)
        if full_text.strip():
            QApplication.clipboard().setText(full_text)
            self.lbl_status.setText("✅ 已将全文复制到剪贴板")
        else:
            self.lbl_status.setText("暂无字幕内容")

    @Slot()
    def open_session_dir(self):
        target_dir = Path(self.active_session_path) if self.active_session_path else SESSIONS_DIR
        if not target_dir.exists():
            target_dir = SESSIONS_DIR
        os.startfile(str(target_dir))

    def closeEvent(self, event):
        if self.session_manager and self.session_manager.recorder.is_recording:
            reply = QMessageBox.question(
                self,
                "正在录音中",
                "课堂录音正在进行中，确认要结束并退出吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.session_manager.end_session()
                event.accept()
            else:
                event.ignore()
                return
        event.accept()


def run_gui():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Modern clean font
    font = QFont("Microsoft YaHei UI", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()
