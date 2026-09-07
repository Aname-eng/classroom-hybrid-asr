# coding: utf-8
"""
课堂实时转写系统 (Hybrid 2-Pass Classroom ASR)
主程序启动入口
"""
import os
import sys
from pathlib import Path

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure proxy settings don't block local server communication
os.environ["NO_PROXY"] = "*"

from app.ui.main_window import run_gui

if __name__ == "__main__":
    run_gui()
