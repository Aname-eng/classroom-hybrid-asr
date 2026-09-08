# coding: utf-8
"""源代码运行入口，也可直接作为 PyInstaller 的脚本入口。"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.main import main


if __name__ == "__main__":
    main()
