# coding: utf-8
"""桌面应用启动入口。"""
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path both from source and from a checkout launched elsewhere.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Localhost model servers should not go through an inherited HTTP proxy.
os.environ.setdefault("NO_PROXY", "*")

from app.ui.main_window import run_gui


def main() -> None:
    run_gui()


if __name__ == "__main__":
    main()
