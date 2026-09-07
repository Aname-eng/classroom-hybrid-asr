import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ui.main_window import MainWindow, UiBridge, SegmentCard
print("[OK] UI modules imported successfully!")
