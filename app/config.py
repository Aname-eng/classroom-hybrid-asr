# coding: utf-8
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
CONFIG_DIR = PROJECT_ROOT / "config"
COURSES_DIR = PROJECT_ROOT / "courses"
SESSIONS_DIR = PROJECT_ROOT / "sessions"
LOGS_DIR = PROJECT_ROOT / "logs"

CAPSWRITER_DIR = Path(r"D:\software\CapsWriter-Offline")
CAPSWRITER_SERVER_EXE = CAPSWRITER_DIR / "start_server.exe"
QWEN_MODEL_DIR = CAPSWRITER_DIR / "models" / "Qwen3-ASR" / "Qwen3-ASR-1.7B"

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = [8, 8, 4] # 8 chunks left (480ms), 8 chunks current (480ms), 4 chunks right (240ms)
CHUNK_STRIDE_SAMPLES = 8 * 960 # 7680 samples (480ms)

WS_SERVER_ADDR = "127.0.0.1"
WS_SERVER_PORT = 6016
WS_SERVER_URI = f"ws://{WS_SERVER_ADDR}:{WS_SERVER_PORT}"

@dataclass
class AppConfig:
    sample_rate: int = SAMPLE_RATE
    channels: int = CHANNELS
    qwen_ws_uri: str = WS_SERVER_URI
    qwen_server_exe: str = str(CAPSWRITER_SERVER_EXE)
    qwen_server_cwd: str = str(CAPSWRITER_DIR)
    qwen_model_path: str = str(QWEN_MODEL_DIR)
    audio_device_index: Optional[int] = None
    selected_course: str = "political_economy"
    chunk_size: List[int] = field(default_factory=lambda: [8, 8, 4])
    vad_max_single_segment_time_ms: int = 30000
    save_raw_audio: bool = True

    @classmethod
    def load(cls) -> 'AppConfig':
        settings_file = CONFIG_DIR / "settings.json"
        if settings_file.exists():
            try:
                with open(settings_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
            except Exception as e:
                print(f"Failed to load settings.json: {e}, using defaults.")
        return cls()

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        settings_file = CONFIG_DIR / "settings.json"
        with open(settings_file, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2, ensure_ascii=False)
