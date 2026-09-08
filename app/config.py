# coding: utf-8
"""应用配置与跨平台路径定义。

开发环境默认把数据放在项目目录，打包后则把可写数据放到各平台的用户数据目录。
模型权重不会被硬编码进程序，用户可以通过 settings.json 或环境变量指定模型/服务地址。
"""
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"  # 兼容旧代码/外部脚本
IS_FROZEN = bool(getattr(sys, "frozen", False))
RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))


def _default_data_root() -> Path:
    """返回应用的可写数据目录。

    非打包开发运行时保留项目目录布局，避免破坏现有测试和课程数据；PyInstaller
    运行时使用各平台标准用户目录，避免把数据写进只读的 .app 或安装目录。
    """
    override = os.environ.get("CLASSROOM_ASR_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    if not IS_FROZEN:
        return PROJECT_ROOT

    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "ClassroomASR"


DATA_ROOT = _default_data_root()
CONFIG_DIR = DATA_ROOT / "config"
COURSES_DIR = DATA_ROOT / "courses"
SESSIONS_DIR = DATA_ROOT / "sessions"
LOGS_DIR = DATA_ROOT / "logs"
MODEL_STORE_DIR = DATA_ROOT / "downloaded_models"
BUNDLED_COURSES_DIR = RESOURCE_ROOT / "courses"


def _optional_path(value: str) -> Optional[Path]:
    value = (value or "").strip()
    return Path(value).expanduser() if value else None


# CapsWriter/Qwen 是可选的第二遍引擎。不要在 Linux/macOS 上假定存在 Windows 路径。
_capswriter_env = _optional_path(os.environ.get("CAPSWRITER_DIR", ""))
_legacy_capswriter = Path(r"D:\software\CapsWriter-Offline")
CAPSWRITER_DIR: Optional[Path] = _capswriter_env
if CAPSWRITER_DIR is None and os.name == "nt" and _legacy_capswriter.exists():
    CAPSWRITER_DIR = _legacy_capswriter
CAPSWRITER_SERVER_EXE: Optional[Path] = (
    CAPSWRITER_DIR / "start_server.exe" if CAPSWRITER_DIR else None
)
QWEN_MODEL_DIR: Optional[Path] = (
    CAPSWRITER_DIR / "models" / "Qwen3-ASR" / "Qwen3-ASR-1.7B" if CAPSWRITER_DIR else None
)

MODELSCOPE_CACHE_DIR = Path(
    os.environ.get("MODELSCOPE_CACHE", str(Path.home() / ".cache" / "modelscope" / "models"))
).expanduser()
VAD_LOCAL_SNAPSHOT = (
    MODELSCOPE_CACHE_DIR
    / "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch"
    / "snapshots"
    / "master"
)
PARAFORMER_LOCAL_SNAPSHOT = (
    MODELSCOPE_CACHE_DIR
    / "iic--speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
    / "snapshots"
    / "master"
)

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = [8, 8, 4]  # 8 chunks left, 8 current, 4 right
CHUNK_STRIDE_SAMPLES = 8 * 960  # 7680 samples (480ms)

WS_SERVER_ADDR = "127.0.0.1"
WS_SERVER_PORT = 6016
WS_SERVER_URI = f"ws://{WS_SERVER_ADDR}:{WS_SERVER_PORT}"


def get_vad_model_path() -> str:
    if VAD_LOCAL_SNAPSHOT.exists():
        return str(VAD_LOCAL_SNAPSHOT)
    return "fsmn-vad"


def get_streaming_model_path(model_name: str = "paraformer-zh-streaming", model_path: str = "") -> str:
    """解析流式 FunASR 模型。

    ``model_path`` 可指向本地快照；否则使用 ``model_name``。Qwen3-ASR 当前是
    离线/分段识别模型，不强行伪装成 100ms 真流式模型，因此仍由第二遍适配器负责。
    """
    explicit = (model_path or "").strip()
    if explicit and Path(explicit).expanduser().exists():
        return str(Path(explicit).expanduser())

    normalized = (model_name or "paraformer-zh-streaming").strip()
    aliases = {
        "paraformer_streaming": "paraformer-zh-streaming",
        "paraformer_zh_streaming": "paraformer-zh-streaming",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in {"paraformer-zh-streaming", "paraformer_streaming"} and PARAFORMER_LOCAL_SNAPSHOT.exists():
        return str(PARAFORMER_LOCAL_SNAPSHOT)
    return normalized


def get_paraformer_model_path() -> str:
    """向旧代码提供兼容的默认流式模型路径。"""
    return get_streaming_model_path()


@dataclass
class AppConfig:
    sample_rate: int = SAMPLE_RATE
    channels: int = CHANNELS
    qwen_ws_uri: str = WS_SERVER_URI
    qwen_server_exe: str = str(CAPSWRITER_SERVER_EXE) if CAPSWRITER_SERVER_EXE else ""
    qwen_server_cwd: str = str(CAPSWRITER_DIR) if CAPSWRITER_DIR else ""
    qwen_model_path: str = str(QWEN_MODEL_DIR) if QWEN_MODEL_DIR else ""
    qwen_backend: str = "auto"
    # 用户可在打包版首次启动时选择；留空时沿用开发目录/平台用户数据目录。
    notes_dir: str = ""
    model_dir: str = ""
    # 每个模型内部的并行文件/分片下载线程数；ModelScope Hub 支持时生效。
    model_download_workers: int = 8
    model_download_max_parallel: int = 4
    audio_device_index: Optional[int] = None
    selected_course: str = "political_economy"
    chunk_size: List[int] = field(default_factory=lambda: [8, 8, 4])
    vad_max_single_segment_time_ms: int = 30000
    save_raw_audio: bool = True

    # 第一遍流式模型。默认仍是可靠的 FunASR Paraformer，可替换为兼容的本地
    # FunASR 中文流式模型；Qwen3-ASR 保留在第二遍，避免把离线模型伪装成真流式。
    streaming_backend: str = "funasr"
    streaming_model: str = "paraformer-zh-streaming"
    streaming_model_path: str = ""
    streaming_device: str = "cpu"

    # ASR 提示词/热词策略
    asr_prompt: str = ""
    auto_generate_hotwords: bool = True
    generated_hotwords_max: int = 30

    # 录音结束后的本地小模型整理。默认是已下载的 Transformers Qwen 小模型；
    # 也兼容 Ollama，没有模型/服务时自动退化为无网络的口头禅清理。
    summary_enabled: bool = True
    summary_provider: str = "local_transformers"
    summary_endpoint: str = "http://127.0.0.1:11434"
    summary_model: str = "Qwen3-0.6B"
    summary_model_path: str = ""
    summary_timeout_sec: float = 12.0
    summary_batch_chars: int = 6000
    summary_remove_off_topic: bool = True

    def notes_root(self) -> Path:
        value = (self.notes_dir or "").strip()
        return Path(value).expanduser() if value else DATA_ROOT

    def courses_root(self) -> Path:
        return self.notes_root() / "courses"

    def sessions_root(self) -> Path:
        return self.notes_root() / "sessions"

    def models_root(self) -> Path:
        value = (self.model_dir or "").strip()
        return Path(value).expanduser() if value else MODEL_STORE_DIR

    @classmethod
    def load(cls) -> "AppConfig":
        settings_file = CONFIG_DIR / "settings.json"
        if settings_file.exists():
            try:
                with open(settings_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                allowed = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
                # 空配置允许通过 CAPSWRITER_DIR 恢复当前机器上的可选服务，避免
                # 把某一台 Windows 电脑的 D:\\ 路径写死到跨平台设置文件。
                if not allowed.get("qwen_server_exe") and CAPSWRITER_SERVER_EXE:
                    allowed["qwen_server_exe"] = str(CAPSWRITER_SERVER_EXE)
                if not allowed.get("qwen_server_cwd") and CAPSWRITER_DIR:
                    allowed["qwen_server_cwd"] = str(CAPSWRITER_DIR)
                if not allowed.get("qwen_model_path") and QWEN_MODEL_DIR:
                    allowed["qwen_model_path"] = str(QWEN_MODEL_DIR)
                # 兼容旧设置文件中的 tuple/非法 chunk_size。
                if not isinstance(allowed.get("chunk_size"), list):
                    allowed.pop("chunk_size", None)
                return cls(**allowed)
            except Exception as e:
                print(f"Failed to load settings.json: {e}, using defaults.")
        return cls()

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        settings_file = CONFIG_DIR / "settings.json"
        payload = dict(self.__dict__)
        # 不把本机自动探测到的旧 CapsWriter 路径写回共享配置；下次启动仍会
        # 通过当前机器的 CAPSWRITER_DIR/legacy detection 恢复它，但 macOS/Linux
        # 不会继承 Windows 盘符。用户显式配置的其他路径照常保存。
        if not os.environ.get("CAPSWRITER_DIR"):
            legacy_values = {
                str(CAPSWRITER_SERVER_EXE) if CAPSWRITER_SERVER_EXE else "",
                str(CAPSWRITER_DIR) if CAPSWRITER_DIR else "",
                str(QWEN_MODEL_DIR) if QWEN_MODEL_DIR else "",
            }
            for key in ("qwen_server_exe", "qwen_server_cwd", "qwen_model_path"):
                if payload.get(key, "") in legacy_values:
                    payload[key] = ""
        with open(settings_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
