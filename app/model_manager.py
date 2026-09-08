# coding: utf-8
"""本地模型目录、下载、导入、启用和删除管理。

下载源默认使用 ModelScope（中国镜像/中国模型社区），不依赖用户手动复制权重。
模型下载本身与推理解耦：即使某个可选推理后端未安装，应用仍可使用流式回退。
"""
from __future__ import annotations

import inspect
import json
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from app.config import AppConfig, CONFIG_DIR, MODELSCOPE_CACHE_DIR


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    kind: str
    modelscope_id: str
    model_name: str
    size_hint: str
    description: str
    default: bool = False

    @property
    def source_url(self) -> str:
        return f"https://www.modelscope.cn/models/{self.modelscope_id}"


MODEL_CATALOG: List[ModelSpec] = [
    ModelSpec(
        key="qwen3_asr_0_6b",
        name="Qwen3-ASR 0.6B",
        kind="offline_asr",
        modelscope_id="Qwen/Qwen3-ASR-0.6B",
        model_name="Qwen3-ASR-0.6B",
        size_hint="约 1–2 GB",
        description="千问中文语音识别小模型，用于停顿后的权威纠错。需要 qwen-asr 后端；没有后端时自动回退。",
        default=True,
    ),
    ModelSpec(
        key="qwen3_0_6b",
        name="Qwen3 0.6B",
        kind="summary",
        modelscope_id="Qwen/Qwen3-0.6B",
        model_name="Qwen3-0.6B",
        size_hint="约 1–2 GB",
        description="千问中文小语言模型，用于课程整理、删除口头禅和自动生成热词。",
        default=True,
    ),
    ModelSpec(
        key="paraformer_zh_streaming",
        name="Paraformer 中文流式",
        kind="streaming_asr",
        modelscope_id="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
        model_name="paraformer-zh-streaming",
        size_hint="约 1 GB",
        description="阿里达摩院 FunASR 中文真流式模型，负责低延迟字幕。",
        default=True,
    ),
    ModelSpec(
        key="qwen3_asr_1_7b",
        name="Qwen3-ASR 1.7B",
        kind="offline_asr",
        modelscope_id="Qwen/Qwen3-ASR-1.7B",
        model_name="Qwen3-ASR-1.7B",
        size_hint="约 3–5 GB",
        description="更大、更准确的千问语音模型，适合内存充足的设备。",
        default=False,
    ),
]


ModelEventCallback = Callable[[str, ModelSpec, str], None]


class ModelManager:
    """管理用户数据目录中的模型副本和当前启用配置。"""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        models_dir: Optional[Path] = None,
        catalog: Optional[List[ModelSpec]] = None,
        registry_file: Optional[Path] = None,
    ):
        self.config = config or AppConfig.load()
        self.models_dir = Path(models_dir or self.config.models_root())
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = Path(registry_file or (CONFIG_DIR / "models.json"))
        self.catalog: Dict[str, ModelSpec] = {
            spec.key: spec for spec in (catalog or MODEL_CATALOG)
        }
        self._lock = threading.RLock()
        self._download_threads: Dict[str, threading.Thread] = {}
        self.download_workers = self._bounded_int(
            getattr(self.config, "model_download_workers", 8), 1, 32, 8
        )
        max_parallel = self._bounded_int(
            getattr(self.config, "model_download_max_parallel", 4), 1, 4, 4
        )
        self._download_slots = threading.Semaphore(max_parallel)
        self.records: Dict[str, Dict[str, object]] = self._load_registry()

    @staticmethod
    def _bounded_int(value: object, minimum: int, maximum: int, fallback: int) -> int:
        try:
            return max(minimum, min(maximum, int(value)))
        except (TypeError, ValueError):
            return fallback

    def _load_registry(self) -> Dict[str, Dict[str, object]]:
        if not self.registry_file.exists():
            return {}
        try:
            data = json.loads(self.registry_file.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            print(f"[ModelManager] Failed to read model registry: {exc}")
            return {}

    def _save_registry(self) -> None:
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.registry_file.write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def list_specs(self) -> List[ModelSpec]:
        return list(self.catalog.values())

    def get_spec(self, key: str) -> Optional[ModelSpec]:
        return self.catalog.get(key)

    def target_dir(self, spec: ModelSpec) -> Path:
        return self.models_dir / spec.key

    def local_path(self, key: str) -> Optional[Path]:
        with self._lock:
            spec = self.get_spec(key)
            if spec is None:
                return None
            record = self.records.get(key) or {}
            if bool(record.get("disabled", False)):
                return None
            path_value = str(record.get("path", "")).strip()
            path = Path(path_value).expanduser() if path_value else self.target_dir(spec)
            if path.exists() and path.is_dir():
                return path
            return None

    def is_downloaded(self, key: str) -> bool:
        return self.local_path(key) is not None

    def status(self, key: str) -> str:
        with self._lock:
            thread = self._download_threads.get(key)
            downloading = bool(thread and thread.is_alive())
        if downloading:
            return "downloading"
        return "installed" if self.is_downloaded(key) else "not_installed"

    def _notify(self, callback: Optional[ModelEventCallback], event: str, spec: ModelSpec, message: str) -> None:
        if callback:
            try:
                callback(event, spec, message)
            except Exception as exc:
                print(f"[ModelManager] callback error: {exc}")

    def _record_installed(self, spec: ModelSpec, path: Path, imported: bool = False) -> None:
        with self._lock:
            self.records[spec.key] = {
                "path": str(path),
                "model_name": spec.model_name,
                "kind": spec.kind,
                "modelscope_id": spec.modelscope_id,
                "imported": imported,
            }
            self._save_registry()

    def download(
        self,
        key: str,
        callback: Optional[ModelEventCallback] = None,
        activate: bool = True,
    ) -> Optional[Path]:
        spec = self.get_spec(key)
        if spec is None:
            raise KeyError(f"Unknown model: {key}")

        # 只保护注册表和同一模型的状态检查；绝不能把网络下载放在全局锁内，
        # 否则默认模型下载时，用户点击另一个模型会一直排队，看起来像 UI 卡死。
        with self._lock:
            if bool((self.records.get(key) or {}).get("disabled", False)):
                self.records.pop(key, None)
            existing = self.local_path(key)
            if existing is not None:
                if activate:
                    self.activate(key)
                self._notify(callback, "installed", spec, f"已存在：{existing}")
                return existing

            target = self.target_dir(spec)
            target.parent.mkdir(parents=True, exist_ok=True)
            temp_target = target.with_name(f".{target.name}.downloading")
            if temp_target.exists():
                shutil.rmtree(temp_target, ignore_errors=True)

        self._notify(callback, "started", spec, f"正在从 ModelScope 下载 {spec.name}…")
        try:
            from modelscope import snapshot_download

            # ModelScope Hub 新版内部采用类似 FDM 的并行文件/分片下载，
            # max_workers 可同时加速多个大权重文件；旧版本没有该参数时回退。
            download_kwargs = {
                "model_id": spec.modelscope_id,
                "cache_dir": str(MODELSCOPE_CACHE_DIR),
                "local_dir": str(temp_target),
                "endpoint": "https://www.modelscope.cn",
            }
            try:
                if "max_workers" in inspect.signature(snapshot_download).parameters:
                    download_kwargs["max_workers"] = self.download_workers
            except (TypeError, ValueError):
                pass
            downloaded = snapshot_download(**download_kwargs)
            downloaded_path = Path(downloaded or temp_target)
            if downloaded_path.resolve() != temp_target.resolve() and downloaded_path.exists():
                # 某些 ModelScope 版本返回 cache snapshot 路径；复制到应用目录。
                shutil.copytree(downloaded_path, temp_target, dirs_exist_ok=True)
            if not temp_target.exists():
                raise RuntimeError("ModelScope 没有返回有效的模型目录")
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            temp_target.replace(target)
            self._record_installed(spec, target, imported=False)
            if activate:
                self.activate(key)
            self._notify(callback, "completed", spec, f"下载完成：{target}")
            return target
        except Exception as exc:
            shutil.rmtree(temp_target, ignore_errors=True)
            self._notify(callback, "failed", spec, f"下载失败：{exc}")
            return None

    def _download_worker(
        self,
        key: str,
        callback: Optional[ModelEventCallback],
        activate: bool,
    ) -> None:
        self._download_slots.acquire()
        try:
            self.download(key, callback=callback, activate=activate)
        finally:
            self._download_slots.release()
            with self._lock:
                current = self._download_threads.get(key)
                if current is threading.current_thread():
                    self._download_threads.pop(key, None)

    def download_async(
        self,
        key: str,
        callback: Optional[ModelEventCallback] = None,
        activate: bool = True,
    ) -> bool:
        with self._lock:
            thread = self._download_threads.get(key)
            if thread and thread.is_alive():
                return False
            thread = threading.Thread(
                target=self._download_worker,
                args=(key, callback, activate),
                daemon=True,
                name=f"ModelDownload-{key}",
            )
            self._download_threads[key] = thread
            thread.start()
            return True

    def ensure_default_models_async(self, callback: Optional[ModelEventCallback] = None) -> None:
        """首次启动并行检查/下载默认模型，不阻塞界面，也不互相排队。"""
        def worker() -> None:
            defaults = [spec for spec in self.catalog.values() if spec.default]
            pending: List[ModelSpec] = []
            failed: List[str] = []
            for spec in defaults:
                with self._lock:
                    disabled = bool((self.records.get(spec.key) or {}).get("disabled", False))
                if disabled:
                    self._notify(callback, "skipped", spec, f"已按上次设置跳过默认模型：{spec.name}")
                    continue
                if self.is_downloaded(spec.key):
                    if self.activate(spec.key):
                        self._notify(callback, "installed", spec, f"已就绪：{spec.name}")
                    continue
                if self.download_async(spec.key, callback=callback, activate=True):
                    pending.append(spec)

            while any(self.status(spec.key) == "downloading" for spec in pending):
                time.sleep(0.2)
            for spec in pending:
                if not self.is_downloaded(spec.key):
                    failed.append(spec.name)
            if defaults:
                message = "默认模型检查完成"
                if failed:
                    message += "；失败：" + "、".join(failed)
                self._notify(callback, "defaults_completed", defaults[-1], message)

        threading.Thread(target=worker, daemon=True, name="DefaultModelBootstrap").start()

    def import_local(self, key: str, source_dir: Path, activate: bool = True) -> Path:
        spec = self.get_spec(key)
        if spec is None:
            raise KeyError(f"Unknown model: {key}")
        source = Path(source_dir).expanduser().resolve()
        if not source.exists() or not source.is_dir():
            raise ValueError("请选择包含模型文件的本地目录")
        self.records.pop(key, None)
        self._record_installed(spec, source, imported=True)
        if activate:
            self.activate(key)
        return source

    def delete(self, key: str) -> bool:
        spec = self.get_spec(key)
        if spec is None:
            return False
        record = self.records.get(key) or {}
        path = self.local_path(key)
        if path is None:
            self.records.pop(key, None)
            self._save_registry()
            return False

        # 导入的目录属于用户，不直接删除，只移除应用登记；应用下载的副本才删除。
        if not bool(record.get("imported", False)):
            if path.resolve() == self.target_dir(spec).resolve():
                shutil.rmtree(path, ignore_errors=False)
        active_attr = {
            "streaming_asr": "streaming_model_path",
            "summary": "summary_model_path",
            "offline_asr": "qwen_model_path",
        }.get(spec.kind)
        if active_attr and str(getattr(self.config, active_attr, "")) == str(path):
            setattr(self.config, active_attr, "")
            if spec.kind == "offline_asr":
                self.config.qwen_backend = "auto"
            self.config.save()
        # 保留删除标记，避免下次启动“默认模型自动下载”立即把用户刚删掉的
        # 权重重新拉回来；用户在模型窗口点击下载时会清除该标记。
        self.records[key] = {
            "disabled": True,
            "model_name": spec.model_name,
            "kind": spec.kind,
            "modelscope_id": spec.modelscope_id,
        }
        self._save_registry()
        return True

    def activate(self, key: str) -> bool:
        with self._lock:
            spec = self.get_spec(key)
            path = self.local_path(key)
            if spec is None or path is None:
                return False

            if spec.kind == "streaming_asr":
                self.config.streaming_backend = "funasr"
                self.config.streaming_model = spec.model_name
                self.config.streaming_model_path = str(path)
            elif spec.kind == "summary":
                self.config.summary_provider = "local_transformers"
                self.config.summary_model = spec.model_name
                self.config.summary_model_path = str(path)
            elif spec.kind == "offline_asr":
                self.config.qwen_backend = "local_qwen"
                self.config.qwen_model_path = str(path)
            self.config.save()
            return True

    def active_model_keys(self) -> Dict[str, str]:
        return {
            "streaming_asr": getattr(self.config, "streaming_model_path", ""),
            "summary": getattr(self.config, "summary_model_path", ""),
            "offline_asr": getattr(self.config, "qwen_model_path", ""),
        }
