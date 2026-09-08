# coding: utf-8
"""可选的本地 Qwen3-ASR 推理后端。

Qwen3-ASR 的官方 Python 推理包并不属于 FunASR；因此采用懒加载，未安装
``qwen-asr`` 时不会影响现有 CapsWriter WebSocket 后端和 Paraformer 回退。
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class LocalQwenASR:
    def __init__(self, model_path: str):
        self.model_path = str(model_path or "")
        self._model = None
        self._lock = threading.Lock()
        self.last_error = ""

    @property
    def available(self) -> bool:
        return bool(self.model_path and Path(self.model_path).expanduser().is_dir())

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.available:
            raise FileNotFoundError(f"Qwen ASR model directory not found: {self.model_path}")

        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise RuntimeError(
                "本地 Qwen3-ASR 需要额外安装 qwen-asr；当前将回退到流式/ CapsWriter。"
            ) from exc

        kwargs = {"device_map": "auto"}
        try:
            import torch

            kwargs["dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        except Exception:
            pass

        try:
            self._model = Qwen3ASRModel.from_pretrained(self.model_path, **kwargs)
        except TypeError:
            # 兼容不同版本 qwen-asr 的参数名。
            kwargs.pop("dtype", None)
            self._model = Qwen3ASRModel.from_pretrained(self.model_path, **kwargs)

    @staticmethod
    def _result_text(result) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            value = result.get("text", "")
        else:
            value = getattr(result, "text", "")
        return "" if value is None else str(value)

    def transcribe(
        self,
        audio: np.ndarray,
        context: str = "",
        language: str = "zh",
    ) -> Tuple[bool, str]:
        try:
            with self._lock:
                self._ensure_loaded()
                # qwen-asr 需要同时得到波形和采样率；只传 ndarray 会被
                # 误判为一批音频或走不兼容的输入解析分支。
                result = self._model.transcribe(
                    audio=(audio, 16000),
                    language="Chinese" if language.startswith("zh") else language,
                    context=context or "",
                )
            if isinstance(result, (list, tuple)):
                text = "".join(self._result_text(item) for item in result)
            else:
                text = self._result_text(result)
            return True, text.strip()
        except Exception as exc:
            self.last_error = str(exc)
            return False, ""
