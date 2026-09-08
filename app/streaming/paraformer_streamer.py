# coding: utf-8
import os
import time
from typing import Optional, Callable, List, Dict, Any, Iterable

import numpy as np
from funasr import AutoModel

from app.config import CHUNK_SIZE, get_streaming_model_path


class ParaformerStreamer:
    """FunASR 中文流式识别器。

    这是一个可配置的 FunASR 流式后端，而不是把离线 Qwen3-ASR 强行切成很小的
    音频片段。默认模型仍然是 Paraformer-zh-streaming；通过 settings.json 的
    ``streaming_model``/``streaming_model_path`` 可以替换为兼容的本地中文流式模型。
    """

    def __init__(
        self,
        chunk_size: Optional[List[int]] = None,
        on_partial_text: Optional[Callable[[int, str, float], None]] = None,
        model_name: str = "paraformer-zh-streaming",
        model_path: str = "",
        device: str = "cpu",
        backend: str = "funasr",
    ):
        self.backend = (backend or "funasr").strip().lower()
        if self.backend not in {"funasr", "paraformer"}:
            raise ValueError(f"Unsupported streaming backend for ParaformerStreamer: {backend}")
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.on_partial_text = on_partial_text
        self.model_name = model_name or "paraformer-zh-streaming"
        self.model_path = model_path or ""
        self.device = device or "cpu"

        # 8 * 960 = 7680 samples (480ms @ 16kHz)
        self.chunk_stride_samples = self.chunk_size[1] * 960
        self._audio_buffer = np.empty(0, dtype=np.float32)

        self.model: Optional[AutoModel] = None
        self.cache: Dict[str, Any] = {}
        self.current_segment_id = 1
        self.current_partial_text = ""
        self._is_initialized = False
        self.prompt = ""
        self.hotwords: List[str] = []
        self._hotword_argument_supported: Optional[bool] = None

        # 统计指标
        self.total_inferences = 0
        self.total_inference_time = 0.0

    @property
    def resolved_model(self) -> str:
        return get_streaming_model_path(self.model_name, self.model_path)

    def set_context(self, prompt: str = "", hotwords: Optional[Iterable[str]] = None) -> None:
        """在新会话开始前设置主题与热词。

        FunASR 不同模型版本对 ``hotword`` 参数的支持并不完全一致，因此真正
        推理时会先尝试带热词调用，若模型不接受该参数则自动回退到标准接口。
        """
        self.prompt = (prompt or "").strip()
        self.hotwords = []
        for word in hotwords or []:
            word = str(word).strip()
            if word and word not in self.hotwords:
                self.hotwords.append(word)
        self._hotword_argument_supported = None

    def initialize(self):
        if self._is_initialized:
            return
        os.environ.setdefault("NO_PROXY", "*")
        model_path = self.resolved_model
        print(
            f"[ParaformerStreamer] Initializing {model_path} on {self.device} "
            f"(streaming_backend={self.backend}, local FunASR)..."
        )
        t0 = time.time()
        self.model = AutoModel(
            model=model_path,
            device=self.device,
            disable_update=True,
        )
        self._is_initialized = True
        print(f"[ParaformerStreamer] Initialized in {time.time() - t0:.2f}s")

    def reset_segment(self, new_segment_id: int):
        """进入新句子时重置流式状态。"""
        self.current_segment_id = new_segment_id
        self.current_partial_text = ""
        self.cache = {}
        self._audio_buffer = np.empty(0, dtype=np.float32)

    def process_chunk(self, chunk: np.ndarray, segment_id: int, timestamp_sec: float) -> str:
        """接收小块音频，累积到流式步长后推理。"""
        if not self._is_initialized:
            self.initialize()

        if segment_id != self.current_segment_id:
            self.flush(self.current_segment_id, timestamp_sec)
            self.reset_segment(segment_id)

        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)

        self._audio_buffer = (
            np.concatenate([self._audio_buffer, chunk])
            if len(self._audio_buffer) > 0
            else chunk.copy()
        )

        while len(self._audio_buffer) >= self.chunk_stride_samples:
            asr_chunk = self._audio_buffer[: self.chunk_stride_samples]
            self._audio_buffer = self._audio_buffer[self.chunk_stride_samples :]
            self._infer_chunk(asr_chunk, timestamp_sec, is_final=False)

        return self.current_partial_text

    def _infer_chunk(self, chunk_data: np.ndarray, timestamp_sec: float, is_final: bool = False):
        t0 = time.monotonic()
        if self.model is None:
            return

        kwargs: Dict[str, Any] = {
            "input": chunk_data,
            "cache": self.cache,
            "is_final": is_final,
            "chunk_size": self.chunk_size,
            "encoder_chunk_look_back": 4,
            "decoder_chunk_look_back": 1,
        }
        hotword = " ".join(self.hotwords)
        if hotword and self._hotword_argument_supported is not False:
            kwargs["hotword"] = hotword

        try:
            try:
                res = self.model.generate(**kwargs)
                if hotword:
                    self._hotword_argument_supported = True
            except TypeError:
                # 老版本/特定流式模型不接受 hotword；只在第一次遇到时重试一次。
                if "hotword" not in kwargs:
                    raise
                kwargs.pop("hotword", None)
                self._hotword_argument_supported = False
                res = self.model.generate(**kwargs)

            dur = time.monotonic() - t0
            self.total_inferences += 1
            self.total_inference_time += dur

            if res and len(res) > 0 and res[0].get("text"):
                text = res[0]["text"].strip()
                if text:
                    self.current_partial_text += text
                    if self.on_partial_text:
                        self.on_partial_text(self.current_segment_id, self.current_partial_text, timestamp_sec)
        except Exception as e:
            print(f"[ParaformerStreamer Error] Chunk inference error: {e}")

    def flush(self, segment_id: Optional[int] = None, timestamp_sec: float = 0.0):
        """句子结束或课堂收尾时冲刷不足一个步长的尾音。"""
        if not self._is_initialized:
            return

        if len(self._audio_buffer) > 0:
            remaining = self._audio_buffer
            self._audio_buffer = np.empty(0, dtype=np.float32)
            self._infer_chunk(remaining, timestamp_sec, is_final=True)
