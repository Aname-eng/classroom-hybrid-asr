# coding: utf-8
import os
import time
from typing import Optional, Callable, List, Dict, Any
import numpy as np
from funasr import AutoModel
from app.config import CHUNK_SIZE

class ParaformerStreamer:
    """
    FunASR Paraformer-zh-streaming 实时流式识别器
    
    第一遍识别：追求极低延迟，实时输出 partial 字幕。
    默认采用 480ms chunk_size [8, 8, 4]
    """
    def __init__(
        self,
        chunk_size: Optional[List[int]] = None,
        on_partial_text: Optional[Callable[[int, str, float], None]] = None
    ):
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.on_partial_text = on_partial_text
        
        self.model: Optional[AutoModel] = None
        self.cache: Dict[str, Any] = {}
        self.current_segment_id = 1
        self.current_partial_text = ""
        self._is_initialized = False

    def initialize(self):
        if self._is_initialized:
            return
        os.environ['NO_PROXY'] = '*'
        print("[ParaformerStreamer] Initializing FunASR paraformer-zh-streaming on CPU...")
        t0 = time.time()
        self.model = AutoModel(
            model="paraformer-zh-streaming",
            device="cpu",
            disable_update=True
        )
        self._is_initialized = True
        print(f"[ParaformerStreamer] Initialized in {time.time() - t0:.2f}s")

    def reset_segment(self, new_segment_id: int):
        """进入新句子时重置流式状态"""
        self.current_segment_id = new_segment_id
        self.current_partial_text = ""
        self.cache = {}

    def process_chunk(self, chunk: np.ndarray, segment_id: int, timestamp_sec: float, is_final: bool = False) -> str:
        """
        处理音频块并输出实时 partial 文本
        """
        if not self._is_initialized:
            self.initialize()

        if segment_id != self.current_segment_id:
            self.reset_segment(segment_id)

        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)

        t0 = time.time()
        try:
            res = self.model.generate(
                input=chunk,
                cache=self.cache,
                is_final=is_final,
                chunk_size=self.chunk_size,
                encoder_chunk_look_back=4,
                decoder_chunk_look_back=1
            )
            latency = time.time() - t0

            if res and len(res) > 0 and res[0].get("text"):
                text = res[0]["text"].strip()
                if text:
                    self.current_partial_text += text
                    if self.on_partial_text:
                        self.on_partial_text(self.current_segment_id, self.current_partial_text, timestamp_sec)
        except Exception as e:
            print(f"[ParaformerStreamer Error] Chunk inference error: {e}")

        return self.current_partial_text
