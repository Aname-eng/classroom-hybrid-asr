# coding: utf-8
import os
import time
from typing import Optional, Callable, List, Dict, Any
import numpy as np
from funasr import AutoModel
from app.config import CHUNK_SIZE, get_paraformer_model_path

class ParaformerStreamer:
    """
    FunASR Paraformer-zh-streaming 实时流式识别器
    
    第一遍识别：追求极低延迟，实时输出 partial 字幕。
    内部集成 480ms (7680 samples) accumulator，将麦克风 100ms 输入平滑聚合并保序推理。
    """
    def __init__(
        self,
        chunk_size: Optional[List[int]] = None,
        on_partial_text: Optional[Callable[[int, str, float], None]] = None
    ):
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.on_partial_text = on_partial_text
        
        # 8 * 960 = 7680 samples (480ms @ 16kHz)
        self.chunk_stride_samples = self.chunk_size[1] * 960
        self._audio_buffer = np.empty(0, dtype=np.float32)
        
        self.model: Optional[AutoModel] = None
        self.cache: Dict[str, Any] = {}
        self.current_segment_id = 1
        self.current_partial_text = ""
        self._is_initialized = False
        
        # 统计指标
        self.total_inferences = 0
        self.total_inference_time = 0.0

    def initialize(self):
        if self._is_initialized:
            return
        os.environ['NO_PROXY'] = '*'
        print("[ParaformerStreamer] Initializing FunASR paraformer-zh-streaming on CPU (pure local snapshot)...")
        t0 = time.time()
        self.model = AutoModel(
            model=get_paraformer_model_path(),
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
        self._audio_buffer = np.empty(0, dtype=np.float32)

    def process_chunk(self, chunk: np.ndarray, segment_id: int, timestamp_sec: float) -> str:
        """
        接收小块音频 (如 100ms)，累积至 480ms (7680 samples) 后执行流式推理
        """
        if not self._is_initialized:
            self.initialize()

        if segment_id != self.current_segment_id:
            # 先 flush 上一个段落未消耗的残留缓冲
            self.flush(self.current_segment_id, timestamp_sec)
            self.reset_segment(segment_id)

        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)

        self._audio_buffer = np.concatenate([self._audio_buffer, chunk]) if len(self._audio_buffer) > 0 else chunk.copy()

        while len(self._audio_buffer) >= self.chunk_stride_samples:
            asr_chunk = self._audio_buffer[:self.chunk_stride_samples]
            self._audio_buffer = self._audio_buffer[self.chunk_stride_samples:]
            self._infer_chunk(asr_chunk, timestamp_sec, is_final=False)

        return self.current_partial_text

    def _infer_chunk(self, chunk_data: np.ndarray, timestamp_sec: float, is_final: bool = False):
        t0 = time.monotonic()
        try:
            res = self.model.generate(
                input=chunk_data,
                cache=self.cache,
                is_final=is_final,
                chunk_size=self.chunk_size,
                encoder_chunk_look_back=4,
                decoder_chunk_look_back=1
            )
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
        """
        句子结束或课堂收尾时，将不足 480ms 的尾音以 is_final=True 推理完成
        """
        if not self._is_initialized:
            return

        target_seg = segment_id or self.current_segment_id
        if len(self._audio_buffer) > 0:
            remaining = self._audio_buffer
            self._audio_buffer = np.empty(0, dtype=np.float32)
            self._infer_chunk(remaining, timestamp_sec, is_final=True)
