# coding: utf-8
import os
import time
from typing import Optional, Callable, List, Tuple
import numpy as np
from funasr import AutoModel

class VADDetector:
    """
    FunASR FSMN-VAD 实时语音端点检测器
    
    实时分析 16kHz PCM 音频流，检测静音与语音边界：
    1. 触发 speech_start: 开始累积单句音频
    2. 持续接收语音帧
    3. 触发 speech_end: 产出单句完整语音段，分配单调递增 segment_id 并分发给离线 Qwen Worker
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        max_segment_sec: float = 25.0,
        min_silence_sec: float = 0.5,
        on_speech_start: Optional[Callable[[int, float], None]] = None,
        on_speech_end: Optional[Callable[[int, float, float, np.ndarray], None]] = None
    ):
        self.sample_rate = sample_rate
        self.max_segment_sec = max_segment_sec
        self.min_silence_sec = min_silence_sec
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end

        self.current_segment_id = 0
        self.in_speech = False
        self.speech_start_time = 0.0
        self.speech_buffer: List[np.ndarray] = []
        
        # 内部 VAD 模型
        os.environ['NO_PROXY'] = '*'
        self.vad_model = AutoModel(
            model="fsmn-vad",
            device="cpu",
            disable_update=True
        )
        self.vad_cache = {}

    def reset(self):
        self.current_segment_id = 0
        self.in_speech = False
        self.speech_start_time = 0.0
        self.speech_buffer.clear()
        self.vad_cache.clear()

    def process_chunk(self, chunk: np.ndarray, timestamp_sec: float):
        """
        处理单块音频 (通常 480ms = 7680 samples)
        """
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)

        chunk_ms = int(len(chunk) / float(self.sample_rate) * 1000)
        chunk_dur = len(chunk) / float(self.sample_rate)

        try:
            res = self.vad_model.generate(
                input=chunk,
                cache=self.vad_cache,
                is_final=False,
                chunk_size=chunk_ms
            )
        except Exception as e:
            print(f"[VADDetector Error] generate error: {e}")
            return

        segments = res[0].get("value", []) if (res and len(res) > 0) else []

        for seg in segments:
            s_ms, e_ms = seg[0], seg[1]
            if s_ms >= 0 and e_ms < 0:
                # Speech start
                self.in_speech = True
                self.speech_start_time = max(0.0, s_ms / 1000.0)
                self.speech_buffer = [chunk]
                if self.on_speech_start:
                    self.on_speech_start(self.current_segment_id + 1, self.speech_start_time)
            elif e_ms >= 0 and s_ms < 0:
                # Speech end
                self.in_speech = False
                speech_end_time = e_ms / 1000.0
                self.speech_buffer.append(chunk)
                full_segment_audio = np.concatenate(self.speech_buffer) if self.speech_buffer else chunk
                self.speech_buffer.clear()
                self.current_segment_id += 1
                if self.on_speech_end:
                    self.on_speech_end(
                        self.current_segment_id,
                        self.speech_start_time,
                        speech_end_time,
                        full_segment_audio
                    )
            elif s_ms >= 0 and e_ms >= 0:
                # Self-contained segment
                self.in_speech = False
                seg_start = s_ms / 1000.0
                seg_end = e_ms / 1000.0
                self.speech_buffer.append(chunk)
                full_segment_audio = np.concatenate(self.speech_buffer) if self.speech_buffer else chunk
                self.speech_buffer.clear()
                self.current_segment_id += 1
                if self.on_speech_end:
                    self.on_speech_end(
                        self.current_segment_id,
                        seg_start,
                        seg_end,
                        full_segment_audio
                    )

        if self.in_speech and not segments:
            self.speech_buffer.append(chunk)
            # 超时强切保护
            current_buffered_sec = sum(len(c) for c in self.speech_buffer) / float(self.sample_rate)
            if current_buffered_sec >= self.max_segment_sec:
                full_segment_audio = np.concatenate(self.speech_buffer)
                self.speech_buffer.clear()
                self.in_speech = False
                self.current_segment_id += 1
                if self.on_speech_end:
                    self.on_speech_end(
                        self.current_segment_id,
                        self.speech_start_time,
                        timestamp_sec + chunk_dur,
                        full_segment_audio
                    )

    def flush(self, timestamp_sec: float):
        """课堂结束时，清空残留音频"""
        if self.speech_buffer:
            full_segment_audio = np.concatenate(self.speech_buffer)
            self.speech_buffer.clear()
            self.in_speech = False
            self.current_segment_id += 1
            if self.on_speech_end:
                self.on_speech_end(
                    self.current_segment_id,
                    self.speech_start_time,
                    timestamp_sec,
                    full_segment_audio
                )
