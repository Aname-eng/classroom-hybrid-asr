# coding: utf-8
import os
import time
from typing import Optional, Callable, List, Tuple
import numpy as np
from funasr import AutoModel
from app.config import get_vad_model_path

class VADDetector:
    """
    FunASR FSMN-VAD 实时语音端点检测器
    
    1. 触发 speech_start: 开始累积单句音频
    2. 持续接收语音帧并动态分析边界
    3. 触发 speech_end: 产出单句完整语音段，分发给离线 Qwen Worker
    4. 超长语音保护 (Continuation Split):
       当讲师连续讲话超过 max_segment_sec 时，主动切割当前句并无缝开启 continuation segment，
       继续保持 in_speech=True，绝不丢弃后续语音。
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

        self.current_segment_id = 1
        self.in_speech = False
        self.speech_start_time = 0.0
        self.speech_buffer: List[np.ndarray] = []
        
        # 内部 VAD 模型 (优先加载本地快照，零网络请求/零校验开销)
        os.environ['NO_PROXY'] = '*'
        self.vad_model = AutoModel(
            model=get_vad_model_path(),
            device="cpu",
            disable_update=True
        )
        self.vad_cache = {}

    def reset(self):
        self.current_segment_id = 1
        self.in_speech = False
        self.speech_start_time = 0.0
        self.speech_buffer.clear()
        self.vad_cache.clear()

    def process_chunk(self, chunk: np.ndarray, timestamp_sec: float):
        """
        处理单块音频 (通常 100ms ~ 480ms)
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
                if not self.in_speech:
                    self.in_speech = True
                    self.speech_start_time = max(0.0, timestamp_sec)
                    self.speech_buffer = [chunk]
                    if self.on_speech_start:
                        self.on_speech_start(self.current_segment_id, self.speech_start_time)
                else:
                    self.speech_buffer.append(chunk)

            elif e_ms >= 0 and s_ms < 0:
                # Speech end
                if self.in_speech:
                    self.in_speech = False
                    speech_end_time = timestamp_sec + chunk_dur
                    self.speech_buffer.append(chunk)
                    full_segment_audio = np.concatenate(self.speech_buffer) if self.speech_buffer else chunk
                    self.speech_buffer.clear()
                    
                    seg_id = self.current_segment_id
                    self.current_segment_id += 1
                    if self.on_speech_end:
                        self.on_speech_end(
                            seg_id,
                            self.speech_start_time,
                            speech_end_time,
                            full_segment_audio
                        )
            elif s_ms >= 0 and e_ms >= 0:
                # Self-contained segment
                self.in_speech = False
                seg_start = timestamp_sec
                seg_end = timestamp_sec + chunk_dur
                self.speech_buffer.append(chunk)
                full_segment_audio = np.concatenate(self.speech_buffer) if self.speech_buffer else chunk
                self.speech_buffer.clear()
                
                seg_id = self.current_segment_id
                self.current_segment_id += 1
                if self.on_speech_end:
                    self.on_speech_end(
                        seg_id,
                        seg_start,
                        seg_end,
                        full_segment_audio
                    )

        if self.in_speech and not segments:
            self.speech_buffer.append(chunk)
            
            # 超时强切保护（无缝 Continuation 切割，绝不重置 in_speech 为 False）
            current_buffered_sec = sum(len(c) for c in self.speech_buffer) / float(self.sample_rate)
            if current_buffered_sec >= self.max_segment_sec:
                full_segment_audio = np.concatenate(self.speech_buffer)
                self.speech_buffer.clear()
                
                split_seg_id = self.current_segment_id
                seg_end_time = timestamp_sec + chunk_dur
                seg_start_time = self.speech_start_time
                
                # 开启新 continuation segment
                self.current_segment_id += 1
                self.speech_start_time = seg_end_time
                # 保持 in_speech = True
                
                if self.on_speech_end:
                    self.on_speech_end(
                        split_seg_id,
                        seg_start_time,
                        seg_end_time,
                        full_segment_audio
                    )
                if self.on_speech_start:
                    self.on_speech_start(self.current_segment_id, self.speech_start_time)

    def flush(self, timestamp_sec: float):
        """课堂结束时，清空残留音频"""
        if self.speech_buffer:
            full_segment_audio = np.concatenate(self.speech_buffer)
            self.speech_buffer.clear()
            self.in_speech = False
            seg_id = self.current_segment_id
            self.current_segment_id += 1
            if self.on_speech_end:
                self.on_speech_end(
                    seg_id,
                    self.speech_start_time,
                    timestamp_sec,
                    full_segment_audio
                )
