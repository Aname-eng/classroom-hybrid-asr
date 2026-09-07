# coding: utf-8
import os
import sys
import time
import json
import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Callable, Dict, Any, List
import numpy as np

from app.config import (
    AppConfig, SESSIONS_DIR, SAMPLE_RATE,
    QWEN_MODEL_DIR, CAPSWRITER_DIR
)
from app.audio.source import AudioSource
from app.audio.recorder import AudioRecorder
from app.vad.vad_detector import VADDetector
from app.streaming.paraformer_streamer import ParaformerStreamer
from app.offline.qwen_adapter import QwenAdapter
from app.offline.qwen_worker import QwenWorker, SegmentTask, SegmentResult
from app.courses.course_manager import CourseManager, CourseInfo

@dataclass
class SegmentRecord:
    segment_id: int
    start_sec: float
    end_sec: float
    online_text: str = ""
    final_text: str = ""
    final_model: str = ""
    latency_sec: float = 0.0
    status: str = "partial" # "partial" or "final"
    final_success: Optional[bool] = None
    fallback_reason: Optional[str] = None

class SessionManager:
    """
    课堂实时转写会话管理器
    
    统一编排录音、VAD断句、流式识别与离线Qwen修正，全量持久化 raw evidence。
    """
    def __init__(
        self,
        config: Optional[AppConfig] = None,
        on_partial_subtitle: Optional[Callable[[int, str, float], None]] = None,
        on_final_subtitle: Optional[Callable[[int, str, str, float, bool, Optional[str]], None]] = None,
        on_status_update: Optional[Callable[[Dict[str, Any]], None]] = None,
        audio_source: Optional[AudioSource] = None
    ):
        self.config = config or AppConfig.load()
        self.on_partial_subtitle = on_partial_subtitle
        self.on_final_subtitle = on_final_subtitle
        self.on_status_update = on_status_update

        self.course_manager = CourseManager()
        self.current_course: Optional[CourseInfo] = None
        self.session_dir: Optional[Path] = None
        self.session_id: Optional[str] = None
        self.session_start_time: float = 0.0
        self.session_start_dt: Optional[datetime.datetime] = None

        self.segments: Dict[int, SegmentRecord] = {}
        self.current_speaking_segment_id: int = 1
        self._pending_speech_ends: List[Dict[str, Any]] = []

        # 初始化核心组件
        self.qwen_adapter = QwenAdapter(
            ws_uri=self.config.qwen_ws_uri,
            server_exe=self.config.qwen_server_exe,
            server_cwd=self.config.qwen_server_cwd
        )
        self.qwen_worker = QwenWorker(
            adapter=self.qwen_adapter,
            on_result=self._on_qwen_result,
            on_queue_change=self._on_qwen_queue_change
        )
        self.paraformer_streamer = ParaformerStreamer(
            chunk_size=self.config.chunk_size,
            on_partial_text=self._on_streaming_partial
        )
        self.vad_detector = VADDetector(
            sample_rate=SAMPLE_RATE,
            max_segment_sec=25.0,
            on_speech_start=self._on_vad_speech_start,
            on_speech_end=self._on_vad_speech_end
        )
        self.recorder = AudioRecorder(
            sample_rate=SAMPLE_RATE,
            device_index=self.config.audio_device_index,
            on_audio_chunk=self._on_audio_chunk,
            source=audio_source
        )

        self._is_session_active = False

    @property
    def is_active(self) -> bool:
        return self._is_session_active

    def select_course(self, course_id: str) -> Optional[CourseInfo]:
        course = self.course_manager.get_course(course_id)
        if course:
            self.current_course = course
            self.config.selected_course = course_id
            self.config.save()
        return course

    def start_session(
        self,
        course_id: str,
        source: Optional[AudioSource] = None,
        device_index: Optional[int] = None
    ) -> str:
        if self._is_session_active:
            raise RuntimeError("A session is already active. Please end the current session first.")

        # 显式更新录音输入设备（支持 device_index=None 作为系统默认麦克风）
        self.recorder.device_index = device_index
        if source is None:
            self.config.audio_device_index = device_index

        course = self.course_manager.get_course(course_id)
        if not course:
            raise ValueError(f"Course '{course_id}' not found in configuration.")
        self.current_course = course

        # 在录音启动前预热/初始化 Paraformer 与 VAD 模型（若未初始化）
        if not self.paraformer_streamer._is_initialized:
            print("[SessionManager] Pre-initializing Paraformer streaming model...")
            self.paraformer_streamer.initialize()

        # 创建 Session 目录
        now = datetime.datetime.now()
        self.session_start_dt = now
        self.session_start_time = time.time()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        self.session_id = f"{date_str}_{self.current_course.id}_{time_str}"
        self.session_dir = SESSIONS_DIR / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.segments.clear()
        self._pending_speech_ends.clear()
        self.current_speaking_segment_id = 1
        self.vad_detector.reset()
        self.paraformer_streamer.reset_segment(1)
        self.qwen_worker.reset_session()
        self.qwen_worker.start()

        # 记录 session_start 事件
        self._log_event("session_start", {
            "session_id": self.session_id,
            "course": self.current_course.name,
            "course_id": self.current_course.id,
            "start_time": now.isoformat(),
            "sample_rate": SAMPLE_RATE,
            "device_index": self.recorder.device_index,
            "hotwords": self.current_course.hotwords
        })

        # 启动主录音机
        audio_wav_path = str(self.session_dir / "audio.wav")
        self.recorder.start(output_wav_path=audio_wav_path, source=source)
        self._is_session_active = True

        print(f"[SessionManager] Session started: {self.session_id} on device {self.recorder.device_index}")
        self._notify_status()
        return self.session_id

    def pause_session(self):
        if self._is_session_active:
            self.recorder.pause()
            self._log_event("session_pause", {"timestamp": self.recorder.total_recorded_seconds})
            self._notify_status()

    def resume_session(self):
        if self._is_session_active:
            self.recorder.resume()
            self._log_event("session_resume", {"timestamp": self.recorder.total_recorded_seconds})
            self._notify_status()

    def end_session(self, timeout: float = 60.0):
        """
        严谨的阶段化收尾流程：
        1. 停止麦克风产生新音频
        2. 排空已录制音频至 WAV 与下游处理
        3. Flush VAD 尾部语音段
        4. 排空所有 pending speech ends 并 flush Paraformer 流式缓冲
        5. 等待 Qwen 离线纠错队列排空 (有界超时，超时自动取消 in-flight)
        6. 关闭录音文件
        7. 保存所有产物文档与元数据
        """
        if not self._is_session_active:
            return

        print("[SessionManager] Step 1: Stopping audio capture stream...")
        self.recorder.stop_capture()

        print("[SessionManager] Step 2: Draining recorder queue to WAV and ASR processing...")
        self.recorder.wait_until_drained(timeout=max(10.0, timeout))

        cur_time = self.recorder.total_recorded_seconds

        print("[SessionManager] Step 3: Flushing VAD speech segment buffer...")
        self.vad_detector.flush(cur_time)

        print("[SessionManager] Step 4: Flushing pending speech ends and Paraformer buffer...")
        self._drain_pending_speech_ends(timestamp_sec=cur_time)
        self.paraformer_streamer.flush(timestamp_sec=cur_time)

        print(f"[SessionManager] Step 5: Waiting for Qwen offline queue to clear (timeout={timeout}s)...")
        qwen_completed = self.qwen_worker.stop(wait_finish=True, timeout=timeout)
        if not qwen_completed:
            print("[SessionManager Warning] Qwen worker timed out during shutdown, flushed as fallback.")

        print("[SessionManager] Step 6: Closing WAV file...")
        self.recorder.close()

        # 先置 session_active = False，确保之后的任何迟到异步结果绝不篡改落盘数据
        self._is_session_active = False

        print("[SessionManager] Step 7: Saving final transcripts and metadata...")
        self._save_transcript_raw()
        self._save_transcript_final_markdown()
        self._save_metadata(qwen_completed=qwen_completed)

        self._log_event("session_end", {
            "session_id": self.session_id,
            "total_duration": self.recorder.total_recorded_seconds,
            "total_segments": len(self.segments),
            "qwen_completed": qwen_completed,
            "end_time": datetime.datetime.now().isoformat()
        })

        print(f"[SessionManager] Session finished successfully. Saved to: {self.session_dir}")
        self._notify_status()

    # --- 内部事件回调 ---

    def _drain_pending_speech_ends(self, timestamp_sec: float = 0.0):
        """排空待处理的句子结束事件（保证当前 chunk 已进入 Paraformer 且尾音已冲刷）"""
        while self._pending_speech_ends:
            pending = self._pending_speech_ends.pop(0)
            seg_id = pending["segment_id"]
            start_sec = pending["start_sec"]
            end_sec = pending["end_sec"]
            seg_audio = pending["audio"]

            # 1. 冲刷该 segment 在 Paraformer 累加器中的尾部不足 480ms 的音频
            self.paraformer_streamer.flush(segment_id=seg_id, timestamp_sec=end_sec)

            # 2. 读取完整包含尾音识别结果的 online_text
            online_text = ""
            if seg_id in self.segments:
                self.segments[seg_id].end_sec = end_sec
                online_text = self.segments[seg_id].online_text

            self._log_event("speech_end", {
                "segment_id": seg_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "samples": len(seg_audio),
                "online_text": online_text
            })

            # 3. 组装热词上下文并提交给后台 Qwen Worker
            context = ""
            if self.current_course and self.current_course.hotwords:
                context = "热词: " + ", ".join(self.current_course.hotwords[:30])

            task = SegmentTask(
                segment_id=seg_id,
                start_sec=start_sec,
                end_sec=end_sec,
                audio=seg_audio,
                online_text=online_text,
                context=context,
                language=self.current_course.language if self.current_course else "zh"
            )
            self.qwen_worker.submit_segment(task)
            self._notify_status()

    def _on_audio_chunk(self, chunk: np.ndarray, timestamp_sec: float):
        if not self._is_session_active:
            return

        # 1. 喂给 VAD 进行断句与端点检测（若断句会登记至 _pending_speech_ends）
        self.vad_detector.process_chunk(chunk, timestamp_sec)

        # 2. 喂给 Paraformer 流式识别器（内部含 480ms accumulator）
        self.paraformer_streamer.process_chunk(
            chunk=chunk,
            segment_id=self.current_speaking_segment_id,
            timestamp_sec=timestamp_sec
        )

        # 3. 在当前 chunk 已经进入 Paraformer 之后，排空并处理所有 pending speech ends
        self._drain_pending_speech_ends(timestamp_sec=timestamp_sec)

    def _on_streaming_partial(self, segment_id: int, partial_text: str, timestamp_sec: float):
        if segment_id not in self.segments:
            self.segments[segment_id] = SegmentRecord(
                segment_id=segment_id,
                start_sec=timestamp_sec,
                end_sec=timestamp_sec,
                online_text=partial_text,
                status="partial",
                final_success=None
            )
        else:
            self.segments[segment_id].online_text = partial_text
            self.segments[segment_id].end_sec = timestamp_sec

        self._log_event("online_partial", {
            "segment_id": segment_id,
            "text": partial_text,
            "timestamp": timestamp_sec
        })

        if self.on_partial_subtitle:
            self.on_partial_subtitle(segment_id, partial_text, timestamp_sec)

    def _on_vad_speech_start(self, segment_id: int, start_sec: float):
        # 确保上一个段落的 pending speech end 已被完全冲刷和提交
        self._drain_pending_speech_ends(timestamp_sec=start_sec)
        self.current_speaking_segment_id = segment_id
        self.paraformer_streamer.reset_segment(segment_id)
        if segment_id not in self.segments:
            self.segments[segment_id] = SegmentRecord(
                segment_id=segment_id,
                start_sec=start_sec,
                end_sec=start_sec,
                status="partial",
                final_success=None
            )
        self._log_event("speech_start", {"segment_id": segment_id, "start_sec": start_sec})

    def _on_vad_speech_end(self, segment_id: int, start_sec: float, end_sec: float, segment_audio: np.ndarray):
        # 只登记 pending end，等待当前 chunk 送入 Paraformer 后再统一 flush 与 submit
        self._pending_speech_ends.append({
            "segment_id": segment_id,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "audio": segment_audio
        })

    def _on_qwen_result(self, res: SegmentResult):
        if not self._is_session_active:
            print(f"[SessionManager Warning] Discarding late Qwen result for seg {res.segment_id} because session is inactive.")
            return

        # 严格按 segment_id 保序记录
        if res.segment_id not in self.segments:
            self.segments[res.segment_id] = SegmentRecord(
                segment_id=res.segment_id,
                start_sec=res.start_sec,
                end_sec=res.end_sec
            )
        seg = self.segments[res.segment_id]
        seg.final_text = res.final_text
        seg.final_model = res.final_model
        seg.latency_sec = res.latency_sec
        seg.status = "final"
        seg.final_success = res.success
        seg.fallback_reason = res.fallback_reason

        self._log_event("qwen_final", {
            "segment_id": res.segment_id,
            "start_sec": res.start_sec,
            "end_sec": res.end_sec,
            "final_text": res.final_text,
            "model": res.final_model,
            "latency_sec": res.latency_sec,
            "success": res.success,
            "fallback_reason": res.fallback_reason
        })

        if self.on_final_subtitle:
            self.on_final_subtitle(
                res.segment_id,
                res.final_text,
                res.final_model,
                res.start_sec,
                res.success,
                res.fallback_reason
            )
        self._notify_status()

    def _on_qwen_queue_change(self, qsize: int):
        self._notify_status()

    def _notify_status(self):
        if not self.on_status_update:
            return
        status_info = {
            "is_active": self._is_session_active,
            "is_paused": self.recorder.is_paused,
            "total_duration": self.recorder.total_recorded_seconds,
            "total_processed": self.recorder.total_processed_seconds,
            "streaming_lag_sec": self.recorder.streaming_lag_sec,
            "total_segments": len(self.segments),
            "qwen_queue_size": self.qwen_worker.queue_size,
            "qwen_success_count": self.qwen_worker.success_count,
            "qwen_fallback_count": self.qwen_worker.fallback_count,
            "session_id": self.session_id,
            "course_name": self.current_course.name if self.current_course else ""
        }
        self.on_status_update(status_info)

    # --- 持久化方法 ---

    def _log_event(self, event_type: str, data: Dict[str, Any]):
        if not self.session_dir:
            return
        event_entry = {
            "timestamp": time.time(),
            "time_iso": datetime.datetime.now().isoformat(),
            "type": event_type,
            **data
        }
        events_file = self.session_dir / "events.jsonl"
        with open(events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_entry, ensure_ascii=False) + "\n")

    def _save_transcript_raw(self):
        if not self.session_dir:
            return
        raw_file = self.session_dir / "transcript_raw.jsonl"
        sorted_segs = sorted(self.segments.values(), key=lambda x: x.segment_id)
        with open(raw_file, "w", encoding="utf-8") as f:
            for seg in sorted_segs:
                entry = {
                    "segment_id": seg.segment_id,
                    "start": round(seg.start_sec, 2),
                    "end": round(seg.end_sec, 2),
                    "online_text": seg.online_text,
                    "final_text": seg.final_text or seg.online_text,
                    "final_model": seg.final_model or "online_interim",
                    "latency_sec": round(seg.latency_sec, 2),
                    "status": seg.status,
                    "final_success": seg.final_success,
                    "fallback_reason": seg.fallback_reason
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _save_transcript_final_markdown(self):
        if not self.session_dir or not self.current_course:
            return
        md_file = self.session_dir / "transcript_final.md"
        date_str = self.session_start_dt.strftime("%Y-%m-%d") if self.session_start_dt else ""
        
        lines = [
            f"# {self.current_course.name}",
            f"**日期**: {date_str}  ",
            f"**会话ID**: `{self.session_id}`  ",
            f"**权威识别模型**: `Qwen3-ASR-1.7B-q4_k (CapsWriter 本地推理)`  ",
            f"**实时流式模型**: `Paraformer-zh-streaming`  ",
            "",
            "---",
            ""
        ]

        sorted_segs = sorted(self.segments.values(), key=lambda x: x.segment_id)
        for seg in sorted_segs:
            text = seg.final_text if seg.final_text else seg.online_text
            if not text.strip():
                continue
            
            # 格式化时间戳 [HH:MM:SS]
            m, s = divmod(int(seg.start_sec), 60)
            h, m = divmod(m, 60)
            ts_str = f"{h:02d}:{m:02d}:{s:02d}"
            tag = " [实时回退]" if seg.final_success is False else ""
            
            lines.append(f"[{ts_str}]{tag}")
            lines.append(f"{text}\n")

        with open(md_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _save_metadata(self, qwen_completed: bool = True):
        if not self.session_dir:
            return
        meta_file = self.session_dir / "meta.json"
        
        qwen_success = sum(
            1 for s in self.segments.values()
            if s.status == "final" and s.final_success is True and (s.final_model or "").startswith("Qwen")
        )
        fallback_cnt = sum(
            1 for s in self.segments.values()
            if s.final_success is False or (s.status == "final" and s.final_success is not True)
        )
        timeout_cnt = sum(1 for s in self.segments.values() if s.fallback_reason == 'QWEN_SHUTDOWN_TIMEOUT')

        status_str = "completed" if (qwen_completed and fallback_cnt == 0) else ("completed_with_fallback" if qwen_completed else "timed_out")

        meta = {
            "session_id": self.session_id,
            "status": status_str,
            "course": self.current_course.name if self.current_course else "",
            "course_id": self.current_course.id if self.current_course else "",
            "start_time": self.session_start_dt.isoformat() if self.session_start_dt else "",
            "end_time": datetime.datetime.now().isoformat(),
            "total_duration_sec": round(self.recorder.total_recorded_seconds, 2),
            "total_segments": len(self.segments),
            "qwen_success_segments": qwen_success,
            "fallback_segments": fallback_cnt,
            "qwen_timeout_segments": timeout_cnt,
            "online_model": "paraformer_zh_streaming",
            "configured_offline_model": "Qwen3-ASR-1.7B-q4_k",
            "configured_capswriter_path": str(CAPSWRITER_DIR),
            "verified_runtime_model": "Qwen3-ASR-1.7B-q4_k (inferred from local CapsWriter config_server.py)",
            "hardware_hint": "Intel Arc / Core Ultra",
            "runtime_verification": "inferred from local CapsWriter config_server.py (model_type=qwen_asr)",
            "audio_file": "audio.wav",
            "events_file": "events.jsonl",
            "transcript_raw": "transcript_raw.jsonl",
            "transcript_final": "transcript_final.md"
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
