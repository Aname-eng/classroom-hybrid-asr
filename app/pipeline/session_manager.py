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

class SessionManager:
    """
    课堂实时转写会话管理器
    
    统一编排录音、VAD断句、流式识别与离线Qwen修正，全量持久化 raw evidence。
    """
    def __init__(
        self,
        config: Optional[AppConfig] = None,
        on_partial_subtitle: Optional[Callable[[int, str, float], None]] = None,
        on_final_subtitle: Optional[Callable[[int, str, str, float], None]] = None,
        on_status_update: Optional[Callable[[Dict[str, Any]], None]] = None
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

        # 初始化组件
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
            on_audio_chunk=self._on_audio_chunk
        )

        self._is_session_active = False

    def select_course(self, course_id: str) -> Optional[CourseInfo]:
        course = self.course_manager.get_course(course_id)
        if course:
            self.current_course = course
            self.config.selected_course = course_id
            self.config.save()
        return course

    def start_session(self, course_id: Optional[str] = None) -> str:
        if self._is_session_active:
            return self.session_id

        if course_id:
            self.select_course(course_id)
        if not self.current_course:
            self.select_course("political_economy")

        # 启动 Qwen 后台 Worker 并确保服务进程就绪
        self.qwen_adapter.ensure_server_running(wait_timeout=35.0)
        self.qwen_worker.start()

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
        self.current_speaking_segment_id = 1
        self.vad_detector.reset()
        self.paraformer_streamer.reset_segment(1)

        # 记录 session_start 事件
        self._log_event("session_start", {
            "session_id": self.session_id,
            "course": self.current_course.name,
            "course_id": self.current_course.id,
            "start_time": now.isoformat(),
            "sample_rate": SAMPLE_RATE,
            "hotwords": self.current_course.hotwords
        })

        # 启动主音频录音机
        audio_wav_path = str(self.session_dir / "audio.wav")
        self.recorder.start(output_wav_path=audio_wav_path)
        self._is_session_active = True

        print(f"[SessionManager] Session started: {self.session_id}")
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

    def end_session(self):
        if not self._is_session_active:
            return

        print("[SessionManager] Ending session, flushing remaining buffers...")
        cur_time = self.recorder.total_recorded_seconds
        self.vad_detector.flush(cur_time)
        self.recorder.stop()

        # 等待后台 Qwen 队列全部处理完毕
        print("[SessionManager] Waiting for Qwen offline queue to clear...")
        self.qwen_worker.wait_completion(timeout=60.0)
        self.qwen_worker.stop(wait_finish=True)

        # 保存 final transcript & meta
        self._save_transcript_raw()
        self._save_transcript_final_markdown()
        self._save_metadata()

        self._log_event("session_end", {
            "session_id": self.session_id,
            "total_duration": self.recorder.total_recorded_seconds,
            "total_segments": len(self.segments),
            "end_time": datetime.datetime.now().isoformat()
        })

        self._is_session_active = False
        print(f"[SessionManager] Session finished. Saved to: {self.session_dir}")
        self._notify_status()

    # --- 内部事件回调 ---

    def _on_audio_chunk(self, chunk: np.ndarray, timestamp_sec: float):
        if not self._is_session_active:
            return

        # 1. 喂给 VAD
        self.vad_detector.process_chunk(chunk, timestamp_sec)

        # 2. 喂给 Paraformer 流式识别器
        self.paraformer_streamer.process_chunk(
            chunk=chunk,
            segment_id=self.current_speaking_segment_id,
            timestamp_sec=timestamp_sec,
            is_final=False
        )

    def _on_streaming_partial(self, segment_id: int, partial_text: str, timestamp_sec: float):
        if segment_id not in self.segments:
            self.segments[segment_id] = SegmentRecord(
                segment_id=segment_id,
                start_sec=timestamp_sec,
                end_sec=timestamp_sec,
                online_text=partial_text,
                status="partial"
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
        self.current_speaking_segment_id = segment_id
        self.paraformer_streamer.reset_segment(segment_id)
        if segment_id not in self.segments:
            self.segments[segment_id] = SegmentRecord(
                segment_id=segment_id,
                start_sec=start_sec,
                end_sec=start_sec,
                status="partial"
            )
        self._log_event("speech_start", {"segment_id": segment_id, "start_sec": start_sec})

    def _on_vad_speech_end(self, segment_id: int, start_sec: float, end_sec: float, segment_audio: np.ndarray):
        online_text = ""
        if segment_id in self.segments:
            self.segments[segment_id].end_sec = end_sec
            online_text = self.segments[segment_id].online_text

        self._log_event("speech_end", {
            "segment_id": segment_id,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "samples": len(segment_audio),
            "online_text": online_text
        })

        # 提交给后台 Qwen Worker
        context = ""
        if self.current_course and self.current_course.hotwords:
            context = "热词: " + ", ".join(self.current_course.hotwords[:30])

        task = SegmentTask(
            segment_id=segment_id,
            start_sec=start_sec,
            end_sec=end_sec,
            audio=segment_audio,
            online_text=online_text,
            context=context,
            language=self.current_course.language if self.current_course else "zh"
        )
        self.qwen_worker.submit_segment(task)
        self._notify_status()

    def _on_qwen_result(self, res: SegmentResult):
        # 确保按 segment_id 保序记录
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

        self._log_event("qwen_final", {
            "segment_id": res.segment_id,
            "start_sec": res.start_sec,
            "end_sec": res.end_sec,
            "final_text": res.final_text,
            "final_model": res.final_model,
            "latency_sec": res.latency_sec,
            "success": res.success
        })

        if self.on_final_subtitle:
            self.on_final_subtitle(res.segment_id, res.final_text, res.final_model, res.end_sec)

        self._save_transcript_raw()
        self._save_transcript_final_markdown()
        self._notify_status()

    def _on_qwen_queue_change(self, queue_len: int):
        self._notify_status()

    def _notify_status(self):
        if self.on_status_update:
            status = {
                "is_recording": self._is_session_active and not self.recorder.is_paused,
                "is_paused": self.recorder.is_paused,
                "recorded_seconds": self.recorder.total_recorded_seconds,
                "queue_length": self.qwen_worker.queue_size,
                "total_segments": len(self.segments),
                "course_name": self.current_course.name if self.current_course else "未选择"
            }
            self.on_status_update(status)

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
                    "status": seg.status
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
            f"**识别权威模型**: `Qwen3-ASR-1.7B-q4_k`  ",
            f"**流式模型**: `Paraformer-zh-streaming`  ",
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
            
            lines.append(f"[{ts_str}]")
            lines.append(f"{text}\n")

        with open(md_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _save_metadata(self):
        if not self.session_dir:
            return
        meta_file = self.session_dir / "meta.json"
        
        meta = {
            "session_id": self.session_id,
            "status": "completed",
            "course": self.current_course.name if self.current_course else "",
            "course_id": self.current_course.id if self.current_course else "",
            "start_time": self.session_start_dt.isoformat() if self.session_start_dt else "",
            "end_time": datetime.datetime.now().isoformat(),
            "total_duration_sec": round(self.recorder.total_recorded_seconds, 2),
            "total_segments": len(self.segments),
            "online_model": "paraformer_zh_streaming",
            "offline_model": "Qwen3-ASR-1.7B-q4_k",
            "hardware": "Intel Arc 140T GPU (Vulkan) + Intel Core Ultra 7 CPU",
            "qwen_model_path": str(QWEN_MODEL_DIR),
            "audio_file": "audio.wav",
            "events_file": "events.jsonl",
            "transcript_raw": "transcript_raw.jsonl",
            "transcript_final": "transcript_final.md"
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
