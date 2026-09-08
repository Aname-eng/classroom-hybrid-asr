# coding: utf-8
import os
import sys
import time
import json
import datetime
import threading
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Callable, Dict, Any, List
import numpy as np

from app.config import (
    AppConfig, SAMPLE_RATE,
    QWEN_MODEL_DIR, CAPSWRITER_DIR
)
from app.audio.source import AudioSource
from app.audio.recorder import AudioRecorder
from app.vad.vad_detector import VADDetector
from app.streaming.paraformer_streamer import ParaformerStreamer
from app.offline.qwen_adapter import QwenAdapter
from app.offline.qwen_worker import QwenWorker, SegmentTask, SegmentResult
from app.offline.summary_processor import SummaryProcessor, CleanupResult
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
    is_manual_edited: bool = False
    cleaned_text: str = ""

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
        audio_source: Optional[AudioSource] = None,
        course_manager: Optional[CourseManager] = None
    ):
        self.config = config or AppConfig.load()
        self.on_partial_subtitle = on_partial_subtitle
        self.on_final_subtitle = on_final_subtitle
        self.on_status_update = on_status_update

        self.course_manager = course_manager or CourseManager()
        self.current_course: Optional[CourseInfo] = None
        self.active_hotwords: List[str] = []
        self.generated_hotwords: List[str] = []
        self.hotwords_generated_by_model: bool = False
        self.summary_result: Optional[CleanupResult] = None
        self.sessions_root = self.config.sessions_root()
        self.session_dir: Optional[Path] = None
        self.session_id: Optional[str] = None
        self.session_start_time: float = 0.0
        self.session_start_dt: Optional[datetime.datetime] = None

        self.segments: Dict[int, SegmentRecord] = {}
        self.current_speaking_segment_id: int = 1
        self._pending_speech_ends: List[Dict[str, Any]] = []
        self.streaming_backend_requested = (self.config.streaming_backend or "funasr").strip().lower()
        self.streaming_backend_effective = self._resolve_streaming_backend(
            self.streaming_backend_requested
        )

        # 初始化核心组件
        self.summary_processor = SummaryProcessor(
            provider=self.config.summary_provider,
            endpoint=self.config.summary_endpoint,
            model=self.config.summary_model,
            model_path=self.config.summary_model_path,
            timeout_sec=self.config.summary_timeout_sec,
            batch_chars=self.config.summary_batch_chars,
            enabled=self.config.summary_enabled,
            remove_off_topic=self.config.summary_remove_off_topic,
        )
        self.qwen_adapter = QwenAdapter(
            ws_uri=self.config.qwen_ws_uri,
            server_exe=self.config.qwen_server_exe,
            server_cwd=self.config.qwen_server_cwd,
            model_path=self.config.qwen_model_path,
            backend=self.config.qwen_backend,
        )
        self.qwen_worker = QwenWorker(
            adapter=self.qwen_adapter,
            on_result=self._on_qwen_result,
            on_queue_change=self._on_qwen_queue_change
        )
        self.paraformer_streamer = ParaformerStreamer(
            chunk_size=self.config.chunk_size,
            on_partial_text=self._on_streaming_partial,
            model_name=self.config.streaming_model,
            model_path=self.config.streaming_model_path,
            device=self.config.streaming_device,
            backend=self.streaming_backend_effective,
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

    @staticmethod
    def _resolve_streaming_backend(name: str) -> str:
        """只接受真正兼容 FunASR streaming API 的第一遍后端。"""
        aliases = {
            "funasr": "funasr",
            "paraformer": "funasr",
            "funasr_streaming": "funasr",
            "paraformer_zh_streaming": "funasr",
        }
        normalized = (name or "funasr").strip().lower()
        if normalized in aliases:
            return aliases[normalized]
        print(
            f"[SessionManager] Unsupported streaming_backend={name!r}; "
            "falling back to FunASR/Paraformer."
        )
        return "funasr"

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

    def reload_model_configuration(self) -> None:
        """模型管理器下载/切换模型后刷新尚未开始的推理组件。"""
        self.streaming_backend_requested = (self.config.streaming_backend or "funasr").strip().lower()
        self.streaming_backend_effective = self._resolve_streaming_backend(
            self.streaming_backend_requested
        )
        self.paraformer_streamer.backend = self.streaming_backend_effective
        self.paraformer_streamer.model_name = self.config.streaming_model
        self.paraformer_streamer.model_path = self.config.streaming_model_path
        self.summary_processor = SummaryProcessor(
            provider=self.config.summary_provider,
            endpoint=self.config.summary_endpoint,
            model=self.config.summary_model,
            model_path=self.config.summary_model_path,
            timeout_sec=self.config.summary_timeout_sec,
            batch_chars=self.config.summary_batch_chars,
            enabled=self.config.summary_enabled,
            remove_off_topic=self.config.summary_remove_off_topic,
        )
        self.qwen_adapter.configure_endpoint(self.config.qwen_ws_uri)
        self.qwen_adapter.server_exe = self.config.qwen_server_exe or ""
        self.qwen_adapter.server_cwd = self.config.qwen_server_cwd or ""
        self.qwen_adapter.configure_model(
            model_path=self.config.qwen_model_path,
            backend=self.config.qwen_backend,
        )

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
            # 容错：自动创建单例 CourseInfo，永不异常中断
            print(f"[SessionManager Warning] Course '{course_id}' not found, generating on-the-fly course info.")
            course = CourseInfo(id=course_id, name=course_id, language="zh")
            self.course_manager.courses[course_id] = course
        self.current_course = course
        self.active_hotwords = list(course.hotwords)
        self.generated_hotwords = []
        self.hotwords_generated_by_model = False
        self.summary_result = None
        # 流式模型支持的提示主要通过 hotword 注入；离线 Qwen 的完整上下文会在
        # 每个句子结束时构造。没有用户提示词时，课程名称就是默认主题。
        self.paraformer_streamer.set_context(
            prompt=self._default_asr_prompt(),
            hotwords=self.active_hotwords,
        )

        # 在录音启动前预热/初始化 Paraformer 与 VAD 模型（若未初始化）
        if not self.paraformer_streamer._is_initialized:
            print("[SessionManager] Pre-initializing Paraformer streaming model...")
            self.paraformer_streamer.initialize()

        # 创建 Session 目录：课程名称_{精确到分钟的时间戳}
        now = datetime.datetime.now()
        self.session_start_dt = now
        self.session_start_time = time.time()
        import re
        clean_course_name = re.sub(r'[\\/:*?"<>|\r\n\t]+', '_', self.current_course.name).strip('_')
        if not clean_course_name:
            clean_course_name = self.current_course.id
        timestamp_min = now.strftime("%Y-%m-%d_%H-%M")
        session_id = f"{clean_course_name}_{timestamp_min}"
        
        target_dir = self.sessions_root / session_id
        base_id = session_id
        counter = 1
        while target_dir.exists():
            session_id = f"{base_id}_{counter}"
            target_dir = self.sessions_root / session_id
            counter += 1

        self.session_id = session_id
        self.session_dir = target_dir
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # 没有手工热词时，优先让本地小模型根据课程主题补充；服务不可用时
        # 使用确定性的课程名/简介词表。词表只属于本次会话，不修改用户原课程配置。
        if self.config.auto_generate_hotwords and self.current_course:
            self.active_hotwords, generated_by_model = self.summary_processor.generate_hotwords(
                course_name=self.current_course.name,
                description=self.current_course.description,
                existing=self.current_course.hotwords,
                max_count=self.config.generated_hotwords_max,
            )
            self.hotwords_generated_by_model = generated_by_model
            if generated_by_model:
                self.generated_hotwords = [
                    word for word in self.active_hotwords if word not in self.current_course.hotwords
                ]
        # 即使用户关闭自动热词，课程名称仍作为最小的主题提示，兼容只支持
        # hotword 而不支持 prompt 参数的流式 ASR 模型。
        if self.current_course.name.strip() and self.current_course.name.strip() not in self.active_hotwords:
            self.active_hotwords.insert(0, self.current_course.name.strip())
        self.paraformer_streamer.set_context(
            prompt=self._default_asr_prompt(),
            hotwords=self.active_hotwords,
        )
        self._save_generated_hotwords()

        self.segments.clear()
        self._pending_speech_ends.clear()
        self.current_speaking_segment_id = 1
        self.vad_detector.reset()
        self.paraformer_streamer.reset_segment(1)
        self.qwen_worker.reset_session()
        self.qwen_worker.start()
        threading.Thread(target=self.qwen_adapter.ensure_server_running, daemon=True, name="CapsWriterPreheatThread").start()

        # 记录 session_start 事件
        self._log_event("session_start", {
            "session_id": self.session_id,
            "course": self.current_course.name,
            "course_id": self.current_course.id,
            "start_time": now.isoformat(),
            "sample_rate": SAMPLE_RATE,
            "device_index": self.recorder.device_index,
            "hotwords": self.active_hotwords,
            "generated_hotwords": self.generated_hotwords,
            "asr_prompt": self._default_asr_prompt(),
            "streaming_model": self.paraformer_streamer.resolved_model,
            "streaming_backend": self.streaming_backend_effective
        })

        # 启动主录音机。先标记 active，避免极速 FileReplay 在 start() 返回前
        # 产生的首个音频块被回调保护条件丢弃。
        audio_wav_path = str(self.session_dir / "audio.wav")
        self._is_session_active = True
        try:
            self.recorder.start(output_wav_path=audio_wav_path, source=source)
        except Exception:
            self._is_session_active = False
            raise

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
        7. 用课程主题/热词调用本地小模型整理稿件（失败则保守清理口头禅）
        8. 保存所有产物文档与元数据
        """
        if not self._is_session_active:
            return

        # timeout 是本次收尾的总预算，而不是只给 Qwen 队列的预算；这样音频
        # 排空耗时较长时，离线服务也不会再额外拖出一个完整 timeout。
        shutdown_deadline = time.monotonic() + max(1.0, float(timeout))

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

        qwen_timeout = max(1.0, shutdown_deadline - time.monotonic())
        print(f"[SessionManager] Step 5: Waiting for Qwen offline queue to clear (timeout={qwen_timeout:.1f}s)...")
        qwen_completed = self.qwen_worker.stop(wait_finish=True, timeout=qwen_timeout)
        if not qwen_completed:
            print("[SessionManager Warning] Qwen worker timed out during shutdown, flushed as fallback.")

        print("[SessionManager] Step 6: Closing WAV file...")
        self.recorder.close()

        # 先置 session_active = False，确保之后的任何迟到异步结果绝不篡改落盘数据
        self._is_session_active = False

        print("[SessionManager] Step 7: Cleaning transcript with local summary model...")
        self._run_postprocessing()

        print("[SessionManager] Step 8: Saving final transcripts and metadata...")
        self._save_transcript_raw()
        self._save_transcript_final_markdown()
        self._save_transcript_cleaned_markdown()
        self._save_metadata(qwen_completed=qwen_completed)

        self._log_event("session_end", {
            "session_id": self.session_id,
            "total_duration": self.recorder.total_recorded_seconds,
            "total_segments": len(self.segments),
            "qwen_completed": qwen_completed,
            "summary_model": self.summary_result.model if self.summary_result else "",
            "summary_used_model": self.summary_result.used_model if self.summary_result else False,
            "end_time": datetime.datetime.now().isoformat()
        })

        print(f"[SessionManager] Session finished successfully. Saved to: {self.session_dir}")
        self._notify_status()

    # --- 内部事件回调 ---

    def _default_asr_prompt(self) -> str:
        """返回 ASR 使用的默认提示词：用户未填写时自动使用课程名称。"""
        custom = (self.config.asr_prompt or "").strip()
        course_prompt = (self.current_course.asr_prompt if self.current_course else "").strip()
        if custom:
            return custom
        if course_prompt:
            return course_prompt
        return f"课程主题：{self.current_course.name}" if self.current_course else ""

    def _save_generated_hotwords(self) -> None:
        if not self.session_dir:
            return
        payload = {
            "course": self.current_course.name if self.current_course else "",
            "generated_by_local_model": self.hotwords_generated_by_model,
            "active_hotwords": self.active_hotwords,
            "generated_hotwords": self.generated_hotwords,
        }
        try:
            with open(self.session_dir / "generated_hotwords.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[SessionManager Warning] Failed to save generated hotwords: {exc}")

    def _run_postprocessing(self) -> None:
        """整理最终稿，但绝不覆盖 transcript_raw.jsonl 或原始分段文本。"""
        if not self.current_course:
            self.summary_result = CleanupResult({}, "deterministic-filler-cleaner", False)
            return

        source = []
        for seg in sorted(self.segments.values(), key=lambda item: item.segment_id):
            text = seg.final_text or seg.online_text
            source.append((seg.segment_id, text))

        self.summary_result = self.summary_processor.clean_segments(
            source,
            course_name=self.current_course.name,
            hotwords=self.active_hotwords,
        )
        for seg_id, seg in self.segments.items():
            seg.cleaned_text = self.summary_result.cleaned_segments.get(
                seg_id, self.summary_processor.remove_fillers(seg.final_text or seg.online_text)
            )

        self._log_event("transcript_postprocessed", {
            "course": self.current_course.name,
            "model": self.summary_result.model,
            "used_model": self.summary_result.used_model,
            "error": self.summary_result.error,
            "removed_segments": sum(
                1 for seg in self.segments.values()
                if (seg.final_text or seg.online_text).strip() and not seg.cleaned_text.strip()
            ),
        })

    @staticmethod
    def _select_relevant_hotwords(all_hotwords: List[str], online_text: str, max_count: int = 15) -> List[str]:
        """
        全自动动态热词优化：
        1. 若当前流式识别文字包含某些热词的字/词片段，优先提取最匹配的热词；
        2. 补齐其他高频热词，严格控制上限在 10~15 个以内，既保证纠错准度，又杜绝大模型复读。
        """
        if not all_hotwords:
            return []
        
        matched = []
        unmatched = []
        online_lower = online_text.lower().strip() if online_text else ""
        
        for hw in all_hotwords:
            hw_clean = hw.strip()
            if not hw_clean:
                continue
            # 检查是否有字符重叠（例如流式输出了"科斯"或"产权"的部分字）
            if online_lower and (hw_clean.lower() in online_lower or any(char in online_lower for char in hw_clean if len(char.strip()) > 0 and char not in "，。！？ 的了是个在")):
                matched.append(hw_clean)
            else:
                unmatched.append(hw_clean)
                
        # 优先把命中相关的热词放在前面，不足时用其他热词补齐至 max_count
        selected = (matched + unmatched)[:max_count]
        return selected

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

            # 计算音频 RMS 能量与有效性保护
            rms = float(np.sqrt(np.mean(seg_audio**2))) if len(seg_audio) > 0 else 0.0
            is_near_silence = (rms < 0.0025 and not online_text.strip())

            # 3. 组装上下文：有显式提示词就使用提示词，否则至少注入课程名称；
            # 静音段不注入任何上下文，避免离线模型复述主题或热词。
            context = ""
            if not is_near_silence and self.current_course:
                context_parts = []
                prompt = self._default_asr_prompt()
                if prompt:
                    context_parts.append(f"识别提示: {prompt}")
                selected_hw = self._select_relevant_hotwords(
                    self.active_hotwords, online_text, max_count=15
                )
                if selected_hw:
                    context_parts.append("专业术语参考: " + "、".join(selected_hw))
                context = "\n".join(context_parts)

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

    def update_segment_text(self, segment_id: int, new_text: str):
        """
        人工实时修正单句转写内容（随听随改）
        立即更新内存状态、记录 manual_edit 事件并同步增量落盘。
        """
        clean_text = new_text.strip()
        if segment_id not in self.segments:
            self.segments[segment_id] = SegmentRecord(
                segment_id=segment_id,
                start_sec=0.0,
                end_sec=0.0
            )
        seg = self.segments[segment_id]
        old_text = seg.final_text or seg.online_text
        seg.final_text = clean_text
        seg.cleaned_text = self.summary_processor.remove_fillers(clean_text)
        seg.status = "final"
        seg.is_manual_edited = True
        
        self._log_event("manual_edit", {
            "segment_id": segment_id,
            "old_text": old_text,
            "new_text": clean_text,
            "timestamp": time.time()
        })
        
        # 实时将人工修改结果持久化落盘
        self._save_transcript_raw()
        self._save_transcript_final_markdown()
        self._save_transcript_cleaned_markdown()
        self._notify_status()

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
        # 若用户此前已进行过人工改错，保留人工内容，不被迟到的模型结果覆盖
        if not seg.is_manual_edited:
            seg.final_text = res.final_text
            seg.final_model = res.final_model
            seg.final_success = res.success
            seg.fallback_reason = res.fallback_reason
            seg.cleaned_text = ""
        seg.latency_sec = res.latency_sec
        seg.status = "final"

        self._log_event("qwen_final", {
            "segment_id": res.segment_id,
            "start_sec": res.start_sec,
            "end_sec": res.end_sec,
            "final_text": seg.final_text,
            "model": res.final_model,
            "latency_sec": res.latency_sec,
            "success": res.success,
            "fallback_reason": res.fallback_reason,
            "is_manual_edited": seg.is_manual_edited
        })

        if self.on_final_subtitle:
            self.on_final_subtitle(
                res.segment_id,
                seg.final_text,
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
                    "fallback_reason": seg.fallback_reason,
                    "is_manual_edited": seg.is_manual_edited,
                    "cleaned_text": seg.cleaned_text
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _save_transcript_final_markdown(self):
        if not self.session_dir or not self.current_course:
            return
        md_file = self.session_dir / "transcript_final.md"
        date_str = self.session_start_dt.strftime("%Y-%m-%d") if self.session_start_dt else ""
        
        qwen_model_label = Path(self.config.qwen_model_path).name if self.config.qwen_model_path else "Qwen3-ASR"
        lines = [
            f"# {self.current_course.name}",
            f"**日期**: {date_str}  ",
            f"**会话ID**: `{self.session_id}`  ",
            f"**权威识别模型**: `{qwen_model_label} ({self.config.qwen_backend})`  ",
            f"**实时流式模型**: `{self.paraformer_streamer.resolved_model}`  ",
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
            
            if seg.is_manual_edited:
                tag = " [人工已修改]"
            elif seg.final_success is False:
                tag = " [实时回退]"
            else:
                tag = ""
            
            lines.append(f"[{ts_str}]{tag}")
            lines.append(f"{text}\n")

        with open(md_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _save_transcript_cleaned_markdown(self):
        """保存课程整理稿；原始权威稿仍保留在 transcript_final.md。"""
        if not self.session_dir or not self.current_course:
            return
        md_file = self.session_dir / "transcript_cleaned.md"
        date_str = self.session_start_dt.strftime("%Y-%m-%d") if self.session_start_dt else ""
        result = self.summary_result
        model_name = result.model if result else "deterministic-filler-cleaner"
        lines = [
            f"# {self.current_course.name}（课程整理稿）",
            f"**日期**: {date_str}  ",
            f"**会话ID**: `{self.session_id}`  ",
            f"**整理模型**: `{model_name}`  ",
            "**处理规则**: 依据课程名称和专业热词删除口头禅及明显无关闲聊；原始稿未被覆盖。  ",
            "",
            "---",
            "",
        ]

        for seg in sorted(self.segments.values(), key=lambda x: x.segment_id):
            text = seg.cleaned_text or ""
            if not text.strip():
                continue
            m, s = divmod(int(seg.start_sec), 60)
            h, m = divmod(m, 60)
            ts_str = f"{h:02d}:{m:02d}:{s:02d}"
            tag = " [人工已修改]" if seg.is_manual_edited else ""
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
            "online_model": self.config.streaming_model,
            "configured_offline_model": Path(self.config.qwen_model_path).name if self.config.qwen_model_path else "Qwen3-ASR",
            "qwen_backend": self.config.qwen_backend,
            "configured_capswriter_path": str(CAPSWRITER_DIR) if CAPSWRITER_DIR else "",
            "verified_runtime_model": Path(self.config.qwen_model_path).name if self.config.qwen_model_path else "Qwen3-ASR (configured)",
            "streaming_backend": self.config.streaming_backend,
            "streaming_backend_requested": self.streaming_backend_requested,
            "streaming_backend_effective": self.streaming_backend_effective,
            "streaming_model": self.paraformer_streamer.resolved_model,
            "asr_prompt": self._default_asr_prompt(),
            "active_hotwords": self.active_hotwords,
            "generated_hotwords": self.generated_hotwords,
            "hotwords_generated_by_model": self.hotwords_generated_by_model,
            "summary_enabled": self.config.summary_enabled,
            "summary_model": self.summary_result.model if self.summary_result else "",
            "summary_used_model": self.summary_result.used_model if self.summary_result else False,
            "summary_error": self.summary_result.error if self.summary_result else "",
            "hardware_hint": "platform-dependent",
            "runtime_verification": "local configuration",
            "audio_file": "audio.wav",
            "events_file": "events.jsonl",
            "transcript_raw": "transcript_raw.jsonl",
            "transcript_final": "transcript_final.md",
            "transcript_cleaned": "transcript_cleaned.md",
            "generated_hotwords_file": "generated_hotwords.json"
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
