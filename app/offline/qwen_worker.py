# coding: utf-8
import time
import queue
import threading
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any
import numpy as np
from app.offline.qwen_adapter import QwenAdapter

@dataclass
class SegmentTask:
    segment_id: int
    start_sec: float
    end_sec: float
    audio: np.ndarray
    online_text: str
    context: str = ""
    language: str = "zh"

@dataclass
class SegmentResult:
    segment_id: int
    start_sec: float
    end_sec: float
    final_text: str
    final_model: str
    online_text: str
    latency_sec: float
    success: bool
    fallback_reason: Optional[str] = None

class QwenWorker:
    """
    Qwen 离线第二遍识别工作线程
    
    1. 采用独立队列异步消费 VAD 产出的完整语音段。
    2. 严格保序与单调递增处理。
    3. 支持有界超时（Bounded Timeout）与显式降级标记（Explicit Fallback）。
    """
    def __init__(
        self,
        adapter: QwenAdapter,
        on_result: Optional[Callable[[SegmentResult], None]] = None,
        on_queue_change: Optional[Callable[[int], None]] = None
    ):
        self.adapter = adapter
        self.on_result = on_result
        self.on_queue_change = on_queue_change
        
        self._task_queue: queue.Queue[Optional[SegmentTask]] = queue.Queue()
        self._is_running = False
        self._is_stopped = False
        self._thread: Optional[threading.Thread] = None
        
        # In-flight 任务状态与取消令牌
        self._task_lock = threading.Lock()
        self._current_in_flight_task: Optional[SegmentTask] = None
        self._active_task_id: Optional[str] = None
        self._cancelled_task_ids: set[str] = set()
        self._session_generation: int = 0
        self._in_flight_generation: int = 0
        
        # 统计计数
        self.success_count = 0
        self.fallback_count = 0
        self.timeout_count = 0

    @property
    def queue_size(self) -> int:
        return self._task_queue.qsize()

    @property
    def unfinished_tasks(self) -> int:
        return self._task_queue.unfinished_tasks

    def reset_session(self):
        """在新 Session 开始时重置会话代际与取消状态"""
        with self._task_lock:
            self._session_generation += 1
            self._cancelled_task_ids.clear()
            self._current_in_flight_task = None
            self._active_task_id = None
            self._is_stopped = False
            self.success_count = 0
            self.fallback_count = 0
            self.timeout_count = 0

    def start(self):
        if self._is_running:
            return
        self._is_running = True
        self._is_stopped = False
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="QwenWorkerThread")
        self._thread.start()

    def submit_segment(self, task: SegmentTask):
        self._task_queue.put(task)
        if self.on_queue_change:
            self.on_queue_change(self._task_queue.qsize())

    @staticmethod
    def _is_hallucinated_prompt_echo(text: str, context: str, online_text: str) -> bool:
        """
        检测模型输出是否为系统提示词/热词回显幻觉（Prompt Echoing Hallucination）
        """
        if not text:
            return False
        
        t = text.strip()
        # 1. 明显的提示词/热词前缀
        if t.startswith(("热词", "热词：", "热词:", "专业术语", "专业术语：", "专业术语:", "Domain terms", "Hotwords", "Keywords")):
            return True
            
        # 2. 如果输出包含大量逗号/顿号分隔的词语列表，且与 context 高度重合
        if context and context.strip():
            import re
            raw_keywords = [w.strip() for w in re.split(r'[,，、:：\s]+', context) if len(w.strip()) >= 2]
            matched_keywords = [w for w in raw_keywords if w in t]
            # 如果匹配到的专有名词超过 3 个，且输出文本中包含了连续逗号或顿号
            if len(matched_keywords) >= 3 and (t.count("，") >= 2 or t.count("、") >= 2 or t.count(",") >= 2):
                # 且如果 online_text 本身并没有这些词，说明是静音下的无中生有幻觉
                if not online_text or not any(w in online_text for w in matched_keywords[:3]):
                    return True

        return False

    def _worker_loop(self):
        while self._is_running:
            try:
                task = self._task_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            # 遇到终止哨兵
            if task is None:
                self._task_queue.task_done()
                break

            cur_gen = self._session_generation
            task_id = f"gen{cur_gen}_seg{task.segment_id}"
            
            with self._task_lock:
                if self._is_stopped or not self._is_running:
                    self._task_queue.task_done()
                    continue
                self._current_in_flight_task = task
                self._active_task_id = task_id

            if self.on_queue_change:
                self.on_queue_change(self._task_queue.qsize())

            # 执行第二遍权威识别
            success, recognized_text, latency = self.adapter.transcribe(
                audio=task.audio,
                task_id=task_id,
                context=task.context,
                language=task.language,
                timeout=25.0
            )

            with self._task_lock:
                is_cancelled = (
                    task_id in self._cancelled_task_ids or
                    cur_gen != self._session_generation or
                    self._is_stopped or
                    not self._is_running
                )
                self._current_in_flight_task = None
                self._active_task_id = None

            if is_cancelled:
                print(f"[QwenWorker] Discarded late response for cancelled/timed-out task {task_id}")
                self._task_queue.task_done()
                if self.on_queue_change:
                    self.on_queue_change(self._task_queue.qsize())
                continue

            if success and recognized_text.strip():
                if self._is_hallucinated_prompt_echo(recognized_text, task.context, task.online_text):
                    print(f"[QwenWorker Warning] Detected prompt/hotword echo hallucination, discarded: '{recognized_text.strip()}'")
                    final_text = task.online_text.strip()
                    final_model = "paraformer_fallback" if final_text else "silence_filtered"
                    fallback_reason = "QWEN_PROMPT_ECHO_FILTERED"
                    res_success = False
                    self.fallback_count += 1
                else:
                    final_text = recognized_text.strip()
                    final_model = "Qwen3-ASR-1.7B-q4_k"
                    fallback_reason = None
                    res_success = True
                    self.success_count += 1
            else:
                final_text = task.online_text.strip()
                final_model = "paraformer_fallback"
                fallback_reason = "QWEN_OFFLINE_OR_ERROR" if not success else "QWEN_EMPTY_OUTPUT"
                res_success = False
                self.fallback_count += 1
                print(f"[QwenWorker Warning] Segment {task.segment_id} fallback ({fallback_reason}): '{final_text}'")

            result = SegmentResult(
                segment_id=task.segment_id,
                start_sec=task.start_sec,
                end_sec=task.end_sec,
                final_text=final_text,
                final_model=final_model,
                online_text=task.online_text,
                latency_sec=latency,
                success=res_success,
                fallback_reason=fallback_reason
            )

            if self.on_result:
                try:
                    self.on_result(result)
                except Exception as e:
                    print(f"[QwenWorker Error] Result callback error: {e}")

            self._task_queue.task_done()
            if self.on_queue_change:
                self.on_queue_change(self._task_queue.qsize())

    def wait_completion(self, timeout: Optional[float] = 60.0) -> bool:
        """
        等待队列所有任务完成，严格遵守 bounded timeout。
        如果超时，返回 False，避免无限挂起。
        """
        deadline = None if timeout is None else (time.monotonic() + timeout)

        while self._task_queue.unfinished_tasks > 0 or self._current_in_flight_task is not None:
            if deadline is not None and time.monotonic() >= deadline:
                print(f"[QwenWorker Warning] wait_completion timed out after {timeout}s! Remaining tasks: {self._task_queue.unfinished_tasks}")
                return False
            time.sleep(0.05)

        return True

    def cancel_in_flight_and_flush_fallback(self):
        """超时后将正在执行 (in-flight) 及滞留在队列的任务强制回退并作废后续迟到结果"""
        with self._task_lock:
            self._is_stopped = True
            
            # 1. 处理正在飞行的任务 (in-flight)
            if self._current_in_flight_task is not None and self._active_task_id is not None:
                in_flight = self._current_in_flight_task
                self._cancelled_task_ids.add(self._active_task_id)
                self._current_in_flight_task = None
                self._active_task_id = None
                
                self.fallback_count += 1
                self.timeout_count += 1
                result = SegmentResult(
                    segment_id=in_flight.segment_id,
                    start_sec=in_flight.start_sec,
                    end_sec=in_flight.end_sec,
                    final_text=in_flight.online_text.strip(),
                    final_model="paraformer_fallback",
                    online_text=in_flight.online_text,
                    latency_sec=0.0,
                    success=False,
                    fallback_reason="QWEN_SHUTDOWN_TIMEOUT"
                )
                if self.on_result:
                    try:
                        self.on_result(result)
                    except Exception as e:
                        print(f"[QwenWorker Error] In-flight fallback callback error: {e}")

            # 2. 清空队列中尚未被取的任务
            while not self._task_queue.empty():
                try:
                    task = self._task_queue.get_nowait()
                except queue.Empty:
                    break
                if task is None:
                    self._task_queue.task_done()
                    continue

                self.fallback_count += 1
                self.timeout_count += 1
                result = SegmentResult(
                    segment_id=task.segment_id,
                    start_sec=task.start_sec,
                    end_sec=task.end_sec,
                    final_text=task.online_text.strip(),
                    final_model="paraformer_fallback",
                    online_text=task.online_text,
                    latency_sec=0.0,
                    success=False,
                    fallback_reason="QWEN_SHUTDOWN_TIMEOUT"
                )
                if self.on_result:
                    try:
                        self.on_result(result)
                    except Exception as e:
                        print(f"[QwenWorker Error] Fallback result callback error: {e}")

                self._task_queue.task_done()

    def stop(self, wait_finish: bool = True, timeout: float = 60.0) -> bool:
        if not self._is_running:
            return True

        completed = True
        if wait_finish:
            completed = self.wait_completion(timeout=timeout)
            if not completed:
                self.cancel_in_flight_and_flush_fallback()

        self._is_running = False
        self._task_queue.put(None)
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

        return completed
