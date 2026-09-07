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
        self._thread: Optional[threading.Thread] = None
        
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

    def start(self):
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="QwenWorkerThread")
        self._thread.start()

    def submit_segment(self, task: SegmentTask):
        self._task_queue.put(task)
        if self.on_queue_change:
            self.on_queue_change(self._task_queue.qsize())

    def _worker_loop(self):
        while self._is_running:
            try:
                task = self._task_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if task is None:
                self._task_queue.task_done()
                break

            if self.on_queue_change:
                self.on_queue_change(self._task_queue.qsize())

            task_id = f"seg_{task.segment_id}_{int(time.time()*1000)}"
            success, recognized_text, latency = self.adapter.transcribe(
                audio=task.audio,
                task_id=task_id,
                context=task.context,
                language=task.language,
                timeout=25.0
            )

            if success and recognized_text.strip():
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

        while self._task_queue.unfinished_tasks > 0:
            if deadline is not None and time.monotonic() >= deadline:
                print(f"[QwenWorker Warning] wait_completion timed out after {timeout}s! Remaining tasks: {self._task_queue.unfinished_tasks}")
                return False
            time.sleep(0.05)

        return True

    def flush_remaining_as_fallback(self):
        """超时后将队列中仍滞留的未处理任务强制回退，防止死锁"""
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
                self.flush_remaining_as_fallback()

        self._is_running = False
        self._task_queue.put(None)
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

        return completed
