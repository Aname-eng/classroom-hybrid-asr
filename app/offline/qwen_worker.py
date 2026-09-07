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

class QwenWorker:
    """
    Qwen 离线第二遍识别工作线程
    
    采用队列驱动，异步消费 VAD 产生的分段，保证麦克风录音与实时字幕绝对不被阻塞。
    支持失败回退至 Paraformer 在线结果，记录队列堆积情况与延迟。
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

    @property
    def queue_size(self) -> int:
        return self._task_queue.qsize()

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
                # 退出信号
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
            else:
                # 失败回退至 Paraformer 在线结果
                final_text = task.online_text.strip()
                final_model = "paraformer_fallback"
                print(f"[QwenWorker Warning] Segment {task.segment_id} fell back to online text: '{final_text}'")

            result = SegmentResult(
                segment_id=task.segment_id,
                start_sec=task.start_sec,
                end_sec=task.end_sec,
                final_text=final_text,
                final_model=final_model,
                online_text=task.online_text,
                latency_sec=latency,
                success=success
            )

            if self.on_result:
                try:
                    self.on_result(result)
                except Exception as e:
                    print(f"[QwenWorker Error] Result callback error: {e}")

            self._task_queue.task_done()
            if self.on_queue_change:
                self.on_queue_change(self._task_queue.qsize())

    def wait_completion(self, timeout: Optional[float] = None):
        """等待所有积压任务完成"""
        self._task_queue.join()

    def stop(self, wait_finish: bool = True):
        if not self._is_running:
            return
        if wait_finish:
            self.wait_completion()
        self._is_running = False
        self._task_queue.put(None)
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
