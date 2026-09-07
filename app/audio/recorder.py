# coding: utf-8
import os
import time
import queue
import threading
from typing import Optional, Callable, List, Dict, Any
import numpy as np
import sounddevice as sd
import soundfile as sf
from app.config import SAMPLE_RATE, CHANNELS
from app.audio.source import AudioSource, MicrophoneAudioSource

class AudioRecorder:
    """
    高可靠双队列录音机与音频调度器
    
    架构设计：
    1. 录音证据母带最高优先级：
       AudioSource (100ms) -> _capture_queue -> WavWriterThread -> audio.wav
       写盘线程绝不执行任何 ASR / VAD 计算，确保即便转写阻塞，母带也绝不丢音。
    2. 下游处理解耦分发：
       WavWriterThread -> _processing_queue -> ProcessingWorkerThread -> VAD / Paraformer
    3. 严谨的阶段化收尾协议：
       stop_capture() -> wait_until_drained() -> close()
    """
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        device_index: Optional[int] = None,
        on_audio_chunk: Optional[Callable[[np.ndarray, float], None]] = None,
        source: Optional[AudioSource] = None
    ):
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.on_audio_chunk = on_audio_chunk
        self.custom_source = source
        
        self._source: Optional[AudioSource] = None
        self._is_recording = False
        self._is_paused = False
        self._wav_file: Optional[sf.SoundFile] = None
        self._wav_path: Optional[str] = None
        
        # 队列定义
        self._capture_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue()
        self._processing_queue: queue.Queue[Optional[tuple[np.ndarray, float]]] = queue.Queue()
        
        # 线程句柄
        self._writer_thread: Optional[threading.Thread] = None
        self._processing_thread: Optional[threading.Thread] = None
        
        # 统计指标
        self._total_captured_samples = 0
        self._total_processed_samples = 0
        self._start_wall_time = 0.0

    @staticmethod
    def list_input_devices() -> List[Dict[str, Any]]:
        devices = []
        try:
            device_list = sd.query_devices()
            for idx, dev in enumerate(device_list):
                if dev.get('max_input_channels', 0) > 0:
                    devices.append({
                        'index': idx,
                        'name': dev.get('name', f'Device {idx}'),
                        'channels': dev.get('max_input_channels', 1),
                        'default_samplerate': dev.get('default_samplerate', 16000),
                        'is_default': (idx == sd.default.device[0])
                    })
        except Exception as e:
            print(f"[AudioRecorder Error] Failed to query devices: {e}")
        return devices

    @property
    def total_recorded_seconds(self) -> float:
        return self._total_captured_samples / float(self.sample_rate)

    @property
    def total_processed_seconds(self) -> float:
        return self._total_processed_samples / float(self.sample_rate)

    @property
    def streaming_lag_sec(self) -> float:
        return max(0.0, self.total_recorded_seconds - self.total_processed_seconds)

    @property
    def capture_queue_size(self) -> int:
        return self._capture_queue.qsize()

    @property
    def processing_queue_size(self) -> int:
        return self._processing_queue.qsize()

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    def start(self, output_wav_path: Optional[str] = None, source: Optional[AudioSource] = None):
        if self._is_recording:
            return

        self._wav_path = output_wav_path
        if self._wav_path:
            os.makedirs(os.path.dirname(self._wav_path), exist_ok=True)
            self._wav_file = sf.SoundFile(
                self._wav_path,
                mode='w',
                samplerate=self.sample_rate,
                channels=CHANNELS,
                subtype='PCM_16'
            )

        self._total_captured_samples = 0
        self._total_processed_samples = 0
        self._start_wall_time = time.time()
        self._is_recording = True
        self._is_paused = False

        # 1. 启动 WAV 写盘线程
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="AudioWavWriterThread"
        )
        self._writer_thread.start()

        # 2. 启动下游 ASR / VAD 分发处理线程
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name="AudioProcessingWorkerThread"
        )
        self._processing_thread.start()

        # 3. 启动音频输入源
        if source is not None:
            self._source = source
        elif self.custom_source is not None:
            self._source = self.custom_source
        else:
            self._source = MicrophoneAudioSource(
                sample_rate=self.sample_rate,
                device_index=self.device_index,
                blocksize=1600 # 100ms
            )

        self._source.start(self._on_source_audio)
        print(f"[AudioRecorder] Recording started, output={self._wav_path}")

    def pause(self):
        self._is_paused = True
        print("[AudioRecorder] Recording paused.")

    def resume(self):
        self._is_paused = False
        print("[AudioRecorder] Recording resumed.")

    def _on_source_audio(self, chunk: np.ndarray):
        """音频源回调（运行在 PortAudio 或 FileReplay 内部线程）"""
        if self._is_recording and not self._is_paused:
            self._capture_queue.put(chunk)

    def _writer_loop(self):
        """WAV 专用写盘线程：保证录音母带最高优先级，不受 ASR 推理耗时影响"""
        while self._is_recording or not self._capture_queue.empty():
            try:
                chunk = self._capture_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if chunk is None:
                # 退出哨兵
                self._capture_queue.task_done()
                break

            current_timestamp = self._total_captured_samples / float(self.sample_rate)
            self._total_captured_samples += len(chunk)

            # 1. 写 WAV
            if self._wav_file is not None:
                try:
                    self._wav_file.write(chunk)
                except Exception as e:
                    print(f"[AudioRecorder Error] Failed to write WAV: {e}")

            # 2. 推送至下游分发处理队列
            self._processing_queue.put((chunk, current_timestamp))
            self._capture_queue.task_done()

    def _processing_loop(self):
        """下游计算分发线程：执行 VAD 与 Paraformer 流式推理"""
        while self._is_recording or not self._processing_queue.empty():
            try:
                item = self._processing_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                self._processing_queue.task_done()
                break

            chunk, timestamp = item
            self._total_processed_samples += len(chunk)

            if self.on_audio_chunk:
                try:
                    self.on_audio_chunk(chunk, timestamp)
                except Exception as e:
                    print(f"[AudioRecorder Error] Processing callback error: {e}")

            self._processing_queue.task_done()

    def stop_capture(self):
        """阶段 1：停止音频采集，不再产生新音频"""
        if self._source is not None:
            try:
                self._source.stop()
            except Exception as e:
                print(f"[AudioRecorder Error] Source stop error: {e}")
            self._source = None
        self._is_recording = False
        self._is_paused = False

    def wait_until_drained(self, timeout: float = 10.0) -> bool:
        """阶段 2：等待所有已采集音频完成写盘与下游分发"""
        deadline = time.monotonic() + timeout

        # 等待 capture 队列排空
        while self._capture_queue.unfinished_tasks > 0:
            if time.monotonic() >= deadline:
                print(f"[AudioRecorder Warning] Capture queue drain timed out! Remaining: {self._capture_queue.unfinished_tasks}")
                return False
            time.sleep(0.02)

        # 等待 processing 队列排空
        while self._processing_queue.unfinished_tasks > 0:
            if time.monotonic() >= deadline:
                print(f"[AudioRecorder Warning] Processing queue drain timed out! Remaining: {self._processing_queue.unfinished_tasks}")
                return False
            time.sleep(0.02)

        return True

    def close(self):
        """阶段 3：关闭写盘文件与工作线程"""
        # 发送退出哨兵
        self._capture_queue.put(None)
        self._processing_queue.put(None)

        if self._writer_thread is not None:
            self._writer_thread.join(timeout=3.0)
            self._writer_thread = None

        if self._processing_thread is not None:
            self._processing_thread.join(timeout=3.0)
            self._processing_thread = None

        if self._wav_file is not None:
            try:
                self._wav_file.flush()
                self._wav_file.close()
            except Exception as e:
                print(f"[AudioRecorder Error] WAV close error: {e}")
            self._wav_file = None

        print(f"[AudioRecorder] Recording closed cleanly. Total recorded: {self.total_recorded_seconds:.2f}s, Processed: {self.total_processed_seconds:.2f}s")

    def stop(self, timeout: float = 10.0) -> bool:
        """完整三阶段收尾协议"""
        self.stop_capture()
        drained = self.wait_until_drained(timeout=timeout)
        self.close()
        return drained
