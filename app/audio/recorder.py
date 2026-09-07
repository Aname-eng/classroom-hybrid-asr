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

class AudioRecorder:
    """
    非阻塞麦克风录音机与音频主分发器
    
    1. 持续采集麦克风 16kHz float32 音频流
    2. 同步写入主课堂录音文件 (audio.wav)，确保绝不丢音
    3. 向 VAD 和流式 ASR 分发 PCM 数据块
    """
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        device_index: Optional[int] = None,
        on_audio_chunk: Optional[Callable[[np.ndarray, float], None]] = None
    ):
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.on_audio_chunk = on_audio_chunk
        
        self._is_recording = False
        self._is_paused = False
        self._stream: Optional[sd.InputStream] = None
        self._wav_file: Optional[sf.SoundFile] = None
        self._wav_path: Optional[str] = None
        
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._writer_thread: Optional[threading.Thread] = None
        self._total_samples = 0
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
        return self._total_samples / float(self.sample_rate)

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    def start(self, output_wav_path: Optional[str] = None):
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

        self._total_samples = 0
        self._start_wall_time = time.time()
        self._is_recording = True
        self._is_paused = False

        # 启动写盘与分发线程
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True, name="AudioWriterThread")
        self._writer_thread.start()

        # 打开音频输入流
        blocksize = 1600 # 100ms 块
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype='float32',
            device=self.device_index,
            blocksize=blocksize,
            callback=self._audio_callback
        )
        self._stream.start()
        print(f"[AudioRecorder] Recording started on device {self.device_index}, output={self._wav_path}")

    def pause(self):
        self._is_paused = True
        print("[AudioRecorder] Recording paused.")

    def resume(self):
        self._is_paused = False
        print("[AudioRecorder] Recording resumed.")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[AudioRecorder Warning] Stream status: {status}")
        if self._is_recording and not self._is_paused:
            # 复制数据压入队列
            chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
            self._audio_queue.put(chunk)

    def _writer_loop(self):
        while self._is_recording or not self._audio_queue.empty():
            try:
                chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # 1. 写入全量主录音 WAV (PCM16 格式)
            if self._wav_file is not None:
                try:
                    self._wav_file.write(chunk)
                except Exception as e:
                    print(f"[AudioRecorder Error] Failed to write WAV: {e}")

            current_timestamp = self._total_samples / float(self.sample_rate)
            self._total_samples += len(chunk)

            # 2. 分发给下游实时处理器 (VAD / Streaming)
            if self.on_audio_chunk:
                try:
                    self.on_audio_chunk(chunk, current_timestamp)
                except Exception as e:
                    print(f"[AudioRecorder Error] Chunk callback error: {e}")

            self._audio_queue.task_done()

    def stop(self):
        if not self._is_recording:
            return

        self._is_recording = False
        self._is_paused = False

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                print(f"[AudioRecorder Error] Stream close error: {e}")
            self._stream = None

        if self._writer_thread is not None:
            self._writer_thread.join(timeout=3.0)
            self._writer_thread = None

        if self._wav_file is not None:
            try:
                self._wav_file.flush()
                self._wav_file.close()
            except Exception as e:
                print(f"[AudioRecorder Error] WAV close error: {e}")
            self._wav_file = None

        print(f"[AudioRecorder] Recording stopped. Total duration: {self.total_recorded_seconds:.2f}s")
