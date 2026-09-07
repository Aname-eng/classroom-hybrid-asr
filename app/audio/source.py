# coding: utf-8
import time
import threading
from abc import ABC, abstractmethod
from typing import Optional, Callable
import numpy as np
import sounddevice as sd
import soundfile as sf
import scipy.signal

class AudioSource(ABC):
    """音频输入源抽象基类"""
    @abstractmethod
    def start(self, callback: Callable[[np.ndarray], None]):
        """开始采集/播放音频，持续将 float32 16kHz PCM 单声道数据块传递给 callback"""
        pass

    @abstractmethod
    def stop(self):
        """停止音频产生"""
        pass

    @property
    @abstractmethod
    def is_active(self) -> bool:
        pass


class MicrophoneAudioSource(AudioSource):
    """真实麦克风音频源"""
    def __init__(self, sample_rate: int = 16000, device_index: Optional[int] = None, blocksize: int = 1600):
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.blocksize = blocksize # 100ms @ 16kHz
        self._stream: Optional[sd.InputStream] = None
        self._callback: Optional[Callable[[np.ndarray], None]] = None
        self._is_active = False

    def start(self, callback: Callable[[np.ndarray], None]):
        if self._is_active:
            return
        self._callback = callback
        self._is_active = True

        def _sd_callback(indata, frames, time_info, status):
            if status:
                print(f"[MicrophoneAudioSource Warning] Stream status: {status}")
            if self._is_active and self._callback:
                chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
                self._callback(chunk.astype(np.float32))

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            device=self.device_index,
            blocksize=self.blocksize,
            callback=_sd_callback
        )
        self._stream.start()

    def stop(self):
        self._is_active = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                print(f"[MicrophoneAudioSource Error] Stream close error: {e}")
            self._stream = None
        self._callback = None

    @property
    def is_active(self) -> bool:
        return self._is_active


class FileReplayAudioSource(AudioSource):
    """
    文件重放音频源（用于严谨的自动化回归测试与回放测试）
    将 WAV 文件以恒定 100ms chunk 持续推入管道，与真实麦克风行为完全一致。
    """
    def __init__(
        self,
        wav_path: str,
        sample_rate: int = 16000,
        chunk_size_samples: int = 1600,
        realtime_factor: float = 1.0 # 1.0 为真实速率，0.0 为全速推流
    ):
        self.wav_path = wav_path
        self.sample_rate = sample_rate
        self.chunk_size_samples = chunk_size_samples
        self.realtime_factor = realtime_factor
        
        self._is_active = False
        self._thread: Optional[threading.Thread] = None
        self._audio_data = self._load_audio(wav_path)

    def _load_audio(self, path: str) -> np.ndarray:
        data, sr = sf.read(path, dtype='float32')
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != self.sample_rate:
            target_len = int(len(data) * self.sample_rate / sr)
            data = scipy.signal.resample(data, target_len).astype(np.float32)
        return data

    @property
    def total_duration_seconds(self) -> float:
        return len(self._audio_data) / float(self.sample_rate)

    def start(self, callback: Callable[[np.ndarray], None]):
        if self._is_active:
            return
        self._is_active = True
        self._thread = threading.Thread(
            target=self._replay_loop,
            args=(callback,),
            daemon=True,
            name="FileReplayAudioSourceThread"
        )
        self._thread.start()

    def _replay_loop(self, callback: Callable[[np.ndarray], None]):
        total_samples = len(self._audio_data)
        idx = 0
        chunk_dur = self.chunk_size_samples / float(self.sample_rate)

        while self._is_active and idx < total_samples:
            t0 = time.monotonic()
            chunk = self._audio_data[idx:idx+self.chunk_size_samples]
            idx += len(chunk)
            callback(chunk)

            if self.realtime_factor > 0:
                sleep_dur = (chunk_dur * self.realtime_factor) - (time.monotonic() - t0)
                if sleep_dur > 0:
                    time.sleep(sleep_dur)
            else:
                # 极速模拟推流
                time.sleep(0.001)

        self._is_active = False

    def stop(self):
        self._is_active = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def is_active(self) -> bool:
        return self._is_active
