# coding: utf-8
import os
import sys
import time
import json
import psutil
import threading
from pathlib import Path
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SESSIONS_DIR
from app.audio.source import AudioSource
from app.pipeline.session_manager import SessionManager


class LoopingSpeechAudioSource(AudioSource):
    """
    Feeds real speech samples interleaved with pauses for a target total duration,
    simulating a multi-segment long lecture.
    """
    def __init__(self, sample_paths: list, total_duration_sec: float = 60.0, realtime_factor: float = 0.6):
        self.sample_rate = 16000
        self.chunk_size_samples = 1600
        self.sample_paths = sample_paths
        self.total_duration_sec = total_duration_sec
        self.realtime_factor = realtime_factor
        self._samples_list = []
        for p in sample_paths:
            data, sr = sf.read(p, dtype='float32')
            if data.ndim > 1:
                data = data.mean(axis=1)
            self._samples_list.append(data)
        self._is_active = False
        self._stop_event = threading.Event()
        self._thread = None

    @property
    def is_active(self) -> bool:
        return self._is_active

    def start(self, callback):
        if self._is_active:
            return
        self._is_active = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._stream_loop,
            args=(callback,),
            daemon=True,
            name="LoopingSpeechAudioSourceThread"
        )
        self._thread.start()

    def stop(self):
        self._is_active = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _stream_loop(self, callback):
        samples_emitted = 0
        target_samples = int(self.total_duration_sec * self.sample_rate)
        chunk_dur = self.chunk_size_samples / self.sample_rate

        sample_idx = 0
        pause_chunk = np.zeros(self.chunk_size_samples, dtype=np.float32)

        while not self._stop_event.is_set() and samples_emitted < target_samples:
            # Emit an audio file in chunks
            data = self._samples_list[sample_idx % len(self._samples_list)]
            sample_idx += 1

            for i in range(0, len(data), self.chunk_size_samples):
                if self._stop_event.is_set() or samples_emitted >= target_samples:
                    break
                chunk = data[i:i + self.chunk_size_samples]
                if len(chunk) < self.chunk_size_samples:
                    chunk = np.pad(chunk, (0, self.chunk_size_samples - len(chunk)))
                
                callback(chunk)
                samples_emitted += len(chunk)
                
                if self.realtime_factor > 0:
                    time.sleep(chunk_dur * self.realtime_factor)

            # Insert 1.5s pause between sentences
            for _ in range(15):
                if self._stop_event.is_set() or samples_emitted >= target_samples:
                    break
                callback(pause_chunk)
                samples_emitted += len(pause_chunk)
                if self.realtime_factor > 0:
                    time.sleep(chunk_dur * self.realtime_factor)

        self._is_active = False


def test_stability_soak(duration_sec: float = 1800.0):
    print(f"\n=== 30-Minute Soak Stability Test ({duration_sec:.0f}s simulated lecture) ===")
    
    samples = [
        r"d:\课程笔记\tests\audio_samples\test_political_economy.wav",
        r"d:\课程笔记\tests\audio_samples\test_hausman.wav"
    ]
    for s in samples:
        assert os.path.exists(s), f"Sample missing: {s}"

    process = psutil.Process(os.getpid())
    initial_rss = process.memory_info().rss / (1024 * 1024)
    print(f"  Initial Process RSS: {initial_rss:.2f} MB")

    # Use accelerated feed (realtime_factor=0.02) to simulate full 1800s lecture in ~40-60s
    source = LoopingSpeechAudioSource(samples, total_duration_sec=duration_sec, realtime_factor=0.02)
    manager = SessionManager()

    max_capture_q = 0
    max_proc_q = 0
    max_qwen_q = 0

    session_id = manager.start_session("political_economy", source=source)
    print(f"  Session started: {session_id}")

    start_time = time.time()
    last_log = start_time

    while source.is_active or manager.recorder._processing_queue.qsize() > 0 or manager.recorder._capture_queue.qsize() > 0:
        time.sleep(0.1)
        cur_time = time.time()

        # Track queue sizes
        max_capture_q = max(max_capture_q, manager.recorder._capture_queue.qsize())
        max_proc_q = max(max_proc_q, manager.recorder._processing_queue.qsize())
        max_qwen_q = max(max_qwen_q, manager.qwen_worker._task_queue.qsize())

        if cur_time - last_log >= 10.0:
            last_log = cur_time
            cur_rss = process.memory_info().rss / (1024 * 1024)
            print(f"  [Soak Monitor] Recorded: {manager.recorder.total_recorded_seconds:.1f}s / {duration_sec:.0f}s | Processed: {manager.recorder.total_processed_seconds:.1f}s | RSS: {cur_rss:.2f} MB | CaptureQ: {manager.recorder._capture_queue.qsize()} | ProcQ: {manager.recorder._processing_queue.qsize()} | QwenQ: {manager.qwen_worker._task_queue.qsize()}")

    print(f"  Audio processing completed ({manager.recorder.total_processed_seconds:.1f}s audio processed). Shutting down...")
    t0 = time.time()
    manager.end_session(timeout=60.0)
    shutdown_time = time.time() - t0
    final_rss = process.memory_info().rss / (1024 * 1024)

    print(f"\n  === 30-Minute Soak Results Summary ===")
    print(f"  Total Audio Simulated: {manager.recorder.total_recorded_seconds:.2f}s (Target: {duration_sec:.2f}s)")
    print(f"  Total Audio Processed: {manager.recorder.total_processed_seconds:.2f}s")
    print(f"  Max Capture Queue Size Observed: {max_capture_q}")
    print(f"  Max Processing Queue Size Observed: {max_proc_q}")
    print(f"  Max Qwen Queue Size Observed: {max_qwen_q}")
    print(f"  Final Capture Queue Size: {manager.recorder._capture_queue.qsize()} (drained to 0)")
    print(f"  Final Processing Queue Size: {manager.recorder._processing_queue.qsize()} (drained to 0)")
    print(f"  Final Qwen Queue Size: {manager.qwen_worker._task_queue.qsize()} (drained to 0)")
    print(f"  Initial RSS: {initial_rss:.2f} MB -> Final RSS: {final_rss:.2f} MB (Delta: {final_rss - initial_rss:+.2f} MB)")
    print(f"  Shutdown Elapsed: {shutdown_time:.2f}s")

    # Assertions
    assert abs(manager.recorder.total_recorded_seconds - duration_sec) < 1.0, f"Recorded audio duration mismatch: {manager.recorder.total_recorded_seconds}s != {duration_sec}s"
    assert abs(manager.recorder.total_recorded_seconds - manager.recorder.total_processed_seconds) < 0.1, "Mismatch in recorded vs processed audio"
    assert manager.recorder._capture_queue.qsize() == 0, "Capture queue not drained to 0"
    assert manager.recorder._processing_queue.qsize() == 0, "Processing queue not drained to 0"
    assert manager.qwen_worker._task_queue.qsize() == 0, "Qwen queue not drained to 0"
    assert shutdown_time < 30.0, f"Shutdown took too long: {shutdown_time:.2f}s"

    session_dir = SESSIONS_DIR / session_id
    with open(session_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    print(f"  Meta segments: total={meta.get('total_segments')}, qwen_success={meta.get('qwen_success_segments')}, fallback={meta.get('fallback_segments')}")
    assert meta.get("total_segments", 0) >= 10, f"Expected at least 10 segments for 30-min soak test, got {meta.get('total_segments')}"

    print(f"  [PASS] Full 30-minute ({duration_sec:.0f}s) soak stability test passed!")


if __name__ == "__main__":
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 1800.0
    test_stability_soak(duration_sec=dur)
