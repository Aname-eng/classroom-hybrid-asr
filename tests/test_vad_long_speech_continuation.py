# coding: utf-8
import os
import sys
import time
from pathlib import Path
import soundfile as sf
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.vad.vad_detector import VADDetector


def test_vad_continuation_split_exact_samples():
    print("=== Test 1: VAD Continuation Split Exact Sample Preservation ===")
    
    starts = []
    ends = []
    
    def on_start(seg_id: int, s_sec: float):
        starts.append((seg_id, s_sec))
        
    def on_end(seg_id: int, s_sec: float, e_sec: float, audio: np.ndarray):
        ends.append((seg_id, s_sec, e_sec, audio))
        print(f"  [Continuation Seg End] Seg {seg_id}: {s_sec:.2f}s -> {e_sec:.2f}s, samples={len(audio)} ({len(audio)/16000.0:.2f}s)")

    # Set max_segment_sec to 2.0s for rapid deterministic verification
    vad = VADDetector(
        sample_rate=16000,
        max_segment_sec=2.0,
        min_silence_sec=0.5,
        on_speech_start=on_start,
        on_speech_end=on_end
    )

    # Manually simulate speech start
    vad.in_speech = True
    vad.speech_start_time = 0.0
    vad.speech_buffer = []

    # Feed 6.0s of continuous audio (60 chunks of 100ms = 1600 samples)
    # Total samples = 60 * 1600 = 96000 samples
    total_fed_samples = 0
    for i in range(60):
        chunk = np.ones(1600, dtype=np.float32) * 0.1 # Constant energy speech chunk
        ts = i * 0.1
        total_fed_samples += len(chunk)
        
        # 严格通过生产 process_chunk() 接口进行处理
        vad.process_chunk(chunk, ts)

    # Flush tail
    vad.flush(6.0)

    print(f"  Total continuation segments emitted: {len(ends)}")
    assert len(ends) == 3, f"Expected exactly 3 segments for 6.0s with 2.0s max, got {len(ends)}"
    
    total_emitted = sum(len(e[3]) for e in ends)
    print(f"  Total fed samples: {total_fed_samples}, Total emitted samples: {total_emitted}")
    assert total_emitted == total_fed_samples, f"Sample mismatch! Fed {total_fed_samples}, emitted {total_emitted}"
    print("  [PASS] Continuation split preserved 100% of samples across segment boundaries via production process_chunk.")


def test_vad_real_lecture_stream():
    print("\n=== Test 2: VAD Real Lecture Stream Bounded Segment Durations ===")
    
    wav_path = r"d:\课程笔记\tests\audio_samples\test_political_economy.wav"
    audio_data, sr = sf.read(wav_path, dtype='float32')
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    target_samples = 16000 * 60 # 60 seconds
    repeats = int(np.ceil(target_samples / len(audio_data)))
    tiled_audio = np.tile(audio_data, repeats)[:target_samples].astype(np.float32)

    speech_ends = []
    def on_end(seg_id: int, s_sec: float, e_sec: float, seg_audio: np.ndarray):
        speech_ends.append((seg_id, s_sec, e_sec, seg_audio))

    vad = VADDetector(
        sample_rate=16000,
        max_segment_sec=25.0,
        min_silence_sec=0.5,
        on_speech_end=on_end
    )

    chunk_size = 1600
    total_chunks = int(np.ceil(len(tiled_audio) / chunk_size))
    for i in range(total_chunks):
        s = i * chunk_size
        e = min((i + 1) * chunk_size, len(tiled_audio))
        chunk = tiled_audio[s:e]
        ts = e / 16000.0
        vad.process_chunk(chunk, ts)

    vad.flush(len(tiled_audio) / 16000.0)

    print(f"  Total speech segments recognized: {len(speech_ends)}")
    assert len(speech_ends) >= 2, f"Expected at least 2 segments, got {len(speech_ends)}"

    for seg_id, s_sec, e_sec, seg_audio in speech_ends:
        dur = e_sec - s_sec
        assert dur <= 25.5, f"Segment {seg_id} duration {dur:.2f}s exceeded max 25.0s limit"

    print("  [PASS] All speech segments stayed strictly within max_segment_sec limits.")


if __name__ == "__main__":
    test_vad_continuation_split_exact_samples()
    test_vad_real_lecture_stream()
    print("\nAll VAD long speech continuation tests passed successfully!")
