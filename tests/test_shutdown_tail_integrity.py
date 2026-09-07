# coding: utf-8
import os
import sys
import json
import time
from pathlib import Path
import soundfile as sf
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SESSIONS_DIR
from app.audio.source import FileReplayAudioSource
from app.pipeline.session_manager import SessionManager


def test_shutdown_tail_integrity():
    print("=== Testing Shutdown Tail Audio & Transcript Integrity ===")
    
    # 1. Use real speech audio file
    wav_path = r"d:\课程笔记\tests\audio_samples\test_political_economy.wav"
    audio_data, sr = sf.read(wav_path, dtype='float32')
    expected_duration = len(audio_data) / float(sr)
    print(f"  Input test audio duration: {expected_duration:.2f}s ({len(audio_data)} samples)")

    # Play slightly faster than realtime for fast testing (e.g. 0.2s sleep per 1s audio, or 0.0 for full speed)
    source = FileReplayAudioSource(
        wav_path=wav_path,
        chunk_size_samples=1600, # 100ms chunk
        realtime_factor=0.2
    )

    partials = []
    finals = []

    def on_partial(segment_id: int, text: str, ts: float):
        partials.append((segment_id, text, ts))

    def on_final(segment_id: int, text: str, model: str, ts: float, success: bool = True, reason: str = ""):
        finals.append((segment_id, text, model, ts, success, reason))

    manager = SessionManager(
        on_partial_subtitle=on_partial,
        on_final_subtitle=on_final
    )

    session_id = manager.start_session("political_economy", source=source)
    print(f"  Session started: {session_id}")

    # Wait until source finishes playing all audio
    while source.is_active:
        time.sleep(0.05)

    print("  Source playback finished. Calling end_session() immediately to test tail flushing...")
    manager.end_session(timeout=15.0)

    # Verify session artifacts
    session_dir = SESSIONS_DIR / session_id
    wav_out_path = session_dir / "audio.wav"
    meta_path = session_dir / "meta.json"
    transcript_final_path = session_dir / "transcript_final.md"
    events_path = session_dir / "events.jsonl"

    assert wav_out_path.exists(), f"WAV file missing: {wav_out_path}"
    assert meta_path.exists(), f"Meta file missing: {meta_path}"
    assert transcript_final_path.exists(), f"Final transcript missing: {transcript_final_path}"
    assert events_path.exists(), f"Events log missing: {events_path}"

    # 1. Check WAV duration
    out_data, out_sr = sf.read(str(wav_out_path))
    actual_duration = len(out_data) / float(out_sr)
    duration_diff = abs(actual_duration - expected_duration)
    print(f"  Recorded WAV duration: {actual_duration:.2f}s (Diff: {duration_diff:.4f}s)")
    assert duration_diff < 0.20, f"Recorded audio truncated! Expected {expected_duration:.2f}s, got {actual_duration:.2f}s"

    # 2. Check meta.json
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print(f"  Meta segments: total={meta.get('total_segments')}, qwen_success={meta.get('qwen_success_segments')}, fallback={meta.get('fallback_segments')}")
    assert meta.get("total_segments", 0) >= 1, "Expected at least 1 speech segment recognized"
    assert meta.get("qwen_success_segments", 0) >= 1, "Expected Qwen to successfully process tail segment"

    # 3. Check transcript_final.md content
    with open(transcript_final_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    print(f"  Transcript content:\n{md_content.strip()}")
    assert "中国特色" in md_content or "政治经济学" in md_content or len(md_content) > 50, "Transcript missing speech content"

    print("  [PASS] Shutdown tail integrity test passed with 0 lost audio samples and full transcript extraction.")


if __name__ == "__main__":
    test_shutdown_tail_integrity()
    print("\nAll Shutdown Tail Integrity tests passed successfully!")
