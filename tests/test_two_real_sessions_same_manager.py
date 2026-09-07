# coding: utf-8
import os
import sys
import json
import time
from pathlib import Path
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SESSIONS_DIR
from app.audio.source import FileReplayAudioSource
from app.pipeline.session_manager import SessionManager


def test_two_real_sessions_same_manager():
    print("=== Multi-Session Lifecycle Test on Same SessionManager ===")
    
    manager = SessionManager()

    # Session 1: Political Economy sample
    print("\n--- Session 1: Political Economy ---")
    wav1_path = r"d:\课程笔记\tests\audio_samples\test_political_economy.wav"
    assert os.path.exists(wav1_path), f"Sample missing: {wav1_path}"
    info1 = sf.info(wav1_path)
    
    source1 = FileReplayAudioSource(wav_path=wav1_path, realtime_factor=0.3)
    s1_id = manager.start_session("political_economy", source=source1)
    print(f"  Session 1 started: {s1_id}")

    while source1.is_active:
        time.sleep(0.05)

    print("  Session 1 audio ended, closing...")
    manager.end_session(timeout=15.0)

    s1_dir = SESSIONS_DIR / s1_id
    assert (s1_dir / "audio.wav").exists(), "Session 1 audio.wav missing"
    assert (s1_dir / "meta.json").exists(), "Session 1 meta.json missing"
    assert (s1_dir / "transcript_final.md").exists(), "Session 1 transcript_final.md missing"

    s1_wav_info = sf.info(str(s1_dir / "audio.wav"))
    print(f"  Session 1 recorded duration: {s1_wav_info.duration:.2f}s (Original: {info1.duration:.2f}s)")
    assert abs(s1_wav_info.duration - info1.duration) < 0.1, "Session 1 audio duration mismatch"

    with open(s1_dir / "meta.json", "r", encoding="utf-8") as f:
        m1 = json.load(f)
    print(f"  Session 1 meta: total_segments={m1['total_segments']}, qwen_success={m1['qwen_success_segments']}, fallback={m1['fallback_segments']}")
    assert m1["total_segments"] >= 1, "Session 1 has no segments"
    assert m1["qwen_success_segments"] >= 1, "Session 1 Qwen should have succeeded"

    # Session 2: Econometrics sample on SAME SessionManager instance
    print("\n--- Session 2: Microeconometrics on SAME SessionManager ---")
    wav2_path = r"d:\课程笔记\tests\audio_samples\test_hausman.wav"
    assert os.path.exists(wav2_path), f"Sample missing: {wav2_path}"
    info2 = sf.info(wav2_path)

    source2 = FileReplayAudioSource(wav_path=wav2_path, realtime_factor=0.3)
    s2_id = manager.start_session("microeconometrics", source=source2)
    print(f"  Session 2 started: {s2_id}")

    while source2.is_active:
        time.sleep(0.05)

    print("  Session 2 audio ended, closing...")
    manager.end_session(timeout=15.0)

    s2_dir = SESSIONS_DIR / s2_id
    assert (s2_dir / "audio.wav").exists(), "Session 2 audio.wav missing"
    assert (s2_dir / "meta.json").exists(), "Session 2 meta.json missing"
    assert (s2_dir / "transcript_final.md").exists(), "Session 2 transcript_final.md missing"

    s2_wav_info = sf.info(str(s2_dir / "audio.wav"))
    print(f"  Session 2 recorded duration: {s2_wav_info.duration:.2f}s (Original: {info2.duration:.2f}s)")
    assert abs(s2_wav_info.duration - info2.duration) < 0.1, "Session 2 audio duration mismatch"

    with open(s2_dir / "meta.json", "r", encoding="utf-8") as f:
        m2 = json.load(f)
    print(f"  Session 2 meta: total_segments={m2['total_segments']}, qwen_success={m2['qwen_success_segments']}, fallback={m2['fallback_segments']}")
    assert m2["total_segments"] >= 1, "Session 2 has no segments"
    assert m2["qwen_success_segments"] >= 1, "Session 2 Qwen should have succeeded on same manager"

    with open(s2_dir / "transcript_final.md", "r", encoding="utf-8") as f:
        md2 = f.read()
    print(f"\n--- Session 2 Transcript Preview ---\n{md2}\n-----------------------------------\n")

    print("  [PASS] Multi-session lifecycle on same SessionManager passed perfectly!")


if __name__ == "__main__":
    test_two_real_sessions_same_manager()
