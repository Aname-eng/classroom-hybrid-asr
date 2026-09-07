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


def test_end_to_end_replay_political_economy():
    print("\n=== End-to-End Replay Test: Political Economy Sample ===")
    wav_path = r"d:\课程笔记\tests\audio_samples\test_political_economy.wav"
    assert os.path.exists(wav_path), f"Sample WAV missing: {wav_path}"

    info = sf.info(wav_path)
    print(f"  Input WAV: duration={info.duration:.2f}s, sr={info.samplerate}, channels={info.channels}")

    # Use 0.5x realtime to simulate playback and verify pipeline under realistic pace
    source = FileReplayAudioSource(wav_path=wav_path, chunk_size_samples=1600, realtime_factor=0.5)
    manager = SessionManager()

    partials_received = []
    finals_received = []

    def on_partial(seg_id, text, ts):
        partials_received.append((seg_id, text, ts))
        print(f"  [Partial] Seg {seg_id} @ {ts:.2f}s: '{text}'")

    def on_final(seg_id, text, model, ts, success, reason):
        finals_received.append((seg_id, text, model, ts, success, reason))
        status = "SUCCESS" if success else f"FALLBACK({reason})"
        print(f"  [Final {status}] Seg {seg_id} @ {ts:.2f}s: '{text}' (Model: {model})")

    manager.on_partial_received = on_partial
    manager.on_final_received = on_final

    print("  Starting session...")
    t_start = time.time()
    session_id = manager.start_session("political_economy", source=source)
    print(f"  Session ID: {session_id}")

    # Wait for audio replay to finish
    while source.is_active:
        time.sleep(0.05)

    print("  Audio source replay completed. Ending session...")
    manager.end_session(timeout=15.0)
    total_elapsed = time.time() - t_start
    print(f"  Total pipeline run time: {total_elapsed:.2f}s")

    # Assertions on outputs
    session_dir = SESSIONS_DIR / session_id
    assert session_dir.exists(), f"Session dir does not exist: {session_dir}"

    wav_out = session_dir / "audio.wav"
    meta_out = session_dir / "meta.json"
    raw_out = session_dir / "transcript_raw.jsonl"
    md_out = session_dir / "transcript_final.md"
    events_out = session_dir / "events.jsonl"

    assert wav_out.exists(), "audio.wav missing"
    assert meta_out.exists(), "meta.json missing"
    assert raw_out.exists(), "transcript_raw.jsonl missing"
    assert md_out.exists(), "transcript_final.md missing"
    assert events_out.exists(), "events.jsonl missing"

    # Verify audio integrity
    out_info = sf.info(str(wav_out))
    print(f"  Recorded audio.wav duration: {out_info.duration:.2f}s (Original: {info.duration:.2f}s)")
    assert abs(out_info.duration - info.duration) < 0.1, "Audio duration mismatch between source and recording"

    # Verify metadata
    with open(meta_out, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print(f"  Meta Summary: total_segments={meta.get('total_segments')}, qwen_success={meta.get('qwen_success_segments')}, fallback={meta.get('fallback_segments')}")
    assert meta.get("total_segments", 0) >= 1, "Expected at least 1 segment"
    assert meta.get("qwen_success_segments", 0) >= 1, "Expected at least 1 authoritative Qwen success"
    assert meta.get("fallback_segments", 0) == 0, f"Expected 0 fallbacks under normal operation, got {meta.get('fallback_segments')}"

    # Verify markdown transcript
    with open(md_out, "r", encoding="utf-8") as f:
        md = f.read()

    print(f"\n--- Final Transcript Preview ---\n{md}\n--------------------------------\n")
    assert "中国特色社会主义" in md or "政治经济学" in md or len(md.strip()) > 50, "Transcript does not contain expected recognized keywords"
    assert "[实时回退]" not in md, "Expected clean transcript without fallback tags"

    print("  [PASS] End-to-end replay test passed perfectly!")


def test_end_to_end_replay_hausman():
    print("\n=== End-to-End Replay Test: Hausman Test Sample ===")
    wav_path = r"d:\课程笔记\tests\audio_samples\test_hausman.wav"
    assert os.path.exists(wav_path), f"Sample WAV missing: {wav_path}"

    info = sf.info(wav_path)
    print(f"  Input WAV: duration={info.duration:.2f}s, sr={info.samplerate}, channels={info.channels}")

    source = FileReplayAudioSource(wav_path=wav_path, chunk_size_samples=1600, realtime_factor=0.5)
    manager = SessionManager()

    finals_received = []
    def on_final(seg_id, text, model, ts, success, reason):
        finals_received.append((seg_id, text, model, ts, success, reason))
        status = "SUCCESS" if success else f"FALLBACK({reason})"
        print(f"  [Final {status}] Seg {seg_id} @ {ts:.2f}s: '{text}' (Model: {model})")

    manager.on_final_received = on_final

    session_id = manager.start_session("econometrics", source=source)
    while source.is_active:
        time.sleep(0.05)

    manager.end_session(timeout=15.0)

    session_dir = SESSIONS_DIR / session_id
    with open(session_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    print(f"  Meta Summary: total_segments={meta.get('total_segments')}, qwen_success={meta.get('qwen_success_segments')}, fallback={meta.get('fallback_segments')}")
    assert meta.get("total_segments", 0) >= 1
    assert meta.get("qwen_success_segments", 0) >= 1
    assert meta.get("fallback_segments", 0) == 0

    with open(session_dir / "transcript_final.md", "r", encoding="utf-8") as f:
        md = f.read()

    print(f"\n--- Final Transcript Preview ---\n{md}\n--------------------------------\n")
    print("  [PASS] Hausman test passed perfectly!")


if __name__ == "__main__":
    test_end_to_end_replay_political_economy()
    test_end_to_end_replay_hausman()
    print("\nAll End-to-End File Replay Tests PASSED!")
