# coding: utf-8
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import json
import soundfile as sf
import numpy as np
import scipy.signal
from app.config import AppConfig, SESSIONS_DIR
from app.pipeline.session_manager import SessionManager

def ensure_16k_mono(audio_path: str) -> np.ndarray:
    data, sr = sf.read(audio_path, dtype='float32')
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != 16000:
        target_len = int(len(data) * 16000 / sr)
        data = scipy.signal.resample(data, target_len).astype(np.float32)
    return data

def main():
    print("=== Testing 2-Pass Classroom Pipeline (Streaming + Qwen Offline) ===")
    
    # Track subtitle updates in test
    displayed_subtitles = {}
    
    def on_partial(segment_id: int, text: str, ts: float):
        displayed_subtitles[segment_id] = f"[Partial #{segment_id}] {text}"
        print(f"  [UI Partial] Seg {segment_id:02d} ({ts:05.2f}s): '{text}'")

    def on_final(segment_id: int, text: str, model: str, ts: float):
        displayed_subtitles[segment_id] = f"[Final #{segment_id} ({model})] {text}"
        print(f"\n  >>> [UI REPLACED TO FINAL] Seg {segment_id:02d} ({ts:05.2f}s) [{model}]:\n      '{text}'\n")

    def on_status(status: dict):
        pass

    manager = SessionManager(
        on_partial_subtitle=on_partial,
        on_final_subtitle=on_final,
        on_status_update=on_status
    )
    manager.select_course("political_economy")

    # Start session
    session_id = manager.start_session("political_economy")
    print(f"Session started: {session_id}")

    # Feed 2 test sentences in sequence with silence in between
    wav1 = r"d:\课程笔记\tests\audio_samples\test_political_economy.wav"
    wav2 = r"d:\课程笔记\tests\audio_samples\test_hausman.wav"
    
    audio1 = ensure_16k_mono(wav1)
    audio2 = ensure_16k_mono(wav2)
    silence = np.zeros(int(16000 * 1.5), dtype=np.float32) # 1.5s silence

    full_audio = np.concatenate([audio1, silence, audio2, silence])
    print(f"Total simulated lecture audio: {len(full_audio)/16000.0:.2f}s")

    # Simulate real-time streaming chunk ingestion (480ms chunk = 7680 samples)
    chunk_size = 7680
    total_chunks = int(np.ceil(len(full_audio) / chunk_size))

    for i in range(total_chunks):
        s = i * chunk_size
        e = min((i + 1) * chunk_size, len(full_audio))
        chunk = full_audio[s:e]
        timestamp = (i * chunk_size) / 16000.0
        
        manager._on_audio_chunk(chunk, timestamp)
        # Sleep for fraction of real-time to simulate rapid or real-time delivery
        time.sleep(0.05)

    print("\nEnding session and waiting for Qwen processing...")
    manager.end_session()

    print("\n=== Verified Final Output Documents ===")
    session_dir = SESSIONS_DIR / session_id
    for f in ["audio.wav", "events.jsonl", "transcript_raw.jsonl", "transcript_final.md", "meta.json"]:
        p = session_dir / f
        if p.exists():
            print(f"  [OK] {f} exists (size: {p.stat().st_size} bytes)")
        else:
            print(f"  [FAIL] {f} missing!")

    print("\n--- Final Markdown Transcript Content ---")
    md_content = (session_dir / "transcript_final.md").read_text(encoding="utf-8")
    print(md_content)

if __name__ == "__main__":
    main()
