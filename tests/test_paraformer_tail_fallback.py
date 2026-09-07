# coding: utf-8
import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SESSIONS_DIR
from app.audio.source import FileReplayAudioSource
from app.pipeline.session_manager import SessionManager


def test_paraformer_tail_fallback():
    print("=== Test Paraformer Tail Audio Flushing & Fallback Timing ===")
    
    wav_path = r"d:\课程笔记\tests\audio_samples\test_political_economy.wav"
    assert os.path.exists(wav_path), f"Sample missing: {wav_path}"

    source = FileReplayAudioSource(wav_path=wav_path, realtime_factor=0.3)
    manager = SessionManager()
    
    # 故意将 Qwen 离线，测试 Paraformer 纯流式 fallback 尾音提取
    manager.qwen_adapter.ws_uri = "ws://127.0.0.1:59999"

    session_id = manager.start_session("political_economy", source=source)
    print(f"  Session started with offline Qwen: {session_id}")

    while source.is_active:
        time.sleep(0.05)

    print("  Audio playback ended, closing session...")
    manager.end_session(timeout=8.0)

    session_dir = SESSIONS_DIR / session_id
    meta_path = session_dir / "meta.json"
    transcript_path = session_dir / "transcript_final.md"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print(f"  Meta verification: total={meta['total_segments']}, qwen_success={meta['qwen_success_segments']}, fallback={meta['fallback_segments']}")
    assert meta["total_segments"] >= 1
    assert meta["fallback_segments"] >= 1
    assert meta["qwen_success_segments"] == 0

    with open(transcript_path, "r", encoding="utf-8") as f:
        md = f.read()

    print(f"\n--- Fallback Transcript Preview ---\n{md}\n-----------------------------------\n")
    
    # 验证尾部关键词是否因 accumulator flush 而成功被 Paraformer 识别捕获
    assert "经济" in md or "制度" in md or "中国" in md, f"Tail keywords missing from fallback transcript: {md}"
    assert "[实时回退]" in md, "Expected [实时回退] tag on fallback segments"

    print("  [PASS] Paraformer tail audio correctly flushed and preserved in fallback transcript!")


if __name__ == "__main__":
    test_paraformer_tail_fallback()
