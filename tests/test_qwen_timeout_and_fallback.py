# coding: utf-8
import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import soundfile as sf

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SESSIONS_DIR
from app.offline.qwen_adapter import QwenAdapter
from app.offline.qwen_worker import QwenWorker, SegmentTask, SegmentResult
from app.audio.source import FileReplayAudioSource
from app.pipeline.session_manager import SessionManager


def test_qwen_worker_fallback_unit():
    print("=== Test 1: QwenWorker Fallback on Unreachable Server ===")
    
    # 1. Create adapter pointing to dead port
    dead_adapter = QwenAdapter(ws_uri="ws://127.0.0.1:59999")
    
    results = []
    def on_result(res: SegmentResult):
        results.append(res)
        print(f"  [Qwen Result Callback] Seg {res.segment_id}: final_text='{res.final_text}', success={res.success}, reason='{res.fallback_reason}'")

    worker = QwenWorker(adapter=dead_adapter, on_result=on_result)
    worker.start()

    # 2. Submit tasks
    dummy_audio = np.zeros(16000, dtype=np.float32)
    task1 = SegmentTask(
        segment_id=1,
        start_sec=0.0,
        end_sec=1.0,
        audio=dummy_audio,
        online_text="第一句流式识别候选文本"
    )
    task2 = SegmentTask(
        segment_id=2,
        start_sec=1.0,
        end_sec=2.0,
        audio=dummy_audio,
        online_text="第二句流式识别候选文本"
    )

    worker.submit_segment(task1)
    worker.submit_segment(task2)

    # 3. Stop worker with bounded timeout
    t0 = time.time()
    completed = worker.stop(wait_finish=True, timeout=5.0)
    elapsed = time.time() - t0
    print(f"  Worker stopped in {elapsed:.2f}s (completed={completed})")

    assert elapsed < 6.0, f"Worker shutdown took too long: {elapsed:.2f}s"
    assert len(results) == 2, f"Expected 2 results emitted via fallback, got {len(results)}"
    
    for r in results:
        assert r.success is False, "Expected success=False on unreachable server"
        assert r.fallback_reason is not None, "Expected fallback_reason to be populated"
        assert r.final_text in ["第一句流式识别候选文本", "第二句流式识别候选文本"], f"Unexpected fallback text: {r.final_text}"

    print("  [PASS] QwenWorker cleanly fallen back to online_text with bounded timeout.")


def test_session_manager_with_offline_qwen():
    print("\n=== Test 2: Full SessionManager Pipeline with Offline Qwen ===")
    
    wav_path = r"d:\课程笔记\tests\audio_samples\test_political_economy.wav"
    source = FileReplayAudioSource(wav_path=wav_path, chunk_size_samples=1600, realtime_factor=0.2)

    manager = SessionManager()
    # Point Qwen to dead port
    manager.qwen_adapter.ws_uri = "ws://127.0.0.1:59999"

    session_id = manager.start_session("political_economy", source=source)
    print(f"  Session started: {session_id}")

    while source.is_active:
        time.sleep(0.05)

    print("  Playback ended, closing session with bounded timeout...")
    t0 = time.time()
    manager.end_session(timeout=8.0)
    elapsed = time.time() - t0
    print(f"  end_session() completed in {elapsed:.2f}s")

    assert elapsed < 10.0, f"end_session() took too long: {elapsed:.2f}s"

    session_dir = SESSIONS_DIR / session_id
    meta_path = session_dir / "meta.json"
    transcript_path = session_dir / "transcript_final.md"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print(f"  Meta verification: total={meta.get('total_segments')}, qwen_success={meta.get('qwen_success_segments')}, fallback={meta.get('fallback_segments')}")
    assert meta.get("fallback_segments", 0) >= 1, "Expected fallback_segments >= 1"
    assert meta.get("qwen_success_segments", 0) == 0, "Expected qwen_success_segments == 0"

    with open(transcript_path, "r", encoding="utf-8") as f:
        md = f.read()

    print(f"  Transcript preview:\n{md.strip()}")
    assert "[实时回退]" in md, "Expected [实时回退] tag in markdown transcript"

    print("  [PASS] SessionManager gracefully handled Qwen failure without hang or data loss.")


if __name__ == "__main__":
    test_qwen_worker_fallback_unit()
    test_session_manager_with_offline_qwen()
    print("\nAll Qwen timeout & fallback tests passed successfully!")
