# coding: utf-8
import os
import sys
import json
import time
import asyncio
import threading
from pathlib import Path
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SESSIONS_DIR
from app.audio.source import FileReplayAudioSource
from app.pipeline.session_manager import SessionManager


class FakeHangingWsServer:
    """Fake WebSocket server that accepts connections and receives audio, but NEVER replies."""
    def __init__(self, host="127.0.0.1", port=59123):
        self.host = host
        self.port = port
        self.server = None
        self.loop = None
        self.thread = None
        self.requests_received = 0

    async def _handler(self, websocket):
        try:
            async for message in websocket:
                self.requests_received += 1
                print(f"  [FakeWsServer] Received audio payload ({len(message)} bytes). Intentionally hanging without reply...")
                # Hang forever without closing or sending responses
                while True:
                    await asyncio.sleep(1.0)
        except Exception:
            pass

    def _run_server(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        async def main():
            self.server = await websockets.serve(self._handler, self.host, self.port, subprotocols=["binary"])
            await asyncio.Future() # run forever

        try:
            self.loop.run_until_complete(main())
        except Exception:
            pass

    def start(self):
        self.thread = threading.Thread(target=self._run_server, daemon=True, name="FakeWsServerThread")
        self.thread.start()
        time.sleep(0.5)

    def stop(self):
        if self.loop and self.server:
            self.loop.call_soon_threadsafe(self.server.close)


def test_qwen_inflight_shutdown_timeout():
    print("=== Test: Qwen In-Flight Task Shutdown Timeout & Late-Response Suppression ===")
    
    fake_server = FakeHangingWsServer(port=59123)
    fake_server.start()
    print("  Fake hanging WebSocket server started on ws://127.0.0.1:59123")

    try:
        wav_path = r"d:\课程笔记\tests\audio_samples\test_political_economy.wav"
        source = FileReplayAudioSource(wav_path=wav_path, realtime_factor=0.2, padding_silence_seconds=1.5)
        
        manager = SessionManager()
        manager.qwen_adapter.ws_uri = "ws://127.0.0.1:59123"

        session_id = manager.start_session("political_economy", source=source)
        print(f"  Session started: {session_id}")

        # Wait until the in-flight request reaches the fake server
        for _ in range(50):
            if fake_server.requests_received >= 1:
                break
            time.sleep(0.1)

        assert fake_server.requests_received >= 1, "Fake server did not receive request!"
        print(f"  Request in-flight (server received {fake_server.requests_received} payloads). Now triggering end_session(timeout=2.0s)...")

        t0 = time.time()
        manager.end_session(timeout=2.0)
        elapsed = time.time() - t0
        print(f"  end_session() returned in {elapsed:.2f}s (bounded timeout)")

        assert elapsed < 7.0, f"end_session() exceeded bounded timeout: {elapsed:.2f}s"

        session_dir = SESSIONS_DIR / session_id
        meta_file = session_dir / "meta.json"
        md_file = session_dir / "transcript_final.md"
        raw_file = session_dir / "transcript_raw.jsonl"

        with open(meta_file, "r", encoding="utf-8") as f:
            meta1 = json.load(f)

        with open(md_file, "r", encoding="utf-8") as f:
            md1 = f.read()

        with open(raw_file, "r", encoding="utf-8") as f:
            raw_lines = [json.loads(line) for line in f if line.strip()]

        print(f"  Meta Summary: total={meta1['total_segments']}, qwen_success={meta1['qwen_success_segments']}, fallback={meta1['fallback_segments']}, timeouts={meta1.get('qwen_timeout_segments')}")
        assert meta1["fallback_segments"] >= 1, "Expected at least 1 fallback segment"
        assert meta1["qwen_success_segments"] == 0, "Expected 0 qwen success segments"
        assert meta1.get("qwen_timeout_segments", 0) >= 1, "Expected qwen_timeout_segments >= 1"

        # Check raw entries
        timeout_found = any(entry.get("fallback_reason") == "QWEN_SHUTDOWN_TIMEOUT" for entry in raw_lines)
        assert timeout_found, "Expected fallback_reason == 'QWEN_SHUTDOWN_TIMEOUT' in raw transcript"

        # Check that markdown contains fallback text
        assert "[实时回退]" in md1, "Expected [实时回退] tag in markdown"

        # Now wait for longer than normal adapter timeout to verify late responses NEVER overwrite files
        print("  Waiting 4.0s to verify no late background callbacks alter saved session files...")
        time.sleep(4.0)

        with open(meta_file, "r", encoding="utf-8") as f:
            meta2 = json.load(f)

        with open(md_file, "r", encoding="utf-8") as f:
            md2 = f.read()

        assert meta1 == meta2, "meta.json was modified after session end!"
        assert md1 == md2, "transcript_final.md was modified after session end!"

        print("  [PASS] In-flight shutdown timeout is strictly bounded and late responses are completely suppressed!")

    finally:
        fake_server.stop()


if __name__ == "__main__":
    test_qwen_inflight_shutdown_timeout()
