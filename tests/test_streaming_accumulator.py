# coding: utf-8
import os
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import scipy.signal

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.streaming.paraformer_streamer import ParaformerStreamer


def test_streaming_accumulator_buffer_logic():
    print("=== Testing ParaformerStreamer 480ms Accumulator Logic ===")
    
    emitted_partials = []
    
    def on_partial(segment_id: int, text: str, ts: float):
        emitted_partials.append((segment_id, text, ts))

    streamer = ParaformerStreamer(chunk_size=[8, 8, 4], on_partial_text=on_partial)
    
    # 1. Feed 4 chunks of 100ms (1600 samples each = 6400 samples total)
    chunk_100ms = np.zeros(1600, dtype=np.float32)
    
    for i in range(4):
        streamer.process_chunk(chunk_100ms, segment_id=1, timestamp_sec=0.1 * (i + 1))
        # Buffer should have (i+1)*1600 samples, none consumed yet because 6400 < 7680
        assert len(streamer._audio_buffer) == 1600 * (i + 1), f"Expected {1600*(i+1)} samples, got {len(streamer._audio_buffer)}"

    print("  [PASS] 100ms chunks (1-4) properly accumulated without premature inference.")

    # 2. Feed 5th chunk (1600 samples -> total 8000 samples)
    # 7680 consumed for 1 stride (480ms), 8000 - 7680 = 320 samples remain
    streamer.process_chunk(chunk_100ms, segment_id=1, timestamp_sec=0.5)
    assert len(streamer._audio_buffer) == 320, f"Expected 320 samples remaining in accumulator, got {len(streamer._audio_buffer)}"
    print("  [PASS] 5th chunk triggered stride inference and left correct remainder (320 samples).")

    # 3. Test flush() on segment boundary / session shutdown
    streamer.flush(timestamp_sec=0.55)
    assert len(streamer._audio_buffer) == 0, f"Expected accumulator to be empty after flush, got {len(streamer._audio_buffer)}"
    print("  [PASS] flush() successfully drained remainder samples.")


def test_streaming_with_real_audio():
    print("\n=== Testing ParaformerStreamer with Real Speech Audio in 100ms Chunks ===")
    
    wav_path = r"d:\课程笔记\tests\audio_samples\test_political_economy.wav"
    audio, sr = sf.read(wav_path, dtype='float32')
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        target_len = int(len(audio) * 16000 / sr)
        audio = scipy.signal.resample(audio, target_len).astype(np.float32)

    partials = []
    def on_partial(segment_id: int, text: str, ts: float):
        if text.strip():
            partials.append((segment_id, text, ts))

    streamer = ParaformerStreamer(chunk_size=[8, 8, 4], on_partial_text=on_partial)
    
    # Slice audio into 100ms (1600 samples) chunks
    chunk_len = 1600
    total_chunks = int(np.ceil(len(audio) / chunk_len))
    for i in range(total_chunks):
        s = i * chunk_len
        e = min((i + 1) * chunk_len, len(audio))
        chunk = audio[s:e]
        ts = e / 16000.0
        streamer.process_chunk(chunk, segment_id=1, timestamp_sec=ts)

    # Flush tail
    streamer.flush(timestamp_sec=len(audio) / 16000.0)
    
    print(f"  Total partial updates received: {len(partials)}")
    assert len(partials) > 0, "Expected at least 1 partial text update"
    final_partial = partials[-1][1]
    print(f"  Final streamed text: '{final_partial}'")
    assert "政治经济学" in final_partial or "经济" in final_partial or len(final_partial) >= 5, f"Unexpected stream output: {final_partial}"
    print("  [PASS] Real speech audio streamed successfully in 100ms chunks with accumulator.")


if __name__ == "__main__":
    test_streaming_accumulator_buffer_logic()
    test_streaming_with_real_audio()
    print("\nAll ParaformerStreamer accumulator tests passed successfully!")
