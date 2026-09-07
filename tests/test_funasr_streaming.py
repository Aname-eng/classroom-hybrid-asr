# coding: utf-8
import os
import sys
import time
import soundfile as sf
import numpy as np
import scipy.signal
from funasr import AutoModel

def ensure_16k_mono(audio_path: str) -> np.ndarray:
    data, sr = sf.read(audio_path, dtype='float32')
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != 16000:
        target_len = int(len(data) * 16000 / sr)
        data = scipy.signal.resample(data, target_len).astype(np.float32)
    return data

def test_streaming():
    print("Initializing FunASR streaming Paraformer and FSMN VAD...")
    t0 = time.time()
    # Disable proxy env for smooth ModelScope download if needed
    os.environ['NO_PROXY'] = '*'
    
    # Initialize AutoModel with streaming Paraformer and VAD
    model = AutoModel(
        model="paraformer-zh-streaming",
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device="cpu",
        disable_update=True
    )
    print(f"Model initialized in {time.time() - t0:.2f}s")
    
    wav_path = r"d:\课程笔记\tests\audio_samples\test_political_economy.wav"
    speech = ensure_16k_mono(wav_path)
    audio_dur = len(speech) / 16000.0
    print(f"Audio loaded: {audio_dur:.2f}s ({len(speech)} samples)")
    
    # 480ms chunk size: chunk_size=[8, 8, 4] (center chunk is 8 * 60ms = 480ms = 7680 samples)
    chunk_size = [8, 8, 4]
    chunk_stride = chunk_size[1] * 960 # 8 * 960 = 7680 samples (480ms)
    
    total_chunk_num = int(np.ceil(len(speech) / chunk_stride))
    print(f"Total streaming chunks: {total_chunk_num} (stride={chunk_stride} samples = 480ms)")
    
    cache = {}
    print("\n--- Starting Real-time Streaming Simulation ---")
    t_start = time.time()
    partial_texts = []
    
    for i in range(total_chunk_num):
        s = i * chunk_stride
        e = min((i + 1) * chunk_stride, len(speech))
        speech_chunk = speech[s:e]
        is_final = (i == total_chunk_num - 1)
        
        t_chunk_start = time.time()
        res = model.generate(
            input=speech_chunk,
            cache=cache,
            is_final=is_final,
            chunk_size=chunk_size,
            encoder_chunk_look_back=4,
            decoder_chunk_look_back=1
        )
        chunk_latency = time.time() - t_chunk_start
        
        if res and len(res) > 0 and res[0].get("text"):
            text = res[0]["text"]
            partial_texts.append(text)
            timestamp = (i * chunk_stride) / 16000.0
            print(f"  [{timestamp:05.2f}s] Streaming output: '{text}' (latency={chunk_latency*1000:.1f}ms)")
            
    total_time = time.time() - t_start
    print("\n" + "="*50)
    print("FunASR Streaming Test Finished:")
    print(f"  Final Accumulated Text : {''.join(partial_texts)}")
    print(f"  Total Audio Duration   : {audio_dur:.2f}s")
    print(f"  Total Streaming Time   : {total_time:.2f}s")
    print(f"  Streaming RTF          : {total_time / audio_dur:.3f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_streaming()
