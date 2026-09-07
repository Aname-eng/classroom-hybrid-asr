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

def main():
    os.environ['NO_PROXY'] = '*'
    print("Loading FunASR streaming Paraformer model from local cache...")
    t0 = time.time()
    model = AutoModel(model="paraformer-zh-streaming", device="cpu", disable_update=True)
    print(f"Model loaded in {time.time() - t0:.2f}s")
    
    wav_path = r"d:\课程笔记\tests\audio_samples\test_political_economy.wav"
    speech = ensure_16k_mono(wav_path)
    audio_dur = len(speech) / 16000.0
    print(f"Loaded audio: {audio_dur:.2f}s ({len(speech)} samples)")
    
    chunk_size = [8, 8, 4] # 480ms chunk
    chunk_stride = chunk_size[1] * 960 # 7680 samples
    total_chunks = int(np.ceil(len(speech) / chunk_stride))
    
    cache = {}
    full_text = ""
    print("\n--- Real-Time Streaming Simulation Output ---")
    t_start = time.time()
    
    for i in range(total_chunks):
        s = i * chunk_stride
        e = min((i + 1) * chunk_stride, len(speech))
        chunk = speech[s:e]
        is_final = (i == total_chunks - 1)
        
        t_chunk_start = time.time()
        res = model.generate(
            input=chunk,
            cache=cache,
            is_final=is_final,
            chunk_size=chunk_size,
            encoder_chunk_look_back=4,
            decoder_chunk_look_back=1
        )
        chunk_cost = (time.time() - t_chunk_start) * 1000
        
        if res and len(res) > 0 and res[0].get("text"):
            piece = res[0]["text"]
            full_text += piece
            timestamp = (i * chunk_stride) / 16000.0
            print(f"  [{timestamp:05.2f}s] +'{piece}' -> partial: '{full_text}' ({chunk_cost:.1f}ms)")
            
    total_time = time.time() - t_start
    print("\n" + "="*50)
    print("Paraformer Streaming Results:")
    print(f"  Streaming Partial Text : {full_text}")
    print(f"  Audio Duration         : {audio_dur:.2f}s")
    print(f"  Processing Time        : {total_time:.2f}s")
    print(f"  RTF                    : {total_time / audio_dur:.3f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
