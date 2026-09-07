import os
os.environ['NO_PROXY'] = '*'
import numpy as np
import soundfile as sf
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

wav1 = r"d:\课程笔记\tests\audio_samples\test_political_economy.wav"
wav2 = r"d:\课程笔记\tests\audio_samples\test_hausman.wav"

audio1 = ensure_16k_mono(wav1)
audio2 = ensure_16k_mono(wav2)
silence = np.zeros(int(16000 * 1.5), dtype=np.float32)

full_audio = np.concatenate([audio1, silence, audio2, silence])

vad_model = AutoModel(model="fsmn-vad", device="cpu", disable_update=True)
cache = {}

# Slicing in 480ms chunks (7680 samples)
chunk_size_480 = 7680
sub_chunk_size = 960 # 60ms

in_speech = False
speech_start_time = 0.0
speech_buffer = []
segment_count = 0

for i in range(0, len(full_audio), chunk_size_480):
    chunk = full_audio[i:i+chunk_size_480]
    ts_chunk = i / 16000.0
    
    # Process in 60ms sub-chunks
    for j in range(0, len(chunk), sub_chunk_size):
        sub_chunk = chunk[j:j+sub_chunk_size]
        if len(sub_chunk) < sub_chunk_size:
            sub_chunk = np.pad(sub_chunk, (0, sub_chunk_size - len(sub_chunk)))
        
        ts_sub = (i + j) / 16000.0
        res = vad_model.generate(input=sub_chunk, cache=cache, is_final=False, chunk_size=60)
        segments = res[0].get("value", []) if (res and len(res) > 0) else []
        
        for seg in segments:
            s_ms, e_ms = seg[0], seg[1]
            if s_ms >= 0 and e_ms < 0:
                # Speech start
                in_speech = True
                speech_start_time = s_ms / 1000.0
                speech_buffer = [sub_chunk]
                print(f"--> Speech Start detected at {speech_start_time:.2f}s")
            elif e_ms >= 0 and s_ms < 0:
                # Speech end
                in_speech = False
                speech_end_time = e_ms / 1000.0
                speech_buffer.append(sub_chunk)
                full_seg = np.concatenate(speech_buffer)
                segment_count += 1
                print(f"<-- Speech End detected at {speech_end_time:.2f}s (Seg {segment_count}, dur {len(full_seg)/16000:.2f}s)")
                speech_buffer = []
            elif s_ms >= 0 and e_ms >= 0:
                # Complete segment in one step
                in_speech = False
                segment_count += 1
                speech_buffer.append(sub_chunk)
                full_seg = np.concatenate(speech_buffer)
                print(f"<-> Speech Segment [{s_ms/1000:.2f}s - {e_ms/1000:.2f}s] (Seg {segment_count}, dur {len(full_seg)/16000:.2f}s)")
                speech_buffer = []
        
        if in_speech and not segments:
            speech_buffer.append(sub_chunk)

print("VAD test completed. Total segments detected:", segment_count)
