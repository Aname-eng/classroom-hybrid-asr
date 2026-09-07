# coding: utf-8
import os
import sys
import time
import json
import base64
import asyncio
import subprocess
import soundfile as sf
import numpy as np
import scipy.signal
import websockets

CAPS_SERVER_EXE = r"D:\software\CapsWriter-Offline\start_server.exe"
CAPS_DIR = r"D:\software\CapsWriter-Offline"
WS_URI = "ws://127.0.0.1:6016"

def ensure_16k_mono(audio_path: str) -> np.ndarray:
    data, sr = sf.read(audio_path, dtype='float32')
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != 16000:
        target_len = int(len(data) * 16000 / sr)
        data = scipy.signal.resample(data, target_len).astype(np.float32)
    return data

async def get_ws_connection():
    kwargs = {
        "uri": WS_URI,
        "subprotocols": ["binary"],
        "max_size": None,
        "open_timeout": 3
    }
    if tuple(int(v) for v in websockets.__version__.split(".")[:2]) >= (14, 0):
        kwargs["proxy"] = None
    return await websockets.connect(**kwargs)

async def wait_for_server(max_retries=30, delay=1.0):
    for i in range(max_retries):
        try:
            ws = await get_ws_connection()
            await ws.close()
            print(f"[OK] CapsWriter server is active and verified on {WS_URI}!")
            return True
        except Exception:
            if i % 2 == 0:
                print(f"Waiting for CapsWriter server on {WS_URI}... ({i+1}/{max_retries})")
            await asyncio.sleep(delay)
    return False

async def test_qwen_transcribe(wav_path: str):
    print(f"--- [Gate 1 Test] Loading audio: {wav_path} ---")
    audio = ensure_16k_mono(wav_path)
    audio_dur = len(audio) / 16000.0
    print(f"Audio loaded: length={len(audio)} samples ({audio_dur:.2f}s), sr=16000")
    
    server_proc = None
    ready = await wait_for_server(max_retries=2, delay=0.5)
    if not ready:
        print(f"Starting CapsWriter server from {CAPS_SERVER_EXE}...")
        # Kill any zombie process if necessary
        subprocess.run(["taskkill", "/F", "/IM", "start_server.exe", "/T"], capture_output=True)
        time.sleep(1)
        
        server_proc = subprocess.Popen(
            [CAPS_SERVER_EXE],
            cwd=CAPS_DIR,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        print(f"Server started (PID={server_proc.pid}), waiting for ready signal...")
        ready = await wait_for_server(max_retries=40, delay=1.0)
        if not ready:
            raise RuntimeError("CapsWriter server failed to become ready within 40s!")
            
    print(f"Connecting to {WS_URI} with subprotocol 'binary' ...")
    ws = await get_ws_connection()
    async with ws:
        b64_data = base64.b64encode(audio.tobytes()).decode('utf-8')
        task_id = f"gate1_test_{int(time.time()*1000)}"
        msg = {
            "task_id": task_id,
            "source": "file",
            "data": b64_data,
            "is_final": True,
            "time_start": time.time(),
            "seg_duration": 60.0,
            "seg_overlap": 0.0,
            "context": "",
            "language": "zh"
        }
        
        t0 = time.time()
        print(f"Sending audio task (task_id={task_id})...")
        await ws.send(json.dumps(msg, ensure_ascii=False))
        
        final_text = ""
        while True:
            resp_raw = await ws.recv()
            resp = json.loads(resp_raw)
            print(f"  [Server stream] is_final={resp.get('is_final')} text='{resp.get('text')}'")
            if resp.get("is_final"):
                final_text = resp.get("text", "")
                break
                
        elapsed = time.time() - t0
        print("\n" + "="*55)
        print(f"Gate 1 Qwen3-ASR Recognition Result:")
        print(f"  Expected Text   : 中国特色社会主义政治经济学研究中国特色社会主义经济制度和经济运行规律。")
        print(f"  Recognized Text : {final_text}")
        print(f"  Audio Duration  : {audio_dur:.2f}s")
        print(f"  Inference Time  : {elapsed:.2f}s")
        print(f"  Real-time Factor: {elapsed / audio_dur:.3f}")
        print("="*55 + "\n")
        return final_text

if __name__ == "__main__":
    wav_file = r"d:\课程笔记\tests\audio_samples\test_political_economy.wav"
    if not os.path.exists(wav_file):
        print(f"Error: {wav_file} does not exist!")
        sys.exit(1)
    
    result = asyncio.run(test_qwen_transcribe(wav_file))
    assert len(result.strip()) > 0, "Recognition result is empty!"
    print(f"Gate 1 PASSED WITH SUCCESS!")
