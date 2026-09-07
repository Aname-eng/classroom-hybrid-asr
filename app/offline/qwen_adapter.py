# coding: utf-8
import os
import sys
import time
import json
import base64
import asyncio
import subprocess
import threading
from typing import Optional, Tuple
import numpy as np
import websockets
from app.config import WS_SERVER_URI, CAPSWRITER_SERVER_EXE, CAPSWRITER_DIR

class QwenAdapter:
    """
    Qwen3-ASR-1.7B-q4_k 适配器
    
    负责与本地 CapsWriter-Offline 服务端交互，执行第二遍高精度离线转写。
    支持自动拉起服务进程、连接自检、超时与熔断保护。
    """
    def __init__(self, ws_uri: str = WS_SERVER_URI, server_exe: str = str(CAPSWRITER_SERVER_EXE), server_cwd: str = str(CAPSWRITER_DIR)):
        self.ws_uri = ws_uri
        self.server_exe = server_exe
        self.server_cwd = server_cwd
        self.server_process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    async def _get_ws(self, timeout=3.0):
        kwargs = {
            "uri": self.ws_uri,
            "subprotocols": ["binary"],
            "max_size": None,
            "open_timeout": timeout
        }
        if tuple(int(v) for v in websockets.__version__.split(".")[:2]) >= (14, 0):
            kwargs["proxy"] = None
        return await websockets.connect(**kwargs)

    async def check_server_ready(self, max_retries: int = 3, delay: float = 0.5) -> bool:
        for _ in range(max_retries):
            try:
                ws = await self._get_ws(timeout=1.0)
                await ws.close()
                return True
            except Exception:
                await asyncio.sleep(delay)
        return False

    async def ensure_server_running_async(self, wait_timeout: float = 30.0) -> bool:
        if await self.check_server_ready(max_retries=2, delay=0.3):
            return True
        
        with self._lock:
            if not self.server_process:
                print(f"[QwenAdapter] CapsWriter server not running. Starting from {self.server_exe}...")
                self.server_process = subprocess.Popen(
                    [self.server_exe],
                    cwd=self.server_cwd,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            
        start_time = time.time()
        while time.time() - start_time < wait_timeout:
            if await self.check_server_ready(max_retries=1, delay=0.5):
                print(f"[QwenAdapter] CapsWriter server is ready (PID: {self.server_process.pid if self.server_process else 'running'})")
                return True
            await asyncio.sleep(0.5)
        
        print(f"[QwenAdapter] Server failed to start within {wait_timeout}s")
        return False

    def ensure_server_running(self, wait_timeout: float = 30.0) -> bool:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(self._ensure_server_running_sync_isolated, wait_timeout)
                return fut.result()
        else:
            return self._ensure_server_running_sync_isolated(wait_timeout)

    def _ensure_server_running_sync_isolated(self, wait_timeout: float = 30.0) -> bool:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.ensure_server_running_async(wait_timeout))
        finally:
            loop.close()

    async def transcribe_async(
        self,
        audio: np.ndarray,
        task_id: str,
        context: str = "",
        language: str = "zh",
        timeout: float = 20.0
    ) -> Tuple[bool, str, float]:
        """
        异步转录音频片段 (float32, 16kHz, mono)
        Returns: (success: bool, text: str, latency: float)
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
            
        b64_data = base64.b64encode(audio.tobytes()).decode('utf-8')
        msg = {
            "task_id": task_id,
            "source": "file",
            "data": b64_data,
            "is_final": True,
            "time_start": time.time(),
            "seg_duration": float(len(audio) / 16000.0) + 5.0,
            "seg_overlap": 0.0,
            "context": context,
            "language": language
        }

        t0 = time.time()
        try:
            ws = None
            try:
                ws = await asyncio.wait_for(self._get_ws(timeout=2.0), timeout=2.0)
            except Exception:
                if os.path.exists(self.server_exe):
                    print(f"[QwenAdapter] CapsWriter server not connected. Auto-starting from {self.server_exe}...")
                    await self.ensure_server_running_async(wait_timeout=15.0)
                    ws = await asyncio.wait_for(self._get_ws(timeout=5.0), timeout=5.0)
                else:
                    raise

            async with ws:
                await ws.send(json.dumps(msg, ensure_ascii=False))
                
                final_text = ""
                while True:
                    resp_raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    resp = json.loads(resp_raw)
                    if resp.get("is_final"):
                        final_text = resp.get("text", "")
                        break
                        
                latency = time.time() - t0
                return True, final_text, latency
                
        except Exception as e:
            latency = time.time() - t0
            print(f"[QwenAdapter Error] Failed to transcribe task {task_id}: {e}")
            return False, "", latency

    def transcribe(
        self,
        audio: np.ndarray,
        task_id: str,
        context: str = "",
        language: str = "zh",
        timeout: float = 20.0
    ) -> Tuple[bool, str, float]:
        """同步阻塞调用"""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.transcribe_async(audio, task_id, context, language, timeout)
            )
        finally:
            loop.close()

    def shutdown(self):
        if self.server_process:
            try:
                self.server_process.terminate()
            except Exception:
                pass
            self.server_process = None
