# coding: utf-8
import os
import sys
import time
import json
import base64
import asyncio
import subprocess
import threading
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import websockets
from app.config import WS_SERVER_URI, CAPSWRITER_SERVER_EXE, CAPSWRITER_DIR
from app.offline.local_qwen_asr import LocalQwenASR

class QwenAdapter:
    """
    Qwen3-ASR 本地/ CapsWriter 适配器

    优先支持已下载的本地 Qwen3-ASR（安装 qwen-asr 后启用）；没有本地后端时
    继续使用 CapsWriter WebSocket，并提供超时回退保护。
    """
    def __init__(
        self,
        ws_uri: str = WS_SERVER_URI,
        server_exe: str = str(CAPSWRITER_SERVER_EXE) if CAPSWRITER_SERVER_EXE else "",
        server_cwd: str = str(CAPSWRITER_DIR) if CAPSWRITER_DIR else "",
        model_path: str = "",
        backend: str = "auto",
    ):
        self.ws_uri = ws_uri
        self._auto_start_uri = ws_uri
        self.server_exe = server_exe or ""
        self.server_cwd = server_cwd or ""
        self.model_path = model_path or ""
        self.backend = (backend or "auto").strip().lower()
        self.local_asr = LocalQwenASR(self.model_path) if self.model_path else None
        self.server_process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    @property
    def model_label(self) -> str:
        return Path(self.model_path).name if self.model_path else "Qwen3-ASR"

    def configure_endpoint(self, ws_uri: str) -> None:
        self.ws_uri = ws_uri or WS_SERVER_URI
        self._auto_start_uri = self.ws_uri

    def configure_model(self, model_path: str = "", backend: str = "auto") -> None:
        self.model_path = model_path or ""
        self.backend = (backend or "auto").strip().lower()
        self.local_asr = LocalQwenASR(self.model_path) if self.model_path else None

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
        if self.backend == "local_qwen":
            return True
        if await self.check_server_ready(max_retries=2, delay=0.3):
            return True
        
        executable = Path(self.server_exe).expanduser() if self.server_exe else None
        if executable is None or not executable.exists():
            print(
                "[QwenAdapter] No local Qwen server executable configured; "
                "offline correction will fall back to the streaming transcript."
            )
            return False

        with self._lock:
            if not self.server_process:
                print(f"[QwenAdapter] CapsWriter server not running. Starting from {executable}...")
                popen_kwargs = {
                    "cwd": self.server_cwd if self.server_cwd and Path(self.server_cwd).exists() else None,
                }
                # CREATE_NO_WINDOW 只存在于 Windows；Linux/macOS 不应访问该常量。
                no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                if no_window:
                    popen_kwargs["creationflags"] = no_window
                self.server_process = subprocess.Popen([str(executable)], **popen_kwargs)
            
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

        t0 = time.time()
        if self.backend == "local_qwen" and (
            self.local_asr is None or not self.local_asr.available
        ):
            print("[QwenAdapter] local_qwen selected but no local model directory is available")
            return False, "", time.time() - t0

        use_local = self.backend == "local_qwen" or (
            self.backend == "auto"
            and self.local_asr is not None
            and self.local_asr.available
            and not self.server_exe
        )
        if use_local and self.local_asr is not None:
            success, text = await asyncio.to_thread(
                self.local_asr.transcribe,
                audio,
                context,
                language,
            )
            latency = time.time() - t0
            if success:
                return True, text, latency
            if self.backend == "local_qwen":
                print(f"[QwenAdapter] Local Qwen ASR failed: {self.local_asr.last_error}")
                return False, "", latency

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

        try:
            ws = None
            try:
                ws = await asyncio.wait_for(self._get_ws(timeout=2.0), timeout=2.0)
            except Exception:
                if (
                    self.server_exe
                    and os.path.exists(self.server_exe)
                    and self.ws_uri == self._auto_start_uri
                ):
                    print(f"[QwenAdapter] CapsWriter server not connected. Auto-starting from {self.server_exe}...")
                    # 服务启动也必须服从单句任务的有界超时，不能因为本地服务
                    # 未启动而把 worker 的停机拖到几十秒。
                    ready = await self.ensure_server_running_async(
                        wait_timeout=min(3.0, max(1.0, float(timeout)))
                    )
                    if not ready:
                        raise ConnectionError("Qwen local server is not ready")
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
