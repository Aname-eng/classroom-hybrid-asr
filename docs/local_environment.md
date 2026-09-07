# Local Environment Reconnaissance Report

## 1. Environment & Hardware Specifications
- **Operating System**: Windows 11 / 10 x64 (Build details: Windows x86_64)
- **CPU / RAM**: Intel Core Ultra 7 H-series, 32 GB RAM
- **GPU / Accelerator**: Intel Integrated Graphics (Intel Arc / Graphics), **No NVIDIA GPU / No CUDA** (Do NOT install or invoke CUDA)
- **Primary Workspaces**: `d:\课程笔记` (Classroom Notes Project Workspace)
- **Available Python Interpreters**:
  - Python 3.12 (`C:\Users\david\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe`)
  - Python 3.11 (`C:\Users\david\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\python.exe`)
  - Python 3.14 (`C:\Python314\python.exe`)
  - Package Manager: `uv` (`C:\Users\david\.local\bin\uv.exe`)

---

## 2. CapsWriter-Offline & Qwen3-ASR Assets
- **Original Download Packages**:
  - `C:\Users\david\Downloads\CapsWriter-Offline-20260531.zip` (99.34 MB)
  - `C:\Users\david\Downloads\Qwen3-ASR-1.7B-q4_k.zip` (1,345.24 MB)
- **Extracted Software Directory**: `D:\software\CapsWriter-Offline`
- **Extracted Model Directory**: `D:\software\CapsWriter-Offline\models\Qwen3-ASR\Qwen3-ASR-1.7B`
- **Model Files & Quantization**:
  1. `qwen3_asr_encoder_frontend.onnx` (19.91 MB) — ONNX Runtime (CPU / DirectML)
  2. `qwen3_asr_encoder_backend.onnx` (157.11 MB) — ONNX Runtime (CPU / DirectML)
  3. `qwen3_asr_llm.gguf` (1,223.02 MB) — Q4_K GGUF quantization (llama.cpp engine, Vulkan / CPU)
  - Partial SHA256 header (first 10MB): `dca7bc5fa16417e35904f590964a329895dfe7c668a97657ca1128f391460b5d`

---

## 3. CapsWriter-Offline Runtime & Architecture
- **Runtime Type**: Hybrid ONNX Runtime (Encoder) + llama.cpp C-API / ctypes (Decoder)
- **Hardware Acceleration**:
  - Encoder: ONNX Runtime CPU / DirectML
  - LLM Decoder: `llama.dll` with Vulkan / CPU fallback
- **Server Communication Protocol**:
  - Server Bind: `ws://127.0.0.1:6016` (or `0.0.0.0:6016`)
  - Input Protocol: JSON + Base64 float32 16kHz mono audio (`AudioMessage` in `core.protocol`)
  - Output Protocol: JSON recognition stream / final result (`RecognitionMessage` in `core.protocol`)
- **Direct Python Engine Interface**:
  - `QwenASREngine` in `core.server.engines.qwen_asr_gguf.asr_engine`
  - Internal Engine: `QwenInternalEngine` in `core.server.engines.qwen_asr_gguf.inference.asr`
  - Accepts raw numpy audio arrays: `engine.asr(audio, context, language)`

---

## 4. First-Pass Streaming ASR & VAD Strategy
- **Streaming Model**: FunASR `paraformer-zh-streaming` (or `sherpa-onnx` streaming Paraformer) running locally on CPU.
- **Chunk Size**: 480ms chunk (`[8, 8, 4]`).
- **VAD**: FunASR `fsmn-vad` / `sherpa-onnx` FSMN VAD with speech start/end boundary detection.
- **Second-Pass Offline Correction**: `Qwen3-ASR-1.7B-q4_k` from CapsWriter-Offline called on completed segments via independent background worker process.

---

## 5. Integration Priority
1. **Primary Choice**: WebSocket Client adapter or direct subprocess to CapsWriter Server (`ws://127.0.0.1:6016`) / Direct Module import with dedicated worker process.
2. **Persistence**: Audio recording (`audio.wav`), `events.jsonl`, `transcript_raw.jsonl`, `transcript_final.md`, `meta.json`.
3. **UI & Packaging**: PySide6 desktop GUI with course/glossary selector, real-time subtitle stream with online->final replacement, status indicators, and desktop shortcut launcher.
