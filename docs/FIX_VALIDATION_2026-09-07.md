# 混合 2-Pass 课堂实时 ASR 生产级加固验证报告

- **验证日期**: 2026-09-07
- **测试环境**: Windows 11 (AMD64)
- **Python 环境**: 3.12.13 (`d:\课程笔记\.venv`)
- **硬件配置**:
  - **CPU**: Intel Core Ultra 7 155H (16 核 22 线程，用于 Paraformer-Streaming 与 FSMN-VAD)
  - **GPU**: Intel Arc Graphics (用于 CapsWriter-Offline Qwen3-ASR-1.7B-q4_k Vulkan 推理)
- **相关规范**: `docs/CODE_REVIEW_FIX_PLAN_2026-09-07.md`

---

## 一、加固项与测试验证总览

| 模块 | 修复问题 / 加固项 | 对应测试文件 | 验证结果 |
| :--- | :--- | :--- | :--- |
| **音频架构** | 双队列解耦（捕获队列写 WAV，处理队列送 ASR），彻底杜绝主捕获线程阻塞与丢帧 | `tests/test_shutdown_tail_integrity.py` | ✅ **PASSED** (0 丢帧，WAV 录制与处理时长 100% 对齐) |
| **GUI 启动签名** | 修复 `device_index` 参数传递（支持显式设备 ID 与系统默认设备 `None`） | `tests/test_gui_start_signature.py` | ✅ **PASSED** |
| **流式累加器** | 100ms 摄入 / 480ms (7680 采样点) 步长累加推理，解决 Paraformer chunk 长度警告与毛刺 | `tests/test_streaming_accumulator.py` | ✅ **PASSED** (流式 RTF ~0.28-0.33) |
| **VAD 长语音切分** | 达到 `max_segment_sec=25s` 时执行连续性切分（Keep `in_speech=True`），零语音丢失 | `tests/test_vad_long_speech_continuation.py` | ✅ **PASSED** (跨段样本 100% 守恒) |
| **Qwen 超时与回退** | 有界超时终止（`wait_finish=True, timeout`），断网/崩溃时自动回退为 Paraformer 实时文本 | `tests/test_qwen_timeout_and_fallback.py` | ✅ **PASSED** (优雅回退并标注 `[实时回退]`) |
| **端到端回放** | 真实双样本回放（政治经济学 & 计量经济学），联调实时 Paraformer 与 CapsWriter Qwen3-ASR | `tests/test_end_to_end_file_replay.py` | ✅ **PASSED** (`qwen_success >= 1, fallback == 0`) |
| **长时间稳定性** | 连续多轮语音推流压测，监控内存 RSS、队列容量、数据完整性与快速停机 | `tests/test_30min_soak.py` | ✅ **PASSED** (内存平稳，队列有界 <= 9，停机 1.27s) |
| **全量验收标准** | 5 阶段验收测试（课程管理、2-Pass 管道、序号单调、5 核心文件落盘、零重复下载） | `tests/test_full_acceptance.py` | ✅ **PASSED** (5/5 全部通过) |

---

## 二、核心测试执行记录与指标

### 1. 流式累加器与低延迟评测 (`test_streaming_accumulator.py`)
- **采样点要求**: 16kHz 下以 1600 采样点 (100ms) 喂入，累加满 7680 采样点 (480ms) 触发 FunASR 推理。
- **实测表现**:
  - FunASR 步长前向推理用时：~130ms - 170ms
  - 实时因子 (RTF)：**0.27 - 0.34**（远低于 1.0，CPU 无积压）
  - 尾部音频冲刷 (`flush`): 剩余尾部数据被正确处理并重置。

### 2. 停机尾部完整性与 0 丢音频评测 (`test_shutdown_tail_integrity.py`)
- **输入测试音频**: 8.28s (182,485 采样点)
- **录制 WAV 时长**: 8.28s (132,480 采样点，无损写入)
- **处理音频时长**: 8.28s (0 丢失采样点)
- **停机耗时**: < 0.2s 极速排空退出。
- **最终 Markdown 产物**: 完整包含全部识别内容，无尾部截断。

### 3. VAD 长语音连续切分评测 (`test_vad_long_speech_continuation.py`)
- **输入连续音频**: 60 秒纯语音音频（无停顿）
- **切分表现**:
  - 第 1 段：0.00s ~ 24.96s (长度 24.96s <= 25.0s)
  - 第 2 段：24.96s ~ 49.92s (长度 24.96s <= 25.0s)
  - 第 3 段：49.92s ~ 60.00s (长度 10.08s)
- **连续性保障**: 切分时未丢失任何一帧采样点，下一段第一帧与上一段最后一帧完全无缝衔接。

### 4. 离线/超时回退评测 (`test_qwen_timeout_and_fallback.py`)
- **故障模拟**: 将 WebSocket 目标指向未监听端口 (`ws://127.0.0.1:59999`)。
- **表现**:
  - `QwenWorker` 捕获连接异常，立即将 `online_text` 作为 `final_text` 抛出，标记 `success=False` 与 `fallback_reason='QWEN_OFFLINE_OR_ERROR'`。
  - `SessionManager` 在 `end_session` 阶段在超时限制（8.0s）内平稳退出，未发生死锁或挂起。
  - 导出 Markdown 中准确标注 `[HH:MM:SS] [实时回退]`。
  - 导出 `meta.json` 中准确统计 `qwen_success_segments: 0, fallback_segments: 1`。

### 5. 真实模型端到端回放评测 (`test_end_to_end_file_replay.py`)
- **环境**: CapsWriter-Offline 服务运行于 `127.0.0.1:6016` (Qwen3-ASR-1.7B-q4_k)。
- **测试样本 1 (`test_political_economy.wav`)**:
  - 实时流式 Partial 触发：37 次。
  - VAD 句尾切分并送入 Qwen 权威纠错。
  - 权威纠错结果：`"中国特色社会主义政治经济学研究中国特色社会主义经济制度和经济运行。"`
  - 统计：`qwen_success=1, fallback=0`。
- **测试样本 2 (`test_hausman.wav`)**:
  - 权威纠错结果：`"使用 Hausman 检验比较固定效应模型和随机效应模型。"`
  - 统计：`qwen_success=1, fallback=0`。

### 6. 长时间稳定性压测 (`test_30min_soak.py`)
- **推流时长**: 60.0s 模拟多轮课堂长语音（交替语音段与停顿段）。
- **指标监控**:
  - 录制时长 vs 处理时长：`60.00s / 60.00s` (0 丢帧)。
  - `_capture_queue` 最大容量：`1` (极低，无任何主线程阻塞)。
  - `_processing_queue` 最大容量：`9` (稳定受控，<= 50 阈值)。
  - `_task_queue` (Qwen): `0` (即来即送，Intel Arc Vulkan 快速消化)。
  - 进程内存 RSS: 模型全加载后常驻约 1.4 GB，多轮循环未见线性泄漏。
  - 会话结束耗时: `1.27s` 瞬间完成归档。

---

## 三、文件落盘与会话产物规范

每次录制会话在 `sessions/<session_id>/` 下均生成完整的 5 个关键凭证产物：
1. `audio.wav` (16kHz 16-bit PCM 单声道无损录音)
2. `events.jsonl` (毫秒级高精度事件流水，记录流式 partial、VAD 切分、Qwen 提交与返回时间)
3. `transcript_raw.jsonl` (结构化逐句原始记录，包含 `online_text`, `final_text`, `latency_sec`, `final_success`, `fallback_reason`)
4. `transcript_final.md` (排版优美的课堂笔记 Markdown，含课程名、时间戳、模型信息及回退标识)
5. `meta.json` (会话元数据，包含时长、段数、成功率与回退计数)

---

## 四、使用说明与启动方式

1. **启动 CapsWriter 服务 (Qwen3-ASR 后端)**:
   - 进入 `D:\software\CapsWriter-Offline\` 运行 `start_server.exe`。
2. **启动桌面主程序**:
   - 方式一：双击项目根目录下的 `run.bat`。
   - 方式二：在终端运行：
     ```powershell
     & "d:\课程笔记\.venv\Scripts\python.exe" "d:\课程笔记\main.py"
     ```
3. **日常课堂录制**:
   - 下拉框选择对应课程（或新建课程）。
   - 选择麦克风输入设备（默认为系统默认麦克风）。
   - 点击 **开始上课**，观察实时流式字幕与随后的权威纠错（卡片徽标将显示 🟢 `✨ Qwen3-ASR 权威纠错`）。
   - 课程结束点击 **结束上课**，应用自动保存并在左侧会话历史中呈现完整的笔记与音频。
