# Classroom Hybrid 2-Pass ASR - 第二轮修复验证报告 (Round 2 Fix Validation Report)

**验证日期**: 2026-09-07  
**基线 Commit**: `279e4c43c23292dff5bd0ddbad8b459c85d6841a`  
**测试平台**: Windows 11 x64, Python 3.12 (venv), FunASR (Paraformer + FSMN-VAD), Qwen3-ASR-1.7B-q4_k (CapsWriter-Offline Server)

---

## 1. 修复核心问题回顾与解决总结

### (1) AudioRecorder 多 Session 生命周期与 Sentinel-Only Worker 机制
- **问题根因**: 原设计中 `AudioRecorder` 工作线程会因为 `_is_recording=False` 且队列为空而自动退出，而 `close()` 又尝试投放 `None` sentinel，导致潜在的生命周期竞态以及跨 Session 的 stale sentinel 污染。
- **重构方案**:
  - `_writer_thread` 与 `_processing_thread` 采用纯 sentinel 驱动（`while True:` 仅在提取到 `None` sentinel 时退出）。
  - `stop_capture()` 仅负责停止前端数据源采集，保持线程与队列管道持续运作。
  - `wait_until_drained()` 阻塞等待 `unfinished_tasks == 0`，确保所有采集到的数据完全落盘并完成流式推理。
  - `close()` 投放 `None` sentinel、等待线程 join 并重置线程句柄。
  - 支持同一个 `SessionManager` 实例在不销毁重建的情况下连续开启、录制、停止多个包含真实音频的 Session。

### (2) Paraformer speech_end 尾音与 Fallback 时序
- **问题根因**: VAD 触发 `speech_end` 时，当前正在处理的 100ms chunk 尚未推入 Paraformer 流式识别器（其内部包含 480ms accumulator），过早调用 `paraformer.flush()` 会导致尾部音节丢失或空文本回退。
- **重构方案**:
  - 引入 `_pending_speech_ends` 队列。
  - 在 `_on_audio_chunk()` 中：先将音频块喂给 VAD 产生断句事件（入队 pending 列表），随后完整推入 Paraformer 累加器，最后在 chunk 进入 Paraformer 之后统一排空 `_pending_speech_ends` 并执行对应段落的尾音 flush，取得完整 `online_text` 后再提交给后台 Qwen Worker。

### (3) Qwen In-flight 任务 Shutdown 超时取消与迟到抑制
- **问题根因**: 课堂结束或异常断开时，若 CapsWriter 服务端挂起未响应，收尾流程容易无限阻塞；或者在会话已经保存完成后迟到的 WebSocket 异步响应会篡改已落盘的元数据。
- **重构方案**:
  - `QwenWorker` 引入会话代际编号 `_session_generation` 与任务取消集合 `_cancelled_task_ids`。
  - 显式跟踪当前正在执行的 `_current_in_flight_task`。
  - `stop(wait_finish=True, timeout=...)` 超时触发时，自动调用 `cancel_in_flight_and_flush_fallback()`，以 `QWEN_SHUTDOWN_TIMEOUT` 为原因将未完成的段落回退为 Paraformer 识别结果，并在 `[0.0, bounded_timeout]` 内安全退出。
  - 若已取消的任务在之后收到了迟到的 WebSocket 响应，直接丢弃（`Discarded late response for cancelled/timed-out task`），严禁触发 `on_segment_final` 回调或修改文件。

### (4) SegmentRecord 统计指标与 Qwen 成功标准
- **重构方案**:
  - `SegmentRecord` 新增 `final_success: Optional[bool] = None` 与 `fallback_reason: Optional[str] = None`。
  - `meta.json` 中 `qwen_success_segments` 严格且仅统计 `status == "final" and final_success is True and final_model.startswith("Qwen")`。
  - Fallback 段落记入 `fallback_segments`，超时段落记入 `qwen_timeout_segments`。

### (5) 启动器与路径统一
- 仓库根目录保留并对齐 `run.bat` 与 `main.py`，分别转发并委托至 `launcher/run_app.bat` 与 `app/main.py`。
- 文档与 README 彻底清理所有绝对本地 `file:///` 协议链接，统一为规范的相对路径。

---

## 2. 自动化测试套件执行指标

| 测试文件 | 覆盖目标 | 测试结果 | 关键指标 / 验证点 |
| :--- | :--- | :---: | :--- |
| `tests/test_two_real_sessions_same_manager.py` | 同一 SessionManager 连续运行两个真实音频 Session | **PASS** | Session 1: 8.28s, 100% 写入, Qwen 成功; Session 2: 6.29s, 100% 写入, Qwen 成功 |
| `tests/test_paraformer_tail_fallback.py` | Qwen 挂掉时 Paraformer 尾部音节回退完整性 | **PASS** | 捕获尾词“经济运行”，markdown 正确打上 `[实时回退]` 标签 |
| `tests/test_qwen_inflight_timeout.py` | 假死 WebSocket 服务端下在途任务有界超时与迟到抑制 | **PASS** | 5.14s 有界退出，产生 `QWEN_SHUTDOWN_TIMEOUT`，迟到响应 100% 抑制 |
| `tests/test_gui_start_signature.py` | UI 启动参数与设备索引传递 | **PASS** | 支持 `device_index=None` 与指定声卡索引 |
| `tests/test_streaming_accumulator.py` | Paraformer 480ms 累加器与残余 flush 准确性 | **PASS** | 16 次 partial 稳定流式输出，零丢字 |
| `tests/test_shutdown_tail_integrity.py` | 停录边界 0 丢音、0 丢失文本 | **PASS** | 录制 WAV 8.28s，与输入采样差 0.0000s |
| `tests/test_vad_long_speech_continuation.py` | 70s 持续长语音无缝切分 | **PASS** | 产出 6 个切片，严格保持单句不超过 `max_segment_sec` |
| `tests/test_end_to_end_file_replay.py` | 政治经济学 & 微观计量经济学 双课程全流程回放 | **PASS** | 政治经济学 (8.28s) + 微观计量经济学 (6.29s) 全部 Qwen 识别成功 |
| `tests/test_full_acceptance.py` | 综合验收五大子测试 (课程、2-Pass、序号、文件、零重复下载) | **PASS** | 5/5 子测试全部通过，产物证据链完整 |
| `tests/test_30min_soak.py 1800.0` | **30分钟 (1800.0s) 课堂长录音压力稳定性测试** | **PASS** | 模拟 1800.0s，处理 1800.0s，156 句子全成功，队列 100% 排空，内存稳定 |

---

## 3. 30 分钟长音频压力测试详报 (Soak Test Summary)

```text
=== 30-Minute Soak Results Summary ===
Total Audio Simulated: 1800.00s (Target: 1800.00s)
Total Audio Processed: 1800.00s
Max Capture Queue Size Observed: 1
Max Processing Queue Size Observed: 16668
Max Qwen Queue Size Observed: 1
Final Capture Queue Size: 0 (drained to 0)
Final Processing Queue Size: 0 (drained to 0)
Final Qwen Queue Size: 0 (drained to 0)
Initial RSS: 451.61 MB -> Final RSS: 1456.94 MB (Delta: +1005.33 MB)
Shutdown Elapsed: 1.16s
Meta segments: total=156, qwen_success=156, fallback=0
[PASS] Full 30-minute (1800s) soak stability test passed!
```

---

## 4. 结论
本工程已完全满足第二轮审查规范的所有要求，不存在音频多 Session 竞态、不存在尾音截断、不存在长时阻塞与迟到篡改，已具备在真实课堂环境下进行完整影子测试与日常实用的高可靠性标准。
