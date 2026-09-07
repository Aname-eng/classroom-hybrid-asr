# Classroom Hybrid ASR｜代码审查问题清单与修复方案｜2026-09-07

> 仓库：`Aname-eng/classroom-hybrid-asr`
> 审查基线：`master` at `57844609289954a4a77184a767044c4f9a72af04`
> 目标：在不推翻“Paraformer streaming 实时字幕 + CapsWriter Qwen3-ASR 句末高精度重识别”的核心架构前提下，修复真实课堂路径中的 P0/P1 问题，并建立可信的端到端验收。

---

## 0. 总结

当前工程的总体架构是合理的，且 `QwenAdapter` 对 CapsWriter-Offline WebSocket 协议的字段使用方向基本正确；主要问题集中在**生产音频路径、线程/队列收尾、VAD 强切、Qwen 失败判定，以及验收测试与真实路径不一致**。

因此不要重写整个项目，也不要把 CapsWriter 的 Qwen3-ASR q4_k 模型强行塞进 FunASR `AutoModel`。推荐保留：

```text
麦克风 16kHz mono
  ├─> 原始 WAV 持久化
  ├─> Paraformer-zh-streaming（实时 partial）
  └─> FSMN-VAD（句末）
          └─> CapsWriter Qwen3-ASR（整句 final）
                  └─> UI 原地替换 + transcript/evidence 落盘
```

修复顺序必须遵循：**先 P0，再 P1，再验收，再优化**。

---

# 1. P0：GUI `start_session()` 调用参数不匹配

## 问题

`app/ui/main_window.py` 调用：

```python
self.session_manager.start_session(
    course_id=course_id,
    device_index=device_idx
)
```

但 `app/pipeline/session_manager.py` 当前定义：

```python
def start_session(self, course_id: Optional[str] = None) -> str:
```

因此真实 GUI 点击“开始课堂录音”可能直接抛出：

```text
TypeError: start_session() got an unexpected keyword argument 'device_index'
```

## 推荐修法

不要在 UI 里绕开。让 `SessionManager.start_session()` 正式接受 `device_index`：

```python
def start_session(
    self,
    course_id: Optional[str] = None,
    device_index: Optional[int] = None,
) -> str:
    ...
    if device_index is not None:
        self.config.audio_device_index = device_index
        self.recorder.device_index = device_index
    else:
        self.recorder.device_index = self.config.audio_device_index
```

并在启动前保存配置（可选，但推荐）：

```python
self.config.save()
```

## 验收

- GUI 可以选择任意实际输入设备并启动。
- 默认设备 `None` 也能启动。
- 测试必须通过 GUI 同一调用签名，而不是绕开 UI。

---

# 2. P0：真实麦克风 100ms chunk 直接送入 Paraformer，和模型配置不一致

## 问题

`AudioRecorder` 当前：

```python
blocksize = 1600  # 100ms @ 16kHz
```

所以 `SessionManager._on_audio_chunk()` 每次收到 1600 samples。

但 `ParaformerStreamer` 配置：

```python
chunk_size = [8, 8, 4]
```

这里中心 chunk 应为：

```text
8 * 960 = 7680 samples = 480ms
```

当前生产路径没有 accumulator，就把 100ms chunk 直接调用：

```python
self.model.generate(input=chunk, chunk_size=[8,8,4], ...)
```

这和 FunASR 官方 streaming 用法不一致。

## 推荐修法

**保持 AudioRecorder 100ms 采集不变**，因为较小录音块有利于写盘、VAD 和低延迟；在 `ParaformerStreamer` 内部增加独立 accumulator。

推荐结构：

```python
self._audio_buffer = np.empty(0, dtype=np.float32)
self.chunk_stride_samples = self.chunk_size[1] * 960
```

`process_chunk()`：

```python
self._audio_buffer = np.concatenate([self._audio_buffer, chunk])

while len(self._audio_buffer) >= self.chunk_stride_samples:
    asr_chunk = self._audio_buffer[:self.chunk_stride_samples]
    self._audio_buffer = self._audio_buffer[self.chunk_stride_samples:]
    _infer(asr_chunk, is_final=False)
```

句末/会话结束时增加：

```python
def flush(self):
    if len(self._audio_buffer) > 0:
        padded_or_remaining = self._audio_buffer
        self._audio_buffer = np.empty(0, dtype=np.float32)
        _infer(padded_or_remaining, is_final=True)
```

### 重要

不要简单把 `AudioRecorder.blocksize` 改成 7680 来掩盖问题。录音采集块与 ASR 推理块应该解耦。

## 验收

- 真麦克风与测试文件必须走相同 accumulator。
- 对 `[8,8,4]`，每次非 final Paraformer 推理输入应是 7680 samples。
- 结束时剩余不足 7680 的尾音必须进入 final flush。

---

# 3. P0/P1：结束课堂存在尾音丢失竞态

## 问题

当前 `SessionManager.end_session()` 顺序大致为：

```text
vad.flush()
recorder.stop()
wait Qwen
save transcript
```

但 `AudioRecorder` 有自己的 `_audio_queue` 和 writer thread。调用 `vad.flush()` 时，队列里可能还有尚未进入 VAD/Paraformer 的音频；之后 `recorder.stop()` 才 drain 一部分队列。这样最后一句的尾部可能：

- 已写入 WAV；
- 但没有进入最后一次 VAD flush；
- 没有送 Qwen final。

## 推荐修法

把“停止采集”和“完成下游处理”拆成两个阶段。

推荐结束顺序：

```text
1. stop_input()：先停止 microphone InputStream，不再产生新音频
2. drain recorder queue：等待所有已采集音频写盘并分发到 VAD/streaming
3. flush Paraformer remaining buffer
4. flush VAD remaining speech segment
5. 等待所有 Qwen segment 完成 / 超时
6. 最后写 transcript/meta
7. 关闭 WAV/worker/server（按 ownership 策略）
```

可将 `AudioRecorder.stop()` 拆成：

```python
stop_capture()
wait_until_drained(timeout=...)
close_writer()
```

或者保持一个公共 `stop()`，但必须保证：

- 先 stop/close InputStream；
- `queue.join()` 真正 drain；
- writer thread 退出后才返回；
- 此时 SessionManager 才 flush VAD / streaming。

## 验收

做专门测试：

```text
播放一句话，在最后一个字结束后几乎立刻点击“结束课堂”
```

必须满足：

- WAV 包含完整尾音；
- final transcript 包含最后一句；
- 最后 segment 不为空；
- 没有 orphan partial。

---

# 4. P1：`AudioRecorder.stop()` 不能严格保证队列排空

## 问题

当前 writer thread：

```python
self._writer_thread.join(timeout=3.0)
```

3 秒后无论队列是否处理完都会继续关闭 WAV。

如果下游 `on_audio_chunk` 中 VAD / Paraformer 某次阻塞，writer thread 可能仍存活，造成：

- WAV 关闭后线程仍尝试 write；
- 音频未完成下游分发；
- 课堂尾部丢失。

## 推荐修法

### 最低限度

使用：

```python
self._audio_queue.join()
```

但不要无限等待，应该封装可观察 timeout。

### 更推荐

将职责拆开：

```text
Audio callback -> capture_queue

writer thread: capture_queue -> WAV
processing thread: capture_queue/copy -> VAD + streaming
```

至少保证**写 WAV 不被 ASR 推理拖慢**。

如果暂时不重构双队列，也至少：

- 对队列长度做 metric；
- 对 shutdown 进行 bounded wait；
- 超时则明确记录 `shutdown_incomplete=true`，不能静默完成。

---

# 5. P1：Qwen Worker 的 `timeout` 参数实际上无效

## 问题

当前：

```python
def wait_completion(self, timeout=None):
    self._task_queue.join()
```

传 `timeout=60` 也不会生效，所以 `end_session()` 可以无限卡住。

## 推荐修法

不要用裸 `queue.join()` 实现超时。

增加：

```python
self._pending_count
self._pending_cv = threading.Condition()
```

或者用轮询：

```python
def wait_completion(self, timeout=None):
    deadline = None if timeout is None else time.monotonic() + timeout
    while self._task_queue.unfinished_tasks > 0:
        if deadline is not None and time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True
```

推荐返回 `bool`。

`SessionManager.end_session()`：

```python
completed = self.qwen_worker.wait_completion(timeout=60.0)
if not completed:
    log warning
    把未完成段落标为 pending/fallback/timeout
    不要无限卡 UI
```

## 验收

模拟 Qwen server 不响应，结束课堂应在预设时间内返回，并显示：

```text
Qwen timeout / fallback
```

而不是 GUI 永久卡在“正在收尾”。

---

# 6. P1：VAD 达到最大段长后的强切可能破坏连续讲话

## 问题

当前达到 `max_segment_sec` 后：

```python
self.in_speech = False
self.current_segment_id += 1
on_speech_end(...)
```

如果老师持续讲话没有静音，VAD 未必重新发送 speech-start，因此后续音频可能没有被继续缓存成新 segment。

## 推荐修法

强切应被视为 **continuation split**，而不是 speech 真正结束。

推荐：

```text
当前 segment 达 max length
→ emit segment_end(reason="max_duration")
→ current_segment_id += 1
→ 立即以当前时间作为新 segment start
→ in_speech 继续保持 True
→ 新 speech_buffer 从下一 chunk 开始累积
```

必要时保留 100–300ms overlap，降低边界断词：

```text
segment N tail 200ms -> segment N+1 prefix 200ms
```

如果实现 overlap，最终 transcript 要有去重策略。

## 验收

输入 60 秒连续讲话、无明显静音：

- 应产生多个连续 segment；
- segment 时间轴无大 gap；
- 没有 25/30 秒之后整段消失。

---

# 7. P1：录音、VAD、Paraformer 实际没有完全线程解耦

## 问题

README 声称多个组件完全独立，但当前 `_writer_loop()` 内同时：

```text
写 WAV
→ 调 on_audio_chunk
   → VAD.generate
   → Paraformer.generate
```

因此慢 ASR 可以拖慢 writer loop，最终造成 `_audio_queue` 堆积。

## 推荐修法

### 推荐 v1.1 结构

```text
PortAudio callback
   └─> capture_queue
        ├─> WavWriterThread  -> audio.wav
        └─> ProcessingThread -> VAD + Paraformer
                               └─> QwenWorker（已有独立队列）
```

如果不想本轮大改，至少做到：

- 音频写盘与 ASR 推理解耦；
- capture callback 绝不做推理；
- 监控 `capture_queue` / `processing_queue` backlog；
- backlog 超阈值时 UI 警告“实时字幕落后”，但绝不能牺牲 WAV 母带。

## 核心原则

**录音证据链优先级高于实时字幕。**

哪怕机器暂时算不过来，也必须保证原始 WAV 完整；字幕可以延迟追赶。

---

# 8. P1：综合验收测试没有真正覆盖生产路径

## 问题

`test_full_acceptance.py` 当前：

1. `manager.start_session()` 会启动真实麦克风录音；
2. 测试又直接调用私有 `manager._on_audio_chunk()` 注入 WAV；
3. 所以测试输入和最终 `audio.wav` 并非同一条真实 ingestion path。

此外，测试只验证：

```python
len(partial_events) > 0
len(final_events) > 0
```

但 Qwen 失败 fallback 也会触发 final，因此会出现假阳性。

## 推荐修法

建立统一输入抽象：

```python
class AudioSource:
    start(callback)
    stop()

class MicrophoneAudioSource(AudioSource)
class FileReplayAudioSource(AudioSource)
```

测试时用 `FileReplayAudioSource`，但必须经过**与麦克风相同的 Recorder/Dispatcher/VAD/Paraformer/Qwen 路径**。

不要再从测试直接调用 `_on_audio_chunk()` 私有方法。

### 验收断言必须新增

- `audio.wav` 时长与输入音频误差 < 合理阈值；
- WAV 非静音样本比例合理；
- Qwen final 中至少一个 `success=True`；
- 关键测试不得出现 `paraformer_fallback`；
- segment ID 单调且无重复；
- final segment 完整；
- 关键词/句子准确率达到阈值；
- streaming RTF < 1；
- 处理队列最终归零。

---

# 9. P1：Qwen fallback 必须成为一等状态，不能伪装成“权威定稿”

## 问题

当前 Qwen 失败后：

```python
final_model = "paraformer_fallback"
```

这是好事，但综合测试和 UI 的系统状态没有把 Qwen 不可用当作重要退化状态。

## 推荐修法

新增状态：

```text
QWEN_OK
QWEN_STARTING
QWEN_DEGRADED
QWEN_TIMEOUT
QWEN_OFFLINE
```

每个 `SegmentRecord` 保留：

```python
final_success: bool
fallback_reason: Optional[str]
```

UI：

```text
绿色：Qwen final
灰/橙色：Paraformer fallback
红色：Qwen service unavailable
```

meta.json 统计：

```json
{
  "qwen_success_segments": 80,
  "fallback_segments": 2,
  "qwen_timeout_segments": 1
}
```

验收时关键路径要求 fallback=0。

---

# 10. P2：Qwen 单测只检查非空，不检查识别质量

## 问题

`test_qwen_adapter.py` 目前打印 Expected Text，但最终只断言：

```python
assert len(result.strip()) > 0
```

错误文本一样 PASS。

## 推荐修法

不要要求逐字 100% 相同，可用：

### 中文关键术语 gate

```python
required_terms = [
    "中国特色社会主义政治经济学",
]
assert all(term in result for term in required_terms)
```

### 或 CER gate

对固定合成测试句计算 Character Error Rate，例如：

```text
CER <= 0.10
```

对于 `Hausman` 测试，可要求：

```text
Hausman / 豪斯曼
```

命中至少一个规范表示，具体由 hotword 配置决定。

---

# 11. P2：模型/硬件元数据不能写死

## 问题

当前输出固定写：

```text
Qwen3-ASR-1.7B-q4_k
Intel Arc 140T GPU (Vulkan)
```

但真正启用哪个模型由 CapsWriter server 配置决定；`QwenAdapter` 并没有通过 WebSocket 指定模型文件。

## 推荐修法

分两层记录：

```json
{
  "configured_offline_model": "Qwen3-ASR-1.7B-q4_k",
  "configured_capswriter_path": "...",
  "verified_runtime_model": null,
  "hardware_hint": "Intel Arc / Core Ultra",
  "runtime_verification": "not_exposed_by_server"
}
```

如果 CapsWriter 有可读取的配置文件，就启动时解析实际模型配置；如果没有，就明确写：

```text
runtime model inferred from local CapsWriter config, not independently verified
```

不要使用“权威模型”这类会掩盖实际 runtime 不确定性的标签。

---

# 12. P2：公共仓库中的 WAV 测试音频

当前 `.gitignore` 放行：

```gitignore
!tests/audio_samples/*.wav
```

这是可以接受的，前提是测试 WAV 是：

- 自己录制且明确可公开；或
- 合成语音；或
- 无第三方隐私/课堂内容。

如果是真实课堂或他人声音，应改为：

```text
tests/audio_samples/README.md
```

只记录生成方法，并在本地生成 fixture，不提交真实声音。

---

# 13. 推荐实施顺序

## Phase A：先让真实 GUI 能稳定跑

1. 修 `device_index` 参数。
2. Paraformer 增加 480ms accumulator + final flush。
3. 修 shutdown 顺序，先停采集再 drain 再 flush。
4. 实现 Qwen bounded timeout。
5. 修 VAD max-duration continuation。

完成后先做 5–10 分钟真麦克风测试。

## Phase B：让系统“不丢音”

6. 写盘与 ASR 推理解耦。
7. 增加 queue/backlog metrics。
8. 任何 ASR 退化不得影响 `audio.wav` 完整性。

## Phase C：重做可信验收

9. FileReplayAudioSource / 统一 ingestion。
10. Qwen success / fallback 强断言。
11. 尾音测试、连续讲话强切测试、断开 Qwen 服务测试。
12. 30 分钟 soak test。
13. 最终 90 分钟真实课堂前再做一次长时验收。

---

# 14. 必须新增/更新的测试

建议至少补这些：

```text
tests/
  test_gui_start_signature.py
  test_streaming_accumulator.py
  test_shutdown_tail_integrity.py
  test_vad_long_speech_continuation.py
  test_qwen_timeout_and_fallback.py
  test_end_to_end_file_replay.py
  test_30min_soak.py
```

### `test_streaming_accumulator.py`

验证连续 100ms 输入时，Paraformer 非 final 推理看到的是 480ms chunk。

### `test_shutdown_tail_integrity.py`

音频最后一句后立即结束，最后关键词仍在 final transcript。

### `test_vad_long_speech_continuation.py`

生成 >60 秒连续语音/测试信号，强切后仍持续产出 segment。

### `test_qwen_timeout_and_fallback.py`

模拟 server 不响应：

- UI/manager 在 timeout 后继续结束；
- fallback reason 正确；
- 不死锁。

### `test_end_to_end_file_replay.py`

完全不直接调用 `_on_audio_chunk()`，用统一 AudioSource API 重放 WAV。

### `test_30min_soak.py`

记录：

```text
max_capture_queue
max_processing_queue
max_qwen_queue
streaming_lag_sec
memory_growth_mb
fallback_count
lost_audio_duration_sec
```

---

# 15. 验收 Gate（全部通过才可称“课堂可用 v1”）

## Gate 1：启动

- GUI 选麦克风并启动无异常。
- CapsWriter server 可自动连接/启动。

## Gate 2：实时层

- 真实 100ms capture 经 accumulator 正确转换为 Paraformer streaming stride。
- streaming RTF 稳定 < 1。
- 字幕允许短时延迟，但不能长期越积越慢。

## Gate 3：证据链

- `audio.wav` 是最高优先级证据，完整、可播放、时长正确。
- `events.jsonl`、`transcript_raw.jsonl`、`transcript_final.md`、`meta.json` 均完整。

## Gate 4：Qwen final

- 正常场景 Qwen success，不允许测试静默 fallback。
- 失败场景能明确 fallback，不死锁。

## Gate 5：结束课堂

- 最后一秒讲话也不会丢。
- Qwen timeout 时仍能 bounded shutdown。

## Gate 6：长时稳定

- 30 分钟 soak 通过。
- 再进行 60–90 分钟真实课堂影子测试；同时用独立录音工具保底。
- 在至少一堂完整课堂验证前，不删除独立录音备份。

---

# 16. 本轮不要做的事情

在 P0/P1 修完前，不要：

- 重做 UI 视觉；
- 加 LLM 自动总结；
- 加 GitHub 自动 push；
- 扩展全部课程热词；
- 增加 speaker diarization；
- 重构成复杂微服务；
- 重复下载 Qwen3-ASR 模型；
- 把 CapsWriter q4_k 当成 FunASR `AutoModel` 的直接模型路径。

先把**真实课堂音频可靠采集 + 实时字幕 + 句末 final + 正确收尾**做稳。

---

# 17. 修改完成后的提交要求

本地 AI 修复后，应：

1. 运行所有新增/现有测试。
2. 把测试命令和真实结果写进 `docs/FIX_VALIDATION_2026-09-07.md`。
3. 不得把“import 成功”“文件存在”当成完整验收。
4. 提交到 `master` 前先 `git diff` 自审。
5. commit message 建议：

```text
fix: harden classroom ASR production audio pipeline and shutdown
```

6. push 后把最终 commit SHA 告知远程 ChatGPT，再进行第二轮 code review。
