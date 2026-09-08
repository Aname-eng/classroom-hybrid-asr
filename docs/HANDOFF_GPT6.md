# Classroom ASR 项目交接文档（交给 GPT-6）

更新时间：2026-09-08
项目目录：`D:\课程笔记`

## 0. 交接结论

项目已经完成了跨平台配置、Qwen/Paraformer 双遍识别、模型管理和 PyInstaller 打包框架，但**还没有形成经过真实运行验证的最终发布版**。

最近针对模型下载卡顿问题已经改了一版：

- 不再用一个全局锁包住整个网络下载过程。
- 每个模型使用独立后台线程。
- 默认最多同时下载 4 个模型。
- 每个模型内部向 ModelScope Hub 请求最多 8 个并行文件/分片下载线程。
- UI 支持 `Ctrl/Shift` 多选模型并行下载，并增加“全部并行下载”按钮。
- 当前使用的是 ModelScope Hub 内置的并行文件/Range 下载机制，**不是外部 FDM 程序**。

这部分代码已经通过模拟下载测试，但尚未重新打出并测试新的 exe。

---

## 1. 当前已完成的功能

### 1.1 识别流水线

- 第一遍：FunASR Paraformer 中文真流式模型。
- 第二遍：本地 `qwen-asr` Qwen3-ASR；不可用时回退 CapsWriter WebSocket。
- CapsWriter 地址保持：`ws://127.0.0.1:6016`。
- Qwen 后端懒加载，不会在未选择本地模型时提前加载大模型。
- Qwen 任务具有超时、取消、会话代际保护，迟到结果不会污染新会话。
- 录音原始音频、原始转写、最终转写、整理稿分开保存。

### 1.2 整理与热词

- 默认整理模型：本地 Transformers Qwen3-0.6B。
- 兼容 Ollama 和 OpenAI-compatible 接口。
- 本地模型不可用时使用确定性口头禅/无关内容清理。
- 自动从课程名称、简介和热词生成热词；失败时规则回退。
- 新增 `transcript_cleaned.md`，不覆盖 `transcript_final.md`。

### 1.3 模型管理

文件：`app/model_manager.py`

预设模型：

- `qwen3_asr_0_6b`
- `qwen3_0_6b`
- `paraformer_zh_streaming`
- `qwen3_asr_1_7b`

支持：

- ModelScope 中国站下载：`https://www.modelscope.cn`
- 本地模型导入
- 启用/切换模型
- 删除应用下载的模型
- 删除后保留 disabled 标记，避免启动自动重新下载
- 注册表：`config/models.json`（本地生成文件，已忽略）

### 1.4 下载并行化（最近修改）

文件：`app/model_manager.py`

配置项：

```json
"model_download_workers": 8,
"model_download_max_parallel": 4
```

实现：

- `download_async()` 按模型 key 建立独立线程。
- `_download_slots` 限制同时下载模型数，默认 4。
- 不再让网络下载持有全局 `_lock`。
- `ModelScope snapshot_download()` 如果支持 `max_workers`，传入 8。
- ModelScope Hub 新版内部有并行文件和大文件 Range 分片下载。
- 注册表和配置保存仍用锁保护。

文件：`app/ui/main_window.py`

- 模型表格改为 `ExtendedSelection`。
- “并行下载选中”支持多选。
- “全部并行下载”直接启动全部四个模型。
- 下载开始时不再重建整个推理配置；只有完成后才刷新模型配置。

模拟测试结果：四个 fake 下载任务同时运行，最大并发数为 4。

---

## 2. 当前没有处理完善的问题

### 2.1 新并行下载版 exe 尚未重新构建

当前已有的文件：

```text
dist/ClassroomASR.exe
```

大小约 475 MB，时间戳早于最近的并行下载修改，不能认为它包含最新代码。

最近重新构建命令曾执行：

```powershell
C:\Python314\python.exe build.py > build_parallel.log 2>&1
```

但构建过程在 PyInstaller Analysis 阶段被中止/超时，没有生成新的最终 exe。需要重新构建并检查：

```text
Build complete: D:\课程笔记\dist\ClassroomASR.exe
```

### 2.2 PyInstaller 构建过慢、包体过大

当前 `build.py` 使用：

```text
--collect-all funasr
--collect-all modelscope
--collect-all transformers
--collect-all qwen_asr（安装时）
```

这会把大量训练、CV、CLI、无关模型代码也打入包，导致：

- Analysis 很慢。
- 内存占用很高。
- exe 约 475 MB 或更大。
- 可能继续增长到数百 MB 以上。

需要优化 PyInstaller spec/参数，但不能误删运行时真正需要的 FunASR、Torch、PySide6、Qwen、ModelScope 数据。

### 2.3 下载进度显示不完整

当前只显示“开始下载/下载完成/失败”，没有每个模型的：

- 百分比
- 已下载大小
- 速度
- ETA
- 当前文件名
- 暂停/继续/取消按钮

用户在大模型下载期间仍可能误以为程序卡住。ModelScope Hub 有 `progress_callbacks` 能力，可以接入 Qt signal。

### 2.4 并行下载尚未用真实模型验证

已验证的是模拟 `snapshot_download`，尚未验证：

- 四个真实模型同时下载。
- ModelScope cache 并发锁是否稳定。
- Windows 路径和中文路径。
- 下载中断后能否复用已有缓存。
- 某个模型失败是否影响其他模型。
- 磁盘空间不足时的清理。
- 网络断开后的重试。

### 2.5 “FDM”目前不是外部 FDM 集成

当前方案使用 `modelscope-hub` 内置的并行下载和 Range 分片能力，属于类似 FDM 的下载架构，但没有调用 Free Download Manager 独立程序。

如果 GPT-6 要接入真正的下载器，需要评估：

- 是否有稳定的 Python API。
- 是否能在 Windows/macOS/Linux 一致运行。
- 是否会破坏 ModelScope 鉴权、缓存和断点续传。
- 是否会增加便携版 exe 的外部依赖。

建议优先保留 ModelScope Hub 原生并发，不要引入外部 FDM 可执行文件。

### 2.6 默认启动自动下载策略需要产品化

当前启动时默认并行检查/下载三个默认模型：

- Qwen3-ASR 0.6B
- Qwen3 0.6B
- Paraformer 中文流式

这可能同时占用大量网络、磁盘和内存。需要考虑：

- 首次启动先弹出模型选择页，而不是立即下载。
- 提供“只下载当前需要模型”选项。
- 默认并发数按网络/磁盘自动调整。
- 明确显示“后台下载中”，允许跳过。

### 2.7 下载线程生命周期还需要完善

当前后台线程是 daemon thread：

- 关闭窗口时没有完整取消/等待下载。
- 程序退出时可能留下 `.downloading` 目录。
- 没有统一的取消 token。
- `import_local()`、`delete()` 与并行下载同时操作时需要进一步加锁审查。

### 2.8 实际 Qwen ASR 端到端仍未完成

当前开发环境此前没有 `qwen-asr`；用户刚刚在全局 Python 3.14 环境安装了 `qwen-asr==0.0.6`，但尚未完成：

- Qwen3-ASR 0.6B ModelScope 下载。
- CPU 推理。
- Intel Arc 环境推理。
- 本地 Qwen 第二遍结果替换。
- 模型不存在时正确回退 CapsWriter。

### 2.9 Python 环境不统一

当前有两个环境：

- 项目 `.venv`：Python 3.12，之前用于测试。
- 当前打包环境：`C:\Python314`，Python 3.14.6，PyInstaller 6.22.2，qwen-asr 已安装。

用户打包日志显示使用的是 Python 3.14。发布前必须决定一个固定构建环境。建议优先使用干净的 Python 3.12/3.13 发布环境，除非确认 Python 3.14 下所有 Torch、PySide6、FunASR、qwen-asr 都能稳定运行。

用户安装时出现的 NumPy 警告：

```text
fastembed requires numpy>=2.3 for Python 3.14
magika requires numpy>=2.1 for Python 3.13+
```

这些是全局环境中的其他包冲突，不一定影响 Classroom ASR，但不应使用混杂的全局环境做最终发布。

### 2.10 PyInstaller 警告尚未做最终运行验证

日志中出现：

```text
ModuleNotFoundError: No module named 'addict'
ModuleNotFoundError: No module named 'einops'
```

以及大量第三方库 `SyntaxWarning`。目前多数属于可选模块收集警告，但需要实际运行 exe 验证：

- 主界面是否能打开。
- FunASR VAD/Paraformer 是否能加载。
- Qwen ASR 是否能加载。
- ModelScope 是否能下载模型。
- Transformers 整理模型是否能加载。

必要时安装 `addict`、`einops`，或者在 PyInstaller 中排除无关的 optional modules。

### 2.11 macOS DMG 尚未构建

`build.py` 已设计为：

- Windows/Linux：`--onefile`
- macOS：生成 `.app`，再用 `hdiutil` 生成 `.dmg`

但 macOS 不能在 Windows 上由 PyInstaller 原生交叉构建。需要在实际 Mac 上执行，并进一步处理：

- Apple Silicon / Intel 架构。
- 代码签名。
- Gatekeeper/notarization。
- 麦克风权限和录音权限。
- PySide6、Torch、sounddevice 的 macOS 动态库。

### 2.12 首次目录选择功能只覆盖首次打包启动

打包版首次启动会选择：

- 课程笔记根目录。
- 模型目录。

目录结构为：

```text
用户选择的笔记目录/
├── courses/
└── sessions/
```

模型目录单独存储。当前没有完善的“设置页”用于后续修改这两个目录；修改需要编辑用户数据目录下的 `config/settings.json`。

---

## 3. 可以优化的方向

### 3.1 优先级最高：先做一个可运行发布版

建议顺序：

1. 停止旧构建残留，删除旧 `build/`、`dist/` 后重新构建。
2. 使用干净的 Python 构建环境。
3. 用最小 PyInstaller spec，避免 `collect-all` 全包。
4. 运行新 exe，完成首次目录选择。
5. 测试模型窗口是否能同时启动四个下载。
6. 测试录音、VAD、Paraformer、保存会话。
7. 再做包体和启动速度优化。

### 3.2 优化 PyInstaller

当前入口：`main.py`。

可尝试：

- 只收集 FunASR 实际依赖模块。
- 排除训练模块、CV 模块、CLI、服务器模块、未使用 Transformers 模型。
- 用 `collect_data_files()` 代替 `collect_all()`。
- 保留 Torch 二进制和 FunASR 模型配置文件。
- 增加明确 hidden imports，而不是粗暴收集整个包。
- 分别建立 `build_windows.py` 和 `build_macos.py` 或 `.spec` 文件。
- 保留 `--onefile`，但接受首次启动需要解压到临时目录。
- 打包后测试 exe 的启动内存和模型加载。

### 3.3 下载器优化

建议保留三层并行：

1. 四个模型之间并行。
2. 每个模型的文件之间并行。
3. 大文件内部 Range 分片并行。

可增加：

- 每个模型独立进度卡片。
- `QThreadPool`/worker signal 统一回传 UI。
- 可取消下载。
- 断点续传。
- 指数退避重试。
- SHA256/文件完整性校验。
- 空间预检查。
- 下载速度限制。
- 自动检测磁盘类型和网络速度。
- `max_parallel` 和 `workers_per_model` 在 UI 设置中可调。

不要让 Qt 主线程调用 `snapshot_download()`、`shutil.copytree()` 或大文件校验。

### 3.4 配置与数据目录

建议增加“存储设置”窗口：

- 修改课程笔记目录。
- 修改模型目录。
- 显示当前占用空间。
- 迁移旧目录。
- 校验目录权限。
- 一键打开目录。

并在 `meta.json` 中记录实际模型目录、模型版本和下载源。

### 3.5 识别质量优化

- 用真实课堂音频验证 Paraformer 端点、重复字和尾音。
- 验证课程热词对流式模型的实际影响。
- 比较 Qwen3-ASR 0.6B 和 1.7B 的第二遍纠错效果。
- 记录 RTF、延迟、CPU/RAM、每句超时和回退原因。
- 不要把 Qwen3-ASR 离线模型硬切成 100ms 伪流式。
- Intel Arc 先以 CPU/现有 CapsWriter 回退为稳定基线，不要假定 CUDA。

### 3.6 稳定性与测试

新增或完善：

- 四模型真实并行下载测试。
- 下载中断/恢复测试。
- 删除正在下载的模型测试。
- 关闭窗口时下载取消测试。
- ModelScope cache 损坏测试。
- Windows 中文路径测试。
- macOS 权限测试。
- PyInstaller frozen 模式测试。
- 一个真正的端到端离线模型测试。

现有历史问题：

- Windows 并行测试偶尔出现 `torch.storage.clone` / `asyncio.windows_events.py` access violation。
- `test_qwen_adapter.py::test_qwen_transcribe` 曾把 `wav_path` 当作缺失 fixture。
- 全量 pytest 没有稳定完成，不能只根据定向测试宣布发布。

---

## 4. 关键文件

```text
app/config.py                         跨平台路径、配置、目录选择
app/model_manager.py                  模型目录、并发下载、导入/启用/删除
app/ui/main_window.py                 模型管理 UI、首次目录选择、主窗口
app/pipeline/session_manager.py       录音、VAD、流式、第二遍、保存
app/streaming/paraformer_streamer.py  FunASR Paraformer 流式
app/offline/qwen_adapter.py           Qwen 本地/ CapsWriter 回退
app/offline/local_qwen_asr.py         qwen-asr 懒加载
app/offline/qwen_worker.py             第二遍异步队列
app/offline/summary_processor.py       本地整理/热词/回退
build.py                              Windows onefile / macOS dmg 构建
requirements.txt                      运行时依赖
requirements-build.txt                PyInstaller 依赖
config/settings.json                  开发默认配置
README.md                             使用和构建说明
```

重要本地目录：

```text
config/models.json          模型注册表，忽略文件
models/                     FunASR/VAD 已有模型缓存，忽略文件
sessions/                   开发模式会话输出，忽略文件
downloaded_models/          开发模式模型目录，忽略文件
dist/ClassroomASR.exe      旧 exe，不代表最新并行代码
build_parallel.log          最近一次中止的构建日志
```

不要盲目回滚以下已有用户数据/课程修改：

- `courses/stata/*`
- `courses/course_*/*`
- 其他课程目录

---

## 5. 建议 GPT-6 接手后的第一步

```powershell
cd D:\课程笔记

# 1. 检查当前代码和构建结果
git status --short
python -m py_compile app/model_manager.py app/ui/main_window.py build.py

# 2. 确认并行下载逻辑和 UI 多选逻辑
# 重点查看：app/model_manager.py、app/ui/main_window.py

# 3. 使用同一个干净 Python 环境重新构建
C:\Python314\python.exe build.py > build_gpt6.log 2>&1

# 4. 检查最终输出
Get-Item dist\ClassroomASR.exe

# 5. 运行 exe，验证首次目录选择和四模型并行下载
.\dist\ClassroomASR.exe
```

若构建仍然超过很长时间，优先优化 `build.py` 的 `--collect-all`，不要继续盲等。

---

## 6. 当前验收标准

最终至少应满足：

- [ ] Windows 单个 `ClassroomASR.exe` 可以启动。
- [ ] exe 旁边不需要 Python、脚本或依赖文件夹。
- [ ] 首次启动可以选择课程笔记目录和模型目录。
- [ ] 可以多选四个模型并行下载。
- [ ] 下载时主窗口不冻结。
- [ ] ModelScope Hub 能进行文件/分片并行下载。
- [ ] 一个模型下载失败不影响其他模型。
- [ ] 断网/超时不会破坏已有模型。
- [ ] Paraformer 流式识别可用。
- [ ] Qwen ASR 可用时进行第二遍纠错，不可用时自动回退。
- [ ] 原始录音、原始稿、最终稿、整理稿均保留。
- [ ] macOS 实机可构建 `.dmg` 并获得麦克风权限。
- [ ] 至少一次真实模型端到端验证。
