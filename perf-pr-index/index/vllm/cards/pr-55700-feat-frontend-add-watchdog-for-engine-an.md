# PR #55700: [Feat][Frontend] Add Watchdog for Engine and Worker Monitoring

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @wenjinhust |
| 合并 | 2026-10-21 |
| 分类 | Frontend / Serving \| tier: major |
| Label | ready, verified, scheduler |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未标注 dropped from diff 说明（diff 完整呈现） |
| 链接 | https://github.com/vllm-project/vllm/pull/55700 |

## 一句话结论

为 V1 Engine/Worker 加 watchdog，心跳超时 300s 自动 dump 全线程栈以诊断 hang/死锁。

## 动机

PR 正文 Purpose 说明，vLLM V1 缺少在 engine 或 worker 进程停止推进时自动抓取所有线程栈的机制，hang 与 deadlock 难以诊断。本 PR 标注 Fixes RFC #56816，加入 watchdog 机制，当 engine 或 worker 进程不再推进时用 `faulthandler` 把全线程栈写入日志，便于定位卡死。

## 优化点

新增 `vllm/utils/watch_dog.py` 的 `WatchDog` 类：后台 daemon 线程监控心跳 `feed()`，若超时（默认 300s，每 10s 检查一次）未刷新，则把全线程栈 dump 到每进程日志文件 `VLLM_STACK_DUMP_for_<name>_<pid>.log`，无 feed 的重复超时只记一次；提供 `feed()` / `dump_stack()` / `start()` / `stop()` 与进程级单例 `get_watch_dog()`。EngineCoreProc 启动后创建并启动名为 `engine` 的 watchdog，在处理输入队列与每个 engine step 前 feed，输入队列 `get()` 改用 `timeout=5` 以保持响应同时保留原阻塞语义，shutdown signal handler 会 dump 全线程栈；WorkerProc（multiproc executor）同样创建名为 `worker` 的 watchdog 并在每个 worker method 执行前 feed，signal handler dump 全栈。属可观测性/诊断功能，PR 未给出性能优化机制。

## 关键改动

```text
vllm/utils/watch_dog.py: 新增 WatchDog 类，含 feed/dump_stack/start/stop、get_watch_dog 单例与 faulthandler 全线程栈 dump
vllm/config/watchdog_config.py、vllm/config/vllm.py、vllm/config/__init__.py、vllm/engine/arg_utils.py: 新增 WatchdogConfig（timeout/check_interval/dump_dir）与 --watchdog-config 命令行参数
vllm/v1/engine/core.py: EngineCoreProc 创建/启动 engine watchdog，处理输入队列与每 step 前 feed，队列 get 改 timeout=5，shutdown handler dump 栈
vllm/v1/executor/multiproc_executor.py、vllm/distributed/device_communicators/shm_broadcast.py: WorkerProc 创建/启动 worker watchdog，每 method 前 feed，signal handler dump 栈
tests/config/test_watchdog_config.py、tests/utils_/test_watch_dog.py、tests/v1/engine/*、tests/v1/executor/test_multiproc_executor.py: 配置校验、watchdog 行为与 engine/worker 集成测试
```

```python
    config = WatchdogConfig()
    assert config.timeout == 300
    assert config.check_interval == 10
    assert config.dump_dir == ""
```

（摘自 `tests/config/test_watchdog_config.py`：校验默认 timeout 300s、check_interval 10s。）

## 性能声明

PR 正文未给出定量性能数据。正文 Test Plan & Result 为功能验证：Qwen3-8B TP1 与 TP2 两次 `vllm serve`（`--enforce-eager`、`--dtype float16`、`--max_model_len 20480`、`--no-enable-prefix-caching`、`--watchdog-config '{"timeout": 30, "check_interval": 5, "dump_dir": ...}'`），通过 patch 构造 feed timeout，日志显示 engine 与 worker 的 `VLLM_STACK_DUMP_for_*` 文件生成；无 benchmark 数字，PR 未注明性能测试条件。

## 验证与回归风险

PR 引入/依赖的测试：除正文两次 `vllm serve` 手工验证外，diff 新增/更新 `tests/config/test_watchdog_config.py`（默认值、自定义值、非法非正值与非整数校验）、`tests/utils_/test_watch_dog.py`、`tests/v1/engine/test_engine_core_shutdown_signal.py`、`tests/v1/engine/test_startup_watch_processes.py`、`tests/v1/executor/test_multiproc_executor.py`。PR 未注明回归风险；改动涉及 EngineCoreProc 输入队列 `get(timeout=5)`、shutdown signal handler、WorkerProc 每个 method 的 feed 与 `shm_broadcast`，可能影响 engine/worker 的启动、关闭与心跳响应路径，其他模型/硬件/精度正文未说明。

## 关联

正文标注 Fixes RFC #56816（https://github.com/vllm-project/vllm/issues/56816，issue #56816）；未给出其他 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
