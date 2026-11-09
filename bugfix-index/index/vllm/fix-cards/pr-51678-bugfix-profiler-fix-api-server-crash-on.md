# PR #51678: [Bugfix][Profiler] Fix API server crash on double /stop_profile

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#51678](https://github.com/vllm-project/vllm/pull/51678) |
| 作者 | aijanai |
| 合并日期 | 2026-09-01 |
| labels | bug, ready, verified |
| 热度 | 👍 2 · 💬 20 |
| 分类 | Frontend / Serving |
| bug 类型 | crash |

## 一句话结论

对 /stop_profile 调用第二次(尤其并发)会使 API server 段错误;AsyncLLM 改用 TorchProfilerWrapper 统一管理,二次 stop 变为 no-op。

## 问题现象

以 torch profiler 启动 vLLM(如 `vllm serve Qwen/Qwen3.5-0.8B --profiler-config '{"profiler":"torch","torch_profiler_dir":"vllm-prof-test/"}' ...`),`POST /start_profile` 后产生一些负载,再对 `POST /stop_profile` 调用第二次(两次调用甚至可以并发发起,第二次即触发段错误)。修复前结果序列为 200 / 200 / 500(trace 较大时为段错误);修复后为 200 / 200 / 200,第二次 stop 是 no-op。

## 根因

正文仅写「Fixes #51676」,未披露根因。(由 diff 推断)此前 `AsyncLLM.__init__` 在启用 profiler 时直接构造裸的 `torch.profiler.profile`,未启用时 `self.profiler = None`;重复 `/stop_profile` 的崩溃路径与此裸 profile 对象的重复停止/未定义状态有关。本 PR 把 `AsyncLLM` 的 profiler 换成 `vllm/profiler/wrapper.py` 的 `TorchProfilerWrapper` 统一封装(由 diff 推断:由该 wrapper 吸收二次 stop 使其成为 no-op),但 wrapper 自身的实现不在本 diff 中,确切机制以 #51676 与 wrapper 代码为准。

## 修复方案

改 `vllm/v1/engine/async_llm.py`:

- `AsyncLLM.__init__` 新增 `profiler: Optional[TorchProfilerWrapper]` 参数。
- 启用 torch profiler 时,不再内联构造 `torch.profiler.profile(activities=[CPU], with_stack=..., on_trace_ready=tensorboard_trace_handler(...))`,改为:

```python
self.profiler = TorchProfilerWrapper(
    vllm_config.profiler_config,
    worker_name=worker_name,
    local_rank=0,
    activities=["CPU"],
)
```

- 删除原来的 `else: self.profiler = None` 分支。

## 影响与回归风险

- 影响窗口:正文未给出具体版本/提交范围。
- 回归面:改动集中在 V1 `AsyncLLM` 的 profiler 初始化路径。所有通过 `--profiler-config` 启用 torch profiler 的部署,其 profiler 生命周期管理改由 `TorchProfilerWrapper` 承担;wrapper 的 `with_stack`、trace handler、gzip 等配置行为需与原内联构造等价(diff 中未展示 wrapper 实现,此处依赖其正确性);`self.profiler` 不再为 `None`,任何检查 `profiler is None` 的调用方行为可能受影响;`AsyncLLM.__init__` 新增参数对直接构造 AsyncLLM 的调用方有签名变化。

## 关联

- #51676(修复的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
