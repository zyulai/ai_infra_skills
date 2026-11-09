# PR #20441: Fix Piecewise CUDA Graph crash with `-enable-mixed-chunk`

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#20441](https://github.com/sgl-project/sglang/pull/20441) |
| 作者 | jianyingzhu |
| 合并日期 | 2026-03-28 |
| labels | run-ci |
| 热度 | 👍 1 · 💬 14 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | crash |

## 一句话结论
mixed-chunk 下 forward_mode=MIXED 与 PCG 捕获时的 EXTEND 不一致触发重编译崩溃,replay 时将 MIXED 归一化为 EXTEND 修复。

## 问题现象
以 `--enable-mixed-chunk` 与 Piecewise CUDA Graph(PCG)同时运行时发生 CUDA graph replay 崩溃。复现命令:`python -m sglang.launch_server --model /models/Qwen3-8B --trust-remote-code --host 0.0.0.0 --port 30000 --enable-mixed-chunk --chunked-prefill-size 8192 --max-running-requests 256 --enable-metrics --attention-backend flashinfer`,报 `AssertionError: PCG capture stream is not set`(截图见原帖)。

## 根因
PR 正文披露:开启 mixed-chunk prefill 后,scheduler 为运行时 batch 设置 `forward_mode=ForwardMode.MIXED`,而 PCG 图总是在 `ForwardMode.EXTEND` 下捕获;TorchDynamo 对 `forward_mode` 的严格 guard 因此失配,触发计划外的运行时重编译。重编译期间会再次尝试 CUDA graph capture,但 `get_pcg_capture_stream()` 在专用 PCG 捕获阶段之外返回 None,于是抛出 `AssertionError: PCG capture stream is not set`。

## 修复方案
改动位于 `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` 的 `replay_prepare()`:构造 `static_forward_batch` 前,把 `forward_mode` 与 `global_forward_mode` 中的 `MIXED` 归一化为 `EXTEND`,使 replay 时的 forward_mode 与捕获时一致,从而命中预捕获图而不是进入无法安全重捕获的重编译路径。

关键 diff 摘录:

```python
pcg_forward_mode = (
    ForwardMode.EXTEND
    if forward_batch.forward_mode == ForwardMode.MIXED
    else forward_batch.forward_mode
)
pcg_global_forward_mode = (
    ForwardMode.EXTEND
    if forward_batch.global_forward_mode == ForwardMode.MIXED
    else forward_batch.global_forward_mode
)
static_forward_batch = ForwardBatch(
    forward_mode=pcg_forward_mode,
    ...
    global_forward_mode=pcg_global_forward_mode,
```

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围,未写。
- 回归面:仅影响 PCG replay 的 `replay_prepare()` 中 static batch 的模式字段。MIXED batch 在 replay 时被视为 EXTEND;MIXED 与 EXTEND 共享捕获图的前提是两者在模型前向中行为一致(由 diff 推断,正文以「align replay-time forward_mode with capture-time mode」论证)。decode 等其他模式不受影响。正文验证:开启 `--enable-mixed-chunk` 后不再触发 TorchDynamo 重编译或该 AssertionError,输出结果正确。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
