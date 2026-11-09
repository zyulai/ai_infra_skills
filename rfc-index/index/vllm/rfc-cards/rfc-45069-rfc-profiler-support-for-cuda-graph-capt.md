# RFC #45069: [RFC]: Profiler support for CUDA graph capture tracing and roofline trace annotations

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#45069](https://github.com/vllm-project/vllm/issues/45069) |
| 作者 | devalshahamd |
| 创建日期 | 2026-06-09 |
| 状态 | ⚪ closed(关闭于 2026-10-10) |
| labels | RFC, stale |
| 热度 | 👍 0 · 💬 2 |
| 分类 | CUDA Graph / Compiler |

## 一句话结论
提议两个默认关闭的 ProfilerConfig 开关:导出 CUDA graph 捕获期 trace、在步骤注解中附加 roofline 聚合量,使单一 trace 自含分析所需统计。

## 提案动机
vLLM 的 PyTorch profiler 集成存在两个缺口:其一,使用 CUDA graphs(`cudagraph_mode=FULL`/`PIECEWISE`/`FULL_AND_PIECEWISE`)时,trace 记录的是 graph replay——有 GPU kernel 但无 CPU 侧调用栈和输入 shapes;这些信息只存在于 server 启动时的捕获阶段,profiler 今天看不到,导致常见生产配置下无法把 GPU 时间归因到 module/op/shape。其二,默认 execution-step 注解只编码请求数与 token 数(如 `execute_context_1(839)_generation_63(63)`),不含注意力 roofline 公式所需的 `Σ N_KV`、`Σ (N_Q·N_KV)`、`Σ (N_Q²)` 等 per-phase 聚合量,分析者无法直接从 trace 重建 arithmetic intensity 与 achieved vs. peak,只能重新跑详细日志或关联 trace 文件中不存在的 scheduler 状态。

## 提出的方案
- 新增两个 opt-in、默认关闭的 `ProfilerConfig` flag(原型见 PR #37524);torch profiler 未启用时均为 no-op,默认行为不变。
- Flag 1 `capture_torch_profiler_dir`(str,默认 `""`):设为目录后,`gpu_model_runner.py` 的捕获流程把 graph capture 包进 `torch.profiler` context(仅 rank 0),每张被捕获图(batch size × compilation mode)输出一份 trace;这些 capture trace 保留图内每个 op 的 CPU 调用栈与输入 shapes,下游工具可按 batch size/编译模式把 replay step 匹配回对应 capture trace,回填 shapes 与调用栈。
- Flag 2 `detailed_trace_annotation`(bool,默认 `False`):把每个执行步的 `torch.record_function` 注解扩展为携带 context(prefill)与 generation(decode)两相的 roofline 聚合量:`sq`=Σ N_Q、`sk`=Σ N_KV、`sqsq`=Σ (N_Q·N_Q)(prefill 因果自注意力校正项)、`sqsk`=Σ (N_Q·N_KV)(decode/chunked prefill 主导项);以 `user_annotation` 事件写入,roofline 分析可直接从单一 trace 完成。默认关闭时注解格式不变。

## 优化点与收益
正文未给出吞吐/加速类量化数字。收益体现在可分析性:配合 TraceLens 演示,attention perf model 用与 vLLM `perf.py` 相同的方程直接消费注解字段;示例(batch-size-32 FULL-graph decode step,`graph_full` test trace)输出中 `vllm::unified_attention_with_output` 为 0.90 GFLOPs、Data Moved 108.1 MB、FLOPS/Byte 7.96、占步时 6.8%,`vllm::rocm_unquantized_gemm` 为 0.94 GFLOPs、28.6 MB、31.4、3.6%;该步 perf 参数 `g_sq=32`、`g_sk=55086`、`g_sqsk=55086`——这些值只有 `detailed_trace_annotation` 开启后才存在于 trace 中,否则无法计算该步的 attention roofline。

## 现状与落地
已关闭(关闭于 2026-10-10),labels 含 stale。正文写明两个 flag 已在 PR #37524 中原型化,且注解格式沿用原型、仅按评审意见改进变量命名;未写明 PR 是否合并,落地情况需查关联 PR。

## 关联
- 原型 PR: https://github.com/vllm-project/vllm/pull/37524
- TraceLens: https://github.com/AMD-AGI/TraceLens/
- attention perf model 扩展: https://github.com/AMD-AGI/TraceLens/blob/main/TraceLens/PerfModel/extensions/attention_perf_model_extensions.py

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
