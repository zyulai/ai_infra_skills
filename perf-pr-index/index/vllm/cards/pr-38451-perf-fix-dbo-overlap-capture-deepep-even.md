# PR #38451: [Perf] Fix DBO overlap: capture DeepEP event before yield

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @czhu-cohere |
| 合并 | 2026-03-31 |
| 分类 | MoE (tier: major) |
| Label | ready |
| Diff | diffstat 缺失(pack 未提供 +None/-None across None);可见 diff 仅含 deepep_ht.py 一个文件的 4 行级改动 |
| 链接 | https://github.com/vllm-project/vllm/pull/38451 |

## 一句话结论

DBO 双 batch overlap 中把 DeepEP event 捕获移到 yield 之前,dispatch/combine 不再等整条 compute stream,prefill 吞吐提升约 30%。

## 动机

从 profiling trace 看 DeepEP HT backend 没有充分 overlap 计算与通信:只有一个 ubatch 的 combine 与另一个 ubatch 的非 MoE 计算重叠。根因是传给 DeepEP dispatch/combine 的 previous_event:实现上 DeepEP 会等 event 覆盖的整条 compute stream 结束才发起 dispatch,而 event 是在 `dbo_yield_and_switch_from_compute_to_comm()` yield 之后才捕获的,此时 compute stream 尾部已排入另一个 ubatch 的计算,当前 ubatch 并不依赖它们,overlap 被人为串行化。

## 优化点

两处(均在 deepep_ht.py):_do_dispatch 中把 `dbo_get_previous_event(self.buffer.capture)` 从 yield 之后移到 yield 之前,使 event 只覆盖当前 ubatch 的 compute 尾部,DeepEP 内部依赖关系仍被正确保留;_finalize(combine 路径)做同样前移。核心机制:event 捕获点决定 DeepEP 等待的 stream 范围,"先捕获后 yield"才能既保住当前 ubatch 的依赖、又不把另一个 ubatch 的无关计算圈进来。依赖语义不变,纯调度顺序修正。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py: _do_dispatch/_finalize 中 previous_event 捕获移到 dbo_yield_and_switch_from_compute_to_comm() 之前
```

## 性能声明

prefill 吞吐测试:Qwen/Qwen3-30B-A3B-Instruct-2507-FP8,4x B200(TP1、DP4、enable-expert-parallel),all2all-backend deepep_high_throughput,load-format dummy,random 8192 in/1 out,max-concurrency 64、640 prompts(VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random)。Total token throughput 98524.82 -> 130017.79 tok/s,Request throughput 12.03 -> 15.87 req/s,Mean TTFT 5076.72 -> 3851.45 ms;作者总结约 30% 吞吐提升。正确性:lm-eval gsm8k(5-shot)before flexible-extract 0.8870 / strict-match 0.8772,after 0.8878 / 0.8779(同机同配置)。

## 验证与回归风险

Test Plan 即上述 prefill throughput 与 lm-eval 两项,无新增单测。回归面:仅 DBO(dual-batch overlap)与 DeepEP HT dispatch/combine 路径;改动只影响 event 捕获时机,若 DeepEP 侧对 previous_event 语义有其他假设需复核;非 DBO 或非 deepep_ht 路径不受影响。yield 顺序本身未变,dispatch kernel 仍在 queue 完另一 ubatch 计算后才启动。

## 关联

同区域:DBO(dual-batch overlap)调度与 DeepEP all2all 相关优化(正文与 diff 未给出具体 PR 编号)。
