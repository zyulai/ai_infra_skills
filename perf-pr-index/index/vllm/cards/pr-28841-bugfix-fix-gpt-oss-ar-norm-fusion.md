# PR #28841: [Bugfix] Fix GPT-OSS AR+NORM fusion

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @elvischenv |
| 合并 | 2025-11-25 |
| 分类 | Model-specific Fast Path |
| Label | performance, ready, ci/build, gpt-oss, nvidia |
| Diff | diffstat 缺失(pack 未提供 +/− 统计,仅含源码 diff) |
| 链接 | https://github.com/vllm-project/vllm/pull/28841 |

## 一句话结论

GPT-OSS MoE 输出的 slice 移到 all_reduce 之前,恢复 AR+NORM 融合,TP8 吞吐 +3.8%。

## 动机

GPT-OSS + B200 上,MoE 输出按 padded 宽度(3072)出 all_reduce,再由 `aten.slice` 裁回 hidden_size(2880)。这条 slice 恰好夹在 all_reduce 与 rms_norm 之间,打断了 `flashinfer_trtllm_fused_allreduce_norm` 的融合 pattern 匹配,使 compile 后的图退化为独立的 AR 和 NORM。

## 优化点

在 `FusedMoE.forward_native` 的 `reduce_output` 中把 slice 提前:当 `hidden_size != og_hidden_states` 时先裁剪再做 all_reduce,使 compile 后图序变为 moe → slice → all_reduce+rms_norm 融合单 op(`launch_with_pdl=True`、`fp32_acc=True`);zero-expert 路径改为对 `zero_expert_result` 先 slice 再相加,shared_output/tp 路径同步去掉 reduce 后的 slice。附带修复:symm_mem all_reduce 中非连续输入的 `view(-1)` 改为 `reshape(-1)`。CI 侧为 `tests/compile/distributed/test_fusions_e2e.py` 增加 `openai/gpt-oss-20b` 用例(FLASHINFER backend,断言 allreduce_fusion=49、sequence_parallel=49、async_tp=48),并把 `fused_moe/layer.py` 纳入该 pipeline 的触发文件。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/layer.py: reduce_output 内 slice 前置,zero-expert/shared 路径同步调整
vllm/distributed/device_communicators/symm_mem.py: 非连续输入 view(-1) → reshape(-1)
tests/compile/distributed/test_fusions_e2e.py: 新增 gpt-oss-20b 融合计数用例
.buildkite/test-pipeline.yaml: fused_moe/layer.py 加入触发文件
```

```python
def reduce_output(states: torch.Tensor) -> torch.Tensor:
    # Slice before all_reduce to enable possible fusion
    if self.hidden_size != og_hidden_states:
        states = states[..., :og_hidden_states]
```

## 性能声明

Serving benchmark(TP8, concurrency 8, 40 请求, in/out 各 1024 token):PR 声明 **3.8% improvement**——output token throughput 3,644.69→3,779.37 tok/s,request throughput 3.56→3.69 req/s,Mean TPOT 2.14→2.06 ms,Mean E2EL 2,245.97→2,165.61 ms;Mean TTFT 54.18→54.30 ms 基本持平。Accuracy:aime25(`gpt-oss-20b`,medium reasoning,temp 1.0):PR 0.7167 vs main 0.7375(低约 2.1 个百分点,PR 未解释差异)。

## 验证与回归风险

验证为融合 e2e 测试新增用例与上述 serve/aime 评测。风险:slice 提前使 all_reduce 收到的张量宽度从 padded 变为真实 hidden_size,依赖 AR 实现支持该宽度;改动位于共享的 `FusedMoE.forward_native`,所有 hidden_size 不等于 padded 宽度的 MoE 模型图结构均受影响;aime25 精度低于 main 约 2.1pp 且 PR 未给出分析,属未解释的回归信号。

## 关联

无(正文未引用其他 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
