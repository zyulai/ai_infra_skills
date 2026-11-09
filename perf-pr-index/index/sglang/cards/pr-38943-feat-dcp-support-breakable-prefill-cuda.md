# PR #38943: [Feat] dcp: Support breakable prefill CUDA graphs under decode context parallelism

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @klshuster |
| 合并 | 2026-11-01 |
| 分类 | Speculative Decoding(pack 分类;实际为 prefill CUDA graph × Decode Context Parallelism 特性) |
| Label | run-ci, run-ci-extra, bypass-fail-fast |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 统计;diff 内容本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/38943 |

## 一句话结论

让 breakable prefill CUDA graph 在 DCP(--dcp-size>1)下可用:短 prompt TTFT 约 404ms 降到 133ms,输出吞吐 +14%。

## 动机

`--cuda-graph-backend-prefill breakable` 与 `--dcp-size > 1` 组合在捕获阶段崩溃:dummy extend batch 没有 `attn_dcp_metadata`,MLA extend 路径在 `all_gather_kv_cache_for_mla_extend` 里直接解引用它。auto-disable 规则只覆盖默认 backend,显式选择 breakable(如文档化的 Kimi-K3 配方)会启动失败,DCP 部署只能让 prefill 走 eager。

## 优化点

核心约束是 breakable graph 的 break 函数按捕获期参数 replay,per-batch 状态必须来自稳定来源而非捕获时传入的 batch 对象。三处机制:(1) `forward_mla.py` 把 DCP extend 的 KV 组装改成 `eager_on_graph` break `bcg_dcp_extend_kv_gather`,graph 内从 forward context 读 live batch;当 `dcp_extend_prefix_lens_sum == 0`(无缓存前缀)直接把本地 `k_nope`/`k_pe` 拷进 gather buffer 并跳过集合通信,有前缀命中才保留 all-gather;(2) `prefill_cuda_graph_runner.py` 捕获时为 zero-prefix batch 规划 DCP metadata,并把每个 shape bucket 一个的 gather buffer 存进 `_dcp_kv_buffers[shape_key]`;replay 时 eager 地规划 live batch 的 indices,但把指针指到捕获的 bucket-sized buffer;(3) `can_replay_locally` 增加守卫:DCP 下仅 fresh prefill 走 graph replay,prefix-hit extend(含 chunked-prefill 续段)回退 eager(跨 rank all-gather 形状随 batch 变化),prefix-hit replay 留作 follow-up。

## 关键改动

```text
python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py: DCP extend KV 组装改为 eager break bcg_dcp_extend_kv_gather
python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py: 捕获规划 DCP metadata、按 shape 保存 gather buffer、replay 复用
test/registered/unit/model_executor/test_prefill_cuda_graph_dcp.py: 新增 admission gate / replay buffer 契约 / live-batch gather 三类单测
```

关键片段(无前缀时跳过集合通信):

```python
if metadata.dcp_extend_prefix_lens_sum == 0:
    # No prefix KV on any DCP rank: skip the collective.
    metadata.dcp_kv_buffer[..., :kv_lora_rank] = k_nope
    metadata.dcp_kv_buffer[..., kv_lora_rank:] = k_pe
    return
```

## 性能声明

PR 正文:
- Prefill TTFT(sequential、单输出 token、median ms,disabled → breakable):22 tokens:404 → 133;301:454 → 155;1142:461 → 171;1996:486 → 177;4437(超出最大 bucket):460 → 460。
- 短 prompt rolling benchmark:TPOT 38.0 → 33.6 ms,输出吞吐 +14%,decode 吞吐不变。
- 捕获开销:42 buckets 约 70s,每 GPU 3.6-4.8 GB。
- 测试硬件/模型(Kimi-K3,cutedsl_mla,--dcp-size 8,2 节点)在 Accuracy 段给出;TTFT/rolling benchmark 的具体模型与硬件配置 PR 未另行注明。

## 验证与回归风险

精度验证:Kimi-K3、cutedsl_mla、--dcp-size 8、2 节点,与关闭 prefill graph 的同配置服务对比——graph replay 确定性(重复 fresh prefill 逐位一致),graph-vs-eager logprob 差异不超过 eager 服务自身的 batch 组成波动,标准精度评测与配对 agentic coding benchmark 均与 baseline 持平;另用 draft-model 投机解码与 1M token context 验证。新增单测在 main 上会失败(覆盖回归)。风险面:auto-disable 规则未改,需显式选 breakable;DCP 下 prefix-hit 与 chunked-prefill 续段仍走 eager,收益集中在 fresh prefill;改动触及所有 MLA 模型的 extend 路径(`bcg_dcp_extend_kv_gather` 替换原直接调用),非 DCP 场景语义不变但路径多一层包装。PR CI:Base/Extra 通过,AMD ROCm 10 失败(x)。

## 关联

DeepSeek/MLA attention 区域:PR #42957([DSA] TileLang FP8 KV cache)、PR #34837([AMD] aiter prefill kernel)。场景出处:Kimi-K3 DCP 配方(文档化 recipe)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
