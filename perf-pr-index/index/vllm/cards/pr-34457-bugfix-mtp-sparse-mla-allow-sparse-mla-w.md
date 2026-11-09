# PR #34457: [Bugfix][MTP][Sparse MLA] Allow sparse MLA with MTP to run with FULL cudagraphs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @MatthewBonanni |
| 合并 | 2026-02-17 |
| 分类 | Speculative Decoding(tier: major) |
| Label | bug, documentation, performance, ready, v1, nvidia |
| Diff | diffstat 缺失(素材未提供 +/− 统计);diff 素材完整可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/34457 |

## 一句话结论

DeepSeek-V3.2 sparse MLA + MTP 开启 FULL cudagraph,输出吞吐 943→2116 tok/s(约 2.2 倍)。

## 动机

DeepseekV32IndexerMetadataBuilder 只声明支持 UNIFORM_SINGLE_TOKEN_DECODE,因此 sparse MLA 模型开 MTP 时永远不会捕获 FULL cudagraph,decode 走 eager 路径。实际上底层 DeepGEMM kernel fp8_paged_mqa_logits 支持 MTP 的 num_speculative_tokens=1(next_n=2)场景,是元数据层的声明过于保守。

## 优化点

把 _cudagraph_support 从 UNIFORM_SINGLE_TOKEN_DECODE 改为 UNIFORM_BATCH,reorder_batch_threshold 直接加上 num_speculative_tokens;对 num_speculative_tokens > 1 抛出显式 ValueError(kernel 不支持 next_n > 2),而不是等到 kernel 崩溃。另外在 build 时对 decode block_table 做 clamp_(min=0):cudagraph padding 请求的 block_table 条目为 -1,直接进 DeepGEMM kernel 会越界,padding 请求 seq_lens=0 故 clamp 安全。

## 关键改动

```text
vllm/v1/attention/backends/mla/indexer.py: _cudagraph_support 改 UNIFORM_BATCH;next_n>2 显式报错;block_table clamp
docs/design/cuda_graphs.md: 全 cudagraph 支持表新增 FlashInferMLASparse = UNIFORM_BATCH
```

```python
_cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
...
if self.num_speculative_tokens > 1:
    raise ValueError("Sparse MLA only supports num_speculative_tokens <= 1 ...")
self.reorder_batch_threshold += self.num_speculative_tokens
...
block_table.clamp_(min=0)  # padded CUDA graph requests have -1 entries
```

## 性能声明

DeepSeek-V3.2,-tp 8 -ep,MTP num_speculative_tokens=1,无 prefix caching;vllm bench serve random,输入 128 / 输出 2048,200 prompts,max-concurrency 64,seed 42:

| 指标 | main | 本 PR |
| --- | --- | --- |
| 输出 token 吞吐 | 943.04 tok/s | 2116.20 tok/s |
| 总 token 吞吐 | 1001.98 tok/s | 2248.46 tok/s |
| Benchmark 时长 | 434.34 s | 193.55 s |
| Mean TPOT | 54.12 ms | 24.73 ms |
| Mean ITL | 107.91 ms | 49.27 ms |
| Acceptance rate / length | 99.43% / 1.99 | 99.26% / 1.99 |

## 验证与回归风险

测试为上述端到端 serving benchmark(含 spec decode acceptance 统计)。风险:num_speculative_tokens>1 从静默运行(或 kernel 崩溃)变为启动报错,属行为收紧;cudagraph padding 路径依赖 clamp 的正确性(注释论证 seq_lens=0 时 kernel 输出无意义故安全);仅影响 FlashInferMLASparse / DeepSeek V3.2 indexer 路径。

## 关联

同分类 Speculative Decoding 卡:#59082(CPU MTP 调度)、#63873(bench 投机状态)、#32542(Pixtral Eagle3)。依赖的 kernel 能力约束:DeepGEMM fp8_paged_mqa_logits 仅支持 next_n <= 2(PR 正文注明)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
