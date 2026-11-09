# PR #32846: [Kernel] use flashinfer for gdn prefill

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ZJY0516 |
| 合并 | 2026-02-09 |
| 分类 | Attention / Kernel (tier: major) |
| Label | performance, ready, qwen |
| Diff | 素材未提供 diffstat(pack 记录为 +None/-None across None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/32846 |

## 一句话结论

Qwen3-Next GDN(gated delta net)prefill 换用 FlashInfer 新内核,Qwen3-Next-80B-A3B 上 mean TTFT 降约 18%,吞吐提升约 7%。

## 动机

FlashInfer 上游新增了 GDN prefill 内核(flashinfer-ai/flashinfer#2276),而 vLLM 中 Qwen3-Next 的 prefill 路径此前走的是自带的 FLA(`vllm/model_executor/layers/fla`)`chunk_gated_delta_rule` 实现。对长输入(4K input)场景,TTFT 主要受 prefill 内核性能支配,因此值得切换到更快的 FlashInfer 实现。

## 优化点

改动机制是在 `qwen3_next.py` 中新增 `ChunkGatedDeltaRule` 这一 `CustomOp`,把原来直接函数调用的 `chunk_gated_delta_rule` 改为经 CustomOp 分发:CUDA 且 compute capability 90(Hopper)时走 `forward_cuda` 调用 FlashInfer 的 `flashinfer.gdn_prefill.chunk_gated_delta_rule`,其他情况回退到原 FLA 实现,保证非 Hopper 平台行为不变。FlashInfer 路径做了适配:先对 q/k 做 `l2norm_fwd`,各张量 `squeeze(0)` 后转 contiguous,并把 state、g(取 `exp(g)`)、beta 转成 float32 以匹配 FlashInfer 接口约定。

## 关键改动

```text
vllm/model_executor/models/qwen3_next.py: 新增 ChunkGatedDeltaRule CustomOp 与 fi_chunk_gated_delta_rule 适配函数;_forward_core 改为经 CustomOp 调用
```

关键摘录(Hopper 分发逻辑):

```python
if current_platform.is_cuda() and current_platform.is_device_capability(90):
    logger.info_once(
        "Using FlashInfer GDN prefill kernel on CUDA compute capability 90"
    )
    self._forward_method = self.forward_cuda
else:
    self._forward_method = self.forward_native
```

## 性能声明

PR 正文给出的 serving benchmark(`vllm bench serve`,模型 Qwen/Qwen3-Next-80B-A3B-Instruct,512 条请求,random input len 4096 / output len 512;未注明硬件与精度):

| 指标 | main | 本 PR |
| --- | --- | --- |
| Benchmark duration (s) | 107.53 | 100.25 |
| Request throughput (req/s) | 4.76 | 5.11 |
| Output token throughput (tok/s) | 2437.78 | 2615.02 |
| Total token throughput (tok/s) | 21940.02 | 23535.15 |
| Mean TTFT (ms) | 42234.71 | 34522.29 |
| Median TTFT (ms) | 40749.87 | 33466.90 |
| P99 TTFT (ms) | 78822.57 | 71442.06 |
| Mean TPOT (ms) | 119.60 | 120.45 |

即 mean TTFT 从 42234.71 ms 降至 34522.29 ms(约 -18.3%),request throughput 4.76 → 5.11 req/s(约 +7.4%);TPOT/ITL 基本持平。TPOT 未改善与改动仅覆盖 prefill 路径一致。

## 验证与回归风险

作者附了 lm_eval gsm8k(5-shot,`-tp 4` 服务,280 并发)对比:main flexible-extract 0.8484 / strict-match 0.6376,本 PR flexible-extract 0.8453 / strict-match 0.6376,flexible-extract 有 0.003 的轻微波动。回归风险集中在:仅 Qwen3-Next 模型路径受影响;非 Hopper 与非 CUDA 平台走原 FLA 路径不受影响;FlashInfer 适配层中 float32 转换与 `exp(g)` 的数值行为依赖 FlashInfer 接口语义。卡内未看到新增单元测试。

## 关联

同属 Qwen3-Next / GDN(gated delta rule)内核路径优化;关联上游 FlashInfer PR flashinfer-ai/flashinfer#2276。本批素材中未提供其他同区域 PR 编号,暂无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
