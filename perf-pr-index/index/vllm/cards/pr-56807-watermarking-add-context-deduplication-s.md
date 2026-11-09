# PR #56807: [watermarking] add context deduplication support to speculative decoding

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @TQCB |
| 合并 | 2026-09-30 |
| 分类 | Speculative Decoding (tier: major) |
| Label | documentation, performance, speculative-decoding, mrv2, dflash |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files);diff 可用但被截断(7 个文件未收录) |
| 链接 | https://github.com/vllm-project/vllm/pull/56807 |

## 一句话结论

把水印的上下文去重扩展到投机解码,并将 draft 采样融合为单次词表归约:融合路径相对两遍采样 1.59–1.60×。

## 动机

此前 `deduplicate_contexts != "none"` 与投机解码不兼容,vLLM 会警告"Context deduplication is not supported with speculative decoding"并静默放弃——接受 draft、rejection-recovery 与 bonus token 都不做去重,水印抗去除性在 spec decode 场景退化。同时,带水印的 draft 路径此前是"普通 draft 采样 + 写 cache → 水印采样 → torch.where 选择"的两遍流程,采样段存在可融合的冗余词表归约。

## 优化点

机制分两条线。功能:目标验证上下文由已提交 token 与提议的 draft token 共同构建;`repeated_context_mask` 的 Triton kernel 扩展 `HAS_HISTORY_OFFSETS`/`SPECULATIVE_CONTEXTS`/`DRAFT_CONTEXTS` 三个 constexpr 模式——前者把未提交位置计入 `deduplicate_contexts_max_history`,后两者分别比对同一 spec block 内更早位置的上下文与 `prior_contexts` 中同请求更早 draft step 的上下文;重复上下文处用普通 draft 提议,随后照常运行不变的 `min(1, q/p)` 接受测试(不强制拒绝),rejection-recovery 与 bonus 上下文用现有残差/目标分布的普通抽样;禁用与 greedy 请求走原路径。性能:committed-history 与 speculative-local 上下文检查融合进同一 kernel,避免 launch 密集的 PyTorch 记账;普通与水印 draft 采样融合进一次词表归约(gumbel/philox kernel 增加 `logits_cache_source`,在 `IS_DRAFTING` 下写 cache 时直接取源 logits,保持 pre-temperature draft-logit cache 与 draft 专属 RNG 流位精确一致);`DraftWatermarker.sample` 直接接管 speculator 的采样入口。

## 关键改动

```text
vllm/v1/worker/gpu/sample/watermark.py: _repeated_context_mask_kernel 扩展三种去重模式,新增 draft_watermarking_mask
vllm/v1/watermarking/spec_decode.py: DraftWatermarker 携带去重配置,prepare/sample 重构为融合路径
vllm/v1/worker/gpu/sample/gumbel.py: gumbel_sample/philox kernel 支持 is_drafting 与 logits_cache_source
vllm/v1/worker/gpu/spec_decode/rejection_sampler.py: _watermarking_kwargs 计算 watermarking_skip_mask
vllm/v1/worker/gpu/spec_decode/dspark/speculator.py: _sample_logits 直接走 draft_watermarker.sample
```

kernel 模式开关:

```python
HAS_HISTORY_OFFSETS=history_offsets is not None,
SPECULATIVE_CONTEXTS=local_positions is not None,
DRAFT_CONTEXTS=False,  # draft_watermarking_mask 中为 True
```

## 性能声明

H100 微基准(隔离 eager 执行,fp16 logits、词表 32768,含 temperature 预处理与 draft-logit cache 写入,五次 1000 轮测量取中位):对比"普通 draft 采样+cache 写、水印采样、torch.where"的两遍路径,融合混合路径 bs=1:91.31→57.40 µs(1.59×,每 draft step 省 33.91 µs);bs=8:92.96→58.14 µs(1.60×);bs=32:92.62→58.13 µs(1.59×)。PR 明确说明这是隔离微基准,不是端到端吞吐。

## 验证与回归风险

测试:`tests/watermarking/test_watermarking.py` 51 passed, 18 skipped(本地);水印与 Gumbel 套件在 H100 上 95 passed;融合 draft 采样与旧选择位精确一致(含 draft RNG 流与 pre-temperature logits cache);CPU 参考实现与 Triton kernel 有 parity 测试;Ruff/formatting/mypy 等钩子通过。风险面:去重上下文的重复判定会影响 spec decode 下哪些 token 走普通采样(验证上下文与 skip mask 计算错误会改变接受率与水印强度);`logits_cache_source` 引入新的缓存写源路径,需与既有 cache 列语义对齐;对未开启水印或去重为 "none" 的请求无行为变化。

## 关联

同分类相关:pr-32642(spec decode min_tokens)、pr-14930(rejection sampling 优化)、pr-13365(n-gram matching);labels 中的 dflash/mrv2 表示该实现位于 MRV2 dSpark speculator 路径。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
