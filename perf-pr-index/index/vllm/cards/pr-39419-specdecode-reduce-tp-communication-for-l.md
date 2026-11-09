# PR #39419: [SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @EanWang211123 |
| 合并 | 2026-06-10 |
| 分类 | Speculative Decoding \| tier: major |
| Label | speculative-decoding, ready, v1, llama, qwen, deepseek, verified |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/39419 |

## 一句话结论

投机解码改用 local argmax + gather 替代全词表 all-gather，吞吐最高 +30.63%。

## 动机

正文指出 DFlash、PARD 等投机解码方法常要求 draft 与 target 模型词表大小一致；当前实现为此需在每一步跨 TP ranks all-gather 完整 logits 张量，通信成本为 O(batch_size × vocab_size)。例如 PARD-Qwen3-0.6B 的词表大小为 40960，大词表下通信开销显著。

## 优化点

在 `speculative_config` 中新增 `use_local_argmax_reduction` 选项，改用 local argmax + gather reduction：每个 TP rank 只在自己的 vocab shard 上算 local argmax，仅 gather `(max_value, global_index)` 对，通信量从 O(batch_size × vocab_size) 降为 O(batch_size × 2 × tp_size)，再跨 ranks reduce 出 global argmax。该方法复用 PR #34049 引入的思路。改动方式是在对应模型文件中新增 `get_top_tokens()`。

## 关键改动

```text
vllm/v1/spec_decode/llm_base_proposer.py: 投机解码 proposer 侧接入 local argmax + gather reduction
vllm/model_executor/models/interfaces.py: 新增 get_top_tokens() 接口约定
vllm/model_executor/models/qwen3.py: Qwen3 系列实现 get_top_tokens()
vllm/model_executor/models/deepseek_eagle3.py: DeepSeek EAGLE3 实现 get_top_tokens()
vllm/model_executor/models/llama.py: Llama 系列实现 get_top_tokens()
```

## 性能声明

正文给出 Total token throughput (tok/s) 的 gains：Qwen3-8B + Qwen3-8B-DFlash + tp2 并发 1 为 +9.38%，同配置并发 32 为 +10.21%；Qwen3-8B + Qwen3-0.6B-PARD + tp2 并发 32 为 +11.54%；Qwen3-8B + Qwen3-8B-DFlash + tp4 并发 32 为 +30.63%。表格对应吞吐：199.25 → 217.94、1407.37 → 1551.11、2747.77 → 3064.77、1338.82 → 1748.92 tok/s；Mean TTFT 分别 179.88 → 98.47 ms、885.07 → 558.23 ms、323.18 → 197.74 ms、1366.44 → 479.53 ms。测试条件：环境 4090D，Target model Qwen3-8B，Draft models Qwen3-8B-DFlash-b16 与 PARD-Qwen3-0.6B，Dataset gsm8k，tp2 / tp4，并发 1 / 32；精度 PR 未注明。

## 验证与回归风险

Accuracy：正文说明 "Enabling `use_local_argmax_reduction` does not change the acceptance rate"，并给出 DFlash 并发 1 下 true / false 两组完全一致的 per-position acceptance（Position 0 均为 69.05%，逐位相同）。Test Plan 给出两条 serve 命令（分别 method=dflash 与 method=draft_model / parallel_drafting，均带 `use_local_argmax_reduction: true`）。PR 未注明新增或依赖的自动化单元测试；Open Question 提到几乎所有使用 draft model 的投机解码配置似乎都能受益，但正文未说明对不支持该选项的路径的回归评估。

## 关联

正文注明复用 PR #34049 引入的方法。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
