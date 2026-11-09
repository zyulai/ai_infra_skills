# PR #64479: [XPU][Perf] Reuse GDN metadata across KV cache groups on XPU

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @gc-fu |
| 合并 | 2026-11-04 |
| 分类 | KV Cache / Memory \| tier: major |
| Label | intel-gpu, verified |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/64479 |

## 一句话结论

XPU 跨 KV cache group 复用 GDN metadata，3 组省 2/3 build，Qwen3.5-0.8B 吞吐配对最高 +6.1%。

## 动机

PR Overview 指出，Model Runner V2 已有跨 KV cache group 复用 GDN attention metadata 的路径（#58762），但复用路径及其依赖的 mamba `align` 模式下 aligned state index 预计算都仅限 CUDA，XPU 被两处设备检查排除。混合 GDN 模型（Qwen3.5 / Qwen3.6 家族）每步会为每个 KV cache spec 构建一次 GDN metadata，存在按组的重复构建。本 PR 在两处检查中放行 XPU，使其复用 CUDA 已有的机制。

## 优化点

仅放宽两处设备检查，不改变算法：`GDNAttentionMetadataBuilder.supports_update_block_table` 由 `device.type == "cuda"` 改为 `("cuda","xpu")`；`compute_aligned_state_indices()` 的断言由 `seq_lens.is_cuda` 改为 `is_cuda or is_xpu`。放行后 XPU 在 MRV2 路径下对每个 KV cache spec 每步只 `build()` 一次 GDN metadata，其余 group 通过 `update_block_table()` 重新 gather 状态索引（与 CUDA 一致）；对应 3 个 GDN KV cache group 时每步 3 次构建中有 2 次被替换。aligned-index 预计算是 Triton kernel，在 XPU 上原样运行，PR 强调输出不变。限制：仅 XPU、仅 eager mode，graph mode 未验证，CUDA 行为不变。

## 关键改动

```text
vllm/v1/attention/backends/gdn_attn.py: supports_update_block_table 设备判断由 "cuda" 放宽为 ("cuda","xpu")。
vllm/v1/worker/mamba_utils.py: compute_aligned_state_indices() 断言放宽为允许 is_xpu。

  self.supports_update_block_table = (
      vllm_config.use_v2_model_runner
      and device.type in ("cuda", "xpu")
      and type(self) is GDNAttentionMetadataBuilder
  )
  ...
  -        assert seq_lens.is_cuda
  +        assert seq_lens.is_cuda or seq_lens.is_xpu
```

## 性能声明

测试条件（Validation）：Intel Arc Pro B70，MTP `num_speculative_tokens=3`，eager mode，prefix caching 开（mamba `align` 模式），greedy decoding、`logprobs=1`，baseline 为未改动 main，GDN metadata 调用在 worker 内计数。Metadata 调用：Qwen3.5-0.8B（TP=1, bf16, B1+B8）`build()` 819 → 273、`update_block_table()` 0 → 546，tokens 与 logprobs 在 5 个进程间完全一致；Qwen3.8-27B-FP8（TP=2, B1+B8, per worker）`build()` 2133 → 711、`update_block_table()` 0 → 1422，tokens 在 4 个进程间一致，B8 logprobs 一致、B1 最大 logprob 差 0.093（baseline-vs-baseline 0.075）。吞吐（Qwen3.5-0.8B，128 输入/64 输出 tokens，每进程 5 次取中位数，A-B-B-A 顺序）：Batch 1 由 183.5 / 181.1 → 189.4 / 192.1 tok/s（配对 +3.2% / +6.1%），Batch 8 由 1329.9 / 1342.0 → 1404.9 / 1399.1 tok/s（配对 +5.7% / +4.3%）。PR 说明进程内 run-to-run 波动最高 5%，Qwen3.8-27B-FP8 吞吐差异在 run-to-run 噪声内。

## 验证与回归风险

`pytest tests/v1/attention/test_gdn_metadata_builder.py tests/v1/worker/test_mamba_utils.py` → 70 passed、36 skipped（现有 CUDA-only 用例）；changed-file pre-commit（含 mypy）通过。PR 明确未新增测试：现有覆盖 `update_block_table()` 的 builder 测试与设备无关、可通过，precompute 数值无既有单元测试；端到端对比已在 XPU 覆盖两处改动。风险：改动仅放宽两处设备检查、仅 XPU、仅 eager mode，graph mode 未验证；CUDA 行为不变，输出不变（tokens/logprobs 一致）。

## 关联

PR 正文明确提及：#58762（MRV2 中跨 KV cache group 复用 GDN attention metadata 的前序实现，原本仅 CUDA）、#58851（把共享移入 builder 并去掉共享本身的 CUDA 检查，但保留 aligned-index precompute 仅 CUDA；本 PR 与其独立，且 `compute_aligned_state_indices()` 对 XPU `align` 模式仍然必要）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
