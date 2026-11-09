# PR #25763: [ROCm][Perf] New design on ROCm AITER MHA backend Implementation

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ganyi1996ppo |
| 合并 | 2025-11-04 |
| 分类 | Attention / Kernel（tier: major） |
| Label | rocm, ready, ci/build, v1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 未标注 dropped 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/25763 |

## 一句话结论

ROCm AITER MHA 后端按 decode/cp/pp 重排并重写 KV fetch，Qwen3/Mi308 长提示吞吐约 4~5x。

## 动机

PR 正文指出，旧的 `AiterFlashAttentionImpl` 每次运行都要重新抓取 K/V，带来不必要的显存压力与不可忽略的延迟，在长提示（long prompt）场景尤其明显。该 PR 因此重写 ROCm 上 AITER MHA 后端的执行方式，目标是消除冗余 KV 读取、提升访存局部性与内核占用率。正文明确表述新方案"more memory efficient and fast, especially on the long prompt scenario"，并给出在 Qwen3、Mi308 上的实测对比。

## 优化点

核心机制是 phase-aware 执行与输入重排：把请求划分为 decode、chunk prefill（cp）、pure prefill（pp）三个阶段，并在 Scheduler 的调度阶段与 ModelRunner 的状态更新阶段把输入重排为 `[decode:chunk_prefill:pure_prefill]`，使 token 在内存中连续，改善内核局部性与 occupancy。同时移除后端里冗余的 KV 抓取，重写原有的 "fetch KV" Triton 内核为按 token 分摊、按 `NUM_PRGMS` 切分的 `cp_mha_gather_cache_kernel`，提升 chunked prefill 等场景的占用率。新增 `split_prefill_from_chunk` 到 `SchedulerConfig` 控制该行为，仅在同时设置 `VLLM_ROCM_USE_AITER` 与 `VLLM_ROCM_USE_AITER_MHA` 时开启；`utils.py` 新增 `split_decodes_prefills_and_extends` 计算三段边界。

## 关键改动

```text
vllm/v1/attention/backends/rocm_aiter_fa.py: 新增 cp_mha_gather_cache_kernel 重写 KV gather，引入 phase-aware 路径与 merge_attn_states、cdiv 等
vllm/v1/attention/backends/utils.py: 新增 split_decodes_prefills_and_extends，识别 decode/extend/prefill 三段边界
```

```python
    is_prefill_or_extend = query_lens > decode_threshold
    is_prefill = (seq_lens == query_lens) & is_prefill_or_extend
    first_extend = is_prefill_or_extend.int().argmax(dim=-1).item()
    first_prefill = is_prefill.int().argmax(dim=-1).item()
    num_decodes = first_extend
    num_decode_tokens = query_start_loc[first_extend].item()
    num_extends = first_prefill - num_decodes
    num_prefills = num_reqs - first_prefill
```

## 性能声明

PR 正文给出在 Qwen3、Mi308 上的吞吐提升（测试条件：Qwen3 模型、Mi308 硬件，精度与 workload 未细分，正文以 prompt/output 长度描述）：

- 长提示短输出（`2k prompt, 16 output`）：约 4.x throughput improvement
- 短提示长输出（`128 prompt, 1k output`）：约 2.x throughput improvement
- 极长提示（`192k prompt, 2k output`）：约 5.x throughput improvement

精度方面，PR 在 Qwen3-30B-A3B-FP8 上用 lm_eval 跑 gsm8k：旧实现 flexible-extract 0.7437±0.0120、strict-match 0.8817±0.0089；新实现 flexible-extract 0.7703±0.0116、strict-match 0.8878±0.0087。其余定量条件（并发、批大小、精度细节）PR 未注明测试条件。

## 验证与回归风险

Test Plan 标注 acc 用 lm_eval 做精度验证、perf 用 vllm bench test；Test Result 以 `2k prompt/16 output`、`128 prompt/1k output` 两组 old/new 截图及 gsm8k 精度表格呈现。可能影响的路径：ROCm 上 `VLLM_ROCM_USE_AITER` 与 `VLLM_ROCM_USE_AITER_MHA` 同时开启时的 attention 后端、chunked prefill 与长提示场景；正文给出的 gsm8k 数值在新旧实现间有变化，属需关注项。其余回归信息 PR 未注明。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
