# PR #59098: [Bugfix][ROCm] Fix ROCM_AITER_FA context loss on out-of-order batches

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#59098](https://github.com/vllm-project/vllm/pull/59098) |
| 作者 | phambinhfin |
| 合并日期 | 2026-11-16 |
| labels | bug, mrv2, ready, rocm |
| 热度 | 👍 0 · 💬 48 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
ROCM_AITER_FA 的注意力切分器假设所有 Extend 请求都排在 Prefill 之前,而 MRV2 调度可能把全新 Prefill 排在带缓存 KV 的 Extend 之前,导致这些 Extend 在不读历史 KV 的情况下做注意力,长上下文精度大跌;本 PR 改为按"最后一个带上下文的请求"划界,并补上全 Decode 批次的守卫和缓存上下文 chunk 的大小计算。

## 问题现象
在 1×MI300X 上用 `lm_eval niah_single_2`(30k tokens,500 samples)测 Phi-4-mini:`main` 得分 0.626 ± 0.022,500 根针里丢了 187 根;本 PR 后 1.000。同配方三模型对比:Phi-4-mini 0.62 → 1.00,Mistral-7B 0.74 → 1.00,Qwen3-8B 0.99 → 1.00(短上下文的 Qwen3-8B 几乎不暴露该 bug)。Qwen3-8B 针埋在 token 2048、64 prompts、并发 16 时,40k 从 59/64 恢复到 64/64。带真实后端的乱序批次测试在 `main` 上失败,本 PR 通过。计数器统计显示 `main` 在 Phi 30k 基准里只读了 574,972 个上下文 token,跳过 233,430 个(占 KV 的 28.9%、prefill 注意力的 17.1%),例如 `q=5424/seq=30000/drop=24576` 这类 chunk。

## 根因
正文披露根因:MRV2 可以把一个全新的 Prefill 请求排在带缓存 KV 的 Extend 请求之前,而注意力切分器假设所有 Extend 都在 Prefill 之前。于是切分器把该 Prefill 之后的每个请求都当作 Prefill 处理,其中的 Extend 请求在注意力时不读自己已缓存的 KV 历史(`cu_seqlens_k = query_start_loc`,`seq − query` 个历史 token 被跳过)。另有两个次级问题:(1) 共享工作区的除数把"没有任何上下文可读"的 Extend 行也计入,使所有人的 chunk 变小、chunk 数变多;(2) 全 Decode 批次在自适应投机验证时 `max_query_len` 可能被夸大,原代码跳过全 Decode 快速路径后去找第一个非 Decode 请求,找不到时 `argmax()` 返回 0,错误地报告零个 Decode。

## 修复方案
三处修改,均在共享切分器与 ROCm 后端:

1. **找最后一个带上下文的请求**:把 Prefill 边界放在最后一个 `seq_len > query_len`(即有缓存上下文)的请求之后;边界之前的全新 Prefill 走 Extend 路径(该路径可安全处理零缓存上下文的请求)。核心改写在 `vllm/v1/attention/backends/utils.py` 的 `split_decodes_prefills_and_extends`:

```python
has_context = seq_lens[num_decodes:num_reqs] > query_lens[num_decodes:]
if bool(has_context.any()):
    last_with_context = len(has_context) - 1
    last_with_context -= int(has_context.flip(0).int().argmax().item())
    first_prefill = num_decodes + last_with_context + 1
else:
    first_prefill = num_decodes
```

2. **只按真正读上下文的行分 workspace**(`vllm/v1/attention/backends/rocm_aiter_fa.py`):

```python
num_extends_with_context = max(int((computed_kv_lens > 0).sum()), 1)
max_context_chunk = _CP_TOKENS_PER_ITER_ROCM // num_extends_with_context
```

3. **全 Decode 守卫**:`is_decode.all()` 时直接 `return num_reqs, 0, 0, num_tokens, 0, 0`,避免 `argmax()` 误报。新增测试 `test_rocm_aiter_fa_unordered_batch`(prefill_before_extend / interleaved 两种批次)与切分器、DCP 单测。

## 影响与回归风险
- 精度:长上下文检索类任务显著恢复(0.626 → 1.000);短上下文不受影响(8k/1k、32k/1k 吞吐变化在 ±0.2% 内)。
- 性能:Phi 30k 基准上本 PR 反而更慢(29,911 → 28,034 tok/s,TTFT 7.27 s → 8.00 s)——因为 `main` 在偷懒跳过 KV;修正后需读全部 808,402 个上下文 token。这是"正确的代价",需注意长上下文吞吐对比口径。
- 波及面:`split_decodes_prefills_and_extends` 有两个调用方——`rocm_aiter_fa.py` 和 `cp_utils.py`(NVIDIA DCP)。在 8×H200、Qwen2.5-1.5B、TP4、DCP2、FA3 下,原切分器在 248 次调用中错排 24 次,本 PR 为 0;FA3 输出 16/16 完全一致,该路径无输出变化。
- chunking 变化只影响 workspace 划分,不改变哪些行读上下文。

## 关联
- 本 PR 采用 #63542(作者 @qli88)的 order-independent split,并新增其缺失的全 Decode 守卫(`all_decode_with_draft_tokens` 用例 #63542 单独仍失败)。
- 测试:`tests/v1/attention/test_attention_backends.py`、`tests/v1/attention/test_attention_splitting.py`、`tests/v1/worker/test_cp_utils.py`。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
