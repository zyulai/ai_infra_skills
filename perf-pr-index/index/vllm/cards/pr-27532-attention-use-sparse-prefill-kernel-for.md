# PR #27532: [Attention] Use sparse prefill kernel for fp8 kv-cache in DeepSeek-v3.2

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LucasWilkinson |
| 合并 | 2025-12-12 |
| 分类 | Quantization (tier: major) |
| Label | ready, v1, deepseek, gpt-oss, nvidia, ready-run-all-tests |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files),另有 8 个文件因超长从 diff 中省略 |
| 链接 | https://github.com/vllm-project/vllm/pull/27532 |

## 一句话结论

DeepSeek-V3.2 的 fp8 KV cache 在 DP=8+EP 下改走 bf16 prefill kernel,吞吐 +7%。

## 动机

PR 正文:prefill 时把 KV cache 从 fp8 上转换(up-convert)为 bf16 并调用 bf16 prefill kernel,而非 decode kernel;本 PR 引入 global workspace management,让这块 bf16 workspace 与 MoE workspace buffers 重叠复用,避免额外显存开销。

## 优化点

采用 hybrid 策略,以 `MIN_HEADS_FOR_BF16_PREFILL = 32` 为阈值:TP=8(每 rank 16 heads)沿用原来的 mixed batch 路径(与 main 相同),DP=8+EP(128 heads)走新的 BF16 prefill 路径(正文称 7% faster)。新增 `cp_gather_and_upconvert_fp8_kv_cache` CUDA kernel,把 FP8 格式的 KV cache token(每 token 512 bytes fp8 + 16 bytes scales + 128 bytes rope)按 block_table / seq_lens / workspace_starts 收集并上转换为 BF16 写进 workspace(每行 576 元素),供 bf16 prefill kernel 使用;配套在 csrc/cache.h、csrc/torch_bindings.cpp、vllm/_custom_ops.py、vllm/envs.py、vllm/model_executor/layers/fused_moe/modular_kernel.py 等处接通算子、workspace 管理与开关,并更新/新增多个 kernel 测试。

## 关键改动

```text
csrc/cache_kernels.cu: 新增 cp_gather_and_upconvert_fp8_kv_cache,把 FP8 KV cache 上转换为 BF16 写入 workspace
csrc/cache.h、csrc/torch_bindings.cpp、vllm/_custom_ops.py: 算子声明、绑定与 Python 接口
vllm/model_executor/layers/fused_moe/modular_kernel.py: global workspace 管理,使 bf16 workspace 与 MoE workspace 重叠
vllm/envs.py: 相关开关;另涉及多个 tests/kernels/moe/* 与 tests/v1/attention/test_sparse_mla_backends.py 测试
```

关键机制代码:

```cpp
// Gather and upconvert FP8 KV cache tokens to BF16 workspace
// Similar to cp_gather_cache but specifically for FP8->BF16 conversion
__global__ void cp_gather_and_upconvert_fp8_kv_cache(
    const uint8_t* __restrict__ src_cache,    // [NUM_BLOCKS, BLOCK_SIZE, 656]
    __nv_bfloat16* __restrict__ dst,          // [TOT_TOKENS, 576]
    const int32_t* __restrict__ block_table,  // [BATCH, BLOCK_INDICES]
    const int32_t* __restrict__ seq_lens,     // [BATCH]
    const int32_t* __restrict__ workspace_starts,  // [BATCH]
```

## 性能声明

PR 正文 Benchmark Results(4096 in / 512 out / 100 prompts,DeepSeek-V3.2,fp8 KV cache,命令 `vllm serve deepseek-ai/DeepSeek-V3.2 ... --kv-cache-dtype fp8 --attention-config '{"backend":"FLASHMLA_SPARSE"}'` + `vllm bench serve ... --num-prompts 100 --request-rate inf --random-input-len 4096 --random-output-len 512`):TP=8、FP8 时 Main 1.58 req/s / 806 tok/s、PR 1.58 / 806(0%);DP=8+EP、FP8 时 Main 2.36 / 1207、PR 2.53 / 1293(+7%)。精度 GSM8K(DeepSeek-V3.2,FP8 KV-cache,TP=8):Main 95.30% ± 0.58%、PR(Hybrid)95.53% ± 0.57%。PR 未注明具体 GPU 型号。

## 验证与回归风险

正文给出 GSM8K 精度对比(95.30% → 95.53%)与 TP=8 / DP=8+EP 两组 serving benchmark;diff 涉及多个 tests/kernels/moe/* 测试文件与 tests/v1/attention/test_sparse_mla_backends.py 的改动(具体测试内容未在正文中列出)。影响路径:DeepSeek-V3.2 且使用 FLASHMLA_SPARSE + fp8 KV cache 的 prefill;hybrid 阈值下 TP=8 走原路径不受益,DP=8+EP 才走 bf16 prefill,workspace 与 MoE buffers 重叠是需关注的显存/并发约束点。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
