# PR #36559: [Kernel] Add swap AB optimization to fused_moe_kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xyang16 |
| 合并 | 2026-06-25 |
| 分类 | Attention / Kernel |
| Label | performance, ready |
| Diff | diffstat 缺失(pack 未提供 +/− 统计,仅含源码 diff) |
| 链接 | https://github.com/vllm-project/vllm/pull/36559 |

## 一句话结论

fused_moe_kernel 在小 BLOCK_SIZE_M 时交换 A/B 操作数以更好利用 WGMMA,e2e 吞吐 +2–4%。

## 动机

Triton `fused_moe_kernel` 的 GEMM 形态为 `a[M, K] @ b[K, N] -> c[M, N]`。在小 BLOCK_SIZE_M(decode/小 batch 下的典型配置,BLOCK_SIZE_M=16)时,WGMMA 硬件单元利用不佳。与 #20396 的思路相同:交换 A/B 的角色,让 tl.dot 映射到 WGMMA 的有利布局。

## 优化点

对 BLOCK_SIZE_M<64 的配置,把计算改为 `b[N, K] @ a[K, M] -> c[N, M]`:A/B 指针按转置布局重排(a_ptrs 变为 `[K, M]` 索引、b_ptrs 变为 `[N, K]` 索引),accumulator 形状改为 `(BLOCK_SIZE_N, BLOCK_SIZE_M)`,load mask 随之调整,K 循环结束后用 `tl.trans(accumulator, (1, 0))` 转回再走 dequant。启用条件由 `enable_swap_ab(BLOCK_SIZE_M, BLOCK_SIZE_N)`(lru_cache)给出:仅 SM90(device capability 90)、BLOCK_SIZE_M<64 且 BLOCK_SIZE_N>=64,且只在 `use_fp8_w8a8` 时生效,int8_w8a16 等路径保持 SWAP_AB=False,行为不变。M>=64 的 config 完全不受影响。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/fused_moe.py: fused_moe_kernel 增加 SWAP_AB constexpr 分支(pointer/mask/dot/转置)
vllm/model_executor/layers/fused_moe/utils.py: 新增 enable_swap_ab 门控(SM90 且 M<64、N>=64)
```

```python
if SWAP_AB:
    accumulator += tl.dot(b, a) * b_scale[:, None] * a_scale[None, :]
else:
    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
...
if SWAP_AB:
    accumulator = tl.trans(accumulator, (1, 0))
```

## 性能声明

- Micro bench(H200,`benchmarks/kernels/benchmark_moe.py`,fp8_w8a8):Qwen3-Next-80B-A3B-Instruct-FP8 bs=1 kernel time 29.97→26.72 us、bs=16 99.91→93.03 us、bs=64 189.83→181.28 us;DeepSeek-R1-0528 bs=1 77.48→73.48 us、bs=64 1212.98→1154.64 us。BLOCK_SIZE_M=64 的行(BS>=96)无回退(如 R1 bs=4096 3807.75→3808.10 us)。
- e2e(`vllm bench serve`,ShareGPT,output-len 300,ignore-eos):Qwen3-Next-80B-A3B-Instruct-FP8 TP1/max-num-seqs 16,conc=1/8/16 输出吞吐 165.14→172.51、719.03→737.82、1033.46→1049.39 tok/s;DeepSeek-R1-0528 TP8,conc=1/8/16 为 85.93→99.57、383.6→385.78、494.5→502.4 tok/s。
- 总结声明:**2% to 4% e2e output token throughput improvement**。

## 验证与回归风险

测试计划为 `pytest -s -v tests/kernels/moe/test_moe.py`,结果 unit tests passed。回归风险主要在 fp8_w8a8 小 M 路径的数值行为:mask 与转置逻辑改动若出错会静默出错位,但启用电门限定 SM90+M<64+N>=64,且大 M config 不受影响;其他量化路径(int8_w8a16 等)SWAP_AB 恒为 False。PR 未附精度评测(GSM8K 等),仅单元测试。

## 关联

同类 swap AB 优化:#20396(正文明确"Similar as #20396")。同批:#29213(FlashInfer DeepGEMM swapAB 用于 W8A8 linear,同为 SM90 小 M 场景)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
