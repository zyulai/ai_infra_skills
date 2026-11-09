# PR #55090: [ROCm][Perf] Pad fp16/bf16 GEMM weight row stride for L2 cache

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgehre-amd |
| 合并 | 2026-10-15 |
| 分类 | Attention / Kernel |
| Label | rocm, ready |
| Diff | diffstat 缺失(pack 未提供) |
| 链接 | https://github.com/vllm-project/vllm/pull/55090 |

## 一句话结论

gfx11 上把 GEMM 权重 row stride 偏移一个 128B cache line,避开 L2/MALL 哈希冲突,AMD Strix Halo 上 TTFT 最高 -13.20%、TPOT 最高 -3.39%。

## 动机

在 gfx11(RDNA3/3.5)上,当 dense GEMM 的权重 row stride 是 2048 字节的倍数时,多行 global load 会撞上 L2/MALL 的 channel 哈希,损失大量可用带宽。对 bf16/fp16,判定条件 `(K * element_size) % 2048 == 0` 等价于 `K % 1024 == 0`,覆盖 Llama/Qwen dense 家族的大多数线性层(Qwen3-8B 的 qkv/o/gate_up K=4096 → 8192B、down K=12288 → 24576B,四层全部命中)。

## 优化点

在 `UnquantizedLinearMethod.process_weights_after_loading` 中对 ROCm 权重一次性做 row-stride padding:`cache_aliasing_pad_elems(k, element_size)` 在 stride 为 2048B 倍数时返回 `128 // element_size`(bf16/fp16 为 64 元素,fp8 为 128、fp32 为 32),否则返回 0;`maybe_pad_weight_avoid_cache_aliasing` 用 `F.pad(weight, (0, pad))[..., :-pad]` 生成形状不变、row stride 加宽一个 cache line 的视图,使每行仍保持 cache-line 对齐同时错开哈希。函数幂等(padded stride 不再命中条件,二次调用返回原指针,CUDA graph 捕获的指针不受影响);仅作用于 2D 连续权重。配套放宽 `rocm_unquantized_gemm_impl` 的 skinny GEMM 分派:`wvSplitK` 允许 row stride(padded 权重,在 gfx11 上要求 `stride(1) == 1`),LLMM1 仍要求连续。仅 gfx11 gate;gfx12 可能同样受益,留作后续测量后启用。

## 关键改动

```text
vllm/model_executor/layers/utils.py: cache_aliasing_pad_elems + maybe_pad_weight_avoid_cache_aliasing,wvSplitK/LLMM1 分派拆分
vllm/model_executor/layers/linear.py: UnquantizedLinearMethod.process_weights_after_loading 接入 padding
tests/model_executor/layers/test_rocm_unquantized_gemm.py: pad 元素数参数化测试 + padded 权重实内核测试
```

```python
if (k * element_size) % 2048 != 0:
    return 0
return 128 // element_size
```

## 性能声明

AMD Strix Halo (gfx1151),`--num-prompts 5`,每格 2 次交错重复:Qwen3-8B 1920/128(100% 命中)TTFT 918.02 → 852.39 ms(-7.15%),TPOT 72.15 → 69.83 ms(-3.21%);Qwen3-8B 128/128 TTFT 119.35 → 103.59 ms(-13.20%),TPOT -3.39%;Qwen2.5-VL-3B-Instruct 1920/128(70.7% 命中)TTFT -3.81%;Qwen3-4B(10.4% 命中)TTFT 不可分辨、TPOT 38.85 → 38.03 ms(-2.11%);Qwen2-1.5B(0% 命中)无效果。显存:仅受影响层小幅增加,Qwen3-8B 15.27 → 15.47 GiB(+171 MiB,权重显存的 1.29%)。gsm8k 全量 1319 题 5-shot greedy(Qwen3-8B,gfx1151):88.931% → 89.159%(视为噪声内一致)。

## 验证与回归风险

新增测试覆盖:pad 元素数参数化(bf16/fp8/fp32 各 K)、非 gfx11 跳过、非 2D/非连续/不命中跳过、幂等性、padded 权重仍走 `wvSplitK` 实内核(mock 与真内核两套)。风险:padding 只在 gfx11 生效,`UnquantizedLinearMethod` 的所有未量化线性层权重都会经此检查;padding 视图改变 weight stride,依赖 `weight.is_contiguous()` 的其他路径(如 LLMM1、部分 aiter/tgemm 分支)需要保持兼容(本 PR 已相应放宽/拆分分派条件);CUDA graph 场景依赖幂等性保证指针稳定。

## 关联

无(本批索引中无其他 ROCm GEMM 布局类 PR 编号;PR 引用 skinny GEMM 上下文 PR #40687)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
