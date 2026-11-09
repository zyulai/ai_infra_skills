# PR #64064: [ROCm][Model] Enable ROCm AITER unified attention for MiMo-V2.6

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @vllmellm |
| 合并 | 2026-10-29 |
| 分类 | Speculative Decoding \| tier: major |
| Label | performance, rocm, ready, ci/build |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/64064 |

## 一句话结论

让 MiMo-V2.6 可用 ROCM_AITER_UNIFIED_ATTN，128k prefill 快 1.43x、decode 1.17x。

## 动机

正文指出 MiMo-V2.6 的 K/V head dims 不对称（192 / 128），目前把它钉死在 DiffKV attention backends：任何其他 `--attention-backend` 都会被静默替换为 DiffKV auto-pick，且原生 KV-cache 写会用 K 的 head width 去 copy V。本 PR 使 `--attention-backend ROCM_AITER_UNIFIED_ATTN` 对该模型可用（opt-in，defaults unchanged）并修复写 kernel。

## 优化点

`reshape_and_cache_flash_kernel` 原本从 key/key_cache 推出一个 head size、page stride、head stride 与元素数，并全部复用到 value 侧；此前所有 caller 都是对称的故无碍。unified attention 是首个传入 `v_head_size != head_size` 的 caller（MiMo 为 192/128，按 `(head_size + head_size_v)` 打包成一个 content dim），此时 V 会按 K 的 per-head offset 读、按 K 的宽度 copy 并写越界。本 PR 为 asymmetric 情形提供独立 kernel，按 V 自身的 head stride 与长度寻址；对称 caller 的既有 kernel 与 NHD fast path 保持 byte-identical。model 侧把 backend 选择 gate 从"列 backend 名字"改为查询 backend capability `supports_head_size_v`：DiffKV 声明支持任意 K/V 宽度关系，unified attention 声明 `v <= k`，不能服务该 shape 的 backend 仍走原来的 warning + DiffKV auto-pick。

## 关键改动

```text
csrc/cache_kernels.cu: 为 asymmetric K/V head dims 提供独立 reshape_and_cache_flash kernel
vllm/model_executor/models/mimo_v2.py: backend 选择改为按 supports_head_size_v 能力判定
vllm/v1/attention/backends/rocm_aiter_unified_attn.py: ROCm AITER unified attention backend
vllm/v1/attention/backend.py: backend 能力声明
tests/models/test_mimo_v2_attention_backend.py: 校验所选 backend 的 impl 与写出 cache 的布局
```

原问题代码（摘自 PR 正文 Details 段）：

```cpp
cache_t* v_dst_h = value_dst + head * head_stride;             // K's stride
vectorize_with_alignment(value_src, value_dst, n_elems, ...);  // K's width
```

## 性能声明

Claims 概述：长上下文 prefill 在 128k 快 1.43x、32k 快 1.21x，decode 每 token 快 1.17x（TPOT p50 2.94 vs 3.45 ms），output throughput 提升 21-45%；8k 并发 8 时两者 parity。Bench serve（MiMo-V2.6-Flash TP4，两侧均 `--block-size 16`，`--max-model-len 270336`，MTP `num_speculative_tokens: 2`，random dataset）：128k in/32 out c1 TTFT p50 7633.8 → 5317.7 ms（-30%，1.44x），output throughput 4.10 → 5.96 tok/s（+45%）；32k in/128 out c1 TTFT p50 890.9 → 738.2 ms（-17%，1.21x），TPOT p50 3.45 → 2.94 ms（-15%），output throughput 96.1 → 116.0 tok/s（+21%）；8k in/256 out c8 ITL p50 8.42 → 8.48 ms（parity）、output throughput 929 → 947 tok/s（+2%）。MiMo-V2.6-Pro TP8：128k c1 TTFT 9665.0 → 8576.9 ms（-11%，1.13x），ITL p50 19.52 → 11.77 ms（-40%，1.66x），output throughput 3.23 → 3.68 tok/s（+14%）。lm_eval gsm8k n=1319：unified 0.9750 / 0.9742（strict / flexible）对 DiffKV 0.9719 / 0.9712。硬件为 ROCm，缓存精度涉及 bf16 与 fp8。

## 验证与回归风险

`pytest tests/kernels/attention/test_cache.py -k asymmetric` 由 33 failed → 33 passed，`pytest tests/models/test_mimo_v2_attention_backend.py` 由 2 failed、5 passed → 7 passed；整个 test_cache.py 保持 694 passed、602 skipped。asymmetric kernel 测试跑 MiMo 的 K 192 / V 128、4 KV heads，在 block size 16 与 64、bf16 / fp8 cache、per-tensor / per-head scales 下参数化；model 测试新增两条检查（所选 backend 的 impl 必须是 unified attention、写出的 cache 须逐 slot 与 torch 构造的 cache 一致），两者在 main 上都失败。另有 lm_eval gsm8k、NIAH（265,673 tokens、3/3）与 MTP acceptance parity（128k 53.33/2.07 → 56.67/2.13）。回归风险：对称 caller 代码不变；已知限制——需 aiter >= 0.1.25（main pin 0.1.26）提供 V-dim-masked kernel，且 unified attention 只支持 V 窄于 K，`v_head_dim > head_dim` 的请求回退 DiffKV auto-pick 并 warning。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
