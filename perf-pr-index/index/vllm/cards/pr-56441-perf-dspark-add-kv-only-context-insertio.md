# PR #56441: [Perf][DSpark] Add KV-only context insertion across V4.1 cache formats

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @0z5a |
| 合并 | 2026-09-16 |
| 分类 | Speculative Decoding (tier: major) |
| Label | performance, ready, deepseek, dflash, DSv4.1 |
| Diff | diffstat 缺失(pack 未记录);diff 覆盖 5 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/56441 |

## 一句话结论

DSpark 上下文 KV 插入改用零 query head 的专用 op,免去 dummy query 分配与计算,插入内核提速 2-6 倍。

## 动机

DSpark(spec-decode)把已 kv_norm 的上下文 KV 写入 paged cache 时,复用了 DSV4 的 fused QK-norm/RoPE/KV-insert op——该 op 同时处理 query,上下文 token 没有 query,只能分配一个 `(n_ctx, n_local_heads, head_dim)` 的全零 dummy query 填进去再丢弃。这带来无谓的显存分配、query 分支计算与三个 cache 格式各自的 dummy 变体调用。#55654 为 V4 做过 uint8 KV-only 插入并新增 fused kv_norm,但 BF16/普通 FP8 路径仍保留 dummy-query 回退。

## 优化点

新增 CUDA op `fused_deepseek_v4_kv_rope_insert`:不分配 query,直接复用现有内核的 KV 分支(保留其 RoPE 舍入与量化),按 cache dtype 分派三条路径——packed uint8(`fp8_ds_mla` UE8M0 paged layout,`launchFusedDeepseekV4Templated<..., 0, false>`,支持 kv_mxfp8)、BF16 与普通 FP8 e4m3fn(均走 `launchFullCacheKernel`,FP8 要求单一 float32 KV scale)。调度上,零 query head 时保留 warp-per-slot grid(原 `NUM_TOKEN_CUTOFF` 之下的 reduced grid 靠在单 CTA 内迭代 Q heads获益,Q heads 为零则一个 warp 会包办全部工作)。op 内做完整的设备/dtype/layout 校验(cache 行宽 584/528/512 按 dtype 匹配),保留 padded page stride;`slot_mapping` 可少于 kv 行数(允许 padding 行)。DSpark 侧 `_insert_context_kv` 简化为单次 op 调用,projection、`kv_norm`、调度与接受逻辑不变。

## 关键改动

```text
csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:
  新增 fused_deepseek_v4_kv_rope_insert(零 Q head 三格式分派 + 校验);
  零 Q head 时的 grid 选择分支
csrc/libtorch_stable/ops.h、torch_bindings.cpp: op 声明与 binding
vllm/models/deepseek_v41/nvidia/dspark.py: _insert_context_kv 改调新 op,
  删除 dummy query 构造与三分支
tests/kernels/test_compressor_kv_cache.py: test_dspark_context_kv_matches_query_insert
```

关键机制摘录(kernel .cu,零 Q head 分派):

```cpp
// Zero query heads schedules only the existing KV branch, preserving its
// RoPE rounding and quantization without allocating any query tensors.
if (packed) {
  launchFusedDeepseekV4Templated<scalar_t, 0, false>(
      nullptr, nullptr, input, cache, slots, positions, cos_sin, 0.0f, ...)
```

## 性能声明

插入微基准(H100,16 local query heads,固定 shape,仅隔离插入操作):512 tokens 下 uint8/fp8_ds_mla 21.344→7.360 µs(2.90×)、BF16 14.240→7.296 µs(1.95×)、FP8 e4m3fn 16.032→7.392 µs(2.17×);2048 tokens 下分别为 43.936→10.624(4.14×)、36.384→8.416(4.32×)、47.264→7.968 µs(5.93×)。

E2E serving A/B(DeepSeek-V4.1-Flash,4×H100 SXM 80GB,TP4+EP,V2 runner,eager,Engram/CPU offload,DSpark 5 草稿,并发 4,16 timed requests/长度):输出吞吐 512 in 12.8313→14.7392 tok/s(+14.87%)、2048 in 8.3804→9.4507(+12.77%)。作者明确声明:这是 FRESH_SMOKE_ONLY 的单对非交错 A/B,无重复与置信区间;两臂 draft acceptance 不同(512/2048 分别 6.57%→7.70%、2.66%→3.86%)、512 TTFT 反升(3313.51→3472.94 ms),故总变化不能单独归因于 KV 插入或 PDL,单次结果不构成稳定加速的证明;本地比率不得外推到整模型吞吐;未声称显存节省。

## 验证与回归风险

正确性:焦点 CUDA 测试 24 passed(3-4 格式 × 1/17/1023/1024 tokens × eager/graph,cache 字节级一致,含 graph replay 与全 -1 slots);真实模型 worker 内 4 workers × 12 insertions 共 48/48 字节一致(含未写行);CUDA launch observer 记录 72 次 `cudaLaunchKernelExC`,零 query head、带 PDL launch attribute(作者注明这验证 launch 属性与正确性,而非 PDL 吞吐收益)。serving 冒烟 6/6 问题回答一致、若干 continuation token 级一致(logprobs 非完全一致,共享前缀最大绝对差 0.04-0.19)。风险:rebase 跨越 `deepseek_v4_1→deepseek_v41` 目录改名后,GPU/model 测试未对最新 head 重跑;与 #55654 的 uint8 插入 API 存在重叠,作者建议二者都推进时合并;limited serving 检查不能替代标准精度评测。

## 关联

- 前序:PR #56320(原 feature 分支,经 #56214 的 dsv41-feat 合入 main 后由本 PR 重提)
- 相关:#55654(V4 uint8 KV-only 插入 + fused kv_norm,目标重叠)、#54674(stacked projections,独立操作)
- 同区域:DSpark/DSv4.1 spec-decode 系列(如 PR #60731 mono decode、PR #63921 latency_specialized)

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
