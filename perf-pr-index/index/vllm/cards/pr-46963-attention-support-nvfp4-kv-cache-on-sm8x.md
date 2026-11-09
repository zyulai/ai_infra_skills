# PR #46963: [Attention] Support NVFP4 KV cache on SM8x and SM12x with FlashInfer

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @lesj0610 |
| 合并 | 2026-10-07 |
| 分类 | Quantization(tier: major) |
| Label | ci/build, v1, nvidia, mrv2 |
| Diff | diffstat 缺失(pack 未提供加/删行数);diff 显示改动 2 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/46963 |

## 一句话结论

把 `--kv-cache-dtype nvfp4` 从 SM100 扩展到 SM8x/SM12x,写路径用 FlashInfer 量化 kernel、读路径走原生 fa2,sm_80 上 KV 容量 3.32 倍。

## 动机

NVFP4 KV cache 此前只有 SM100 上的 paged KV 写入路径(trtllm-gen 原生 store),SM8x/SM12x 完全无法使用该 dtype,4-bit KV 带来的容量收益在这些存量卡上不可得。需要在不新增 CUDA kernel 的前提下复用 FlashInfer 的量化写入与 fa2 读取。

## 优化点

机制是"fa2 写读"组合:SM8x/SM12x 上写缓存调用 FlashInfer 的 `nvfp4_quantize_append_paged_kv_cache_with_slot_mapping`,读回用原生 fa2 prefill/decode wrapper,无需内核改动。新增 `_nvfp4_kv_views` helper 按 stride 判别 NHD/HND 布局,把每页切成 K/V 两半并经既有 `nvfp4_split_data_scale` 拆分 data/scale;SM100 也改用同一 helper 构视图、保留 trtllm-gen kernel。fa2 路径上 query/output 保持模型 dtype,不施加 q_scale、不分配 FP8 暂存缓冲。SM12x 的 NVFP4 用原生 decode wrapper 替代 XQA,且只捕获单 token decode graph(与 SM8x 一致)。安全护栏:head_size 64(fa2 prefill 会误读 NVFP4 10-15%,flashinfer#5965)与 attention sinks(sink prefill wrapper 读不了 block scales,flashinfer#5966)在 `supports_combination` 直接拒绝,DCP 在 metadata builder 拒绝;`nvfp4_4over6` 等 scale-search 变体仍限 SM100。

## 关键改动

```text
vllm/v1/attention/backends/flashinfer.py: nvfp4_fa2 分支、_nvfp4_kv_views helper、supports_combination 拒绝逻辑、do_kv_cache_update 写路径
tests/kernels/attention/test_flashinfer.py: 新增 4 种 layout 的 NVFP4 KV 写入/反量化测试
```

```python
if self.nvfp4_fa2:
    data, scales, kv_layout = _nvfp4_kv_views(kv_cache, self.num_kv_heads)
    nvfp4_quantize_append_paged_kv_cache_with_slot_mapping(
        key, value, slot_mapping, data, scales,
        layer._k_scale, layer._v_scale, kv_layout=kv_layout)
    return
```

## 性能声明

sm_80 验证机上 KV cache 容量从 1,446,932 增至 4,805,405 tokens(**3.32x**)。吞吐性能未测(PR 明言 SM86/SM89/SM120 与 performance 均未测试)。精度:GSM8K(sm_80,FLASHINFER 后端)Qwen3-0.6B bf16 KV 0.384 vs nvfp4 0.174(明显退化);Qwen3-4B(32 query/8 KV heads)0.920 vs 0.880。MRCR(Qwen3.5-4B,30 samples,16.6k-33.2k tokens):auto 0.8523 vs nvfp4 0.8452 match ratio,prefix hit rate 1.0000 vs 0.9667。

## 验证与回归风险

`pytest tests/kernels/attention/test_flashinfer.py -k nvfp4`:在 4 种 block-compact layout(LBNHC/LBHNC/BLNHC/BLHNC)上经 `do_kv_cache_update()` 写入、按同视图反量化并与输入比对相对误差 <0.2,且相邻层不被误写。GSM8K/MRCR eval 覆盖读路径;SM121 上用 stock `flashinfer-python==0.7.0.post1` 验证。风险点:两个上游修复(flashinfer#6024/#6021)未发布前拒绝逻辑必须保留;SM100 视图重构仅做 CPU 对比且 SM100 未实跑;fa2 路径精度在小模型上退化显著,需要按模型评估。

## 关联

同区域:#46329、#54772、#56550(同样面向 SM12x);RFC #47684(整体追踪);#26107(cudnn fp4 GEMM via flashinfer,FP4 计算侧)。依赖 pinned FlashInfer 0.7.0,两个 entry point 已在 pin 内,无需版本 bump。
