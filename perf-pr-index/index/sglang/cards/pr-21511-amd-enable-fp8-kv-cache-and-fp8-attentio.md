# PR #21511: [AMD] Enable FP8 KV cache and FP8 attention kernel for NSA on MI300/MI355 with TileLang backend

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @1am9trash |
| 合并 | 2026-04-03 |
| 分类 | Quantization (tier: major) |
| Label | amd, run-ci |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 数;diff 本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/21511 |

## 一句话结论

在 MI300/MI355 上用 TileLang backend 为 NSA 启用 FP8 KV cache 与 FP8 attention kernel,MI300 吞吐提升超 10%、MI355 超 5%,并发越高收益越大,gsm8k 无回退。

## 动机

DeepSeek-V3.2 / GLM-5 等 MLA 模型在 AMD GPU 上走 NSA(tilelang prefill/decode backend)时,KV cache 此前只能存 BF16,attention 主 GEMM 也在 bf16 上跑。FP8 KV cache 可把 KV 显存与 attention 访存减半,但 AMD(尤其 MI300 gfx942 与 MI355 gfx950)的 FP8 GEMM 支持依赖 TileLang 上游能力,此前路径未打通。

## 优化点

四块改动:(1) TileLang 升级到 a55a823,使 AMD GPU 具备 FP8 GEMM 能力(rocm.Dockerfile 更新 commit)。(2) 新增 TileLang FP8 attention kernel `sparse_mla_fwd_decode_partial_fp8`,复用既有 `sparse_mla_fwd_decode_combine` reduce kernel:FP8 e4m3(fn/fnuz)下 Q/KV 直接做 4x128 分块的 T.gemm,softmax 分数因有界 [0,1] 先乘 fp8_max_val(240.0/448.0)再量化成 FP8 做 P·V GEMM,GEMM 后再乘逆 scale,K=512 GEMM 有意拆成 4 个 tile 以缩短 MFMA 累积依赖链;按 gfx95/gfx942 选 block_I/threads/CU 并用 `lru_cache` 的 `_pick_inner_iter` 选 inner_iter。(3) 融合量化路径:MI355 复用 `fused_qk_rope_cat_and_cache_mla` 同时完成 Q/KV 量化(forward_mla.py 新增 `_skip_rope_for_nsa_tilelang_fused` 分支,rope 融入该 kernel);MI300 新增 Triton kernel `set_mla_kv_buffer_fp8_quant`,把 BF16/FP16->FP8 cast 融合进 paged KV 写入(HIP 路径使用 nope+rope 原始布局、无 per-block scales)。(4) `calculate_mla_kv_cache_dim` 为 HIP+TileLang 保留默认 MLA KV 维度(不做 scaled KV 布局的 dim 扩展)。启用方式:server 启动加 `--kv-cache-dtype fp8_e4m3` 与 `--nsa-prefill-backend tilelang --nsa-decode-backend tilelang`。

## 关键改动

```text
docker/rocm.Dockerfile: TILELANG_COMMIT 升级到 a55a823(AMD FP8 GEMM 支持)
python/sglang/srt/layers/attention/nsa/tilelang_kernel.py: 新增 sparse_mla_fwd_decode_partial_fp8 kernel;tilelang_sparse_fwd 按 KV dtype 分派
python/sglang/srt/mem_cache/utils.py: 新增 set_mla_kv_buffer_fp8_quant Triton kernel(量化融合进 paged KV 写)
python/sglang/srt/mem_cache/memory_pool.py: set_mla_kv_buffer 的 HIP FP8 分支
python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py: gfx95 TileLang NSA 走 fused_qk_rope_cat_and_cache_mla 融合 Q/KV 量化
python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py: HIP+TileLang 保持默认 MLA KV cache dim
```

关键片段(FP8 P·V GEMM 的 scale 处理):

```python
fp8_max_val = 240.0 if _is_fp8_fnuz else 448.0
s_inv_scale_const = fp8_max_val
s_scale_const = 1.0 / fp8_max_val
# softmax 分数有界 [0,1],先乘 fp8_max_val 再 FP8 cast,GEMM 后乘逆 scale
```

## 性能声明

基准:`sglang.bench_serving --dataset-name random --random-input 8192 --random-output 1024 --random-range-ratio 1.0`,DSV32,TP8:

- MI300:吞吐较 BF16 提升 10% 以上(conc=1 时 110.69%,conc=4 112.82%,conc=16 118.80%,conc=64 127.95%,即 conc=64 时 4723.46 tok/s);并发越高收益越大
- MI355:吞吐提升 5% 以上(conc=1 105.19%,conc=4 105.45%);conc=16/64 因 BF16 基线未采集而未列百分比(conc=64 时 7632.95 tok/s)
- 准确性(gsm8k):DSV32 BF16 0.945 vs FP8 0.946;GLM5 BF16 0.946 vs FP8 0.950,均无回退

## 验证与回归风险

PR 正文以 gsm8k 数字作为准确性验证,未注明新增单元测试;Checklist 项均未勾选。回归面:HIP 分支的 KV 布局为 nope(512 fp8)+rope(64 fp8) 原始布局、无 per-block scales,与 CUDA 侧 scaled KV 布局不同,`calculate_mla_kv_cache_dim` 对 HIP+TileLang 特判保留默认维度;forward_mla 的 fused rope 路径仅限 `_use_aiter_gfx95` + nsa + tilelang backend;CUDA/其他 attention backend 路径不变;`tilelang_sparse_fwd` 对 non-FP8 KV 保留原 bf16 kernel(gfx942 block_I=32/threads=128)。

## 关联

同区域:DeepSeek MLA / NSA 在 AMD 上的既有 TileLang bf16 kernel(`sparse_mla_fwd_decode_partial` + `combine`,本 PR 的 FP8 kernel 直接复用其 combine 与 partial 结构);依赖上游 tile-ai/tilelang commit a55a823。pack 未列出同分类其他 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
