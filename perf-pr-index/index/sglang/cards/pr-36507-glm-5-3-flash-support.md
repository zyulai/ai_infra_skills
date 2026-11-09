# PR #36507: GLM-5.3-Flash support

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @JustinTong0323 |
| 合并 | 2026-09-06 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | quant, amd, Multi-modal, deepseek, hicache, npu, run-ci, apple-silicon, jit-kernel, bypass-fastfail, run-ci-extra, release-highlight, unified-radix-cache, memory-pool |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 被截断(89 个文件未收入),收录部分含 configs/kernels/model_config 等支撑改动 |
| 链接 | https://github.com/sgl-project/sglang/pull/36507 |

## 一句话结论

为 GLM-5.3-Flash(KDA 混合线性注意力 + no-PE MLA/DSA 索引 + MTP)打通 SGLang 全链路支持,属模型接入 PR,正文未附性能数字。

## 动机

GLM-5.3-Flash(`Glm5NextForConditionalGeneration`)采用与现有 MoE-DSA 模型不同的混合架构:多数层为 KDA 线性注意力(带 short conv、gate lower bound),少量层为 no-PE MLA(`qk_rope_head_dim=0`,即无 rope tail 的 FP8 MLA KV)配 DSA indexer(`index_kpool`/`index_topk_pattern`/`index_skip_topk_offset` 等),并带 1 层 NextN(MTP)与可选 mHC 结构(`hc_mult`)。此前 SGLang 的 kernel 与配置链路假定 MLA 必有 64 维 rope、topk 固定 2048,无法直接跑该模型。

## 优化点

接入工作覆盖 config、kernel 与调度三层:新增 `Glm5NextConfig`/`Glm5NextTextConfig`(含 KDA layer 推导、`mamba2_cache_params` 状态形状、draft 模型映射为 `Glm5NextForConditionalGenerationNextN` 并清空 linear_attn_config),并把该架构注册进 deterministic inference、breakable cudagraph(新增 "KDA hybrid linear attention" 不兼容规则)、encoder 分离等 hook 的模型名单。kernel 侧的实质改动:`quantize_k_cache_separate` 支持 `dim_rope=0` 的 no-PE MLA(empty rope),新增 `_gather_dsa_kv_scales` Triton kernel 按 kv_indices gather FP8 KV scales;TileLang sparse attention kernel 放宽约束——dim 允许 64 的倍数、`tail_dim=0` 走 v1 kernel(跳过 tail GEMM)、topk 从固定 2048 放宽为 64 的倍数;FLA 的 KDA kernel(`recompute_w_u_fwd`)统一 `DOT_PRECISION="ieee"` 并扩大 BK/BV autotune 空间;DSA metadata kernel 中 page_table 行索引改为显式 `int64`,修复大上下文下的溢出风险。`model_config.py` 补齐 `uses_kda_attention`、`get_dsa_index_kpool`、`get_dsa_mtp_topk_width`(index_topk + index_kpool - 1)等形状派生与 GLM5Next 的 spec/hc hidden size 解析。

## 关键改动

```text
python/sglang/srt/configs/glm5_next.py: 新增 GLM5Next 系列 config(KDA/线性注意力/MTP/mHC 参数与 layer 推导)
python/sglang/kernels/ops/attention/dsa/quant_k_cache.py: no-PE MLA(dim_rope=0)支持 + gather_dsa_kv_scales kernel
python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py: sparse attention 支持 tail_dim=0 与 topk 64 倍数(放宽 2048 硬编码)
python/sglang/srt/configs/model_config.py: uses_kda_attention / index_kpool / MTP topk width 等派生与 NextN 架构映射
python/sglang/kernels/ops/attention/fla/kda.py: KDA kernel ieee 精度 + autotune 空间扩展
```

```python
kernel_factory = (
    sparse_attention_fwd_kernel_v1 if tail_dim == 0
    else sparse_attention_fwd_kernel_v2
)
```

## 性能声明

PR 正文仅一句 "Support GLM-5.3-Flash.",**未附任何性能或精度 benchmark 数字**(pack 的 perf claim 检测也为空)。性能相关信息仅能从改动本身推断:no-PE MLA 省去 rope tail GEMM、KDA 线性注意力层降低 KV/状态成本,均未量化。

## 验证与回归风险

diff 被大量截断,测试文件不可见,PR 正文未列 Accuracy Tests/Speed Tests 结果。风险面较广:KDA/FLA kernel 的 autotune 空间与 ieee 精度选择会影响其他使用 KDA 的模型(如 KimiLinear);`quantize_k_cache_separate`、TileLang sparse kernel、DSA metadata 被所有 DSA 系模型共用,放宽约束属于共享路径变更;新架构涉及 memory-pool/hicache/unified-radix-cache 等 label 指向的存储组件;CI 状态 Base/Extra/AMD ROCm 7.2 三个 run 均记录为 `no_entry_sign`(未运行/被跳过),绿色验证情况未注明。

## 关联

同区域:GlmMoeDsa(DSA 系)、KimiLinear(KDA/线性注意力状态池)、LongcatFlash(spec hidden size 处理)等模型的既有支持;DSA attention kernel 与 #30575 等 sparse MLA 后端共享。pack 未列出直接前后序 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
