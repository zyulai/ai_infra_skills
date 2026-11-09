# PR #46528: [Fix] [AMD] aiter: Fix garbage output with an fp8 KV cache and calibrated KV scales

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#46528](https://github.com/sgl-project/sglang/pull/46528) |
| 作者 | jiaqiang-dot-liu |
| 合并日期 | 2026-10-27 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论

ROCm aiter 后端三处误用 fp8 KV cache 的 KV scales(未反量化/拿 k_scale 当 q scale),带标定 scales 的 checkpoint 输出全错,GSM8K 归零。

## 问题现象

`--attention-backend aiter`(ROCm 上 MHA 模型的默认后端)配 fp8 KV cache,checkpoint 自带标定 KV scales 时输出垃圾:GSM8K 掉到 0。无需特殊 flag 即可命中:默认 `--kv-cache-dtype auto` 下,`configure_kv_cache_dtype` 只要 checkpoint 的 quant config 声明了 fp8 KV(如 `nm-testing/Llama-3.2-1B-Instruct-FP8-KV`)就会打开 fp8 KV cache;未声明的 checkpoint(如 `amd/Llama-3.1-8B-Instruct-FP8-KV`、Quark `amd/gpt-oss-20b-MoE-Quant-...-FP8-KV-FP8`)按其预期服务方式加 `--kv-cache-dtype fp8_e4m3` 也会命中。实测:MI355X 上 Llama-3.1-8B-FP8-KV 默认 decode 修前 0.005/0.000/0.000,修后 0.755/0.760/0.760。

## 根因

正文披露:KV cache 本身写入正确(`K / k_scale`),但三个 attention 调用点错误使用 scales——

1. **默认 decode(`paged_attention_ragged`)**:给内核传的是后端全局单位 scale,而不是该层的 k/v scales,KV cache 从未被反量化。
2. **unified-attention decode(`unified_attention`,滑动窗口模型如 gpt-oss 及 `SGLANG_USE_AITER_UNIFIED_ATTN=1` 时)**:q 用 `k_scale` 量化,但 `k_scale` 是为 K 标定的,不是 q 的 scale。
3. **NHD prefill(`mha_batch_prefill_func`,gfx942 全部、gfx950 上 HD128 ASM prefill 不适用时如 head_dim 64)**:q 无 scale 直接 cast 成 fp8,却把 KV cache 的 `k_scale` 当作 q 的 descale 传入,每个 logit 被 `k_scale` 多乘一次。

没有 KV scales 的模型 `k_scale == 1`,三种错误都不可见。

## 修复方案

`python/sglang/srt/layers/attention/aiter_backend.py`:

1. 新增 `_quantize_q_fp8`:用静态单位 q scale 量化 q(sglang 不加载 checkpoint 的 q scales)并返回该 scale 作 `q_descale`,与 gfx950 HD128 ASM prefill 的既有约定一致;量化 kernel 会饱和(此前 `.to(fp8)` 在 e4m3fnuz 上对 |q|>240 产生 NaN),且输入先做 contiguous(q 是 fused qkv 输出的 strided view,strided 输入会被量化 kernel 静默读错元素)。NHD prefill 与 unified-attention decode 均改用此函数。
2. `paged_attention_ragged` decode 改传层的 `k_descale`/`v_descale`(缺省回退单位 scale)。

```diff
-                    q_descale = (
-                        layer.k_scale if layer.k_scale is not None else self.k_scale
-                    )
-                    q, _ = scaled_fp8_quant(q, q_descale)
+                    q, q_descale = self._quantize_q_fp8(q)
```

新增 AMD 单测 `test/registered/attention/test_aiter_fp8_kv_scales_paths.py`(真实 aiter 内核对比 fp32 参考,scales 远离 1);`test_aiter_fp8_q_unified_attention.py` 改为断言单位 q scale。gfx950 head_dim-256 fp8 prefill 路径存在同类问题但未改,留作后续。

## 影响与回归风险

- 回归面:aiter 后端三条路径(默认 paged decode、unified-attention decode、NHD prefill)的 fp8 KV 数值行为;对无 KV scales 的模型数值不变(单位 scale),但量化 kernel 由 `.to(fp8)` 换为饱和式 `scaled_fp8_quant`。性能上无 scales 模型无可测开销,unified 路径新增 q 量化拷贝在测试中 ≤0.6% 差异。

## 关联

- #31856(FP8-Q unified-attention decode,沿用同一 kernel 与成本)
- 由 [Hyperloom](https://github.com/AMD-AGI/Hyperloom) 发现

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
