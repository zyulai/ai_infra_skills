# PR #34158: [Bugfix] Relax TRTLLM KV cache contiguity assertion for cross-layer layout

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#34158](https://github.com/vllm-project/vllm/pull/34158) |
| 作者 | Etelis |
| 合并日期 | 2026-03-16 |
| labels | bug, nvidia, ready, v1 |
| 热度 | 👍 5 · 💬 7 |
| 分类 | KV Cache / Memory |
| bug 类型 | general |

## 一句话结论
TRTLLM kernel 本身支持非连续 KV cache 布局,vLLM 侧整张量连续性断言过严;放宽为仅要求内层两维连续。

## 问题现象
`kv_cache_permute` 上的 `is_strictly_contiguous` 断言拒绝了非规范 strides 的 view,使跨层(cross-layer)统一分配的 KV cache(如 shape `(167623, 2, 8, 16, 16, 64)`)在 TRTLLM 路径上触发断言失败;此前 #33192 的绕法是在启用 KV transfer 时整体禁用 TRTLLM。

## 根因
正文披露:作者查阅 FlashInfer kernel 源码确认,`csrc/trtllm_fmha_kernel_launcher.cu` 中的 TRTLLM kernel 实际读取张量 strides(`key_cache.stride(-2)`、`stride(-3)`、`stride(0)` 等),本身能处理非连续布局——整张量连续性断言只是 vLLM 侧的守卫,并非 kernel 硬需求。唯一真正需要连续性的是 Triton FP8 dequant kernel(`_trtllm_prefill_attn_kvfp8_dequant`),它从 shape 推算 strides;外层维度的退化(degenerate)strides 还会破坏 TMA 描述符(见 flashinfer-ai/flashinfer#2232)。

## 修复方案
修改 `vllm/v1/attention/backends/flashinfer.py` 的 `forward`:prefill 与 decode 两条 HND 路径均删除 `assert is_strictly_contiguous(kv_cache_permute)`,改为在需要处断言内层两维(block_size, head_size)连续——`kv_strides[-1] == 1` 且 `kv_strides[-2] == shape[-1]`,允许外层维度带非规范 strides;该内层断言置于 fp8 kv cache 的 mock KV/dequant 调用前(prefill)与 decode 断言组内(decode):

```diff
-                assert is_strictly_contiguous(kv_cache_permute)
+                kv_strides = kv_cache_permute.stride()
+                assert (
+                    kv_strides[-1] == 1
+                    and kv_strides[-2] == kv_cache_permute.shape[-1]
+                ), (
+                    "KV cache inner dims (block_size, head_size) must be "
+                    f"contiguous, got strides {kv_strides}"
+                )
```

测试:B200(SM100)、vLLM 0.15.1、FlashInfer 0.6.1、Llama-3.2-1B-Instruct——`test_cpu_offloading[FLASHINFER-48]` PASSED;`test_flashinfer_trtllm_attention` 224 passed / 16 skipped;CPU offloading 路径 GPU hit 19.26ms、CPU hit 40.50ms。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:FlashInfer 后端 TRTLLM prefill/decode 的 KV cache 布局校验——外层非连续布局从必败变为放行;fp8 kv cache 的 dequant 路径新增内维断言,若上游 kernel 对外层退化 strides 的容忍度有变(如 TMA 描述符问题 flashinfer-ai/flashinfer#2232 修复状态变化),此处断言可能需要再调(由 diff 推断);非 TRTLLM 路径与 `is_strictly_contiguous` 对 query/workspace/block_tables 的其余断言不变。

## 关联
- Fixes #33572(正文首行)
- #33192(旧绕法:KV transfer 启用时整体禁用 TRTLLM)
- flashinfer-ai/flashinfer#2232(diff 注释中引用的 TMA 描述符问题)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
