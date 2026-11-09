# PR #60372: [Kernel] Integrate FA4 SM90 FP8 MMA for Gemma-4 full attention

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @katec846 |
| 合并 | 2026-11-05 |
| 分类 | Speculative Decoding |
| Label | ready, ci/build |
| Diff | diffstat 缺失(pack 未提供);diff 可用,涉及 FA 后端路由(fa_utils/flash_attn/flash_attn_diffkv)、vllm_flash_attn 接口、cmake FA 版本、路由测试与 lm-eval CI 配置 |
| 链接 | https://github.com/vllm-project/vllm/pull/60372 |

## 一句话结论

为 Gemma-4 的 d512 full attention 接入 FA4 SM90 FP8 MMA 内核(保留 FP8 KV 存储),H200 serving 吞吐 +4.3%,full-attention 内核本身快约 12%。

## 动机

Gemma-4 的 full attention head dim 为 512、SWA 为 256。此前 SM90 上 FP8 KV 的 FA4 路径是"读 FP8 KV、kernel 内反量化、再做 FP16 MMA"(flash-attention PR #164);flash-attention PR #199 提供了真正的 FP8 MMA kernel(SM90 d512,补偿式 FP8 MMA + FP32 累加,Q 在 kernel 内量化)。本 PR 把它接入 vLLM,并量化其相对 dequant 基线与 BF16 参考的收益/回退。

## 优化点

路由与门控:(1) cmake 的 vllm-flash-attn 升到 #199 的 commit,SM90 + FP8 KV + head_size 512 时 FA4 走 FP8 MMA kernel——Q 保持原生 FP16/BF16、由 kernel 内量化(`supports_quant_query_input = False`,vLLM 不得预量化),K/V descale 在 kernel 内折叠,输出直接写 Q 的原生 dtype;(2) d256 的 SWA 层继续走 FA3 FP8(FA4 FP8 SWA 不支持),`flash_attn_supports_kv_cache_dtype` 要求 FA4 FP8 时 Q/K/V head_size 同为 512,`supports_combination` 加 block size % 64、block sparsity 拒绝,Diff-KV 后端显式校验,DCP(非因果)raise;(3) 接口层把 `fa4_fp8_kv_dequant` 判定前移并收紧为 q/k/v 均 512,非因果 full attention 因已知精度问题 raise;(4) FA4 full-attention 层允许自动选 splits(claims:解除这些层此前 CUDA-graph cap 32,FA3 保留);descale 布局归一。

## 关键改动

```text
vllm/vllm_flash_attn/flash_attn_interface.py: SM90 d512 FP8 MMA 路由判定与不支持的组合 raise
vllm/v1/attention/backends/flash_attn.py: supports_combination 门控 + quant query 能力位
vllm/v1/attention/backends/fa_utils.py: FA4 FP8 要求 head_size_v=512
cmake/external_projects/vllm_flash_attn.cmake: FA 源码升级到 #199
tests/v1/attention/test_flash_attn_sm90_fp8_mma_selection.py: 新增路由/门控测试(免 GPU)
```

```python
+        fa4_fp8_kv_dequant = False
+        if (
+            fa_version == 4
+            and k.dtype == torch.float8_e4m3fn
+            and torch.cuda.get_device_capability()[0] == 9
+        ):
+            fa4_fp8_kv_dequant = q.shape[-1] == k.shape[-1] == v.shape[-1] == 512
+            if not fa4_fp8_kv_dequant:
+                raise NotImplementedError(...)
```

## 性能声明

Serving(gemma_4_31b_it_h200 8k-in-1k-out,单卡 H200 TP=1、锁 GPU 时钟、3 轮取中位,FP8 权重):Before(`5d35ce3934`+FA `b273a02`)796.5 tok/s / TPOT 126.5 ms / ITL 75.1 ms → 本 PR + FA#199:**831.1 tok/s(1.04x,+4.3%)** / 123.2 ms(-2.6%) / 69.3 ms(-7.8%)。

Nsight 剖析(25K 输入、TP4、MTP、cc64):full attention 6.681 vs 7.584 ms(**快 11.9%**),但 SWA 慢 30.3%、KV 写慢 66.2%,整周期反而比 BF16 慢 1.7%(reuse-off 下 2.6%)——PR 明确报告 FP8 相对 BF16 的整体回退。

精度:AIME 2025(avg@64)87.19%、GPQA Diamond(avg@16)85.73%、MMLU-Pro 86.86%(FP8 MMA),与 FP8 KV+FP16 MMA(86.88/86.55/86.64)和 BF16 Triton(87.29/85.42/86.44)相当。MTP 接受长度三配置在 2.96-3.01 之间无系统性差异。

## 验证与回归风险

新增免 GPU 的 mock 路由测试(head_size/dtype/FA 版本组合、DCP 拒绝、wrapper 对不支持组合 raise);CI 增加 gemma-4-31B FP8 的 lm-eval 配置(GSM8K 0.937 阈值)。风险面:仅覆盖 SM90 d512 + FP8 KV + causal full attention 的窄组合,SWA/多模态 bidirectional attention 走 Triton/FA3;FP8 全路径相对 BF16 在 SWA+KV 写上仍有净回退(PR 如实呈现);FP8 精度依赖 kernel 内补偿累加,非因果路径因精度问题被禁用。

## 关联

依赖 flash-attention PR #199(基线为 PR #164 的 dequant kernel);精度参考基线来自 vLLM PR #48666。同批无其他 FA4 卡。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
