# PR #12437: Revert "fix llama4 kv cache layout"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#12437](https://github.com/sgl-project/sglang/pull/12437) |
| 作者 | b8zhong |
| 合并日期 | 2025-10-31 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | KV Cache / Memory |
| bug 类型 | regression |

## 一句话结论
回滚 Llama4 强制 KV cache 降为 bfloat16 的 workaround,因 PR #12307 才是正确解法,并恢复 trtllm_mha 的 FP8 支持标记。

## 问题现象
正文未描述具体现象,只说明回滚原因:因为 https://github.com/sgl-project/sglang/pull/12307 才是正确的解决方案。被回滚的改动是先前 「fix llama4 kv cache layout」PR 的一部分:在 sm100 上 Llama4 + `trtllm_mha` 后端时强制把 `kv_cache_dtype` 覆写为 `bfloat16`(原因是当时缺少 FlashInfer TRTLLM MHA 的 FP8 KV cache kernel,tileSizeKv=128)。该强制覆写随本 PR 撤销,FP8 KV cache 能力恢复。

## 根因
正文披露:先前的「fix llama4 kv cache layout」方案不是正解,PR #12307 才是正确方案,故将被回滚改动撤销。至于先前方案具体如何引入问题,正文未进一步披露。

## 修复方案
两处改动:
1. `python/sglang/srt/server_args.py` 的 `_handle_model_specific_adjustments`:整体删除如下分支——`is_sm100_supported() and self.attention_backend == "trtllm_mha"` 时执行 `self.kv_cache_dtype = "bfloat16"` 并打印警告(Llama4 专属 workaround,TODO 注明等 TRTLLM MHA FP8/tileSizeKv=128 kernel 可用后移除)。删除后 Llama4 + trtllm_mha 不再强制降级 KV cache 精度。
```diff
-            if is_sm100_supported() and self.attention_backend == "trtllm_mha":
-                # TODO(brayden): remove this once TRTLLM MHA kernel for FP8 w/ tileSizeKv=128 is available.
-                # This is a Llama 4 specific issue only.
-                self.kv_cache_dtype = "bfloat16"
-                logger.warning(
-                    "Setting kv_cache_dtype to bfloat16 for Llama4 with trtllm_mha backend, due to a missing FlashInfer TRTLLM MHA kernel for FP8 KV Cache"
-                )
```
2. `docs/advanced_features/attention_backend.md`:TRTLLM MHA 行的 FP8 支持列由 ❌ 改为 ✅,与恢复后的能力一致。

## 影响与回归风险
- 影响窗口:正文未说明(仅指明依赖 PR #12307 作为正解落地)。
- 回归面:Llama4 在 sm100 + trtllm_mha 后端的 KV cache dtype 选择——用户显式指定的 `--kv-cache-dtype`(含 fp8)不再被静默覆写为 bfloat16;若 FP8 kernel 支持实际不完备,该场景可能回到覆写前的原始问题(依赖 #12307 的修复兜底)。其余模型/后端组合不受影响;文档矩阵变更仅影响说明。

## 关联
- https://github.com/sgl-project/sglang/pull/12307

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
