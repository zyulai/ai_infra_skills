# PR #54805: [ROCm][BugFix] Revert AITER PA gluon decode from ROCM_AITER_FA

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#54805](https://github.com/vllm-project/vllm/pull/54805) |
| 作者 | ukannika |
| 合并日期 | 2026-10-02 |
| labels | bug, rocm, ready |
| 热度 | 👍 0 · 💬 10 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
ROCM_AITER_FA 的 gluon decode kernel 假设批内 query 长度一致,spec draft 截断产生 ragged batch 即 abort;回退该路径,保留 shuffle layout 修复。

## 问题现象
使用 MiniMax-M3 + EAGLE3 + shuffle KV cache layout 时,在第一个 ragged decode step 上直接 abort。触发条件:gluon decode kernel 要求整个 batch 的 query 长度一致(uniform),但 scheduler 在 speculative draft 被截断、或 token budget 裁掉某个请求时,会发出 ragged decode batch(各行 query 长度不一),此时 gluon 路径无法处理而中止。凡启用 ROCM_AITER_FA 且命中 gluon decode 分发的场景都可能受影响。

## 根因
正文披露:gluon decode kernel 需要 batch 内 query 长度一致,而 scheduler 会在 draft 截断或 token budget 裁剪请求时产生 ragged decode batch,MiniMax-M3 + EAGLE3 + shuffle KV layout 在第一个 ragged step 即 abort。同时,给该 kernel 提供 per-row query lengths 的 AITER 侧改动尚在开发中(kernel changes work in progress),当前 AITER 版本无法支撑 ragged 输入,因此在 vLLM 侧先回退。gluon PA decode backend 由 #52849 引入。

## 修复方案
在 `vllm/v1/attention/backends/rocm_aiter_fa.py` 回退 PA gluon 相关改动,只保留 shuffle layout 修复:删除 `_pa_gluon_supports`/`_PA_GLUON_*` 常量、`pa_decode_gluon` 内核导入、`AiterFlashAttentionDecodeMetadata.uniform_query_len` 字段及其在 `build`/`build_for_drafting` 中的计算、`forward` 中 `use_gluon` 分发与 gluon 调用路径(恢复走 `paged_attention_common`),`get_supported_kernel_block_sizes` 恢复只返回 `[16, 32]`;`supported_kv_cache_layouts` 保留 shuffle 启用时的 `(LHBNC, LBHNC)` 声明。后续等 AITER 支持落地后再由 follow-up PR 恢复 gluon decode 路径。

```diff
-    # Only gluon serves 128-token pages; the pa_fwd_asm/ll4mi decode
-    # fallback is limited to 16 and 32. Advertise 128 only when gluon can
-    # run so selection never picks a page we cannot serve.
-    vllm_config = get_current_vllm_config_or_none()
-    if vllm_config is not None and vllm_config.model_config is not None:
-        ...
-                return [16, 32, 128]
         return [16, 32]
```

## 影响与回归风险
- 影响窗口:自 #52849 引入 PA gluon backend 至本回退之间,启用 ROCM_AITER_FA(+shuffle KV layout)的 ROCm 版本。
- 回归面:`rocm_aiter_fa.py` 的 decode/元数据构建/内核选择路径。回退后 decode 统一走 `pa_fwd_asm`/`paged_attention_common` 回退路径;128-token page 不再被声明;shuffle layout 代码路径保留(作者用 llama2-70b fp4_gptq + quark + fp8 KV cache 的 serve 配置验证了 shuffle 路径仍正常);依赖 gluon 128 page 的配置会回到 16/32 block。

## 关联
- #52849(引入 PA gluon backend 的 PR)
- AITER 侧 per-row query lengths 改动(正文的 "AITER PR",未给出链接);follow-up PR 将在 AITER 落地后恢复 gluon decode 路径(正文提及,未附链接)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
