# PR #47144: [Bugfix][ROCm] Change AttentionCGSuppoort in TritonMLA to UNIFORM_SINGLE_TOKEN_DECODE

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47144](https://github.com/vllm-project/vllm/pull/47144) |
| 作者 | music-dino |
| 合并日期 | 2026-07-09 |
| labels | bug, rocm, ready, v1 |
| 热度 | 👍 1 · 💬 4 |
| 分类 | Attention / Kernel |
| bug 类型 | hang |

## 一句话结论
TritonMLA 声明了过强的 UNIFORM_BATCH cudagraph 支持而 query_len 仍是 SINGLE_ONLY,开投机解码后 cudagraph 捕获触发 decode-only 断言;降级为 UNIFORM_SINGLE_TOKEN_DECODE。

## 问题现象
开启 speculative decoding 时,`MLACommonMetadataBuilder.build_for_cudagraph_capture` 中的 decode-only 断言失败:
```
assert m.max_query_len <= self.reorder_batch_threshold  # decode only
```
该问题在 ROCm CI 上暴露:失败模型没有 AITER MLA kernel 时 TritonMLA 是唯一可用后端;NVIDIA 上 TritonMLA 不会被选中,所以问题未显现。

## 根因
正文披露:`TritonMLAMetadataBuilder` 在 #42885 中声明了 `_cudagraph_support = AttentionCGSupport.UNIFORM_BATCH`,同时却保留继承来的 `query_len_support = SINGLE_ONLY`(即 `reorder_batch_threshold = 1`)。开投机解码时,UNIFORM_BATCH 的声明绕过了 `resolve_cudagraph_mode_and_sizes` 中的降级逻辑(该降级只在 `min_cg_support < UNIFORM_BATCH` 时触发),于是 full decode cudagraph 捕获以 `max_query_len = 2` 进行,撞上 decode-only 断言——声明与实际能力不匹配。

## 修复方案
改 `vllm/v1/attention/backends/mla/triton_mla.py` 的 `TritonMLAMetadataBuilder`,把 `_cudagraph_support` 从 `AttentionCGSupport.UNIFORM_BATCH` 改为 `AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE`,与 `SINGLE_ONLY` 的 query_len 限制对齐。

```diff
-    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
+    _cudagraph_support: ClassVar[AttentionCGSupport] = (
+        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
+    )
```

## 影响与回归风险
- 影响窗口:自 #42885 引入 UNIFORM_BATCH 声明之后(正文提及该出处,未给出具体版本)。
- 回归面:使用 TritonMLA 后端(典型为 ROCm)的 cudagraph 路径。该后端在投机解码下不再绕过降级逻辑,cudagraph 模式/尺寸解析可能选择更保守的模式;对非投机、单 token decode 场景,UNIFORM_SINGLE_TOKEN_DECODE 的语义仍匹配,预期不变。NVIDIA 平台因不选 TritonMLA 基本不受影响。

## 关联
- [PR #42885](https://github.com/vllm-project/vllm/pull/42885)(正文提及为 UNIFORM_BATCH 声明的引入处)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
