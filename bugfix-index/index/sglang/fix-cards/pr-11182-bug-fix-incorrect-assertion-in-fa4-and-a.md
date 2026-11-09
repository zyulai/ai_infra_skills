# PR #11182: [Bug] Fix incorrect assertion in FA4 and add UT.

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#11182](https://github.com/sgl-project/sglang/pull/11182) |
| 作者 | lifuhuang |
| 合并日期 | 2025-10-06 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
FA4 的 kvcache 断言误把 `rotary_interleaved` 当必为 None 的张量参数检查,合法调用被拒;从断言中移除该项并补 kvcache 单测。

## 问题现象
正文未描述具体现象(Motivation / Modifications 等节均为空模板)。标题仅说明:FA4(flash attention ver=4)中存在不正确的 assertion,本 PR 修复之并补充单元测试。受影响接口为 `sgl_kernel.flash_attn.flash_attn_with_kvcache`(ver=4)。

## 根因
正文未披露根因。(由 diff 推断)`flash_attn_with_kvcache` 中 "FA4 does not support rotary embedding." 的断言要求 `rotary_cos`、`rotary_sin`、`rotary_interleaved`、`rotary_seqlens` 四者同时为 None。但 `rotary_interleaved` 是布尔开关(如 `rotary_interleaved=False`/`True` 的取值)而非"传了才存在"的张量输入,因此在未使用 rotary(cos/sin 均为 None)的正常调用里,该断言也会因 `rotary_interleaved` 非 None 而误报,把合法调用当不支持的用法拒绝。

## 修复方案
改 `sgl-kernel/python/sgl_kernel/flash_attn.py` 的 `flash_attn_with_kvcache`:断言不再检查 `rotary_interleaved`,只保留真正的张量输入检查:

```diff
         assert (
-            rotary_cos is None
-            and rotary_sin is None
-            and rotary_interleaved is None
-            and rotary_seqlens is None
+            rotary_cos is None and rotary_sin is None and rotary_seqlens is None
         ), "FA4 does not support rotary embedding."
```

同时在 `sgl-kernel/tests/test_flash_attention_4.py` 新增 `test_flash_attn_kvcache` 大规模参数化 UT(覆盖 mha/mqa/gqa、learnable sink、local window、page_size None/128 等组合,`flash_attn_with_kvcache` 以 `partial(..., ver=4)` 调用,与参考实现对比 max/mean diff 并断言 KV cache 写回一致),并附 `_generate_block_kvcache` 辅助函数;该测试在 Hopper 上跳过。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:仅 sgl-kernel FA4 的 `flash_attn_with_kvcache` 入口断言。放开的只是 `rotary_interleaved` 一项,FA4 仍拒绝 `rotary_cos`/`rotary_sin`/`rotary_seqlens` 非 None 的调用,rotary 不支持的限制本身不变;新增 UT 属于纯增量。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
