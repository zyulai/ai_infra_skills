# PR #27428: [debug] Register #27338 EAGLE draft kv_indices revert in pr_fix_toggle

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27428](https://github.com/sgl-project/sglang/pull/27428) |
| 作者 | hnyls2002 |
| 合并日期 | 2026-06-06 |
| labels | (none) |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Speculative Decoding |
| bug 类型 | regression |

## 一句话结论
#27338(EAGLE draft kv_indices topk 欠分配修复)未注册进 pr_fix_toggle;补注册以支持回归 A/B 测试。

## 问题现象
属调试设施缺口而非线上故障:`pr_fix_toggle` 机制支持 `SGLANG_DEBUG_REVERT_PR=<n>` 反向应用某已合入 PR 的修复,用于回归 A/B 测试;#27338(EAGLE draft cuda-graph `kv_indices` 的 topk 欠分配修复)此前未注册,无法通过该机制还原复现。正文未描述用户侧故障现象。

## 根因
正文披露:#27338 的修复此前未在 `pr_fix_toggle` 中注册,本次补上注册项。还原生效的原理(正文说明):revert 会把 buffer 分配中的 `* self.topk` 去掉,于是 #27338 在 `common_template` 中加入的常开尺寸 invariant 会确定性地触发——即用还原态验证该 invariant 确实能拦住欠分配回归。

## 修复方案
`python/sglang/srt/debug_utils/pr_fix_toggle.py` 新增 `_PR_REVERT_YAML_27338` 并登记到 `_PR_FIX_REVERT_YAML`(键 27338)。补丁目标为 `FlashInferMultiStepDraftBackend.init_cuda_graph_state`,把分配式中的 `max_bs * self.topk * self.max_context_len` 替换为 `max_bs * self.max_context_len`(即模拟修复前的欠分配状态)。

```diff
+_PR_REVERT_YAML_27338 = """
+patches:
+  - target: sglang.srt.layers.attention.flashinfer_backend.FlashInferMultiStepDraftBackend.init_cuda_graph_state
+    edits:
+      - match: |
+          (self.speculative_num_steps, max_bs * self.topk * self.max_context_len),
+        replacement: |
+          (self.speculative_num_steps, max_bs * self.max_context_len),
+"""
```

## 影响与回归风险
- 回归面:纯调试注册,默认不影响任何生产路径;仅在设置 `SGLANG_DEBUG_REVERT_PR=27338` 时生效。生效时 kv_indices 缓冲被有意缩小并触发尺寸 invariant(设计目的即复现欠分配),不可用于正常服务。

## 关联
- [#27338](https://github.com/sgl-project/sglang/pull/27338)(被注册反向应用的修复 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
