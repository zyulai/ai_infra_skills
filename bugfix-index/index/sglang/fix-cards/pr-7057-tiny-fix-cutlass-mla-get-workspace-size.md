# PR #7057: Tiny fix cutlass_mla_get_workspace_size stub incorrect signature

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#7057](https://github.com/sgl-project/sglang/pull/7057) |
| 作者 | fzyzcjy |
| 合并日期 | 2025-06-10 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

`cutlass_mla_get_workspace_size` 在 CUDA < 12.4 下的 stub 缺少 `num_kv_splits` 参数,与真实实现签名不一致;补齐该参数。

## 问题现象

正文未描述具体现象(Motivation/Modifications 均为模板占位,未填写)。问题由 diff 体现:`sgl-kernel/csrc/attention/cutlass_mla_kernel.cu` 中,CUDA 版本 < 12.4 时编译的兜底 stub `cutlass_mla_get_workspace_size` 只接受 `(max_seq_len, num_batches, sm_count)` 三个参数,而真实实现已有第四个参数 `num_kv_splits`,两侧签名不一致。

## 根因

正文未披露根因。(由 diff 推断)为不支持 cutlass MLA 的 CUDA 版本(< 12.4)提供的编译期兜底 stub,在真实实现增加 `num_kv_splits` 参数时没有同步更新,导致 stub 与实现签名不匹配;在该 CUDA 版本下调用此函数会因参数个数不符而出错。

## 修复方案

为 stub 补上第四个参数,使其与真实实现签名一致:

```diff
-int64_t cutlass_mla_get_workspace_size(int64_t max_seq_len, int64_t num_batches, int64_t sm_count) {
+int64_t cutlass_mla_get_workspace_size(int64_t max_seq_len, int64_t num_batches, int64_t sm_count, int64_t num_kv_splits) {
   TORCH_CHECK(false, "CUDA version must be >= 12.4 for cutlass_mla_get_workspace_size");
 }
```

stub 本身的行为不变(进入即 `TORCH_CHECK(false)` 报错),只是签名对齐。

## 影响与回归风险

- 影响窗口:正文未说明。
- 回归面:仅 `sgl-kernel/csrc/attention/cutlass_mla_kernel.cu` 中 CUDA < 12.4 分支的 stub 声明;真实实现分支(`#else` 之后)不受影响。改动后该 stub 的调用方可以按与真实实现相同的参数个数编译通过,报错时机与信息不变。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
