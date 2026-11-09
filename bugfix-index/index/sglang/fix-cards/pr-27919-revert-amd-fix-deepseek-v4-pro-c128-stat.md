# PR #27919: Revert "[AMD] Fix DeepSeek V4 Pro c128 state tensor dtype mismatch error and c4_sparse_raw_indices attribute error in cuda graph phase"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27919](https://github.com/sgl-project/sglang/pull/27919) |
| 作者 | At1a8 |
| 合并日期 | 2026-06-11 |
| labels | jit-kernel |
| 热度 | 👍 1 · 💬 0 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | correctness |

## 一句话结论
回滚 #27529:该修复引入 kernel 性能回退,且与 #27380 的 cuda graph 修复逻辑冲突会再次导致 crash,整体还原。

## 问题现象
本 PR 是对 #27529 的整体回滚(Revert),正文给出的回滚理由即问题现象:其一,#27529 引入了 kernel 性能回退(perf regression),在 #27529 的 review 评论中被指出;其二,cuda graph launch error 原本已被 #27380 解决,但 #27529 的逻辑与 #27380 相互冲突,会导致 crash 问题再次出现。

## 根因
正文明确披露:#27529(为 DeepSeek V4 Pro 修 c128 state tensor dtype mismatch 与 c4_sparse_raw_indices attribute error)带来了两个副作用——kernel 性能回退,以及与 #27380 已落地的 cuda graph 修复逻辑冲突,使 cuda graph crash 复现。因此回滚是唯一被采纳的处理方式。

从 diff 可以描述被回滚改动所暗示的机制:#27529 曾给 C128/C4 compress kernel 引入独立的 `BufFloat` 模板参数,让 fp32 的 kv_score_buffer 与输入 dtype 分离并逐元素 cast;回滚后 kv buffer 直接使用 `InFloat`,并删除了 `compressor.py` 中 `_use_aiter` 下把 ape/norm 权重 cast 到 bf16 的 hotfix。(由 diff 推断)

## 修复方案
整体还原 #27529 的全部改动,涉及四个文件:

- `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh` / `c4_v2.cuh`:删除 `BufFloat` 模板参数,`c128_forward`/`c4_forward`/`c128_write_decode`/`c4_write_decode` 及各 kernel 的 `kv_buffer` 恢复为 `InFloat` 类型,删去加载后逐元素转 fp32 的分支,TensorMatcher 的 `with_dtype<BufFloat>()` 恢复为 `with_dtype<InFloat>()`。
- `python/sglang/jit_kernel/dsv4/compress.py`:`_jit_compress_module` 移除 `dtype_buf` 参数,JIT 编译参数恢复为 `(head_dim, dtype_in, dtype_out, ...)`。
- `python/sglang/srt/layers/attention/dsv4/compressor.py`:删除 `apply_ape_hotfix` 中 `_use_aiter` 时把 `ape`/`norm.weight` cast 到 bfloat16 的代码。

```diff
-template <typename Trait, bool kUsePDL, typename BufFloat, typename InFloat, typename OutFloat>
+template <typename Trait, bool kUsePDL, typename InFloat, typename OutFloat>
 SGL_DEVICE void c128_forward(
-    const BufFloat* kv_buf,  // [128n, 128n + 127]
-    const InFloat* kv_src,   // ragged pointer at position = 128n + 127
+    const InFloat* kv_buf,  // [128n, 128n + 127]
+    const InFloat* kv_src,  // ragged pointer at position = 128n + 127
```

## 影响与回归风险
- 回归面:影响 DeepSeek V4 Pro 在 AMD/ROCm 上的 C128/C4 compress JIT kernel 及 dsv4 compressor 路径;回滚后 #27529 原本要修的 c128 state dtype mismatch / c4_sparse_raw_indices attribute error 问题在受影响场景下可能重新暴露,需依赖 #27380 及后续 PR 重新处理。

## 关联
- https://github.com/sgl-project/sglang/pull/27529 (被回滚的 PR)
- https://github.com/sgl-project/sglang/pull/27529#pullrequestreview-4469436235 (perf regression 记录)
- https://github.com/sgl-project/sglang/pull/27380 (解决 cuda graph launch error 的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
