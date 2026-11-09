# PR #11579: [Fix] Fix accuracy bug in CSGMV kernel caching key.

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#11579](https://github.com/sgl-project/sglang/pull/11579) |
| 作者 | lifuhuang |
| 合并日期 | 2025-10-15 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
CSGMV LoRA 内核缓存键错误:Triton 对未声明 constexpr 的 num_segs 隐式特化,缓存命中错误内核导致精度错误。

## 问题现象
作者在把默认 LoRA 后端切换为 CSGMV 时发现该 bug:kernel caching 优化因错误的 kernel key 产生错误缓存。此前未被察觉的原因是现有 CI 绝大多数只针对默认 LoRA 后端(triton)运行,CSGMV 路径缺乏覆盖。现象层面即使用 CSGMV 后端时 LoRA 计算结果错误(精度 bug);正文未给出具体复现日志。

## 根因
正文披露:Triton 会对 kernel 参数做隐式特化(implicit specialization)——即使 `num_segs` 参数没有声明为 `tl.constexpr`,Triton 也会按其取值隐式特化编译内核;而外层 `cached_triton_kernel` 的缓存键只取 `(NUM_SLICES, BLOCK_M)`,不含 `num_segs` 的实际取值,于是不同 `num_segs` 的调用命中同一份按旧取值特化编译的内核缓存,导致计算错误。CI 只跑 triton 默认后端,故长期未暴露(后续由 PR #11488 解决覆盖问题)。

## 修复方案
1. `python/sglang/srt/lora/triton_ops/chunked_sgmv_expand.py`:`@triton.jit` 改为 `@triton.jit(do_not_specialize=["num_segs"])`,禁止 Triton 对 `num_segs` 隐式特化;缓存键保持 `(NUM_SLICES, BLOCK_M)`。
2. `python/sglang/srt/lora/triton_ops/chunked_sgmv_shrink.py`:同样加 `do_not_specialize=["num_segs"]`;缓存键 lambda 由 `(NUM_SLICES, BLOCK_M)` 扩为 `(K, NUM_SLICES, BLOCK_M)`,把 K 维也纳入 key。

```diff
-@cached_triton_kernel(lambda _, kwargs: (kwargs["NUM_SLICES"], kwargs["BLOCK_M"]))
-@triton.jit
+@cached_triton_kernel(
+    lambda _, kwargs: (kwargs["K"], kwargs["NUM_SLICES"], kwargs["BLOCK_M"])
+)
+@triton.jit(do_not_specialize=["num_segs"])
 def _chunked_lora_shrink_kernel(
```

## 影响与回归风险
- 影响窗口:引入 CSGMV kernel caching 优化之后至本修复之前(具体提交正文未给)。
- 回归面:使用 CSGMV 后端的 LoRA expand/shrink 两个 chunked sgmv 内核:`do_not_specialize` 会减少特化编译(可能略影响性能,换取正确性);shrink 缓存键新增 `K`,缓存条目划分变细、编译次数可能增多;默认 triton 后端不受影响。

## 关联
- Accuracy test run:https://github.com/sgl-project/sglang/actions/runs/18454668488
- 相关 PR(补 CI 覆盖):https://github.com/sgl-project/sglang/pull/11488

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
