# PR #63118: [Bugfix][Kernel] Fix cooperative_topk illegal memory access on NaN scores

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#63118](https://github.com/vllm-project/vllm/pull/63118) |
| 作者 | LopezCastroRoberto |
| 合并日期 | 2026-10-23 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
nvcc 把 topk 有序键的 `bits | 0x80000000u` 优化成 FP 运算 `-|x|`,NaN 被 canonicalize 成 -0.0 的键,与粗分桶排序分歧导致共享内存越界写;用不透明 `mov.b32` 读位修复。

## 问题现象
自 #59645 起,`cooperative_topk` 在一行 64K 列或更多、且 NaN 数量超过 `top_k` 时(如 256K 行的 1%,或整行全 NaN)崩溃,报 `cudaErrorIllegalAddress`,直接杀死进程。NaN 分数可在 warmup 与 CUDA graph capture 期间的 decode top-k 中出现——此时 indexer 会读未初始化的 KV blocks(见 `candidate_blocks.py` 中的注释)。#59645 之前这类行只会产出垃圾索引,现在则进程崩溃。compute-sanitizer 定位为 `unbounded_round0_single_key` 中非法的 `__shared__` 写。

## 根因
PR 正文披露:nvcc 把 `convert_to_uint32_v2` 中的整数表达式 `bits | 0x80000000u` 编译成 FP 运算 `-|x|`:

```
FADD R10, -|R2|, -RZ    // "bits | 0x80000000u" emitted as -|x|
```

该变换对有限值与 ±inf bit 级等价,但 FADD 会把 NaN canonicalize 成 `0x7FFFFFFF`——即 -0.0 的有序键;而 FP16 粗分桶把 NaN 排在 +inf 之上(bin 1023),两种排序不一致。当 NaN 溢出阈值 bin 时,精确恢复(exact recovery)走单 key 路径并得到 -0.0 pivot:所有正分数都算作"在其之上",`collect_exact_candidates` 中未检查的 append 越过 `s_topk` 写入,造成共享内存越界、`cudaErrorIllegalAddress`。

## 修复方案
`csrc/topk_histogram_4096.cuh` 中 `convert_to_uint32_v2` 改用不透明内联汇编读位,键保持 bit 精确,NaN 在粗键与精确键中都排在 +inf 之上(与 `torch.topk` 一致):

```diff
-  uint32_t bits = __float_as_uint(x);
+  // Read the bits through an opaque move: otherwise nvcc lowers
+  // `bits | 0x80000000u` to the FP op -|x|, which canonicalizes NaN to
+  // 0x7FFFFFFF (the key of -0.0) while the FP16 coarse bin puts NaN on top.
+  uint32_t bits;
+  asm("mov.b32 %0, %1;" : "=r"(bits) : "f"(x));
```

同时删除散布在 `persistent_topk.cuh`、`sampled_topk.cuh`、`FilteredTopKTraits` 中的重复本地实现(`convert_to_uint32_v2`/`ordered`),统一改为调用 `topk_histogram_4096::convert_to_uint32_v2`。新增回归测试 `test_cooperative_topk_overflow_nan`(NaN 多于 top_k 时索引合法且全为 NaN)与 `test_topk_nan_ordering_matches_torch`(cooperative/persistent 各行数档位下,NaN+inf+最大有限值混合时排序与 `torch.topk` bit 级一致)。

## 影响与回归风险
- 影响窗口:#59645 引入该实现起至本 PR 之间的 main 分支;影响 cooperative_topk(SM90+,非 SM120 家族)及共享同一转换函数的 persistent_topk、sampled_topk 等路径。
- 回归面:有序键生成函数的编译方式变化(数值上仅 NaN 的键从 `0x7FFFFFFF` 恢复为高于 +inf 的正确有序键,有限值与 ±inf bit 级不变);多处重复实现收敛为单一实现,行为应一致但依赖 `asm mov.b32` 的编译器兼容性。测试覆盖 64K/256K/1M 行长与 512/2048 的 top_k。

## 关联
- https://github.com/vllm-project/vllm/pull/59645(引入该 topk 实现的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
