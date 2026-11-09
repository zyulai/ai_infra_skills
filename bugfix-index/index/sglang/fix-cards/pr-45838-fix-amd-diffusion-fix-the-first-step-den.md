# PR #45838: [Fix] [AMD] diffusion: Fix the first-step denoise regression from compiling ENABLE_CK=0 BF16 attention

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#45838](https://github.com/sgl-project/sglang/pull/45838) |
| 作者 | yichiche |
| 合并日期 | 2026-10-24 |
| labels | run-ci, diffusion |
| 热度 | 👍 0 · 💬 0 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | regression |

## 一句话结论
ENABLE_CK=0 时 BF16 attention 在 torch.compile 内触发不可 trace 的 Triton kernel,首步 11.8s 重 trace;CK 关闭时改走 eager 路径修复。

## 问题现象
MI355 上进程设置 `ENABLE_CK=0` 时(包括 FLUX.2 720p run),scored denoise 的第一步耗时 11.8s(Dynamo 重 trace),之后各步保持 2.02 it/s。profiled CK GEMM 与 Triton attention kernel 与旧镜像一致(总 kernel 时间 1.481s vs 旧 1.512s),说明回归不在 kernel 时间而在首步重 trace。gfx1250(已走 eager)与 gfx950 + CK enabled 不受影响。

## 根因
正文披露:AITerImpl.forward 在 torch.compile 内调用 `aiter.flash_attn_func`;`ENABLE_CK=0` 时该 dispatch 会打开 Triton config JSON 并构建 Dynamo 无法重建的 Triton kernel。#44671 将 BF16 调用拆成 `_forward_bf16`(编译)与 `_forward_bf16_eager`(`@torch.compiler.disable`),但只把 `USE_AITER_GFX1250` 送入 eager 分支;gfx942/gfx950 保持编译的依据是 FlyDSL 在那里返回 None、调用落到 CK `@compile_ops` kernel(Dynamo 记为不透明节点),该前提仅在 `ENABLE_CK=1` 时成立。`ENABLE_CK=0` 时 gfx950 经 FlyDSL 落到 `aiter.ops.triton.attention.mha.flash_attn_func`,而 `_forward_bf16` 仍在编译图内调用它;graph break 位于 DiT block 循环内,导致 Dynamo 在第一个真实 step 跳过 `Flux2Transformer2DModel.forward`。

## 修复方案
`python/sglang/multimodal_gen/runtime/layers/attention/backends/aiter.py`:引入 `ENABLE_CK as _AITER_ENABLE_CK`,`forward` 中只要 CK 关闭就走 eager BF16 路径,使不可 trace 的 Triton attention 留在 Dynamo 之外:

```diff
-        if USE_AITER_GFX1250:
+        if USE_AITER_GFX1250 or not _AITER_ENABLE_CK:
             return self._forward_bf16_eager(query, key, value)
         return self._forward_bf16(query, key, value)
```

测试 `test_aiter_attention_impl.py` 改为参数化三用例:gfx1250、`ENABLE_CK=0` 时断言 `torch.compiler.is_compiling()` 为 False,CK 路径为 True;编译后输出与 `q.clone()` `assert_close`。`pytest` 8 passed。

## 影响与回归风险
- 影响窗口:自 #44671 引入 compiled/eager 拆分起,且进程设置 `ENABLE_CK=0`。
- 回归面:multimodal_gen diffusion 的 AITer BF16 attention 路径——`ENABLE_CK=0`(gfx950/MI355)从编译图内调用改为 eager 调用,移除首步重 trace;`ENABLE_CK=1` 的 gfx950 仍留在编译图内(有测试锁定)。eager 路径调用同一 `aiter.flash_attn_func`、参数不变,无 weight/scale/kernel-config 改动。修复后端到端 FLUX.2 数字作者称尚不在该分支。

## 关联
- 引入回归的 PR:sgl-project/sglang#44671

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
