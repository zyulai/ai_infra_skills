# PR #20840: [AMD] Fix dpsk-v32 accuracy issue on mi355

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#20840](https://github.com/sgl-project/sglang/pull/20840) |
| 作者 | 1am9trash |
| 合并日期 | 2026-03-18 |
| labels | run-ci |
| 热度 | 👍 2 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
gfx950 上 (7168,2304) 形状 fp8 GEMM 被 #19826 改派到 ck 内核输出错误;把该形状加入 triton tuned 列表强制走 triton。

## 问题现象
MI355(gfx950)上运行 DeepSeek-V3.2 时精度从约 95% 崩塌到 0%(nightly CI: https://github.com/sgl-project/sglang/actions/runs/23227025664/job/67511792991 )。复现用 server:TP8、NSA prefill/decode 后端 tilelang、关闭 radix cache。修复前 gsm8k(1319 题)Accuracy 0.012、Invalid 0.133、Output throughput 3232.187 token/s;修复后 Accuracy 0.959、Invalid 0.000、Output throughput 1125.136 token/s。

## 根因
正文披露:PR #19826 改变了 gfx950 上 fp8 block-scaled GEMM 的分发逻辑——对于不在 triton tuned 列表中的形状,现在回退(fallback)到 ck(Composable Kernel)内核;而形状 (N=7168, K=2304) 恰好不在列表内,其 ck 内核输出不正确,导致精度崩塌。DeepSeek-V3.2 的 MoE 路由 GEMM 会命中该形状。

## 修复方案
`python/sglang/srt/layers/quantization/fp8_utils.py` 的 `use_aiter_triton_gemm_w8a8_tuned_gfx950(n, k)` tuned 形状列表中新增 `(7168, 2304)` 一项,使该形状被判定为 tuned、强制使用 triton 内核而非 ck 内核。作者说明这是快速 workaround(「May remove later」),正式修复待 ck 内核或分发逻辑修正后再移除。

```diff
         (4608, 7168),
         (512, 7168),
         (7168, 2048),
+        (7168, 2304),
         (7168, 16384),
         (7168, 256),
         (8192, 1024),
```

## 影响与回归风险
- 影响窗口:#19826 合并之后至本修复之前(gfx950/MI355 上的 fp8 block-scaled GEMM 分发)。
- 回归面:gfx950 上 (7168, 2304) 形状的 fp8 GEMM 从 ck 内核切回 triton 内核,该形状的性能可能与 ck 回退路径不同(修复后吞吐从 3232 降至 1125 token/s,但同时 Invalid 输出消失,端到端延迟反而缩短);其他 tuned 列表内形状不受影响;若未来 ck 内核修复,需按作者说明移除该 workaround。

## 关联
- 引入问题的 PR: https://github.com/sgl-project/sglang/pull/19826
- nightly CI 失败:https://github.com/sgl-project/sglang/actions/runs/23227025664/job/67511792991

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
