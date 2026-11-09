# PR #33771: [Revert] Fix performance regression for GLM-4.7-GPTQ decode and MTP acceptance rate

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#33771](https://github.com/vllm-project/vllm/pull/33771) |
| 作者 | aabbccddwasd |
| 合并日期 | 2026-02-08 |
| labels | ready, v1, nvidia |
| 热度 | 👍 0 · 💬 10 |
| 分类 | Speculative Decoding |
| bug 类型 | regression |

## 一句话结论
回退近期两个 commit,恢复 GLM-4.7-GPTQ(INT4/INT8 混合量化 + BF16 MTP)在 4×RTX PRO 6000 Blackwell 上的 decode 吞吐(86→95 tps)与 MTP 接受率;pack 中的 diff 仅含 flashinfer.py 中 `seq_lens_cpu` 获取方式的回退。

## 问题现象
GLM-4.7-GPTQ-INT4-INT8MIX(model body 为 int4/int8 混合量化,MTP 模块(第 92 层)手动配置为未量化 BF16 以提高接受率)出现性能回归:4×RTX PRO 6000 BLACKWELL 实测,基线 v0.13.0 + BF16 MTP 为 95 tps;引入 `e0327c9db`(#31773)后降到 86 tps(-9.5%);再叠加 `654a71fc3`(#32805)后降到 77 tps(-19.0%)。回归表现为 decode 吞吐下降与 MTP(Multi-Token Prediction)接受率下降、未量化 MTP 层内核选择错误(应为 fused_triton)。

## 根因
PR 正文披露:回归由两个近期 commit 引入——`e0327c9db`(#31773)导致 decode 性能回归;`654a71fc3`(#32805)导致 MTP 接受率回归(注意:正文原文中第二条带有删除线标记)。正文未逐条解释两个 commit 的错误机制。从 diff 可描述回退所暗示的 decode 回归机制(由 diff 推断):#31773 在 Flashinfer attention 的 `build` 中把 `seq_lens_cpu` 改为按需调用 `common_attn_metadata.seq_lens.cpu()`,在异步模式下该同步拷贝开销大且并非总是必要,回退后恢复直接使用 metadata 中已有的 `seq_lens_cpu` 字段。

## 修复方案
回退引入回归的 commit。pack 所含 diff 为 `vllm/v1/attention/backends/flashinfer.py` 中 `build` 的一处:

```diff
-        seq_lens_cpu = (
-            common_attn_metadata.seq_lens.cpu() if needs_seq_lens_cpu else None
-        )
+        seq_lens_cpu = common_attn_metadata.seq_lens_cpu if needs_seq_lens_cpu else None
```

验证结果(正文):回退后吞吐从 77 tps 恢复到 95 tps;MTP 接受率恢复(约 30% 提升);内核选择恢复正确——量化层用 Marlin kernels,未量化 MTP 层恢复使用 fused_triton kernels。注:正文说明性能数据是在没有优化 Triton 配置文件的情况下测得,配合适当的 Triton 优化预期还有额外提升。

## 影响与回归风险
- 影响窗口:正文披露受影响范围为 `e0327c9db`(#31773)与 `654a71fc3`(#32805)合并之后、本回退之前;具体版本号未给出。
- 回归面:pack 中的 diff 仅显示 `flashinfer.py` 一处改动(Flashinfer attention 的 metadata 构建路径):`seq_lens_cpu` 恢复为直接读取 metadata 预计算字段,不再走 `.cpu()` 同步拷贝。使用 `needs_seq_lens_cpu`(DCP、cascade 或非纯 TRTLLM decode)的路径行为依赖 `common_attn_metadata.seq_lens_cpu` 字段的可用性。正文标题与测试结果还涉及 MTP 接受率与内核选择的恢复,但 pack 的 diff 部分未包含对应文件改动。

## 关联
- #31773(被回退 commit `e0327c9db` 所属 PR): https://github.com/vllm-project/vllm/pull/31773
- #32805(被回退 commit `654a71fc3` 所属 PR): https://github.com/vllm-project/vllm/pull/32805

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
