# PR #47902: [CI Bug Fix] Temp fix for v3.2 accuracy

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47902](https://github.com/vllm-project/vllm/pull/47902) |
| 作者 | yewentao256 |
| 合并日期 | 2026-07-07 |
| labels | bug, ready |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
TP8+EP 无 DP 时 sequence parallel MoE 被误启用致 GSM8K 归零,加 DP>1 条件临时止血。

## 问题现象
CI 中 DeepSeek-R1-TP 与 DeepSeek-V3.2-TP 的 GSM8K 正确性测试失败,断言显示 `GSM8K metric too low: 0.0000 < 0.9500 - 0.0800 = 0.8700`,即精度完全归零(附 BuildKite build 76393 链接)。复现方式:`vllm serve deepseek-ai/DeepSeek-V3.2 --trust-remote-code --kernel-config.enable_flashinfer_autotune=False --enable-expert-parallel --tensor-parallel-size 8 ...`(TP=8 + EP)。修复后 lm_eval gsm8k flexible-extract 0.9545、strict-match 0.9530,恢复正常。

## 根因
正文只说是对 v3.2 精度问题的临时修复("Temple fix for v3.2 accuracy"),并称"我会进一步排查并彻底修复",未披露完整根因机制。(由 diff 推断)`vllm/config/parallel.py` 中 `use_sequence_parallel_moe` 属性此前只要求 flashinfer autotune 关闭 + EP 开启 + TP>1,使得 TP8+EP 但无数据并行的部署也启用了 sequence parallel MoE,精度因此崩到 0;本次为该属性追加 `data_parallel_size > 1` 条件,避免在无 DP 的部署里误启用该路径。

## 修复方案
在 `vllm/config/parallel.py` 的 `use_sequence_parallel_moe` property 中追加一个条件,使 sequence parallel MoE 仅在数据并行规模大于 1 时启用:

```diff
             )
             and self.enable_expert_parallel
             and self.tensor_parallel_size > 1
+            and self.data_parallel_size > 1
         )
```

## 影响与回归风险
- 影响窗口:正文未说明;作者自述这是临时修复(temp fix),彻底修复待后续跟进。
- 回归面:所有依赖 `use_sequence_parallel_moe` 判定的路径——原先在"EP + TP>1 但 DP=1"配置下会走 sequence parallel MoE 的部署,现在回退到非 sequence parallel 路径;真正 DP>1 的部署不受影响。若存在依赖该配置才正确的场景,行为会改变;作者已用 V3.2 GSM8K 验证精度恢复。

## 关联
- https://buildkite.com/vllm/ci/builds/76393/list?sid=019f3415-e06a-4286-8b5b-eae99cb5095e&tab=output

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
