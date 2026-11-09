# PR #34476: [BUGFIX] Fix accuracy regression for NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 with TP>1

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#34476](https://github.com/vllm-project/vllm/pull/34476) |
| 作者 | vadiklyutiy |
| 合并日期 | 2026-02-15 |
| labels | bug, nvidia, ready |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
TP>1 时量化 Mamba 模型的 scale 参数被默认连续切分,与权重行错位导致精度回归;回退统一 loader 改动,恢复 MergedColumnParallelLinear。

## 问题现象
NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 模型在 TP>1 下出现精度回归。作者在 8×B200、TP=8 上跑 gsm8k 5-shot(limit=500)验证:修复前精度异常(本 PR 只给出修复后 3 次运行 flexible-extract 0.430/0.416/0.424、strict-match 0.760/0.746/0.730,3/3 GOOD),修复后连续 3 次运行全部正常。影响面是使用 NVFP4/FP8 等量化的 Mamba 结构模型且 TP>1 的部署。

## 根因
正文披露:PR #33257(commit `a372f3f40`)是两个逻辑改动的 squash-merge,其中第二个改动(`d5d6d0b88`,"Unify MambaMixer2 TP sharding to use custom weight loader")把所有情况下的 `MergedColumnParallelLinear` 换成了 `ColumnParallelLinear` + 自定义 `mamba_v2_sharded_weight_loader`,但只覆盖了 `in_proj.weight` 的 weight_loader。对量化模型(NVFP4、FP8),还存在 scale 参数(`weight_scale`、`input_scale`、`weight_scale_2`),它们仍走 `ColumnParallelLinear` 的默认连续切分。mamba `in_proj` 的复合权重布局为 `[gate, intermediate, B_groups, C_groups, dt_heads]`,自定义 loader 按分量分别切分,而 scale 参数只做简单连续切分,造成 weight-scale misalignment——第 N 行的反量化 scale 对应的不再是第 N 行权重所属的模型分量。`MergedColumnParallelLinear` 知道每个分量的输出大小,能对所有参数(权重和 scale)正确切分,因此旧实现是对的。

## 修复方案
回退 `d5d6d0b88`(即 #33257 的最后一个 commit),改动集中在 `vllm/model_executor/layers/mamba/mamba_mixer2.py`:恢复 `n_groups % tp_size == 0` 情况下用 `MergedColumnParallelLinear`(conv1d 与 in_proj 都传 `output_sizes=[...]`),`n_groups == 1` 的量化 TP 支持分支保留 `ColumnParallelLinear` + 自定义 loader 不变。

```diff
+        if n_groups % self.tp_size == 0:
+            self.conv1d = MergedColumnParallelLinear(
+                input_size=conv_kernel_size,
+                output_sizes=[
+                    intermediate_size,
+                    self.groups_ssm_state_size,
+                    self.groups_ssm_state_size,
+                ],
...
+            self.in_proj = MergedColumnParallelLinear(
+                input_size=hidden_size,
+                output_sizes=[
+                    intermediate_size,
+                    intermediate_size,
+                    self.groups_ssm_state_size,
+                    self.groups_ssm_state_size,
+                    self.num_heads,
+                ],
```

## 影响与回归风险
- 影响窗口:自 PR #33257(commit `a372f3f40`,含 `d5d6d0b88`)合入至本修复之间,量化 Mamba 模型 + TP>1 的版本。
- 回归面:`mamba_mixer2.py` 的 TP 分片构建路径。`n_groups % tp_size == 0` 的模型(量化与非量化)从自定义 loader 路径切回 `MergedColumnParallelLinear`,权重加载行为需与 #33257 之前一致;`n_groups == 1` 分支(如 Falcon-H1R-7B FP8)逻辑保留,不受影响;非 TP 场景 tp_size==1 两种实现等价。

## 关联
- #34151(本 PR 的替代方案)
- #33257(引入问题的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
