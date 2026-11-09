# PR #48113: [Bugfix][Spec Decode] Fix DFlash draft/target layer-count mismatch

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#48113](https://github.com/vllm-project/vllm/pull/48113) |
| 作者 | njhill |
| 合并日期 | 2026-07-10 |
| labels | bug, qwen, ready |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

DFlash 测试误配 Qwen3.5-4B 目标模型(层数不足),drafter 收到 4 路而非 5 路 aux 特征报 matmul 错;改测试配对并加清晰报错。

## 问题现象

DFlashDraftModel 测试把 drafter `z-lab/Qwen3-4B-DFlash-b16` 与目标 `Qwen/Qwen3.5-4B` 配对运行时,在 drafter 的 `fc` 层报出难以理解的错误:

```text
mat1 and mat2 shapes cannot be multiplied (Nx10240 and 12800x2560)
```

该错误在 Model Runner V2 下于 engine 初始化时快速失败(V2 在 memory profiling 的 `profile_run -> _dummy_run` 中就会执行 drafter 的 `propose()`);V1 不对 drafter 做 profiling,初始化测试能通过,同样的错误只会推迟到第一个 decode step 才出现。

## 根因

PR 正文披露:这是不兼容的模型配对而非 runner bug。该 drafter 的 `dflash_config.target_layer_ids` 为 `[1, 9, 17, 25, 33]`,解析时 +1 得 aux 层 (2, 10, 18, 26, 34),要求目标模型至少 35 层;Qwen3.5-4B 只有 32 层,第 34 层的 aux hidden state 永远不会被产生,drafter 的 `fc` 按 5 路拼接 aux 特征设计却只收到 4 路,于是 matmul 形状不匹配。drafter 真正的目标是 Qwen3-4B(36 层),配对正常。

## 修复方案

1. `tests/models/registry.py`:把 DFlashDraftModel 测试的目标模型从 `Qwen/Qwen3.5-4B` 改为其真实目标 `Qwen/Qwen3-4B`。
2. `vllm/model_executor/models/qwen3_dflash.py` 的 `combine_hidden_states`:在拼接送入 `fc` 前校验特征数,不匹配时抛出带解释的 `ValueError` 而非晦涩的 matmul 失败(V1、V2 均受益):

```diff
+        expected = self.model.fc.input_size
+        if hidden_states.shape[-1] != expected:
+            raise ValueError(
+                f"DFlash drafter expects {expected} concatenated aux hidden "
+                f"features but received {hidden_states.shape[-1]}. This usually "
+                "means the draft model's target_layer_ids reference layers that "
+                "do not exist in the target model (incompatible draft/target pair)."
+            )
         result = self.model.fc(hidden_states)
```

## 影响与回归风险

- 影响窗口:正文未说明具体版本/提交范围;V2 行为差异(初始化即失败 vs 首个 decode 失败)与 Model Runner V2 的 drafter profiling 相关。
- 回归面:`combine_hidden_states` 新增的前置校验只在不匹配时抛错,正常配对路径行为不变;测试注册表仅改测试用模型配对,不影响生产代码路径。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
