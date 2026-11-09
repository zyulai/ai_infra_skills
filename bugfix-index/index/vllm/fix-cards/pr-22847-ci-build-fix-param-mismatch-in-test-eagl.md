# PR #22847: [CI/Build] Fix param mismatch in `test_eagle_correctness`

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#22847](https://github.com/vllm-project/vllm/pull/22847) |
| 作者 | DarkLight1337 |
| 合并日期 | 2025-08-13 |
| labels | v1 |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
#22611 移除 eagle 用例后漏删 parametrize 的 `ids` 条目,测试参数与用例不匹配;注释掉 `qwen3_eagle3` id 修复。

## 问题现象
正文未描述运行时症状,问题出在 CI 测试代码层面:`tests/v1/e2e/test_spec_decode.py` 中 eagle correctness 测试的 parametrize `ids` 列表仍包含 `"qwen3_eagle3"`,而对应的测试用例已在 #22611 中被移除(用例处还有因 CI OOM 而 skip 的标记),造成参数 id 与实际用例不匹配。作者说明会在获批后 force-merge,因为可以在本地直接跑这些测试验证。影响面为 v1 e2e speculative decoding 正确性测试的 CI 运行。

## 根因
PR 正文披露:前置 PR #22611 在移除 qwen3_eagle3 相关测试用例时,忘记同时删除 parametrize `ids` 列表中对应的参数条目,导致 `ids` 与实际用例不一致,出现 param mismatch。

## 修复方案
只改 `tests/v1/e2e/test_spec_decode.py`:在 parametrize 的 `ids` 中注释掉 `"qwen3_eagle3"` 一行,并加 TODO 注明待 `tests/models/test_initialization.py` 修复后重新启用(关联 PR #22333、#22611);同时把其余 id(`llama3_eagle`、`llama3_eagle3`、`llama4_eagle`)逐行排版。

```diff
-        "qwen3_eagle3", "llama3_eagle", "llama3_eagle3", "llama4_eagle",
+        # TODO: Re-enable this once tests/models/test_initialization.py is fixed, see PR #22333 #22611  # noqa: E501
+        # "qwen3_eagle3",
+        "llama3_eagle",
+        "llama3_eagle3",
+        "llama4_eagle",
```

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:仅改动测试文件的参数化 `ids` 列表,不涉及运行时代码,不影响推理行为;`qwen3_eagle3` 用例维持注释(跳过)状态,后续恢复依赖 tests/models/test_initialization.py 的修复(见 #22333)。

## 关联
- PR #22611(引入问题的前置改动)
- PR #22333(TODO 中提到的待修复 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
