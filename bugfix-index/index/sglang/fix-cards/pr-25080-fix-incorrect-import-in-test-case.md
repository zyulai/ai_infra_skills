# PR #25080: Fix incorrect import in test case

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#25080](https://github.com/sgl-project/sglang/pull/25080) |
| 作者 | liuxianglong17 |
| 合并日期 | 2026-05-15 |
| labels | npu, run-ci |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
Ascend NPU 的 `test_npu_qwen3_30b_attn_cp.py` 因 import 路径错误在 daily build 失败;修正为 `sglang.test.ascend.test_ascend_utils`。

## 问题现象
`test/registered/ascend/llm_models/test_npu_qwen3_30b_attn_cp.py` 在 daily build pipeline 中失败,原因是 import error。正文未附具体报错日志。

## 根因
正文未披露根因分析,仅说明该测试文件中存在不正确的 import。diff 显示被修改的行是 `from python.sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_WEIGHTS_PATH`,模块路径带了错误的 `python.` 前缀,应属 import error 的直接来源(由 diff 推断)。

## 修复方案
`test/registered/ascend/llm_models/test_npu_qwen3_30b_attn_cp.py`:仅修改一行 import(正文 Modifications;与 diff 一致)。

```diff
-from python.sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_WEIGHTS_PATH
+from sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_WEIGHTS_PATH
```

## 影响与回归风险
- 回归面:仅该 Ascend NPU 测试文件的 import;不触及任何运行时代码,对模型输出与推理路径无影响(由 diff 推断)。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
