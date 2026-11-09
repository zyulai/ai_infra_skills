# PR #34610: Revert "[Misc] fix qwen3.5 config"

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#34610](https://github.com/vllm-project/vllm/pull/34610) |
| 作者 | ywang96 |
| 合并日期 | 2026-02-16 |
| labels | qwen |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Model-specific Fast Path |
| bug 类型 | regression |

## 一句话结论
#34604 对 qwen3.5 配置的修改是为 transformers v5 准备的;回退以避免当前版本不兼容。

## 问题现象
正文未描述具体现象,仅说明被回退的 #34604 "It was meant for transformers v5"(它是为 transformers v5 准备的)。

## 根因
正文:被回退的改动本意面向 transformers v5,不适合当前代码库(具体失败症状未披露)。从 diff 看,涉及 `qwen3_5.py` 与 `qwen3_5_moe.py` 配置类中 `ignore_keys_at_rope_validation` 的容器类型:回退把它从 set(`{...}`)恢复为 list(`[...]`)(由 diff 推断)。

## 修复方案
回退 PR #34604:`vllm/transformers_utils/configs/qwen3_5.py` 与 `vllm/transformers_utils/configs/qwen3_5_moe.py` 的 `__init__` 中,`kwargs["ignore_keys_at_rope_validation"]` 由 set 恢复为 list,元素仍为 `"mrope_section"` 与 `"mrope_interleaved"`。

```diff
-        kwargs["ignore_keys_at_rope_validation"] = {
+        kwargs["ignore_keys_at_rope_validation"] = [
             "mrope_section",
             "mrope_interleaved",
-        }
+        ]
```

## 影响与回归风险
- 影响窗口:正文未写(自 #34604 合入起,本 PR 为其回退)。
- 回归面:qwen3.5 与 qwen3.5 MoE 配置类构造参数;rope 校验忽略键的容器类型恢复为 list。transformers v5 场景的适配随后另行处理(正文未展开)。

## 关联
- #34604(被回退的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
