# PR #15279: [Bugfix] Fix incorrect resolving order for transformers fallback

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#15279](https://github.com/vllm-project/vllm/pull/15279) |
| 作者 | Isotr0py |
| 合并日期 | 2025-03-21 |
| labels | ready |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
模型多架构归一化时 transformers 回退项位置不定,可能排在受支持架构之前;改为先过滤出已注册架构,再把回退项固定放到列表末尾。

## 问题现象
正文未描述具体现象,只说明修复目标:确保 transformers fallback 总是被最后解析(made sure transformers fallback are always resolved at last),起因是 PR #15023 中一条 review 评论(#discussion_r2007002093)指出的解析顺序问题。

## 根因
正文未披露根因,仅链接到 PR #15023 的 review 评论。
(由 diff 推断)原 `_normalize_archs` 对 config 中的 architectures 逐个遍历,凡不在注册表中的架构就被原地替换为 `"TransformersModel"`。这样回退项保留在原列表中的位置,可能出现在受支持架构之前;当模型同时声明受支持与不受支持的架构时,解析顺序不再是「优先原生实现、回退兜底」,首选项可能落到通用 TransformersModel 上。

## 修复方案
改 `vllm/model_executor/models/registry.py` 的 `_normalize_archs`:先用 `filter` 只保留注册表中存在的架构;若有架构被过滤掉,再把 `"TransformersModel"` 追加到列表末尾,保证回退始终殿后。

```diff
-        normalized_arch = []
-        for model in architectures:
-            if model not in self.models:
-                model = "TransformersModel"
-            normalized_arch.append(model)
+        # filter out support architectures
+        normalized_arch = list(
+            filter(lambda model: model in self.models, architectures))
+
+        # make sure Transformers fallback are put at the last
+        if len(normalized_arch) != len(architectures):
+            normalized_arch.append("TransformersModel")
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:所有经 `_normalize_archs` 归一化架构列表的模型解析路径,典型是 config.json 声明多个 architectures 且含未注册架构、依赖 transformers 回退的模型。列表顺序与内容变化会影响依赖该顺序的 `inspect_model_cls` 等后续解析逻辑;对全部架构都已注册的模型,行为不变(长度相等时不追加)。

## 关联
- [PR #15023 review 评论](https://github.com/vllm-project/vllm/pull/15023#discussion_r2007002093)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
