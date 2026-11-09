# PR #27133: [Bugfix] Fix incorrect kv cache metrics in grafana.json

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#27133](https://github.com/vllm-project/vllm/pull/27133) |
| 作者 | fangpings |
| 合并日期 | 2025-10-23 |
| labels | documentation |
| 热度 | 👍 1 · 💬 4 |
| 分类 | KV Cache / Memory |
| bug 类型 | correctness |

## 一句话结论

示例 Grafana/Perses 仪表盘引用过时指标名 `vllm:gpu_cache_usage_perc` 导致面板取不到数据;统一改为 `vllm:kv_cache_usage_perc`。

## 问题现象

正文未用文字描述现象,以修复前后 dashboard 截图对比展示(Before 截图中面板无数据,After 中有数据)。正文根因段指出:当前 grafana.json 使用的是过时的 prometheus 指标名。作者在自己 grafana dashboard 上验证。

## 根因

正文披露:`examples/online_serving/prometheus_grafana/grafana.json` 使用过时的 prometheus 指标名 `vllm:gpu_cache_usage_perc`;最新指标名为 `vllm:kv_cache_usage_perc`(正文引用 vllm/v1/metrics/loggers.py 的代码位置佐证)。

## 修复方案

将 grafana.json 与 perses 两个仪表盘(`performance_statistics.yaml`、`query_statistics.yaml`)查询中的 `vllm:gpu_cache_usage_perc` 全部替换为 `vllm:kv_cache_usage_perc`,共 7 处(grafana.json 1 处 + perses 6 处),查询结构与其他参数不变。

```diff
-          "expr": "vllm:gpu_cache_usage_perc{model_name=\"$model_name\"}",
+          "expr": "vllm:kv_cache_usage_perc{model_name=\"$model_name\"}",
```

## 影响与回归风险

- 影响窗口:正文未说明(指标改名发生在 vllm/v1/metrics/loggers.py,正文未给出引入改名的具体 PR)。
- 回归面:仅 examples 目录下的示例仪表盘文件(grafana.json、perses performance_statistics.yaml、query_statistics.yaml),不涉及运行时代码。

## 关联

正文未列出 issue/PR 链接(仅引用 grafana.json 与 loggers.py 的代码文件链接)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
