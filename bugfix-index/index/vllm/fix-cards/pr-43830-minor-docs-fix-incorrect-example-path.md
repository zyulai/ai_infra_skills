# PR #43830: minor docs: fix incorrect example path

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#43830](https://github.com/vllm-project/vllm/pull/43830) |
| 作者 | JINO-ROHIT |
| 合并日期 | 2026-05-28 |
| labels | documentation, ready, verified |
| 热度 | 👍 1 · 💬 4 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
mooncake connector 使用文档中的 proxy 脚本路径多了一层 `disaggregated_serving/`,照抄会找不到文件;改为实际路径。

## 问题现象
正文未描述具体现象(无日志或报错),作者仅说明:在测试 mooncake 时发现文档里的 proxy 文件路径是错的。影响面为照抄 `docs/features/mooncake_connector_usage.md` Proxy 小节命令的用户——按文档路径执行会因文件不存在而失败。

## 根因
正文未披露根因,只陈述路径不正确这一事实。由 diff 推断:文档中 proxy 脚本路径写成了 `examples/disaggregated/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py`,多了一层 `disaggregated_serving/` 目录,与仓库实际文件位置 `examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py` 不符,疑似目录调整后文档未同步。

## 修复方案
`docs/features/mooncake_connector_usage.md` 的 Proxy 小节,单行修正示例命令中的脚本路径:

```diff
-python examples/disaggregated/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py --prefill http://192.168.0.2:8010 --decode http://192.168.0.3:8020
+python examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py --prefill http://192.168.0.2:8010 --decode http://192.168.0.3:8020
```

命令的其余参数(`--prefill`、`--decode` 及 `vllm serve` 部分均不变。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:纯文档改动,不影响任何代码路径;仅 mooncake connector 使用文档中 Proxy 示例命令的路径文本。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
