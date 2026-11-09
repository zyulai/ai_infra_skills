# PR #8168: Fix suffix mismatch for the metrics.

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#8168](https://github.com/sgl-project/sglang/pull/8168) |
| 作者 | CharlesRiggins |
| 合并日期 | 2025-07-19 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Frontend / Serving |
| bug 类型 | correctness |

## 一句话结论
sgl-router 的 duration 指标后缀匹配器写成 `"duration"`,匹配不到 `*_duration_seconds` 指标,histogram 退化成 summary;后缀改为 `"duration_seconds"`。

## 问题现象
后缀匹配器匹配不到指标名 `sgl_router_request_duration_seconds` 与 `sgl_router_pd_request_duration_seconds`,导致匹配器不按预期为这些指标设置 bucket;没有 bucket 信息时,histogram 指标退化为 summary 指标,实际输出形如:

```
sgl_router_generate_duration_seconds{route="/generate",quantile="0.9"} 10
sgl_router_generate_duration_seconds{route="/generate",quantile="0.95"} 20
sgl_router_generate_duration_seconds{route="/generate",quantile="0.99"} 15
```

而原本期望的输出是带 `_bucket{le=...}` 的 histogram 形式(正文以 `sglang:time_to_first_token_seconds_bucket{le="0.1",...}` 示例)。

## 根因
PR 正文披露:后缀 `"duration"` 不匹配以 `duration_seconds` 结尾的指标名,正确后缀应为 `"duration_seconds"` 才能匹配 `sgl_router_request_duration_seconds` 等指标;匹配失败即不会设置 bucket,histogram 退化为 summary。

## 修复方案
`sgl-router/src/metrics.rs` 中一行修正:

```diff
-    let duration_matcher = Matcher::Suffix(String::from("duration"));
+    let duration_matcher = Matcher::Suffix(String::from("duration_seconds"));
```

匹配成功后,这些 duration 指标将使用紧随其后的 duration_bucket 分桶配置(0.001 … 240)。

## 影响与回归风险
- 影响窗口:正文未给出受影响版本范围。
- 回归面:仅 `sgl-router` 的 Prometheus 指标初始化一处字符串。行为变化:duration 类 histogram 指标从输出 quantile summary 恢复为输出 bucket histogram;若有监控系统/告警按 `quantile=` 标签读取这些指标,抓取形态改变后需要相应调整。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
