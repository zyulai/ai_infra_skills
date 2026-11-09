# PR #20371: [PD] Fix the infinite loop in deocde resolve_pending_reqs

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#20371](https://github.com/sgl-project/sglang/pull/20371) |
| 作者 | ShangmingCai |
| 合并日期 | 2026-03-11 |
| labels | (none) |
| 热度 | 👍 0 · 💬 12 |
| 分类 | Other |
| bug 类型 | hang |

## 一句话结论

PD 分离下 decode 解析 pending 请求死循环;按 bootstrap 地址分组、失败即 abort 请求并把重试从 20 次降到 5 次。

## 问题现象

PD(prefill-decode)分离部署中,decode 侧 `_resolve_pending_reqs` 出现无限循环导致调度卡死(标题与正文,修复 issue #20252)。正文说明:只有使用 sglang-router 时才能复现——mini-lb 会在 prefill 已死时立即拒绝请求,这类请求因此进入 `pending_reqs` 并滞留;直接用 mini-lb 的场景不会触发。

## 根因

正文披露存在无限循环并给出修复计划(修复死循环 + 把过长的重试缩短),但未逐步展开机制。由 diff 可描述:原实现只取 `pending_reqs[0]` 的 bootstrap_addr 调 `ensure_parallel_info`,失败就整体 `return`,这批请求原样留在 `pending_reqs` 里,下轮循环继续以同样方式失败;加上 `ensure_parallel_info` 默认 `max_retries=20`、间隔 1 秒,最多阻塞调度约 20 秒,且请求永不清理,形成无限循环(由 diff 推断)。

## 修复方案

1. `python/sglang/srt/disaggregation/common/conn.py`:`ensure_parallel_info` 的 `max_retries` 从 20 改为 5(正文:20 秒太长,sglang-router 自带重试机制,无需在调度路径阻塞重试 20 次)。

```diff
-        self, bootstrap_addr: str, max_retries: int = 20, retry_interval: float = 1.0
+        self, bootstrap_addr: str, max_retries: int = 5, retry_interval: float = 1.0
```

2. `python/sglang/srt/disaggregation/decode.py` `_resolve_pending_reqs`:按 `bootstrap_addr` 把 pending 请求分组,逐组 ensure;某组失败时对该组请求 `prepare_abort`(HTTP 500)并 `stream_output` 返回,同时累加 `bootstrap_failed_reqs` 指标,不再整体 return 滞留;成功路径逐组查询 `query_prefill_dp_ranks`,未解析出的留入 remaining。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:PD 分离 decode 端 pending 请求处理路径。行为变化:解析失败(如 prefill 已死)的请求会被 abort 并以 500 返回,而不是无限重试占住队列;重试上限缩短后,瞬时抖动场景下失败返回会更快,依赖内部 20 秒重试窗口的部署需关注。

## 关联

- https://github.com/sgl-project/sglang/issues/20252

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
