# PR #33565: [Fix] Restore data_parallel_rank alias on native /generate (dp-aware gateway routing is silently dropped)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#33565](https://github.com/sgl-project/sglang/pull/33565) |
| 作者 | sshleifer |
| 合并日期 | 2026-08-08 |
| labels | run-ci, bypass-fastfail, run-ci-extra |
| 热度 | 👍 1 · 💬 5 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
#29214 清理时从 GenerateReqInput 删掉了 data_parallel_rank 别名,而网关 dp-aware 模式仍注入旧字段名,原生 /generate 的 DP 路由静默退化为负载均衡;修复为恢复该别名映射到 routed_dp_rank。

## 问题现象
`sgl-model-gateway` 的 dp-aware 模式给每个转发的请求注入 `data_parallel_rank` 字段,但在原生 `/generate` 路径上,当前服务端静默丢弃该字段,网关的路由决策到不了 DP controller,分发退回负载均衡——正文称「dp-aware routing over native /generate is currently a silent no-op」。作者在生产 dp=4 部署(网关 dp-aware、session-keyed 手动路由)中命中:路由被静默丢弃时 prefix-cache 命中率约 0.49,而 rank 生效时约 0.97,agentic 负载 decode 吞吐差约 2 倍。机械验证:`/generate` 带 `routed_dp_rank=2` 只动 rank 2 的 prefill 计数器,而 `data_parallel_rank=2` 移动的是负载均衡选中的任意 rank。

## 根因
正文披露:#19268 把 `GenerateReqInput.data_parallel_rank` 改名为 `routed_dp_rank`,并在 `normalize_batch_and_arguments()` 保留带 DeprecationWarning 的旧名别名;#29214(2026-06-25 的清理 PR)删除了该别名——但网关仍发旧字段名,且另外两条入口(OpenAI 兼容层 `_migrate_deprecated_dp_rank`、`Engine.generate()`)都还保留别名,原生 HTTP `/generate` 是唯一丢失该别名的入口。静默的原因:FastAPI 把 `GenerateReqInput` 解析为忽略未知键的 pydantic dataclass,不报错;`routed_dp_rank` 保持 None;DP controller 对 None 有合法回退(负载均衡)。故障模式是纯性能退化(prefix-cache 命中率与路由质量悄悄变差),难以察觉和定位。

## 修复方案
按正文「Restore exactly what #29214 removed」:(1) `python/sglang/srt/managers/io_struct.py` 的 `GenerateReqInput` 恢复弃用字段 `data_parallel_rank`;(2) 在 `normalize_batch_and_arguments()` 恢复映射——显式 `routed_dp_rank` 优先,别名只在其为 None 时生效,映射后置空并发出原 DeprecationWarning。关键 diff:

```python
+        if self.data_parallel_rank is not None:
+            import warnings
+            warnings.warn(
+                "'data_parallel_rank' is deprecated, use 'routed_dp_rank' instead.",
+                DeprecationWarning,
+                stacklevel=2,
+            )
+            if self.routed_dp_rank is None:
+                self.routed_dp_rank = self.data_parallel_rank
+            self.data_parallel_rank = None
```

并在 `test/registered/unit/managers/test_io_struct.py` 新增 3 个单测:别名映射到 routed_dp_rank、别名不覆盖显式 routed_dp_rank、别名传播到 batch items(pytest 该文件 30 passed)。

## 影响与回归风险
- 影响窗口:正文给出兼容矩阵——#19268→#29214 之间正常(有别名);#29214→本修复之间在原生 /generate 上静默失效;≥本修复恢复(别名)。
- 回归面:影响所有构造 `GenerateReqInput` 的路径(原生 /generate、Vertex endpoint、engine API 等):带 `data_parallel_rank` 的请求会触发 DeprecationWarning 并映射到 routed_dp_rank,且别名字段在归一化后被置 None;已直接用 routed_dp_rank 的请求不受影响(显式值优先)。

## 关联
- [#19268](https://github.com/sgl-project/sglang/pull/19268)(改名引入别名)
- [#29214](https://github.com/sgl-project/sglang/pull/29214)(删除别名导致回归)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
