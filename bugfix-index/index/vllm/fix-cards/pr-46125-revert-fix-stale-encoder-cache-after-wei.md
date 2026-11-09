# PR #46125: Revert "Fix Stale Encoder Cache After Weight Update"

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#46125](https://github.com/vllm-project/vllm/pull/46125) |
| 作者 | SumanthRH |
| 合并日期 | 2026-06-20 |
| labels | frontend, v1 |
| 热度 | 👍 1 · 💬 3 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
回滚 #45093:每次权重更新后不应静默重置 prefix/encoder cache,是否重置应由用户决定。

## 问题现象
正文未描述崩溃类或输出错误类现象;问题出在行为层面:#45093 在 `finish_weight_update()` 中自动执行 `reset_prefix_cache()` 与 `reset_encoder_cache()`,导致每次权重更新结束后缓存都被静默清空。维护者认为这一行为不合理——是否重置 prefix cache 应由用户决定,不应在每次权重更新后悄悄发生。

## 根因
正文披露:这是设计层面的回归——#45093 的方案错误地假定权重更新后必须自动失效缓存。正文原话:"The decision to reset prefix cache should be left to the user. We should not be silently resetting prefix cache after every weight update"。作者同时表示对 encoder cache 拿不准(多模态 async RL 场景下重置 encoder cache 是否是典型做法尚不确定),因此整段回滚。

## 修复方案
直接 revert vllm-project/vllm#45093,删除两处缓存重置调用:

- `vllm/entrypoints/llm.py` 的 `finish_weight_update()`:移除 `self.llm_engine.reset_prefix_cache()` 与 `self.llm_engine.reset_encoder_cache()`(连同说明注释);
- `vllm/v1/engine/async_llm.py` 的 `finish_weight_update()`:移除 `await self.reset_prefix_cache()` 与 `await self.reset_encoder_cache()`。

```diff
     def finish_weight_update(self) -> None:
         """Finish the current weight update."""
         self.llm_engine.collective_rpc("finish_weight_update")
-        # Invalidate cached state computed with the old weights so it isn't
-        # reused for subsequent requests:
-        # - prefix cache: KV blocks computed with the old weights
-        # - encoder cache: multimodal embeddings keyed only by mm_hash
-        self.llm_engine.reset_prefix_cache()
-        self.llm_engine.reset_encoder_cache()
```

## 影响与回归风险
- 影响窗口:#45093 合并之后至本回滚之前,所有调用 `finish_weight_update()` 的权重更新流程。
- 回归面:回滚后权重更新结束不再自动清缓存——同步 `LLM` 与异步 `AsyncLLM` 两条入口路径均受影响。按正文注释所述,旧权重计算的 prefix cache(KV blocks)与仅按 mm_hash 键控的 encoder cache 可能被后续请求复用;依赖自动失效的 async RL/权重更新用户需自行显式调用 reset。而 #45093 原本要解决的 stale cache 问题,在用户不主动重置时会重新存在(正文已知悉并接受,决定权交还用户)。

## 关联
- PR #45093(被回滚的改动)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
