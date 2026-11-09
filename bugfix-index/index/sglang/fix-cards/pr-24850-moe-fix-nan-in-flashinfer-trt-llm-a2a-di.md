# PR #24850: [MoE] Fix NaN in flashinfer TRT-LLM A2A dispatch by sanitizing padding slots

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#24850](https://github.com/sgl-project/sglang/pull/24850) |
| 作者 | minosfuture |
| 合并日期 | 2026-05-11 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
flashinfer TRT-LLM A2A dispatch 未标记 padding slot,垃圾 expert ID 致下游 NaN;传入 -1 使 sanitize kernel 清洗填充槽。

## 问题现象
启用 flashinfer TRT-LLM all-to-all(A2A)dispatch 的 MoE 路径上出现 NaN。触发条件:dispatch 收到的 buffer 中,超出每个源 rank `recv_counters` 的 padding slot 从未被清洗,里面残留未初始化的垃圾 expert ID 与 hidden states。下游 TRTLLM MoE kernel 会照常处理这些垃圾条目,读取未初始化的 hidden states,最终产生 NaN。正文记录:修复后 NaN 问题消失,内部测试显示精度评估与其他非 a2a 代码路径(TPEP)持平。

## 根因
正文明确披露:`MoeAlltoAll.dispatch()` 调用时没有传 `invalid_token_expert_id`,该参数默认为 `None`,导致 dispatch 内部的 sanitize kernel 完全跳过清洗;于是超出 `recv_counters` 的 padding slot 里保留未初始化的垃圾 expert ID,下游 TRTLLM MoE kernel 按正常 token 处理这些垃圾项,读到未初始化的 hidden states 而输出 NaN。

## 修复方案
在 `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py` 的 `dispatch()` 调用处,给 `MoeAlltoAll.dispatch()` 传入 `invalid_token_expert_id=-1`,让 sanitize kernel 把所有 padding slot(每个源 rank 上超出 `recv_counters` 的槽位)的 expert ID 标记为 -1,下游 kernel 即可跳过这些无效槽位。

```diff
             self.dummy_topk_ids_current_rank if self.has_dummy_token else topk_ids,
             payloads,
             self.runtime_max_tokens_per_rank,
+            invalid_token_expert_id=-1,
             expert_id_payload_index=expert_id_payload_index,
         )
```

改动为单行参数传递,未改动 sanitize kernel 或下游 TRTLLM kernel 本身。

## 影响与回归风险
- 回归面:仅影响 flashinfer token dispatcher(`token_dispatcher/flashinfer.py`)的 A2A dispatch 调用路径;`invalid_token_expert_id=-1` 使 padding slot 被 sanitize kernel 显式标记,TRTLLM MoE kernel 对标记为 -1 的 expert ID 的跳过逻辑成为新的依赖点,若上游 flashinfer 接口对该参数语义有变需同步关注。

## 关联
正文未列出
