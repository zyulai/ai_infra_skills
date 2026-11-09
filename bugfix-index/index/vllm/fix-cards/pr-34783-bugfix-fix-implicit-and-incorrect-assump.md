# PR #34783: [BugFix] Fix implicit and incorrect assumption on ECConnector is_producer

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#34783](https://github.com/vllm-project/vllm/pull/34783) |
| 作者 | furionw |
| 合并日期 | 2026-03-04 |
| labels | bug, documentation, kv-connector, ready, v1 |
| 热度 | 👍 0 · 💬 9 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
`is_producer` 被隐含当作"只编码不 prefill",聚合节点 `ec_both` 角色下误判导致不分配 KV cache、forward 失败;改为以 consumer 语义判断。

## 问题现象
正文描述:在此 PR 之前,聚合(aggregated)节点开启 `ec_both` 会直接崩溃——vLLM 认为自己只做 encode(实际同时做 prefill/输出 token)而不分配 KV cache;同时 `engine.py` 不更新 `model_executed`,forward pass 也无法工作。

## 根因
正文披露:#25233 引入 disagg encoder 时隐含假设 `is_producer` 指只做编码(不做 prefill)的 worker;#34182 引入后,`ec_both` 角色的聚合节点可以同时是 producer 和 consumer 并输出 token,这类 P/PD worker 天然需要 KV cache,"`is_producer` → 只编码"的假设不再成立,相关检查随之失效。

## 修复方案
- `gpu_model_runner.py`、`engine/core.py`、`ray_executor.py`:把 `is_producer` / `not is_ec_producer` 判断全部换成 `not is_consumer` / `is_ec_consumer` 语义——例如 `get_kv_cache_spec` 的空返回条件由 `is_producer` 改为 `not is_consumer`,`core.py` 中 `is_ec_producer` 改为 `is_ec_consumer`(config 缺省时视为 consumer),`step_with_batch_queue` 与 `ray_executor.uses_sampler` 同步翻转。
- `scheduler.py`:encoder cache 分配循环(`encoder_inputs_to_schedule` 两处)之后调用 `self.ec_connector.update_state_after_alloc(request, i)`,保持 connector 状态同步;`example_connector.py` 的 `update_state_after_alloc` 内部改为仅在 `is_consumer` 且 `has_cache_item(mm_hash)` 时记录加载项。
- 新增单机 `ec_both` 示例脚本 `examples/online_serving/ec_both_encoder/ec_both_encoder.sh`;测试相应更新(`update_state_after_alloc` 现在对所有路径调用,local 命中不调用、no_where 场景调用但 connector 内 no-op)。
- 基准(单节点 ec_both,200 请求):吞吐 8.29 req/s,Mean TTFT 12857.62 ms,Mean TPOT 108.71 ms。

## 影响与回归风险
- 影响窗口:#25233/#34182 引入 EC connector 与 `ec_both` 角色起至本 PR。
- 回归面:所有 EC transfer(ec connector)的角色判断路径——producer/consumer 语义翻转影响 KV cache 分配、`model_executed` 判定、sampler 启用(ray executor)与 encoder cache 的 connector 状态同步;disagg 纯 encoder/prefill/decode 拓扑(`disagg_1e1p1d`、`disagg_1e1pd`)与单 ec_consumer 实例经正文测试验证。

## 关联
- #25233(引入 disagg encoder 与 is_producer 假设)
- #34182(引入 ec_both 聚合节点能力)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
