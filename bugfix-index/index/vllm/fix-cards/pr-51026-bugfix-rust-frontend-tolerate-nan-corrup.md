# PR #51026: [Bugfix][Rust Frontend] Tolerate NaN-corrupted logprobs in engine-core

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#51026](https://github.com/vllm-project/vllm/pull/51026) |
| 作者 | almersawi |
| 合并日期 | 2026-09-16 |
| labels | bug, ready, rust |
| 热度 | 👍 3 · 💬 6 |
| 分类 | Frontend / Serving |
| bug 类型 | correctness |

## 一句话结论
NaN logits 使 sampled rank 为 0,Rust 客户端拒绝解码导致全服 503;改为接受并保留 rank 0 修复。

## 问题现象
Rust 前端的 engine-core 客户端存在整体性宕机 bug:只要有一个请求带 `logprobs`/`prompt_logprobs` 且模型产生 NaN logits,客户端就被永久打死——`/health` 返回 503,后续所有请求(包括不带 logprobs 的)都报 `engine-core error: messagepack ext value decode failed: token_ranks must be >= 1 for decoded engine-core logprobs`,直到进程重启。解码失败是帧级(frame-scoped)的:批内一行坏数据会让整批 `EngineCoreOutputs` 载荷解码失败;它发生在共享的 output-reader 路径上,并被记为 sticky health error,于是一个请求的数据异常升级为全服故障。

## 根因
正文披露:sampler 计算 rank 的方式是 `(logprobs >= sampled_logprob).sum(-1)`(`vllm/v1/sample/sampler.py`),当被采样 logprob 为 NaN 时所有 `>=` 比较均为 false,sum 结果为 0——rank 0 本是引擎的合法输出。而 Rust 端 `PositionLogprobs::from_decoded_row` 对 `sampled_rank == 0` 直接 `bail_ext_value_decode!` 拒绝解码,与引擎语义不符,形成故障链。

## 修复方案
改 `rust/src/engine-core-client/src/protocol/logprobs.rs`:删除 `sampled_rank == 0` 的拒绝分支,接受并保留 rank 0 与引擎原始 logprob 值,使同批其他请求的解码不受影响:

```diff
-        if sampled_rank == 0 {
-            bail_ext_value_decode!("token_ranks must be >= 1 for decoded engine-core logprobs");
-        }
```

HTTP 层不变:既有 Rust OpenAI serving helper 在构造响应时已把 NaN 与负无穷转成 `-9999.0`;engine-core 解码层则为所有消费方保留数值原样。新增两个回归测试:`zero_sampled_rank_does_not_fail_frame_mates` 验证批内 rank 0/NaN/-inf 行不再拖垮健康请求,`chat_logprobs_clamps_nan_and_negative_infinity_for_json` 锁定 HTTP 侧 -9999.0 转换。`cargo nextest run` 118 passed,clippy `-D warnings`、fmt 等全过;无需 GPU。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围;按指认,受影响的是带 Rust 前端 engine-core 客户端且客户端对该 rank 校验存在的版本。
- 回归面:wire 载荷解码容错变化,只放宽一处校验;健康载荷行为不变(正文明确)。rank 0 现在会流入下游,依赖「rank 恒 >= 1」假设的消费方需自行处理;HTTP 响应仍经 -9999.0 钳制。作者说明不与 #48585 重复(那是 Python 前端、logprob 值而非 rank)。

## 关联
- [#48585](https://github.com/vllm-project/vllm/pull/48585)(同类缺陷的 Python 前端修复)
- [#50725](https://github.com/vllm-project/vllm/pull/50725)(不同代码路径,仅爆炸半径模式相同)
- [#50002](https://github.com/vllm-project/vllm/pull/50002)(被 #50283 取代的 NaN 中止提案)
- [#50283](https://github.com/vllm-project/vllm/pull/50283)
- [#45060](https://github.com/vllm-project/vllm/pull/45060)(确认 all-NaN logits 是引擎既有处理的真实状态)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
