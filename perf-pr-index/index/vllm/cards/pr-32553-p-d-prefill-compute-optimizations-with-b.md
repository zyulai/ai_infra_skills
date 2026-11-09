# PR #32553: [P/D] Prefill compute optimizations with bi-directional KV cache transfers between P and D nodes

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @snadampal |
| 合并 | 2026-04-30 |
| 分类 | KV Cache / Memory |
| Label | documentation, performance, frontend, ready, v1, kv-connector |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 部分可用,pack 注明另 dropped 1 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/32553 |

## 一句话结论

P 节点可回拉 D 节点 KV 免去重复 prefill 计算,多轮对话 TTFT 最高 >2x 改善。

## 动机

P/D 分离部署中 KV 传输是单向的:prefill 完成后 P 把 KV 推给 D,而 D 上累积的完整上下文 KV 无法回流。多轮对话里,下一轮 prompt 与上一轮高度重叠,但 P 节点要么因本地前缀缓存已被驱逐而重复计算全部前缀,要么在关闭 prefix caching 时完全重算。D 节点上明明还保留着这些块,却只能白白释放。另一场景是 P 本地缓存被驱逐的块仍可从 D 恢复。

## 优化点

在 NixlConnector 的 scheduler 侧打通 D→P 方向的 KV 拉取。D 节点 `request_finished` 在双向模式开启时也返回 `kv_transfer_params`(携带 `remote_block_ids` 与新增的 `remote_num_tokens = num_computed_tokens`),角色标志按 P/D 语义重排(`do_remote_prefill=is_p_node, do_remote_decode=is_d_node`)。P 节点 `get_num_new_matched_tokens` 看到带 `remote_block_ids` 的请求时,把 `min(remote_num_tokens, num_prompt_tokens) - num_computed_tokens` 计为 external tokens 走 `WAITING_FOR_REMOTE_KVS` 异步加载;并引入 `kv_recompute_threshold`(默认 64 tokens)低于阈值时放弃拉取、直接重算,以摊薄传输延迟。开关由 kv_connector extra config 控制:`bidirectional_kv_xfer`(默认 False)。`update_state_after_alloc` 增加 `_remote_blocks_processed` 标记防止 reschedule 后重复拉取,并兼容"本地 full prefix hit 但仍需通知远端释放"的路径。另附一个按 `conversation_id` 在 D 完成响应后缓存其块信息、下一轮转交给 P 触发回拉的 FastAPI 代理示例。

## 关键改动

```text
vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py: D 侧 finished 返回 remote_block_ids/remote_num_tokens;P 侧计 external tokens、阈值判断、防重复拉取
examples/online_serving/disaggregated_serving/disagg_proxy_multiturn.py: 按 conversation_id 跨轮缓存 D 块信息并转交 P 的代理示例
tests/v1/kv_connector/unit/test_bidirectional_kv_transfer.py: 两轮生命周期、abort、部分覆盖、阈值等单测
benchmarks/multi_turn/benchmark_serving_multi_turn.py: 请求携带 conversation_id
```

关键片段(P 侧决定是否回拉):

```python
remote_num_tokens = params.get("remote_num_tokens") or 0
count = min(remote_num_tokens, request.num_prompt_tokens) - num_computed_tokens
if count > 0:
    if (self.kv_recompute_threshold > 0
            and count < self.kv_recompute_threshold):
        return 0, False
    return count, True
```

## 性能声明

多轮对话 benchmark(vllm benchmarks/multi_turn),AWS p5en(Trainium 上增益相近),llama-3.3-70b-instruct,2 clients、24 conversations、12-18 turns:

- P 节点关闭 prefix caching(`--no-enable-prefix-caching`)、开启从 D 拉 KV:TTFT > 2x 改善(mean 1049.77 → 498.15 ms);runtime 1483.9 → 1445.1 s。
- P 节点开启 prefix cache 时:TTFT mean -18%(396.73 → 334.39 ms);median -28.6%(307.86 → 219.61 ms);p75 -28.6%(342.86 → 244.65 ms);min -41%(243.50 → 143.76 ms)。TPOT 基本不变(~21.8 ms)。

## 验证与回归风险

新增 `tests/v1/kv_connector/unit/test_bidirectional_kv_transfer.py`(cpu_test 标记):覆盖 P 节点两轮完整生命周期、首轮回退本地 prefill、abort 时块释放、remote tokens 超 prompt 的截断、部分覆盖(最后一块半满)、`_remote_blocks_processed` 防 double read、阈值跳过等;另回归运行 `test_nixl_connector.py`(56 项)与 `test_nixl_connector_hma.py`(HMA/hybrid/mamba 侧)全部通过。风险集中在 NixlConnector 调度语义:`request_finished` 的角色判断与状态机改动影响所有 P/D 部署;`remote_num_tokens` 依赖 `num_computed_tokens`,与 MLA/HMA 布局组合需靠现有单测兜底;默认关闭(bidirectional_kv_xfer=False),未开启时代码路径与 main 等价。

## 关联

同区域:PR #42296、#42258(SimpleCPUOffloadBackend,本地 CPU KV offload 的另一类前缀复用);同属 kv-connector / NixlConnector P/D 传输体系。pack 内未给出更明确的前序/后续 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
