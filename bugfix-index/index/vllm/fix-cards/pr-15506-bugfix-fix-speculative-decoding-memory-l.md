# PR #15506: [BugFix] fix speculative decoding memory leak when speculation is disabled

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#15506](https://github.com/vllm-project/vllm/pull/15506) |
| 作者 | noah-yoshida |
| 合并日期 | 2025-05-01 |
| labels | bug, ready, speculative-decoding |
| 热度 | 👍 0 · 💬 9 |
| 分类 | Speculative Decoding |
| bug 类型 | resource |

## 一句话结论
`--speculative-disable-by-batch-size` 走 `_run_no_spec` 时从不释放 `previous_hidden_states`,CUDA 内存持续增长最终 OOM;补充 prune 调用修复。

## 问题现象
使用 `--speculative-disable-by-batch-size` 时,`spec_decode_worker` 上的 `self.previous_hidden_states` 存在内存泄漏。可通过循环调用 `llm.generate` 并打印 `torch.cuda.memory_allocated()` 观察:也可在 speculative decoding worker 的 `execute_model` 里硬编码 `no_spec=True`,模拟模型超过 max speculative batch size 后长期停留的情形。现实影响:一旦 batch 超过阈值 X,高负载往往持续很久(如整个工作日 8 小时),期间显存不停增长,最终 CUDA OOM、性能越来越差。作者复现配置:speculative_model + quantization fp8、enable_prefix_caching=False、enable_chunked_prefill=True、num_speculative_tokens=3、max_tokens=100。

## 根因
正文披露:这些张量只在 speculative decoding 生效的代码路径里被清理;当 speculation 因 batch size 达到阈值被禁用后,执行流走 `_run_no_spec` 函数,该路径从不释放 hidden states,于是持续累积。作者认为此前没被发现是因为测试场景是「禁用后很快又重新启用」——重新启用时通过 `self.previous_hidden_state = None` 把内存清掉了;而真实负载中长期停留在禁用状态才会暴露泄漏。

## 修复方案
改 `vllm/spec_decode/spec_decode_worker.py` 的 `_run_no_spec`:在遍历 `seq_group_meta_with_hidden` 调用 `self.previous_hidden_states.update(...)` 之后,补一行 `self.previous_hidden_states.prune(seq_group_meta_with_hidden)`,让 no-spec 路径也及时修剪已完成的 hidden states。同时新增回归测试 `tests/spec_decode/test_memory_usage.py`:以 batch 5 大于 disable 阈值 2 的配置跑 100 轮,断言 `torch.cuda.memory_allocated()` 不增长。

```python
self.previous_hidden_states.update(hidden_states,
                                   seq_group_meta_with_hidden)
+                self.previous_hidden_states.prune(seq_group_meta_with_hidden)
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:`spec_decode_worker.py` 的 `_run_no_spec` 路径,即使用 `--speculative-disable-by-batch-size` 且当前 batch 超阈值的所有 speculative decoding 部署;prune 只针对本请求已更新的 hidden states 条目,speculation 正常启用路径不变;新增测试依赖 CUDA 显存断言,仅在 GPU 测试环境生效。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
