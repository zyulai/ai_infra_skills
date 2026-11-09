# PR #45715: [LoRA] Gate all_gather on fully_sharded_loras inside _mcp_apply; rewrite regression test

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#45715](https://github.com/vllm-project/vllm/pull/45715) |
| 作者 | lcheng321 |
| 合并日期 | 2026-06-23 |
| labels | bug, ci/build, cpu, deepseek, documentation, frontend, gpt-oss, intel-gpu, kv-connector, llama, multi-modality, new-model, nvidia, performance, qwen, ready, rocm, rust, speculative-decoding, tool-calling, v1 |
| 热度 | 👍 1 · 💬 9 |
| 分类 | Parallelism / Communication |
| bug 类型 | regression |

## 一句话结论

_mcp_apply 无条件 all_gather 把 rank 维撑大,输出静默出错,现按 fully_sharded_loras 门控。

## 问题现象

TP > 1 且 `fully_sharded_loras=False` 时,LoRA 输出静默错误(不崩溃、无报错)。触发条件(正文照抄):`enable_lora=True`、`tensor_parallel_size > 1`、`fully_sharded_loras=False`,且 LoRA 挂在合并的 column-parallel 模块(如 `gate_proj`/`up_proj`)上。作者在 2xGPU(TinyLlama-1.1B, TP=2, gate_proj/up_proj LoRA)上用 debug 日志验证修复后每次 forward 都跳过 all_gather;端到端输出差异无法演示——`add_expand` 内核对 rank 维不匹配不报错、静默产出错误输出,且所用 LoRA 权重(lora_B std=0.01, bfloat16)太小,损坏的 LoRA 贡献不足以改变贪心 token 选择。

## 根因

按正文披露:`_mcp_apply()` 无条件对 shrink buffer 调用 `tensor_model_parallel_all_gather()`。在 `fully_sharded_loras=False` 时,lora_A 是每 rank 复制完整 `max_lora_rank` 的,all_gather 会把 rank 维膨胀到 `max_lora_rank * tp_size`,再传给期望 `max_lora_rank` 的 `add_expand`,从而静默产生错误的 LoRA 输出。

## 修复方案

按正文:在 `_mcp_apply()` 内把 all_gather 门控在 `fully_sharded_loras` 上,保留所有调用方的双流路径:

```python
if layer.lora_config.fully_sharded_loras and layer.tp_size > 1:
    buffers = tensor_model_parallel_all_gather(buffers)
```

`*WithShardedLoRA` 子类继续按原样调用 `_mcp_apply()`。diff(节选,见下方说明)还显示:

- `vllm/lora/layers/column_parallel_linear.py`:`_mcp_apply` 文档串改为 "Fully-sharded (S-LoRA) apply path",并新增断言 `assert layer.lora_config.fully_sharded_loras, "_mcp_apply is only used for fully sharded LoRA"`;
- 某非分片 LoRA 列并行类的 `apply()` 由 `return _mcp_apply(x, bias, self)` 改为 `return super().apply(x, bias)`,使非 fully-sharded 场景不再进入该函数。

新增单元测试 `test_mcp_apply_all_gather_gated_by_fully_sharded_loras`,用 mock 的 all_gather 验证两个分支,并断言到达 `add_expand` 的 buffer rank 维在两种情况下都等于 `max_lora_rank`。

说明:pack 中 diff 在 `can_replace_layer` 处截断结束,正文中给出的门控代码片段未完整出现在 pack 的 diff 节选里,以正文描述为准。

## 影响与回归风险

- 影响窗口:由 issue #45691 对应的回归引入起至本修复前;正文未写具体版本,不写。
- 回归面:仅 LoRA 列并行的 apply 路径。`fully_sharded_loras=True`(S-LoRA)路径行为不变;`fully_sharded_loras=False` + TP>1 时不再执行 all_gather,rank 维保持 `max_lora_rank`,输出恢复正确;新增 assert 会让误用 `_mcp_apply` 的调用方显式报错。

## 关联

- [issue #45691](https://github.com/vllm-project/vllm/issues/45691)(本 PR 修复的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
