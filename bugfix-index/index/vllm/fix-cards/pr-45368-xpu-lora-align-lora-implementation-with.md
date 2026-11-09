# PR #45368: [xpu][lora]: Align LoRA implementation with Punica GPU: fix _apply_expand rank mismatch, add_inputs hardcode, and MoE EP

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#45368](https://github.com/vllm-project/vllm/pull/45368) |
| 作者 | chaojun-zhang |
| 合并日期 | 2026-07-01 |
| labels | intel-gpu, ready, ci/build |
| 热度 | 👍 1 · 💬 0 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
修复 XPU LoRA 相对 Punica GPU 实现的四处偏差:`_apply_expand` TP>1 rank 维度不匹配、`add_inputs` 硬编码/未透传、MoE EP 路径对齐块大小计算错误。

## 问题现象
正文未描述具体现象(无复现步骤或日志),仅说明这是 PR #42290 与 #41184 之后遗留的多个 bug 的集中修复,涉及:TP > 1 且非 fully-sharded LoRA 时 `_apply_expand` 的 rank 维度不匹配;`add_lora_linear` 中 `add_inputs` 被硬编码——调用方传入的值被忽略(双流 dual-stream LoRA 传 `add_inputs=False` 被破坏);`add_lora_fused_moe` 未转发 `add_inputs`;`moe_lora_align_block_size` 的 MoE LoRA EP 路径问题。

## 根因
正文列出四点:1) `_apply_expand` rank 维度不匹配(TP > 1、非 fully-sharded LoRA);2) `add_lora_linear` 中 `add_inputs` kwarg 被 pop 出来后,`add_expand` 仍固定以 `add_inputs=True` 调用,忽略调用方取值;3) `add_lora_fused_moe` 未把 `add_inputs` 传下去;4) `moe_lora_align_block_size` 的 EP 路径有错。具体机制:diff 注释披露(由 diff 推断)——TP all-gather 后 `x.size(1)` 变成 `max_lora_rank * tp_size` 而 lora_b 只用 `max_lora_rank`,XPU C++ kernel 要求二者相等,需截断;EP 下调用方传 `local_num_experts` 但 `topk_ids` 携带全局 expert 索引,kernel 按 `num_experts` 建 bucket 表会溢出,须按 `expert_map.numel()`(全局专家数)取尺寸,并由 kernel 内的 `expert_map` 做全局到局部转换。

## 修复方案
- `vllm/lora/punica_wrapper/punica_xpu.py` `_apply_expand`:rank 维不匹配时截断并 `x[:, :rank].contiguous()`;`add_expand`/`add_lora_logits` 修正 `y = y.view_as(y_org)` 赋值;`add_lora_linear` 改为断言 buffer 由函数内创建,并 `add_inputs = kwargs.pop("add_inputs", True)` 后透传给 `add_expand`;
- `moe_lora_align_block_size`:新增 `token_lora_mapping` 参数覆盖全局映射(EP+LoRA 在 all-to-all 后注入 per-rank 本地映射);`kernel_num_experts = expert_map.numel() if expert_map is not None else num_experts` 用于 padded 尺寸与 kernel 调用;`expert_ids` 用 `torch.full(..., -1)` 初始化,防止空块把 Triton kernel 引到错误 bucket;返回值由 `None` 改为 `token_lora_mapping`;
- `add_lora_w13`/`add_lora_w2` 透传 `add_inputs`、`token_lora_mapping`,`naive_block_assignment` 加 `not fully_sharded` 条件;
- `vllm/lora/ops/xpu_ops/lora_ops.py`:bgmv 三个算子经 `direct_register_custom_op` 注册为 custom op(含 fake impl);同步更新 CI deselect 列表。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:XPU LoRA 全部执行路径——TP>1 的 expand、dual-stream(非默认 `add_inputs`)、MoE LoRA 尤其 EP 场景的对齐与 kernel 调用、以及算子注册方式(改为 torch custom op,影响 compile/fake 路径);`add_lora_linear`/`add_lora_logits` 现在断言 `buffer is None`,外部传 buffer 会直接报错。

## 关联
- https://github.com/vllm-project/vllm/pull/42290
- https://github.com/vllm-project/vllm/pull/41184

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
