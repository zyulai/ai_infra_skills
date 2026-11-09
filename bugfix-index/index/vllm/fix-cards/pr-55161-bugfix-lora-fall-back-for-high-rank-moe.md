# PR #55161: [Bugfix][LoRA] Fall back for high-rank MoE LoRA

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#55161](https://github.com/vllm-project/vllm/pull/55161) |
| 作者 | xiaoyu-xyz |
| 合并日期 | 2026-10-02 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 32 |
| 分类 | MoE |
| bug 类型 | general |

## 一句话结论
`max_lora_rank>128` 时非 fully-sharded MoE LoRA 仍走限rank 128 的 one-shot kernel,CUDA graph warm-up 即失败;超限 rank 改走 shrink/expand 两 kernel 回退路径。

## 问题现象
`LoRAConfig` 允许配置大于 128 的 `max_lora_rank`,但启动即挂:引擎初始化在 CUDA graph warm-up 阶段失败,即使实际加载的 adapter rank 只有 4——因为 MoE LoRA 缓冲会 pad 到配置的最大 rank。复现方式(正文测试计划):以 `--max-lora-rank 256` 启动 #55158 中的公开 Nemotron MoE 模型。

## 根因
正文披露:非 fully-sharded 的 MoE LoRA 分发无条件选择 one-shot Triton kernel,而该 kernel 的 rank 维度上限为 128(rank 维中间量保存在寄存器中,更大 rank 需要 rank tiling,由 diff 注释印证(由 diff 推断));缓冲却按配置的最大 rank pad,于是断言在 warm-up 时触发。正文还指出 PR #42290 的 review 已识别出缺失的 rank 回退与 dual-stream 风险,但合并前未处理,当时 main 仍存在此不匹配。

## 修复方案
`vllm/lora/ops/triton_ops/fused_moe_lora_op.py`:
- 新增常量 `_FUSED_MOE_LORA_ONE_SHOT_MAX_RANK = 128`;
- 分发条件由 `if not fully_sharded:` 改为 `if not fully_sharded and max_lora_rank <= _FUSED_MOE_LORA_ONE_SHOT_MAX_RANK:`,rank ≤128 保留 one-shot 快路径,更大 rank 走既有两 kernel shrink/expand 回退路径;
- `_fused_moe_lora_expand` 新增 `add_inputs` 参数,`ADD_INPUTS=add_inputs` 替换硬编码 `True`,保住 dual-stream 等调用方的 `add_inputs=False` 契约;相应断言改为 `assert add_inputs or not fully_sharded`;
- 保留 one-shot 内部 `rank <= 128` 断言作为防御性 kernel 本地守卫;
- 新增 `test_fused_moe_lora_kernel_large_rank_fallback` 数值测试。

## 影响与回归风险
- 影响窗口:正文指出 PR #42290 合并后的 main 即存在此不匹配(未给出版本号)。
- 回归面:非 fully-sharded MoE LoRA 的 kernel 分发路径——rank>128 时改走两 kernel 路径,数值结果需与 PyTorch 参考一致(atol/rtol 2e-2);`add_inputs=False` 现在在非 fully-sharded 回退路径也被支持,`fully_sharded=True` 仍拒绝 `add_inputs=False`。作者选择复用既有回退以避免改动 kernel 寄存器压力/性能特征,rank tiling 留待将来。验证:A100-SXM4-80GB 数值测试 8 passed,2xA100 TP=2 E2E(PIECEWISE 与 FULL CUDA graph 均捕获成功,/health 与 LoRA 请求均 200)。

## 关联
- Fixes https://github.com/vllm-project/vllm/issues/55158
- https://github.com/vllm-project/vllm/pull/42290

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
