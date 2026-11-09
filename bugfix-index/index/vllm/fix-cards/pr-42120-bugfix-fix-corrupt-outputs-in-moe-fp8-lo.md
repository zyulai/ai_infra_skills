# PR #42120: [Bugfix] Fix corrupt outputs in MoE FP8 LoRA responses and MoE base model responses when LoRAs are loaded

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#42120](https://github.com/vllm-project/vllm/pull/42120) |
| 作者 | nv-nedelman-1 |
| 合并日期 | 2026-06-19 |
| labels | bug, ci/build, frontend, performance, ready, rocm, v1 |
| 热度 | 👍 2 · 💬 8 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
修两个 MoE+LoRA 输出损坏 bug:无 LoRA 请求时 `add_lora_fused_moe` 仍跑 kernel 污染输出;FP8 下 LoRA shrink 收到的是已量化激活而非原精度,量级全错。

## 问题现象
两个输出损坏 bug(正文分别对应 issue #42007 与 #42008):加载 LoRA 后,MoE FP8 的 LoRA 响应与 MoE 基座模型响应损坏。正文说明这些 bug 最初在 vLLM 0.20.0 被发现/引入,近期的 MoE 重构轻微改变了触发前提与表现形态,但根因仍在,由本 PR 修复。正文未附端到端失败日志。

## 根因
正文正文未逐条展开根因,仅给出修复思路与两个 issue;从 diff 内注释可描述机制(由 diff 推断):
1. **#42008(基座响应损坏)**:本批次无活跃 LoRA 请求时(`no_lora_flag_cpu=True`,mapping 全为 -1),`add_lora_fused_moe` 仍进入 LoRA kernel 并修改输出张量——wrapper 内部的 stale LoRA ID 映射未被短路。
2. **#42007(FP8 LoRA 响应损坏)**:LoRA shrink 的输入应是未量化的原始 hidden states;prepare 步骤可能已把激活量化,`apply_w13_lora` 拿到的是缺 activation scale 的 raw quantized 值,激活量级错误,LoRA delta 随之损坏(diff 中 `lora_context.py` 新字段注释:"so the LoRA kernel sees correct-magnitude activations instead of raw quantized values that are missing the activation scale")。

## 修复方案
1. `vllm/lora/punica_wrapper/punica_gpu.py`:`add_lora_fused_moe` 从 `meta_args` 取出 `no_lora_flag`,numel==1 且为 True 时直接 `return`,不调 `fused_moe_lora`、不改输出(早退短路)。
2. 原精度激活保留两条路径:`modular_kernel.py` 的 `apply` 在 `_prepare()` 前把原始 hidden states 暂存到 `MoELoRAContext.original_hidden_states`,调用结束后清空;`triton_moe.py` 新增 `expects_unquantized_inputs` property(LoRA 活跃 + 有量化 dtype + DP/EP all2all 时声明,把激活量化推迟到 apply),apply 内自行量化一份供基座 GEMM、原张量留给 LoRA shrink;非推迟路径回退到 context stash,并以行数相等作守卫防止 DP-gathered 布局越界。`_lora_w13_fn` 两处 `x=hidden_states` 改为 `x=lora_x`。
新增回归测试 `test_add_lora_fused_moe_early_exit`:先模拟一个有 LoRA 的批次再切到全 -1 的基座批次,断言 `no_lora_flag_cpu=True` 时 `add_lora_fused_moe` 不修改输出张量。

## 影响与回归风险
- 影响窗口:正文未列具体版本范围;正文指出 bug 最初在 vLLM 0.20.0 发现/引入。
- 回归面:`punica_gpu.py`(MoE LoRA 入口早退)、`modular_kernel.py`(context 暂存)、`triton_moe.py`(量化时机与 LoRA 输入选择)、`lora_context.py`(新字段);正文提到 `lora_shrink_op.py`/`lora_expand_op.py` 等的优先级使用 `no_lora_flag_cpu`(正文 Implementation Details 提及)。新测试修复前失败(证明 #42008)、修复后通过;**dtype 错误路径(#42007)作者自述未加测试**(尝试均显得繁琐/不合位置,如需要可再补)。量化推迟仅在 `use_all2all_kernels` 时生效,非 all2all 布局行为不变。

## 关联
- #42007(所修 issue)
- #42008(所修 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
