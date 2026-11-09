# PR #57092: [Bugfix] Make online FP8 MoE experts batch-invariant under VLLM_BATCH_INVARIANT

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#57092](https://github.com/vllm-project/vllm/pull/57092) |
| 作者 | ybwbqg9379 |
| 合并日期 | 2026-10-23 |
| labels | bug, ready, quantization |
| 热度 | 👍 0 · 💬 43 |
| 分类 | MoE |
| bug 类型 | general |

## 一句话结论
VLLM_BATCH_INVARIANT=1 下在线 FP8 MoE 专家输入用整批单一动态 scale、且 1 维 weight scale 的 stride 传 0,输出随 batch 组成变化;改为 per-token 量化并修 launcher stride。

## 问题现象
`VLLM_BATCH_INVARIANT=1` 且 `--quantization fp8`(在线 per-tensor FP8,Triton experts)时,MoE 模型输出仍取决于同 batch 的其他请求:greedy token 在前约 10 个位置内翻转,prefix-cached K/V 随写入者不同而不同。Dense 层不受影响,只有 experts 不是 batch 不变的。报告者的复现:修复前 A 单独跑 vs A 与 B 同 batch,首个 token 在位置 11 翻转(logprob 差 0.166),warm vs cold 在位置 11 翻转(差 0.347);修复后所有对比逐位一致。

## 根因
正文披露两处:
1. `Fp8PerTensorOnlineMoEMethod`(以及 FP8 序列化 per-tensor checkpoint 动态激活时的 `Fp8MoEMethod`)用 `kFp8DynamicTensorSym` 对专家输入以整个激活张量一个动态 scale 量化——即对同 batch 所有请求所有 token 取 max。该 scale 依赖 batch 组成,破坏 batch 不变性。
2. 换成 per-token 激活后暴露出潜在的 launcher bug:`invoke_fused_moe_triton_kernel` 对 1 维 `[E]` weight scale 传 `stride_bse=0`(`[E,1,1]` 则 `stride_bsn=1`),kernel 的 `per_channel_quant` 分支于是对每个专家都读专家 0 的 scale,或索引越过缓冲区。组合测试此前把「per-tensor 权重 + per-token 激活」当作 NYI 跳过,尽管 `TritonExperts._supports_quant_scheme` 声称支持该组合。

## 修复方案
- `vllm/model_executor/layers/quantization/online/fp8.py`:`VLLM_BATCH_INVARIANT` 开启时 `activation_key = kFp8DynamicTokenSym` 并置 `per_act_token_quant=True`;`fp8.py` 的 `Fp8MoEMethod` 在 static scheme 之外同样按 flag 选择 `kFp8DynamicTokenSym`,并把 `per_act_token_quant` 传入 `get_fused_moe_quant_config`——对齐 dense 层 `Fp8LinearMethod` 的既有做法。
- `vllm/model_executor/layers/fused_moe/fused_moe.py` 的 `invoke_fused_moe_triton_kernel`:`stride_bse` 改为只要 `B_scale is not None` 就传 `stride(0)`;`stride_bsn` 在 `size(1) > 1` 时才传 stride,否则传 0(广播单例维度)。`fused_batched_moe.py` 的 batched launcher 同样修 `stride_bsn`。
- `test_modular_kernel_combinations.py` 的 NYI 判定收窄为「per-channel 权重 + per-tensor 激活」,使「per-tensor 权重 + per-token 激活」组合进入覆盖;新增 e2e 逐位一致性测试与 kernel 单测。

## 影响与回归风险
- 影响窗口:正文提及报告者的 bf16 复现发生在 0.27.1,独立验证在 0.28.0+cu129 wheel 上复现失败;自 #53585 起在线 per-tensor FP8 以 `--quantization fp8_per_tensor` 选择。
- 回归面:在线 FP8 MoE 的激活量化粒度在 batch-invariant 模式下从 per-tensor 变为 per-token(算力开销略增、kernel 误差反而更小);fused_moe Triton launcher 的 stride 计算对其他 scale 布局通用。gsm8k 5-shot 1B 模型 34.3%→32.7%,作者归因于采样噪声;H20(sm_90)上独立复现并消除 #57016 症状。

## 关联
Fixes #57016;相关 #55881(XPU MoE 变体,不涉及本路径), #53585(fp8_per_tensor 选择);独立验证 comment https://github.com/vllm-project/vllm/pull/57092#issuecomment-5730213177

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
