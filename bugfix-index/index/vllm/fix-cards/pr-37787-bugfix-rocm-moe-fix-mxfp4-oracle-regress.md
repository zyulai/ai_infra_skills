# PR #37787: [Bugfix][ROCm][MoE] Fix mxfp4 oracle regressions from #37128

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#37787](https://github.com/vllm-project/vllm/pull/37787) |
| 作者 | AndreasKaratzas |
| 合并日期 | 2026-03-25 |
| labels | bug, ci/build, gpt-oss, ready, rocm |
| 热度 | 👍 1 · 💬 13 |
| 分类 | MoE |
| bug 类型 | regression |

## 一句话结论
#37128 重构丢失 gfx950 门控与对齐/padding 检查,CK mxfp4 在 gfx942 上被误选致 gpt-oss ROCm 崩溃;恢复三处检查并回落 Triton。

## 问题现象
#37128 引入的回归在 ROCm 上破坏了 gpt-oss:CK mxfp4 后端在 gfx942 上被选中即崩溃;intermediate_size 未对齐 256 的模型(如 gpt-oss-20b 的 2880)在 aiter shuffle_scale_a16w4 中触发 reshape 错误;CK 路径丢失 hidden_pad/intermediate_pad 参数。实测(MI325X (gfx942)):test_gpt_oss_speculative_reasoning_leakage 通过;GPQA eval 56.76%(1584 题,effort=low);非 gfx950 上后端正确回落 Triton。

## 根因
正文披露:#37128 的重构丢掉了三处原有逻辑——(1) 旧代码只在 gfx950 上经 `on_gfx950()` 选择 CK 后端,重构后丢失该门控,CK 在 gfx942 上被选中;(2) `CK_MXFP4_MOE_DIM_ALIGNMENT`(256)对齐检查丢失;(3) hidden_pad/intermediate_pad 在旧代码传给 `rocm_aiter_ops.fused_moe()`,重构中丢失。

## 修复方案
- `rocm_aiter_fused_moe.py`:`_supports_quant_scheme` 对 `kMxfp4Static` 权重新增 `on_gfx950()` 检查,非 gfx950 返回 False。
- 正文提及为 AiterExperts 新增 `is_supported_config`,使后端选择器回落 Triton(对齐检查由选择器接住)。
- `FusedMoEQuantConfig` 新增 `hidden_pad`/`intermediate_pad` 字段,经 `make_mxfp4_moe_quant_config`、`mxfp4.py` 的 create_weights/get_fused_moe_quant_config 接线到 `rocm_aiter_ops.fused_experts`。
- `layer.py`:quant method 可能在 `moe_config` 中上调 hidden_dim/intermediate_size,`__init__` 后同步 `self.hidden_size`,使 LoRA 等下游看到 padding 后的值。
- 另修 ROCm Triton LoRA kernel 中 `tl.dot` 两侧 dtype 不一致(合并 if/else 分支时 Triton 推断不同类型),统一 `a.to(tl.bfloat16), b.to(tl.bfloat16)`;若干 `w1.size()`→`w1.shape` 修正;测试侧 ROCm 跳过 Mxfp4 LoRA(spawn 兼容问题)。

## 影响与回归风险
- 影响窗口:#37128 合并起至本 PR 的 ROCm gpt-oss 路径。
- 回归面:ROCm mxfp4 MoE 后端选择(gfx942 改走 Triton)、CK 路径 fused_moe 的 padding 参数、FusedMoE hidden_size 同步(影响 LoRA 看到的 shape)、fused_moe_lora Triton kernel dtype;CUDA 侧 Mxfp4 LoRA 行为不变。

## 关联
- #37811
- #37784
- #37786
- #37128(引入回归的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
