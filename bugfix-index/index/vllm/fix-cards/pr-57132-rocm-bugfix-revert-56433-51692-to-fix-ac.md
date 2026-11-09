# PR #57132: [ROCm][Bugfix] Revert #56433 + #51692 to fix accuracy breakdown for DeepSeek-V4

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#57132](https://github.com/vllm-project/vllm/pull/57132) |
| 作者 | shen-shanshan |
| 合并日期 | 2026-09-16 |
| labels | DSv4, bug, deepseek, ready, rocm |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
#56433+#51692 使 DeepSeek-V4 在 AMD GPU 上 gsm8k 精度归零,两 PR 一并 revert 后恢复 0.95+。

## 问题现象
在 AMD GPU 上用最新 main 对 DeepSeek-V4 测 gsm8k,exact_match 掉到接近 0(flexible-extract 0.0053±0.0020、strict-match 0.0008±0.0008)。仅 revert #56433 时服务启动失败:`aiter.py` 第 584 行抛 `NotImplementedError: AiterPreshuffledFp8BlockScaledMMKernel overrides apply_weights and does not use apply_block_scaled_mm`;把 #56433 与 #51692 一并 revert 后精度恢复:conc=64/5-shot 0.9538±0.0058 / 0.9545±0.0057,conc=256/30-shot 0.9568±0.0056 / 0.9575±0.0056。复现命令照抄正文(`VLLM_ROCM_USE_AITER=1`、`--moe-backend aiter`、TP8、`--kv-cache-dtype fp8`、FULL_AND_PIECEWISE 编译等,lm_eval local-completions)。

## 根因
正文通过实验定位:DeepSeek-V4 ROCm AITER 路径的精度崩溃由 #56433 与 #51692 两个 PR 的组合引入——两者互相耦合,单 revert #56433 会因 AiterPreshuffledFp8BlockScaledMMKernel 被其他路径引用而启动失败,必须一并 revert 才能恢复;具体精度破坏机制正文未进一步披露。
由 diff 推断的线索:被删除的 `AiterPreshuffledFp8BlockScaledMMKernel.apply_block_scaled_mm` docstring 自述「As 必须列主序;行主序不会报错,只会对 M>1 返回错误数值」,且 `_split_qkv_and_norm` 的 fused norm+quant 路径需按 kernel 期望输出特定字节序的激活 scale——激活 scale 字节序/布局错配会静默产生错误数值,与精度崩溃现象相符(由 diff 推断,正文未确认)。

## 修复方案
整体 revert #56433 + #51692 的相关改动:
- `vllm/model_executor/kernels/linear/scaled_mm/aiter.py`:整段删除 `AiterPreshuffledFp8BlockScaledMMKernel`(bpreshuffle 权重 kernel);`kernels/linear/__init__.py` 同步移除其导入、注册与 ROCm 后端 kernel 列表;
- `vllm/models/deepseek_v4/amd/rocm.py`:删除 `weight_already_preshuffled` 及其防重复 shuffle 判断(恢复无条件 `shuffle_weight`);`_wq_b_act_scale_transpose` 恢复为 `_wq_b_uses_aiter_block_scaled`(只判断是否都是 aiter block-scaled kernel),fused norm+quant 固定 `transpose_scale=False`;
- `vllm/models/deepseek_v4/amd/model.py`:`prepare_gateup_preshuffle` 移除防重复 shuffle 判断;
- `vllm/model_executor/layers/attention/mla_attention.py` + `linear.py`:kv_b_proj 由 `skip_weight_relayout` 恢复为 `_cpu_skip_gemm_dispatch` 语义;
- `vllm/_aiter_ops.py`:删除 `is_blockscale_bpreshuffle_tuned`,`_load_gemm_tuned_configs` 恢复为按 `q_dtype_w` 过滤的旧签名;
- `tests/models/test_deepseek_v4_vl_rocm.py`:删除随 #56433 添加的 preshuffle 相关测试。

## 影响与回归风险
- 影响窗口:#56433 与 #51692 合并之后至本 PR(2026-09-16)合并前的 ROCm main
- 回归面:ROCm AITER FP8 linear 路径失去 bpreshuffle 优化 kernel,相关形状的 GEMM 性能可能回退到 #56433 之前的水平;wq_b fused norm+quant 快路径条件收窄为「两个 wq_b 均为 aiter block-scaled kernel」且 transpose_scale 固定 False;`_load_gemm_tuned_configs` 签名回退会影响其他调用点的匹配方式;被删测试不再覆盖 preshuffle 行为。

## 关联
- [#56433](https://github.com/vllm-project/vllm/pull/56433)(被 revert 的 PR)
- [#51692](https://github.com/vllm-project/vllm/pull/51692)(被 revert 的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
