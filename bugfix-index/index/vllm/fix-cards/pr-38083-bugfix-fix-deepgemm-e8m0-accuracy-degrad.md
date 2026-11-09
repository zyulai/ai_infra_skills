# PR #38083: [Bugfix] Fix DeepGemm E8M0 accuracy degradation for Qwen3.5 FP8 on Blackwell

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38083](https://github.com/vllm-project/vllm/pull/38083) |
| 作者 | vadiklyutiy |
| 合并日期 | 2026-03-26 |
| labels | bug, qwen, ready |
| 热度 | 👍 1 · 💬 5 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
DeepGemm 强制 E8M0 scale 导致 Qwen3.5 FP8 在 B200 上 GSM8K 掉约 12pp,改为对 Qwen3.5 在 Blackwell 上自动禁用 DeepGemm 回退 CUTLASS。

## 问题现象
`Qwen/Qwen3.5-35B-A3B-FP8` 在 Blackwell(B200)上、DeepGemm 激活时,GSM8K 精度下降约 12 个百分点(1319 题、5-shot、B200 DP=2 EP、commit `2e67fa75` 实测):DeepGemm ON 平均 0.6917,DeepGemm OFF(`VLLM_USE_DEEP_GEMM=0`)平均 0.8122。影响所有使用 FP8 block 量化的 Qwen3.5 模型(35B-A3B 等约 80 层 attention 投影与 shared-expert FFN)。

## 根因
正文披露:DeepGemm 强制使用 E8M0(2 的幂向上取整)scale 格式,每个 FP8 block 量化层都会损失精度,误差跨 Qwen3.5 的约 80 个 attention 投影层与 shared-expert FFN 层逐层累积,最终造成明显精度回退。

## 修复方案
- 在 `vllm/utils/deep_gemm.py` 增加模型类型排除表 `qwen3_5_text`、`qwen3_5_moe_text` 与 `should_auto_disable_deep_gemm()`:仅当平台为 SM100+(Blackwell)且模型类型命中排除表时返回 True。
- 在 `vllm/config/vllm.py` 配置最终化阶段检查并置 `quant_config.use_deep_gemm = False`,输出一次性告警(回退 CUTLASS,`VLLM_USE_DEEP_GEMM=0` 可全局关闭);环境变量仍可覆盖。
- `Fp8Config` 新增 `use_deep_gemm: bool | None` 字段,经 `Fp8LinearMethod`、`W8A8BlockFp8LinearOp`(`fp8_utils.py`)传递;禁用时跳过加载时的 UE8M0 权重预转换与 `maybe_post_process_fp8_weight_block`;`input_quant_fp8.py` 的激活量化路径同样加上 `self.use_ue8m0` 判断。
- 扩充 Blackwell GSM8K CI:加入 `Qwen3.5-35B-A3B-FP8` 与 `Qwen3.5-397B-A17B-NVFP4`,并把全局 `TOL = 0.08` 改为每配置 `tolerance` 字段。

关键 diff 摘录(vllm/config/vllm.py):
```python
model_type = getattr(self.model_config.hf_text_config, "model_type", None)
if should_auto_disable_deep_gemm(model_type):
    self.quant_config.use_deep_gemm = False
    logger.warning_once(
        "Auto-disabled DeepGemm for model_type=%s on Blackwell. "
        ... Falling back to CUTLASS. ...)
```

## 影响与回归风险
- 回归面:SM100+ 上 FP8 block 量化模型的配置初始化路径(VllmConfig.__post_init__)、`Fp8LinearMethod.process_weights_after_loading`(禁用时不再做 UE8M0 预转换/后处理)、`input_quant_fp8.py` 激活量化路径、`W8A8BlockFp8LinearOp` 构造参数;非 Qwen3.5 模型与非 Blackwell 平台行为不变。GSM8K CI 阈值调整可能放宽个别模型断言。

## 关联
- #37804(fixes)
- #37806
- #37889

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
