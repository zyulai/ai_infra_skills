# PR #7699: [AMD] Fix Llama 4 FP8 accuracy issues on MI300X

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#7699](https://github.com/sgl-project/sglang/pull/7699) |
| 作者 | hubertlu-tw |
| 合并日期 | 2025-08-22 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
MI300X 上 Llama 4 FP8 精度异常;为 aiter 增加 `apply_router_weight_on_input` 的 FP8 per-channel MoE 路径(AITER asm_moe_tkw1 + 权重预 shuffle)并允许 aiter attention backend。

## 问题现象
MI300X 上以 `SGLANG_USE_AITER=1` 启动 `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`(`--tp 8 --attention-backend aiter`)出现精度问题(对应 issue #5362)。修复后正文给出验证:GSM8K 0-shot flexible-extract exact_match 0.9636(±0.0052,strict-match 0.0000);MMLU-Pro 与 toy prompt "What is Deep Learning" 的结果以截图形式附于正文。

## 根因
正文未披露具体根因(Motivation 只说 "To fix Llama 4 FP8 accuracy issue on MI300X",Modifications 留空)。(由 diff 推断)Llama 4 的 router 权重作用在输入侧(`apply_router_weight_on_input`),而 compressed-tensors FP8 per-channel MoE 原先统一走通用 `fused_experts`,其将 topk_weights 乘在第二层 FC 输出上,与 Llama 4 期望的"乘在第一层 FC GEMM 输出"不符;且 aiter tkw1 kernel 要求 (16,16) 预 shuffle 的权重布局,原路径未做,导致 FP8 MoE 数值错误。

## 修复方案
- 新增 `python/sglang/srt/layers/moe/rocm_moe_utils.py`:封装 AITER `asm_moe_tkw1` 为自定义 op `rocm_aiter_asm_moe_tkw1`;`rocm_fused_experts_tkw1` 在 per-channel quant + `apply_router_weight_on_input` + fp8_w8a8 时调用 AITER tkw1 kernel(topk_weights 作用于第一层 FC 输出,要求 topk=1)。
- `compressed_tensors_moe.py`:`process_weights_after_loading` 在 `_use_aiter` 下对 `w13_weight`/`w2_weight` 做 `shuffle_weight(..., (16,16))` 预 shuffle;`apply` 按 `_use_aiter + CHANNEL 策略 + apply_router_weight_on_input` 分流到新路径,否则保持原 `fused_experts`。
- `server_args.py`:Llama4 的 attention backend 断言从仅允许 `fa3` 放宽为 `{fa3, aiter}`。
- `fp8.py`:`process_weights_hip_scale_padding` 保持 ROCm 列向缩放(仅空行整理)。

关键 diff(compressed_tensors_moe.py apply 分流):

```python
        if (
            _use_aiter
            and self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and moe_runner_config.apply_router_weight_on_input
        ):
            topk_weights, topk_ids, _ = topk_output
            return rocm_fused_experts_tkw1(...)
```

## 影响与回归风险
- 影响窗口:正文未给版本/提交范围。
- 回归面:`SGLANG_USE_AITER=1` 的 HIP 环境:compressed-tensors MoE 的 FP8 per-channel 权重加载后会被预 shuffle 并清空缓存,.apply 分流到 AITER kernel;Llama4 允许 `--attention-backend aiter`;非 HIP / 非 aiter 环境走原 `fused_experts` 分支,行为不变(diff 如实所示)。新路径断言仅支持 topk=1,其他 topk 配置在该分支会触发 assert。

## 关联
- https://github.com/sgl-project/sglang/issues/5362

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
