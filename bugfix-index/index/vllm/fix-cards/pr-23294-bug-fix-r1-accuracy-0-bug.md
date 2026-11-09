# PR #23294: [Bug] Fix R1 Accuracy 0 Bug

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#23294](https://github.com/vllm-project/vllm/pull/23294) |
| 作者 | yewentao256 |
| 合并日期 | 2025-08-21 |
| labels | ready |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
FP8 MoE apply 分支走错 fused_experts 实现,DeepSeek-R1 gsm8k 精度跌至 0,改为优先使用 self.fused_experts 修复。

## 问题现象
DeepSeek-R1-0528 在以下部署下输出完全错误、精度归零(lm_eval gsm8k 5-shot,修复前 flexible-extract exact_match 0.005±0.005、strict-match 0.000±0.000;修复后 0.9037±0.0081 / 0.9022±0.0082)。复现命令(照抄正文):
`vllm serve --model="deepseek-ai/DeepSeek-R1-0528" --max-num-seqs 512 --trust-remote-code --data-parallel-size 8 --enable-expert-parallel --gpu-memory-utilization 0.75 --port 9256 --disable-log-requests --no-enable-prefix-caching -O '{"full_cuda_graph": true}' --cuda-graph-sizes 16 32 48 64 96 128 160 192 256 512`,评测用 `lm_eval --model local-completions --model_args "base_url=http://127.0.0.1:9256/v1/completions,model=deepseek-ai/DeepSeek-R1-0528,num_concurrent=256" --tasks gsm8k`。

## 根因
正文未披露根因(仅写 "Fixes #23282" 并给出前后精度对比)。
由 diff 推断:`vllm/model_executor/layers/quantization/fp8.py` 中 Fp8MoEMethod.apply 的 else 分支此前无条件调用全局 `fused_experts` 并携带 `use_fp8_w8a8=True`、`block_shape`、`allow_deep_gemm` 等 FP8 专属参数,而忽略实例上已设置的 `self.fused_experts` 实现;走错 MoE kernel 路径导致输出错误、精度归零(由 diff 推断)。

## 修复方案
改 `vllm/model_executor/layers/quantization/fp8.py` 的 apply else 分支:把公共参数抽成 `common_kwargs = dict(hidden_states=x, w1=layer.w13_weight, w2=layer.w2_weight, ...)`,然后:
```python
if self.fused_experts is not None:
    return self.fused_experts(**common_kwargs)
else:
    from vllm.model_executor.layers.fused_moe import fused_experts
    return fused_experts(
        **common_kwargs,
        use_fp8_w8a8=True,
        block_shape=self.quant_config.weight_block_size,
        allow_deep_gemm=self.allow_deep_gemm,
        allow_cutlass_block_scaled_grouped_gemm=(
            self.allow_cutlass_block_scaled_grouped_gemm),
    )
```
即实例上配置了专用 fused_experts 时优先使用(不带 FP8 专属参数),否则回退原全局实现,参数不变。

## 影响与回归风险
- 影响窗口:正文未说明
- 回归面:所有走 Fp8MoEMethod.apply 的 FP8 block-quant MoE 模型(DeepSeek 系列、Qwen3 MoE FP8 等);当 `self.fused_experts` 被设置的部署(如特定后端/图模式组合)行为改变,可能恢复或切换到预期 kernel 路径;未设置时路径与参数完全不变。

## 关联
- [#23282](https://github.com/vllm-project/vllm/issues/23282)(修复的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
