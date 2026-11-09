# PR #14454: [ROCm][Kernel] MoE weights padding

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @gshtras |
| 合并 | 2025-03-24 |
| 分类 | MoE (tier: major) |
| Label | ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);diff 可见 5 个文件(环境开关、fused_moe、layer、fp8 与测试) |
| 链接 | https://github.com/vllm-project/vllm/pull/14454 |

## 一句话结论

ROCm 上对半精度 MoE 权重做内存 padding,Mixtral-8x22B 延迟最多改善 10%。

## 动机

PR 正文:该优化从 ROCm/vllm 移植而来,为 MoE 权重应用 padding。原理与 #13231 的 FP8 padding 类似,区别是这里针对半精度(half precision)类型。作者说明该优化更偏实验性、并不适用于任意 MoE 模型,因此默认关闭(正文表述),并扩展了单元测试以覆盖 padding 情况。

## 优化点

在 `FusedMoE` 权重加载后的 `process_weights_after_loading` 阶段新增 `_maybe_pad_weight`:当 `VLLM_ROCM_MOE_PADDING` 开启、平台为 ROCm、权重最后一维 stride 为 1 且 `stride(-2) * element_size % 512 == 0` 时,用 `F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]`(`num_pad = 256 // element_size`)让不同权重张量在内存中彼此错开足够远,以改善 ROCm 访存;对 w13_weight / w2_weight 都调用。同时把 `fused_experts_impl` 对 w1/w2 的 `is_contiguous()` 断言放宽为 `stride(-1) == 1`,并把 `invoke_fused_moe_kernel` 传入的 K 维从 `A.shape[1]` 改为 `B.shape[2]` 以适配 padding 后的形状;FP8 路径里原 `add_padding_to_weight` 也一并重命名为 `_maybe_pad_weight` 复用。新增环境变量 `VLLM_ROCM_MOE_PADDING`(diff 中默认值 1/True,且带 `current_platform.is_rocm()` 守卫)。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/layer.py: 新增 _maybe_pad_weight,对 w13_weight / w2_weight 做 ROCm padding
vllm/envs.py: 新增 VLLM_ROCM_MOE_PADDING 开关(默认 1)
vllm/model_executor/layers/fused_moe/fused_moe.py: 放宽 stride 断言、K 维改取 B.shape[2]
vllm/model_executor/layers/quantization/fp8.py: 把 FP8 的 add_padding_to_weight 重命名为 _maybe_pad_weight
tests/kernels/test_moe.py: 增加 padding 参数化用例
```

关键机制代码:

```python
    def _maybe_pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        # Pad the weight tensor. This is an optimization on ROCm platform, which
        # can benefit from tensors located far enough from one another in memory
        if (envs.VLLM_ROCM_MOE_PADDING and current_platform.is_rocm()
                and weight.stride(-1) == 1
                and (weight.stride(-2) * weight.element_size()) % 512 == 0):
            num_pad = 256 // weight.element_size()
            weight = F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]
            torch.cuda.empty_cache()
        return weight
```

## 性能声明

PR 正文:启用该特性后,在 `mistralai/Mixtral-8x22B-Instruct-v0.1`、配置 bs=64 / in=256 / out=256 / tp=8 下,延迟数字最多可观察到 up to 10% improvement(10% improvement in latency numbers)。正文未注明具体硬件型号、精度与测试工具。

## 验证与回归风险

Test Plan:扩展 `tests/kernels/test_moe.py` 的单元测试以覆盖 padding 情况(新增 `padding` 参数化、float16/bfloat16)。作者明确该优化为实验性、并不适用于所有 MoE 模型;正文称默认关闭,而 diff 中 `VLLM_ROCM_MOE_PADDING` 默认值为 1(True)且仅在 ROCm 生效,两处表述以素材为准并列列出。潜在影响路径:ROCm 上启用该开关的半精度 MoE 权重与 fused_moe kernel,且改动放宽了权重连续性断言并调整了 K 维取值,需关注非 ROCm 平台与未 padding 权重路径。

## 关联

前序 PR #13231(正文所引 FP8 padding,原理类似,本 PR 为其半精度版本);正文说明该优化移植自 ROCm/vllm。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
