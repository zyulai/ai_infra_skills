# PR #44109: [Kernel] Add weightless RMSNorm CUDA kernels for has_weight=False (#41430)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @hello-args |
| 合并 | 2026-06-17 |
| 分类 | Attention / Kernel (tier: major) |
| Label | performance, rocm, intel-gpu, ready, cpu, nvidia |
| Diff | diffstat 缺失(pack 未记录 +None/-None,None files);diff 内容可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/44109 |

## 一句话结论

`rms_norm`/`fused_add_rms_norm` 的 CUDA/CPU kernel 支持 weight=null 的无权重路径,FlashNorm 折叠 checkpoint 不再用全 1 张量做无谓的加载与乘法。

## 动机

`RMSNorm(has_weight=False)`(FlashNorm-folded checkpoint,#40117)此前只在 native 路径跳过 per-channel 乘法;CUDA/CPU `_C` kernel 强制要求 weight 张量,层只能替换为全 1 张量再跑带权 kernel——结果正确,但每个元素多一次权重加载与乘法。对应 issue #41430。

## 优化点

机制:不新增 `*_weightless` 独立 op,而是把现有 `rms_norm`/`fused_add_rms_norm` 的 `weight` 参数改为 optional(`Tensor?`),CUDA kernel 以 `bool HasWeight` 模板参数实例化——`if constexpr (HasWeight)` 在编译期剪掉乘法与 weight 向量加载(fused_add 版本同时剪掉 weight 对齐检查对 vectorized 判定的影响);CPU 版本同样按 `has_weight` 分支跳过 `fp32_w` 乘法。`layernorm.py` 删除猜测 native 是否被分派的启发式(#39370),`has_weight=False` 时直接传 `weight=None`;`vllm_c.py`/`xpu_ops.py` 移除 ones-substitution,XPU 无权路径回退 native 实现(weightless `_C` 路径仅 CUDA/ROCm);需要 weight 的后端(如 oink)经 IR op priority 回退。带权热路径代码不变。

## 关键改动

```text
csrc/libtorch_stable/layernorm_kernels.cu: CUDA kernel 模板化 HasWeight,weight 可为 null
csrc/cpu/layernorm.cpp: CPU 实现 has_weight 分支
vllm/model_executor/layers/layernorm.py: has_weight=False 时传 weight=None,删除 native 分派猜测
vllm/kernels/vllm_c.py / xpu_ops.py: 移除全 1 张量替换;XPU 回退 native
tests/kernels/core/test_layernorm.py: 新增 test_rms_norm_weightless
```

## 性能声明

无量化性能数据;PR 未注明测试条件。收益定性:has_weight=False 的 checkpoint(FlashNorm 折叠)在 CUDA/ROCm/CPU 上省去每元素一次 weight 加载与一次乘法,且 fused_add 路径不再因 ones 张量参与对齐判定。带权路径性能声明为不变。

## 验证与回归风险

新增 `test_rms_norm_weightless`:参数化 num_tokens/hidden_size/add_residual/dtype,与 `forward_native` 对比(ir.ops.rms_norm 的容差),并用 `opcheck` 验证 `weight=None` 的 op schema。风险面:op schema 变更(`Tensor` → `Tensor?`)触及所有调用方;XPU 无权路径改走 native(行为差异);IR op priority 回退逻辑依赖各后端对 `weight=None` 的声明;CUDA libtorch_stable ABI 与 CPU torch::Tensor 两套声明需同步(已在 ops.h 中保留双份)。

## 关联

同区域:#41430(对应 issue)、#40117(FlashNorm 折叠,引入 has_weight=False)、#41431(配套的 FlashNorm-folded Llama checkpoint loader)、#39370(weight 传递不一致问题,本 PR 删除其启发式)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
