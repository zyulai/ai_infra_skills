# PR #35256: [Bugfix] Fix dtype mismatch in RMSNormGated.forward_native() during torch.compile

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#35256](https://github.com/vllm-project/vllm/pull/35256) |
| 作者 | haosdent |
| 合并日期 | 2026-03-01 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 11 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | correctness |

## 一句话结论

torch.compile 下 RMSNormGated 走 forward_native,float16 的 mean() 提升为 float32 且不回转,传给 fp16 的 out_proj 触发 mm dtype 不匹配崩溃;改为显式上采浮点后再转回原 dtype。

## 问题现象

`torch.compile` 激活时(默认 Inductor backend,vLLM 设 `custom_ops=["none"]` 禁用自定义算子),`RMSNormGated` 分派到 `forward_native` 而非 `forward_cuda`;float16 输入下其输出变成 float32,传给权重为 float16 的 `out_proj`(`RowParallelLinear`)时触发 `mm(float32, float16)` 崩溃(对应 issue #35238)。

## 根因

正文披露:`forward_native` 实现未保持输入 dtype——PyTorch 对 float16 张量调用 `mean()` 会把结果提升为 float32,该提升沿后续所有运算传播(`rsqrt`、权重乘、gating),最终输出 float32。修复采用同文件中 `RMSNorm.forward_static` 与 `GemmaRMSNorm._forward_static_no_residual` 已有的模式:为数值稳定性显式上采到 float32 计算,返回前转回原 dtype。

## 修复方案

改 `vllm/model_executor/layers/layernorm.py` 的 `RMSNormGated.forward_native`:

```python
orig_dtype = x.dtype
x = x.float()
weight = self.weight.float()
z = z.float() if z is not None else None
...
return out.to(orig_dtype)
```

标准与分组两条 RMS 归一化路径统一使用上采后的 `weight` 参与乘法,gating 的 silu 也在 float32 域进行,返回时 `out.to(orig_dtype)`。测试:新增 `test_rmsnorm_gated_forward_native_dtype`,216 个参数化用例覆盖 dtype 保持(float16/bfloat16/float32)、对 `rms_norm_ref` 的数值正确性、有无 gating(z)、有无 group_size、两种 `norm_before_gate` 模式;`test_fla_layernorm_guard.py` 的 `DTYPES` 加入 `torch.float16` 扩大 Triton kernel 测试覆盖。216 个新测试与 1337 个既有测试全部通过。

## 影响与回归风险

- 影响窗口:正文未给出具体版本/提交范围;触发条件为 torch.compile(Inductor)开启、custom ops 被禁用且模型含 RMSNormGated(如 FLA 类模型)。
- 回归面:仅 `RMSNormGated.forward_native`(CPU/编译路径),`forward_cuda` 不变;float32 计算域的引入会轻微改变数值结果(上采后再回转,预期更接近参考实现),`forward_native` 的直接调用方与 torch.compile 图中的 dtype 约束需一致;测试新增 float16 参数化对相关 Triton kernel 提出新的 dtype 覆盖要求。

## 关联

- #35238(修复的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
