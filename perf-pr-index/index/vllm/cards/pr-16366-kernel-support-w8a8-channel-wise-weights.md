# PR #16366: [Kernel] Support W8A8 channel-wise weights and per-token activations in triton fused_moe_kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2025-04-11 |
| 分类 | Quantization(major) |
| Label | performance, ready |
| Diff | diffstat 缺失(素材未提供 +additions/-deletions 与文件数;diff 本身可用,2 个文件因超长被省略) |
| 链接 | https://github.com/vllm-project/vllm/pull/16366 |

## 一句话结论

把 triton fused_moe kernel 的 scale 处理扩展为三种模式,新增支持 W8A8 FP8/INT8 的 per-channel 权重 × per-token 激活与 INT8 blocked 量化。

## 动机

triton `fused_moe_kernel` 此前的 W8A8 支持只有两档:per-tensor scale(group_n/group_k 为 0)与 block-wise 量化(group_n/group_k > 0,即 DeepSeek 式 FP8 block 量化)。逐通道(per-channel)权重量化 + per-token 激活量化的 W8A8 组合——FP8 与 INT8 皆是——无法落入现有分支,依赖这类量化格式的 MoE 模型跑不起来。这些能力来自 sglang 团队的先行开发,本 PR 将其移植进 vLLM。

## 优化点

`fused_moe_kernel` 新增 `use_int8_w8a8` 与 `per_channel_quant` 两个 constexpr,把 scale 加载扩成三分支:block-wise(按 `offs_bsn = offs_bn // group_n` 取 block scale)、channel-wise(b_scale 按输出列 `offs_bn` 加载,a_scale 按 token 加载并做外积乘)、tensor-wise(标量)。累加路径相应区分:fp8 用 `tl.dot(a, b, acc=accumulator)` 以启用 fp8_fast_accum,int8 则 `accumulator += tl.dot(a, b)`(int8 不能走 fp8 fast-accum 累加器)。外层 `invoke_fused_moe_kernel`/`fused_experts` 等一整条签名链路补齐 `use_int8_w8a8`、`per_channel_quant` 参数;激活量化分别复用 `per_token_group_quant_int8`/`per_token_quant_int8` 与 `scaled_fp8_quant(use_per_token_if_dynamic=True)`。测试侧把 block matmul 的 native 参考实现抽到 dtype 无关的 `tests/kernels/utils_block.py`。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/fused_moe.py: kernel 三分支 scale 模式 + int8 w8a8 累加路径 + 签名链路
tests/kernels/test_triton_moe_ptpc_fp8.py: 新增 per-token per-channel FP8 fused_moe 测试
tests/kernels/test_int8_kernel.py: 新增 per-channel INT8 测试(移植自 sglang)
tests/kernels/test_block_int8.py: 新增 INT8 blocked W8A8 测试(移植自 sglang)
tests/kernels/utils_block.py: dtype 无关的 native block matmul 参考实现
```

关键片段(kernel 内 channel-wise 分支):

```python
elif per_channel_quant:
    b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
    b_scale = tl.load(b_scale_ptrs)
    a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
    a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
```

## 性能声明

PR 正文未给出 benchmark 数字(perf claims 检测为 none);本 PR 定位为扩展支持的量化格式(enabling),正文感谢 sglang 团队的开发。加速效果 PR 未注明。

## 验证与回归风险

新增三个测试文件,均与 native torch 参考实现对拍:FP8 ptpc(rel_diff < 0.05,需 CUDA capability >= 9.0)、INT8 per-channel(rel_diff < 0.05,需 >= 7.0)、INT8 blocked(matmul < 0.001、fused_moe < 0.06),覆盖 M ∈ {1, 33, 64, 222}、E ∈ {8, 24}、topk ∈ {2, 6} 等;`test_block_fp8.py` 重构引用路径后语义不变。风险面:kernel 的 constexpr 组合从 2 种扩到 4 种,fp8 fast-accum 与 int8 累加分支若混淆会出精度问题;`invoke_fused_moe_kernel` 删除了部分入参断言,异常输入的报错时机后移。

## 关联

同分类 Quantization 下的 MoE 量化链路;后续 block-FP8 MoE 调优见 PR #46642。素材注明移植自 sglang(test 文件头部标注 adapted from sgl-project/sglang)。PR 未注明其他前序/后续编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
