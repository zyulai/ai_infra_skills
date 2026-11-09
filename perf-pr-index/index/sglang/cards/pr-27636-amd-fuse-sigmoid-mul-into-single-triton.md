# PR #27636: [AMD] Fuse sigmoid + mul into single Triton kernel for shared expert gating

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yichiche |
| 合并 | 2026-06-16 |
| 分类 | MoE(tier: major) |
| Label | documentation, run-ci, jit-kernel |
| Diff | diffstat 缺失(素材未统计行数);diff 覆盖 jit kernel、qwen2_moe 模型与单测 3 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/27636 |

## 一句话结论

MI355X 上把 shared expert gating 的 sigmoid+mul 两个 elementwise kernel 融合为一个 Triton kernel,每 decode 迭代省约 264 us。

## 动机

MI355X(HIP/ROCm)上 `Qwen2MoeSparseMoeBlock._forward_shared_experts` 的 shared expert gating 跑成两个独立 elementwise kernel:`sigmoid_kernel`(约 3.8 us)与 `elementwise_mul`(约 4.6 us);MI355X 上每次 kernel launch 约 4 us 开销,这些微小 pointwise 算子是 launch-bound。本 PR 将两者融合为单个 Triton kernel,每个 A-type decode 层省一次 launch。它是 #27630(同 shape attention output gate 场景)kernel 的 broadcast 变体:shared expert gate 形状 (N,1)、输出 (N,D),需要 broadcast-aware kernel。

## 优化点

新增 `sigmoid_gate_mul_broadcast(x, gate)`:Triton kernel 每 program 处理一行,加载该行标量 gate 后升 fp32 做 sigmoid,再与该行 (N,D) 的 x 逐元素相乘并写回(同样升 fp32 计算保证数值稳定),消掉一次独立 launch 与中间张量。wrapper 按 hidden_dim 选 BLOCK_SIZE(next_power_of_2)与 num_warps(HIP 上限 16,CUDA 32)。在 `qwen2_moe.py` 的 `_forward_shared_experts` 中新增 `_is_hip` 分支调用该 kernel;CUDA 路径保持原 `F.sigmoid` + `*` 不变,改动被 `_use_aiter` 分支限定。另保留 #27630 的同 shape `sigmoid_gate_mul` 变体并补齐文档字符串。

## 关键改动

```text
python/sglang/jit_kernel/triton/sigmoid_gate_mul.py: 新增 _sigmoid_gate_mul_broadcast_kernel 与 sigmoid_gate_mul_broadcast wrapper
python/sglang/srt/models/qwen2_moe.py: _forward_shared_experts 新增 _is_hip 分支调用融合 kernel
test/registered/kernels/test_sigmoid_gate_mul.py: 40 个单测(两种变体 x bf16/fp16/fp32 x 多形状,含输入不可变/输出 dtype/contiguity)
```

```python
@triton.jit
def _sigmoid_gate_mul_broadcast_kernel(out_ptr, gate_ptr, x_ptr,
                                       hidden_dim, BLOCK_SIZE):
    row = tl.program_id(0)
    g = tl.load(gate_ptr + row).to(tl.float32)
    g = tl.sigmoid(g)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_dim
    x = tl.load(x_ptr + row * hidden_dim + offs, mask=mask).to(tl.float32)
    tl.store(out_ptr + row * hidden_dim + offs, (x * g).to(...), mask=mask)
```

## 性能声明

内核级(每个 A-type decode 层):before 为 `sigmoid_kernel_cuda` + `elementwise_kernel BinaryFunctor(mul)` 2 个 kernel 共 8.4 us;after 为 `_sigmoid_gate_mul_broadcast_kernel` 1 个 4.0 us,节省 4.4 us,x60 A-type 层约 264 us/iteration。

E2E(Qwen3.5-397B-A17B-MXFP4,TP2,MI355X,IL8k/OL1k,conc=4,40 prompts):总吞吐 2986.62 -> 2982.67 tok/s(-0.1%),Median TTFT 391.24 -> 390.09 ms(-0.3%),Median ITL 10.31 -> 10.33 ms(+0.2%),Median TPOT 11.42 -> 11.40 ms(-0.2%)。正文说明期望的 ITL 改善(约 264 us / 10310 us = 2.6%)在 conc=4 下淹没于运行间噪声,未观察到回退。即 E2E 中性,收益主要体现在 kernel trace 层面。

## 验证与回归风险

新增 40 个单测覆盖 element-wise 与 broadcast 两个变体,跨 bf16/fp16/fp32 与 6 种形状,并校验输入不可变、输出 dtype 与 contiguity(`register_amd_ci`,suite `jit-kernel-unit-test-amd`)。精度:Qwen3.5-397B-A17B-MXFP4 TP2 MI355X,GSM8K 2000 题 accuracy 0.909、invalid 0.3%,阈值 0.88,PASS。影响面:仅 HIP 且 `_use_aiter` 的 qwen2_moe 系列 shared expert gating 路径;CUDA 路径未动。CI 状态显示 Base 与 Extra 均失败(素材所示)。

## 关联

前序:#27630(同 shape 版本的 fused sigmoid-gate-mul kernel)。同分类其他 PR 编号素材未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
