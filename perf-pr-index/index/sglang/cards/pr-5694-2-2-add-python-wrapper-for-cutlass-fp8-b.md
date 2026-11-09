# PR #5694: [2/2] Add python wrapper for CUTLASS FP8 Blockscale MoE Kernel.

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @elfiegg |
| 合并 | 2025-05-16 |
| 分类 | MoE(tier: major) |
| Label | high priority |
| Diff | diffstat 缺失(pack 未记录);diff 可用,含 6 个被截断文件(详见 PR) |
| 链接 | https://github.com/sgl-project/sglang/pull/5694 |

## 一句话结论

为 CUTLASS FP8 Blockscale MoE 内核接上 Python 侧 dispatch,小 batch 下比 Triton fused_experts 快约 30%-40%。

## 动机

DeepSeek V3/R1 这类 FP8 blockwise 量化 MoE 模型此前只有 Triton `fused_experts` 路径。作者用 PR 自带的 benchmark 发现,CUTLASS 4.0 的 fused_expert 在 CUDA graph 模式下小 batch 场景明显快于 Triton,值得作为可选路径接入。同时作者注明:仓库当前的 CUTLASS 3.9 会遇到 kernel hang 和性能回退,需要另开 PR 升级 CUTLASS 依赖。

## 优化点

新增 `cutlass_moe.py` 封装完整的 FP8 blockwise fused MoE 流程:输入先用 `sglang_per_token_group_quant_fp8` 按 128-group 量化,`prepare_moe_input`(新增 CUDA kernel)生成 expert offset/permutation map,按 map 重排激活与 scale 后调用两次 `fp8_blockwise_scaled_grouped_mm`(up-proj 与 down-proj),中间插入 `silu_and_mul`,最后按 topk_weights 加权求和。机制上的关键改动是把所有张量分配(workspace、指针数组、strides、expert_offsets、problem_sizes)移出 kernel 实现之外——在 `create_weights` 阶段一次性预分配,`apply` 时直接复用,消除每步的分配开销。CUDA 侧把 `hw_info.sm_count` 从 1 改为 148(SM100 实际 SM 数),并把 stream 显式传入 GEMM 的 initialize/run;tile 配置从两档改为三档 dispatch(MmaConfig1: 128x32x128 小 M,MmaConfig2: 128x128x128 大 M 且 K>=2048,MmaConfig3: 64x128x128)。

## 关键改动

```text
python/sglang/srt/layers/moe/cutlass_moe.py: 新增 CUTLASS MoE Python 封装(cutlass_fused_experts)
python/sglang/srt/layers/quantization/fp8.py: CUTLASS_MOE 环境变量 + SM100 下走 cutlass_fused_experts,预分配 strides/workspace/ptrs
python/sglang/srt/layers/quantization/fp8_utils.py: 新增 is_sm100_supported()(SM100 + CUDA >= 12.8)
sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu: 三档 tile dispatch、sm_count=148、stream 传入、参数加 ptrs/workspace
python/sglang/test/test_cutlass_moe.py: 新增 CUTLASS vs Triton benchmark 与 correctness check
```

启用条件(diff 中 fp8.py):

```python
if (
    get_bool_env_var("CUTLASS_MOE")
    and self.cutlass_fp8_supported
    and self.block_quant
    and is_sm100_supported()
):
    return cutlass_fused_experts(...)
```

## 性能声明

条件:DeepSeek V3/R1 配置(E=256, topk=8, H=7168, I_shard=512, bf16, block_shape=[128,128]),`test_cutlass_moe.py` 用 `do_bench_cudagraph`,CUTLASS 4.0 vs Triton,CUDA graph 模式,小 batch 约 30%-40% speedup。部分数字(median,ms):batch 1: 0.106 vs 0.148;batch 4: 0.155 vs 0.207;batch 8: 0.234 vs 0.287;batch 16: 0.350 vs 0.372;batch 32 起反超(0.534 vs 0.532),batch 512: 1.325 vs 0.850——即大 batch 时 Triton 更快。端到端精度:DeepSeek-R1,server `CUTLASS_MOE=1 --enable-dp-attention --tp 8 --dp 8`,GSM8K 8-shot 1319 题,accuracy 0.955,invalid 0.000,latency 477.968 s。

## 验证与回归风险

`test_cutlass_moe.py` 提供 `--check` 模式,断言 CUTLASS 与 Triton 输出最大相对误差 < 5e-1(容差较宽,FP8 常规);GSM8K 精度评估如上。路径默认关闭,需显式设置 `CUTLASS_MOE=1`,且仅 SM100 + CUDA >= 12.8 生效,默认 Triton 路径不受影响。风险点:作者自述仓库当前 CUTLASS 3.9 有 kernel hang 与性能回退问题,需等另开的 CUTLASS 升级 PR;`fp8_blockwise_scaled_grouped_mm` 的 TORCH 签名变更(新增 ptrs/workspace 参数)影响 sgl-kernel ABI。

## 关联

系列 PR 的 [2/2];[1/2](应为 sgl-kernel 侧内核)编号在正文中未注明。同区域相关:FP8 blockwise MoE(Quantization/MoE 分类),CUTLASS 依赖升级 PR 由作者在正文提及但未给编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
