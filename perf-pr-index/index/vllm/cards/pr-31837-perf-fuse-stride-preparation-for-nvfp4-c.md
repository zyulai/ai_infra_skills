# PR #31837: [Perf] Fuse stride preparation for NVFP4 cutlass_moe

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2026-01-07 |
| 分类 | MoE(major) |
| Label | performance, ready, nvidia |
| Diff | diffstat 缺失(diff 全文已提供,未附行数统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/31837 |

## 一句话结论

把 NVFP4 cutlass MoE 的 3 个 per-expert stride 填充 kernel(`torch::full`)并入已有的 group GEMM 起始地址 kernel,小批量延迟改善约 5%。

## 动机

对另一 PR 做性能剖析时发现,调用 `ops.cutlass_fp4_moe_mm` 时总有 3 个 `vectorized_elementwise_kernel`,来自为 group GEMM 准备 per-expert stride 数组的 3 个 `torch::full` 构造。这些 stride 对所有 expert 是同一常量,而 `__get_group_gemm_starts` kernel 本来就每 expert 启动一次去填写指针,完全可以顺带写入,省掉 3 次独立 launch。

## 优化点

`nvfp4_blockwise_moe_kernel.cu` 中 `__get_group_gemm_starts` 的签名扩展 `a_strides/b_strides/c_strides` 及对应常量值;kernel 在写 expert 指针、scale 指针与 alpha 指针的同一循环里,用常量直接初始化三个 stride 数组。主机侧 `run_fp4_blockwise_scaled_group_mm_sm100` 与 `run_fp4_blockwise_scaled_group_mm_sm120` 两条路径(SM100 与 SM120)都把 `torch::full` 换成 `torch::empty`,stride 值(`a.stride(0)*2`、`b.stride(1)*2`、`output.stride(0)`)作为 kernel 参数传入。每次 MoE 调用减少 3 个小 kernel 的 launch 与执行。

## 关键改动

```text
csrc/quantization/fp4/nvfp4_blockwise_moe_kernel.cu: __get_group_gemm_starts 增写 stride;sm100/sm120 两条 host 路径去 torch::full
```

kernel 内新增片段:

```cuda
// Initialize strides (constant across all experts, avoids separate kernels)
a_strides[expert_id] = a_stride_val;
b_strides[expert_id] = b_stride_val;
c_strides[expert_id] = c_stride_val;
```

## 性能声明

模型 `nvidia/Qwen3-30B-A3B-NVFP4`,GPU 型号 PR 未注明(diff 涉及 sm100/sm120 kernel)。基线含 #31832(silu_and_mul 融合):

- 延迟型(input-len 100、output-len 100、num-prompts 8):Mean TPOT 7.53→7.14 ms,Mean ITL 7.53→7.14 ms,输出吞吐 1025.21→1079.23 tok/s;PR 称整体 latency 提升约 5%
- 吞吐型(同 len,num-prompts 512):输出吞吐 20044.52→20188.84 tok/s,变化很小(PR 称对吞吐影响更小)
- GSM8K eval:准确率 0.883→0.886,invalid 0.000→0.001

## 验证与回归风险

PR 未列新增单元测试(Test Plan/Test Result 空缺),验证为 serving benchmark 与 GSM8K eval。风险面:改动 sm100 与 sm120 两条 NVFP4 group GEMM 路径,stride 值从 host 传入 kernel,若 expert 数超过 kernel 线程配置(`gridDim.x * blockDim.x`)则对应条目未初始化——原 `torch::full` 无此约束;`torch::empty` 替代后依赖 kernel 全量写满数组。

## 关联

前序:#31832(NVFP4 cutlass MoE 融合 silu_and_mul,本 PR 基于其基线测得);同区域为 NVFP4 cutlass_moe launch 数量削减系列。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
