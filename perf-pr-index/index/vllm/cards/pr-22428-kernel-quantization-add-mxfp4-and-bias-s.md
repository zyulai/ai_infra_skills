# PR #22428: [Kernel] [Quantization] Add MXFP4 and bias support for marlin kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jinzhen-lin |
| 合并 | 2025-08-14 |
| 分类 | Quantization（tier: major） |
| Label | performance, ready, ci/build |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files); pack 末尾标注 27 file(s) dropped from diff for size |
| 链接 | https://github.com/vllm-project/vllm/pull/22428 |

## 一句话结论

marlin kernel 加 MXFP4 与 bias 支持：gpt-oss-20b/A100/TP1 bs128 吞吐 3533→5887.8 tok/s。

## 动机

PR 正文极简：为 marlin kernel 增加 MXFP4 与 bias 支持，目标是 gpt-oss 模型（"This pr add mxfp4 and bias support for marlin kernel. (for gpt-oss)"）。正文未展开改动前的瓶颈，只给出 marlin 与 triton 两条实现在三组硬件/模型配置下的吞吐对比。

## 优化点

从 diff 可见的机制改动：在 `csrc/core/scalar_type.hpp` 新增 `kFE8M0fnu`（8 位指数、无尾数的 E8M0 无符号 scale 类型）；`csrc/moe/marlin_moe_wna16/generate_kernels.py` 为 MXFP4（`kFE2M1f`）增加 `s_type_id` 模板参数并选择 scale 类型——`group_blocks==1` 用 `kFE4M3fn`、`group_blocks==2`（对应 group_size=32）用 `kFE8M0fnu`，且 fp16 下因无法安全把 e8m0 反量化而跳过；同时注明 MXFP4 只支持 group_size=32。`csrc/moe/marlin_moe_wna16/kernel.h` 的 `MARLIN_KERNEL_PARAMS` 增加 `b_bias_ptr` 与 `has_bias`，`marlin_template.h` 同步接收 bias；`benchmarks/kernels/benchmark_machete.py` 的 marlin 调用补 `b_bias=None`。pack 末尾标注 27 个文件因体积被从 diff 中丢弃，其余改动未在素材中完整展示。

## 关键改动

```text
csrc/core/scalar_type.hpp: 新增 kFE8M0fnu（E8M0 无符号 scale 标量类型）
csrc/moe/marlin_moe_wna16/generate_kernels.py: 为 MXFP4 选 s_type（kFE4M3fn / kFE8M0fnu）并把 s_type_id 加入 kernel 模板
csrc/moe/marlin_moe_wna16/kernel.h 与 marlin_template.h: MARLIN_KERNEL_PARAMS 增 b_bias_ptr 与 has_bias，支持 bias
benchmarks/kernels/benchmark_machete.py: marlin_create_bench_fn 调用补 b_bias=None
```
```text
            if scalar_type == "vllm::kFE2M1f" and group_blocks == 1:
                s_type = "vllm::kFE4M3fn"
            elif scalar_type == "vllm::kFE2M1f" and group_blocks == 2:
                s_type = "vllm::kFE8M0fnu"
                if dtype == "fp16":
                    # we cannot safely dequantize e8m0 to fp16, so skip this
                    continue
```

## 性能声明

正文给出三张 tokens/s 表，指标为从 vllm 日志拷贝的 `Avg generation throughput`（正文 Note: "not benchmark test, just copy the `Avg generation throughput:` value from vllm log"）。gpt-oss-20b + RTX4090 + TP1：bs1 triton 97.9 / marlin 97.8，bs32 1513.5 / 1769.5，bs64 2380.4 / 3142，bs128 3659.5 / 4670.6。gpt-oss-20b + A100 + TP1：bs1 124.4 / 125.8，bs32 1395 / 2367.7，bs64 2349.1 / 3756.5，bs128 3533 / 5887.8。gpt-oss-120b + A100 + TP2：bs1 110.4 / 117.5，bs32 1280 / 2054.2，bs64 1868.7 / 3052.7，bs128 2777.8 / 4172.4。测试条件：模型 gpt-oss-20b / gpt-oss-120b，硬件 RTX4090 / A100，TP1 / TP2，batch size 1-128，指标 tokens/s；PR 未注明精度。

## 验证与回归风险

PR 正文未给出专门的正确性测试命令或测试结果，只提供上述吞吐表；diff 中可见 `benchmark_machete.py` 补 `b_bias=None` 以适配新参数。回归风险：改动作用于 marlin MoE wna16 kernel 与 gpt-oss 量化路径；MXFP4 仅支持 group_size=32，fp16 下 e8m0 反量化被跳过；正文明确数字来自日志中的 `Avg generation throughput`、非严格 benchmark 测试；pack 标注 27 个文件因体积从 diff 丢弃，完整改动未在素材中呈现。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
