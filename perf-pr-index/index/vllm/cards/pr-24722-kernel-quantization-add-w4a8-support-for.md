# PR #24722: [Kernel][Quantization] add w4a8 support for marlin kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jinzhen-lin |
| 合并 | 2025-11-29 |
| 分类 | Quantization \| tier: major |
| Label | documentation, performance, ready, ci/build, nvidia |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾标注「41 file(s) dropped from diff for size (see PR for full diff)」 |
| 链接 | https://github.com/vllm-project/vllm/pull/24722 |

## 一句话结论

为 marlin / moe marlin kernel 增加 W4A8（INT8/FP8 激活）与 L2 调度优化；正文未给定量性能数据。

## 动机

正文 Introduction：为 marlin 和 moe marlin kernel 增加 W4A8 支持。支持权重 INT4 without ZP（GPTQ）、INT4 with ZP（AWQ）、MXFP4（NVFP4 暂不支持，groupsize 16 小于当前最小 32）；支持激活 INT8（MXFP4 权重不支持）与 FP8 (E4M3)。W8A8-INT8 / W8A8-FP8 当前 kernel 也支持，但 MoE 中仍有性能问题（比默认快、略慢于精细调优的 triton_fused_moe，需调 stages 与 num_threadblocks_per_sm），本 PR 未做该优化。用法：`export VLLM_MARLIN_INPUT_DTYPE=int8`（或 fp8）后 `vllm serve Qwen/Qwen3-30B-A3B-GPTQ-Int4`；不同层用不同激活类型属 future plan，不在本 PR。

## 优化点

新 Weight Layout：为适配 INT8/FP8 的 m16n8k32 mma 指令、让单个 warp 尽量处理同组权重。L2 Cache Optimization：此前 MoE Marlin 沿用 dense Marlin 的 Basic Stream K 调度，大 batch size 下 L2 命中率低、性能差；本 PR 改用 DP + one-tile SK 与 DP + two-tile SK 组合（参考 arXiv 2301.03598）。去量化：INT4（GPTQ/AWQ）→ INT8 用 `dw = ((w | 0x80) - zp) ^ 0x80`；INT4 without ZP → FP8 需 weight reorder 与 sign bit 预处理，dequant 结果乘 512（由 a_scales 乘 512 补偿）；AWQ → FP8 类似。Groupwise Scales：W4A8-INT8 累加器为 INT32、scales 为 BF16/FP16，需频繁 INT32→float 转换；本 PR 把 scales 量化到更高精度 INT 以减少转换开销（示例误差 0.0009，接近 fp16 与 bf16 间 0.0014），潜在 INT32 溢出可通过调整 scales 量化精度缓解，数值模拟显示概率低（示例 2^31 / max = 142.9164）。

## 关键改动

- `csrc/moe/marlin_moe_wna16/marlin_template.h`、`kernel.h`、`generate_kernels.py`：新 weight layout、W4A8 mma 模板（模板参数由 w_type_id/s_type_id 改为 a/b/c/s_type_id，新增 `a_scales_ptr`、`global_scale_ptr`）、去量化与 DP+SK 调度。
- `benchmarks/kernels/benchmark_machete.py`、`benchmark_marlin.py`：`gptq_marlin_gemm` 调用增加 `a_scales=None` 参数适配。
- 正文另说明 `VLLM_MARLIN_INPUT_DTYPE` 控制激活类型；素材 diff 另有 41 file(s) dropped。

```text
       const int4 *__restrict__ A, const int4 *__restrict__ B,                     \
       int4 *__restrict__ C, int4 *__restrict__ C_tmp,                         \
       const int4 *__restrict__ b_bias_ptr,                                    \
+      const float *__restrict__ a_scales_ptr,                                 \
       const int4 *__restrict__ scales_ptr,                                    \
-      const uint16_t *__restrict__ scale2_ptr,                                \
+      const uint16_t *__restrict__ global_scale_ptr,                          \
```

## 性能声明

PR 正文未给出定量性能数据（无加速比、吞吐或延迟的 benchmark 结果；regex 未检出 perf claims）。正文中的数值仅用于说明去量化逻辑与 scales 量化误差（0.0009、0.0014、142.9164），非性能测量。定性描述：W8A8 在 MoE 中"比默认配置快但比精细调优的 triton_fused_moe 略慢"，大 BS 下旧 Stream K 调度曾导致 L2 命中率低。PR 未注明测试条件，正文未给出 Test Plan / Test Result 段落。

## 验证与回归风险

验证：PR 未注明测试命令、测试结果或新增测试（正文无 Test Plan / Test Result 段落），仅给出用法示例与去量化逻辑演示。可能受影响路径：marlin / moe marlin 的 W4A8 与 W8A8 路径、`VLLM_MARLIN_INPUT_DTYPE=int8/fp8` 配置、cudagraph（正文未提及 fake op）；正文自述风险：INT32 累加器可能溢出（数值模拟概率低）、NVFP4 不支持、MXFP4 权重不支持 INT8 激活、W8A8 MoE 性能待调优、scales 量化精度影响。

## 关联

正文提及参考 arXiv 2301.03598（Stream K 论文，非 PR 编号），未给出同分类其它 vLLM PR 编号或前序/后续 PR，无明确线索。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
