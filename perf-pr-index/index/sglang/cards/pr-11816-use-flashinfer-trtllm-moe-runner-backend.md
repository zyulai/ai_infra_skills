# PR #11816: use flashinfer_trtllm moe runner backend to gain around 10% perf on b200 fp8 dpsk

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @b8zhong |
| 合并 | 2025-10-24 |
| 分类 | MoE(major) |
| Label | ready-to-merge, run-ci |
| Diff | diffstat 缺失(素材未提供统计);diff 实际涉及 3 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/11816 |

## 一句话结论

B200 FP8 DeepSeek 上把默认 MoE runner 从 triton 切到 flashinfer_trtllm,约 10% 性能收益。

## 动机

SM100 + FP8 的 DeepSeek(dpsk)此前默认用 triton MoE runner;FP4 路径已默认 flashinfer_trtllm。作者给出 E2E 对比截图称 FP8 也有约 10% 收益;同时发现即使为 triton 调优 config(kernel 层约 +5%),E2E 仍不如 flashinfer_trtllm。另外 `SGLANG_CUTLASS_MOE` 是唯一游离在 server args 之外的 MoE 后端开关,需要收编为 `--moe-runner-backend cutlass`。

## 优化点

三件事:(1) `server_args.py` 的模型特定调整中,DeepseekV3ForCausalLM 在 sm100 且 `moe_runner_backend == "auto"` 时一律默认 `flashinfer_trtllm`(原先仅 modelopt_fp4 才触发);若用户未显式指定 quantization,则默认为 `fp8` 以满足 flashinfer_trtllm MoE runner 的断言。注意 shared experts fusion 在此路径被禁用。(2) 加入调优后的 triton config(kernel 层 +5%,作为对照保留)。(3) 删除 `Fp8MoEMethod.use_cutlass_fused_experts_fp8` 的环境变量判定,改为 `_should_use_cutlass_fused_experts()`:优先 `SGLANG_CUTLASS_MOE` 强制(标 TODO 未来移除),否则由 `--moe-runner-backend cutlass` + cutlass 支持 + block quant + SM90/100 决定;cutlass 缓冲区从 create_weights 移入 `_ensure_cutlass_buffers_initialized()` 懒初始化。

## 关键改动

```text
python/sglang/srt/server_args.py: DeepSeek sm100 默认 moe runner = flashinfer_trtllm,quant 未指定时默认 fp8
python/sglang/srt/layers/quantization/fp8.py: cutlass 路径判定改后端驱动,缓冲区懒初始化
python/sglang/srt/layers/moe/fused_moe_triton/layer.py: 判定改用 _should_use_cutlass_fused_experts()
```

```python
if self.moe_runner_backend == "auto":
    self.moe_runner_backend = "flashinfer_trtllm"
    logger.info(
        "Use flashinfer_trtllm as MoE runner backend on sm100 for DeepseekV3ForCausalLM"
    )
    if self.quantization is None:
        self.quantization = "fp8"
```

## 性能声明

- 标题口径:flashinfer_trtllm 对比 triton,约 10% 性能提升,条件为 B200、FP8、DeepSeek(dpsk)。E2E 具体数字在 PR 正文中以截图和 Slack 链接给出,本卡素材无法提取,原始表格数字未注明。
- 作者补充:调优后的 triton config 使 kernel 本身约 +5%,但 E2E(用调优 triton)仍更差。
- Accuracy:gsm8k(1319 题,8-shot)Accuracy 0.952,Invalid 0.001。

PR 未注明 token 长度、并发等 benchmark 细节。

## 验证与回归风险

gsm8k 准确性验证通过(0.952)。回归风险面较大:所有 DeepSeek 系列模型在 sm100 + auto 后端时默认行为改变(量化未指定还会被隐式设为 fp8);shared experts fusion 在 flashinfer_trtllm 路径被禁用,依赖该融合的配置会回退;`SGLANG_CUTLASS_MOE` 环境变量仍可强制 cutlass 路径但已计划移除;cutlass 缓冲区改为懒初始化,改变初始化时序。

## 关联

同区域:FP4 路径此前已默认 flashinfer_trtllm(本 PR 将 FP8 对齐);前序依赖 #4278/#4376 一类 SM100 FP8 工作未被本 PR 正文提及,故不列。关联 PR 编号素材未提供,写无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
