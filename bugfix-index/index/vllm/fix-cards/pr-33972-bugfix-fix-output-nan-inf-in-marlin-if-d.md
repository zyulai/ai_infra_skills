# PR #33972: [Bugfix]fix output Nan/Inf in marlin if dtype=float16

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#33972](https://github.com/vllm-project/vllm/pull/33972) |
| 作者 | ir1ka |
| 合并日期 | 2026-03-27 |
| labels | bug, ready |
| 热度 | 👍 1 · 💬 20 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
NVFP4 模型用 dtype=float16 时 marlin GEMM 因 fp16 动态范围小溢出产生 NaN;修复为 GEMM 前动态缩放输入并让 global_scale 走 float32。

## 问题现象
nvfp4 模型支持 `dtype=float16`,但 float16 动态范围比 bfloat16 小,GEMM 中容易溢出导致推理错误。实测(0.16.0rc1.dev78+gb6bb2842c):修复前 dense 模型(/RedHatAI/Qwen3-32B-NVFP4、Qwen3-32B-NVFP4A16、Qwen3-8B-NVFP4)的输入激活出现 NaN,lambada_openai perplexity 为 NaN(acc 恒为 1);MoE 模型(Qwen3-30B-A3B-NVFP4)正常。

## 根因
正文披露:float16 动态范围较小,GEMM 期间数据容易溢出。正文未做更细的机制剖析;从 diff 看,marlin kernel 中 NVFP4 的 global_scale 以输出类型的 `uint16_t`/fp16 表示并在 fp16 累加路径上参与计算,SM75 上 `use_fp16_accum` 对 NVFP4(group_blocks==-1 且 4bit)也生效,fp16 累加放大了溢出风险,修复注释与「动态缩放防止溢出」一致(由 diff 推断)。

## 修复方案
- 修复思路(正文):GEMM 前对输入做动态缩放,防止数据溢出。
- Python 侧 `marlin_utils_fp4.py`:`_nvfp4_compute_scale_factor`/`nvfp4_marlin_process_scales`/`nvfp4_marlin_process_global_scale` 新增 `a_dtype` 参数,`a_dtype` 为 `torch.half` 时不做 rescaling(scale_factor 返回 1.0);weight_global_scale 改存 `torch.float32`,`prepare_fp4_layer_for_marlin`/`prepare_moe_fp4_layer_for_marlin` 全部传入 `param_dtype`。
- CUDA 侧(csrc/quantization/marlin 与 csrc/moe/marlin_moe_wna16):kernel 的 `global_scale_ptr` 从 `uint16_t*` 改为 `float*`,kernel 内用 `float global_scale_f32` 在写回时(float 域)乘 global_scale;SM75 上 `use_fp16_accum` 对 NVFP4 及 group_size==-1 且 4bit 的情况禁用;host 侧断言改为 global_scale 必须是 float;`marlin_gemm` 的架构要求从 CUDA_ARCH >= 8.0 放宽到 >= 7.5,`activation_kernels.cu` 的 `clip` 去掉空实现分支。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围(测试基于 0.16.0rc1.dev78+gb6bb2842c)。
- 回归面:所有 marlin/fp4 量化路径(nvfp4、mxfp4, dense 与 MoE)的权重预处理与 kernel 签名;global_scale 张量 dtype 变化会波及加载与 kernel 调用方。修复后 fp16 结果接近 bf16 基线(Qwen3-32B-NVFP4 ppl 3.8613 vs bf16 3.8664)。

## 关联
Fixes: https://github.com/vllm-project/vllm/issues/33560 , https://github.com/vllm-project/vllm/issues/33461

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
